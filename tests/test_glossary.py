# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Tier 9 — Business Glossary tools.

These run against a routed fake Viya built on :class:`httpx.MockTransport`
rather than an ``AsyncMock`` client, because most of what this tier does is
choose the *right request*: the representation that carries ``resourceId``, the
``Accept-Item`` that carries relationship endpoints, the ``publish`` flag, the
batched ``in(...)`` filter. A mock that answers every call identically cannot
tell a correct request from a wrong one, so the fake dispatches on path and
records every request for the tests to assert against.

The fixture data mirrors a real deployment: a term type whose attributes are
keyed by UUID, a term whose glossary id differs from its catalog entity id, and
``glossaryTermAsset`` relationships with the term on ``endpoint1``.
"""

import json
import re
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastmcp import Client, FastMCP

from sas_mcp_server import viya_client
from sas_mcp_server.helpers import glossary_helpers as gh
from sas_mcp_server.tools import glossary

pytestmark = pytest.mark.asyncio

VIYA = "https://test.viya.com"

# --- fixture data ------------------------------------------------------------

TERM_TYPE_ID = "tt-0001"
TERM_TYPE = {
    "id": TERM_TYPE_ID,
    "name": "BCBS239",
    "label": "BCBS239",
    "description": "Risk data aggregation terms.",
    "usageCount": 3,
    "attributeCount": 4,
    "allowCustomAttributes": False,
    "attributes": [
        {"name": "attr-scope", "label": "Scope", "type": "single-select",
         "required": True, "items": ["Local", "Group"]},
        {"name": "attr-risk", "label": "Used in Risk", "type": "boolean",
         "defaultValue": "false"},
        {"name": "attr-region", "label": "Regions", "type": "multi-select",
         "items": ["EMEA", "APAC", "AMER"]},
        {"name": "attr-asof", "label": "As of", "type": "date"},
        {"name": "attr-seen", "label": "Last seen", "type": "date-time"},
        {"name": "attr-cutoff", "label": "Cutoff", "type": "time"},
        {"name": "attr-oper", "label": "Operational field", "type": "single-line"},
        {"name": "attr-note", "label": "Notes", "type": "multi-line"},
    ],
}

# A term's two identities: the glossary id and the catalog entity id differ.
TERM_ID = "gterm-1111"
TERM_ENTITY_ID = "cent-2222"
TERM = {
    "id": TERM_ID,
    "name": "Currency",
    "label": "Currency",
    "definition": "The ISO 4217 currency of the exposure.",
    "description": "Currency",
    "termTypeId": TERM_TYPE_ID,
    "termTypeLabel": "BCBS239",
    "parentId": None,
    "status": "Published",
    "isDraft": False,
    "assetCount": 1,
    "version": 3,
    "createdBy": "author",
    "modifiedBy": "author",
    "modifiedTimeStamp": "2026-06-11T12:02:35.602Z",
    # Keyed by attribute-definition UUID, and carrying the empty values the
    # glossary stores for every declared attribute.
    # Viya stores a boolean as a JSON boolean, and a multi-select as one
    # comma-joined string — both verified live against a real term type.
    "attributes": {
        "attr-scope": "Group",
        "attr-risk": True,
        "attr-region": "EMEA,APAC",
        "attr-oper": "",
        "attr-note": "",
    },
    "links": [{"rel": "self", "href": f"/glossary/terms/{TERM_ID}"}],
}

TABLE_ID = "cent-table"
TABLE_RESOURCE = "/dataTables/dataSources/Compute~fs~abc~fs~PUBLIC/tables/BCBS_SOURCE"
TABLE_ENTITY = {
    "id": TABLE_ID,
    "name": "BCBS_SOURCE",
    "type": "sasTable",
    "resourceId": TABLE_RESOURCE,
    "attributes": {"rowCount": 100},
}
COLUMN_CURR = {
    "id": "cent-col-curr",
    "name": "CURR_CD",
    "type": "sasColumn",
    "resourceId": f"{TABLE_RESOURCE}/columns/CURR_CD",
    "attributes": {"dataType": "string"},
}
COLUMN_BAL = {
    "id": "cent-col-bal",
    "name": "BAL_AMT",
    "type": "sasColumn",
    "resourceId": f"{TABLE_RESOURCE}/columns/BAL_AMT",
    "attributes": {"dataType": "double"},
}
TERM_ENTITY = {
    "id": TERM_ENTITY_ID,
    "name": "Currency",
    "type": "glossaryTerm",
    "description": TERM["definition"],
    "resourceId": f"/glossary/terms/{TERM_ID}",
    "attributes": {"assetCount": 1, "status": "Published"},
}
# endpoint1 is the term, endpoint2 the asset — the direction the module relies on.
RELATIONSHIP = {
    "id": "rel-9999",
    "instanceType": "relationship",
    "definition": "glossaryTermAsset",
    "endpoint1Id": TERM_ENTITY_ID,
    "endpoint2Id": COLUMN_CURR["id"],
}

# A table the catalog knows about but whose columns it never indexed — the
# shape that makes every column name look wrong.
EMPTY_TABLE_RESOURCE = "/dataTables/dataSources/Compute~fs~abc~fs~PUBLIC/tables/NOT_INDEXED"
EMPTY_TABLE = {
    "id": "cent-empty-table",
    "name": "NOT_INDEXED",
    "type": "sasTable",
    "resourceId": EMPTY_TABLE_RESOURCE,
    "attributes": {},
}

_ENTITIES = {
    e["id"]: e
    for e in (TERM_ENTITY, TABLE_ENTITY, COLUMN_CURR, COLUMN_BAL, EMPTY_TABLE)
}


class FakeViya:
    """A routed stand-in for the Viya REST API.

    ``requests`` records every call so a test can assert on the headers and
    query parameters the tier actually sent, not merely on what it returned.
    """

    def __init__(self, **overrides: Any) -> None:
        self.requests: list[httpx.Request] = []
        self.overrides = overrides
        self.deleted: list[str] = []
        self.posted: list[dict[str, Any]] = []
        self.put_bodies: list[dict[str, Any]] = []
        self.relationships: list[dict[str, Any]] = [RELATIONSHIP]

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _json(payload: Any, status: int = 200) -> httpx.Response:
        return httpx.Response(status, json=payload)

    @staticmethod
    def _ids_in(filter_expr: str, field: str) -> list[str]:
        """Extract the ids from an ``in(field,'a','b')`` clause."""
        match = re.search(rf"in\({field},((?:'[^']*',?)+)\)", filter_expr or "")
        if not match:
            return []
        return re.findall(r"'([^']*)'", match.group(1))

    def _instances(self, request: httpx.Request) -> httpx.Response:
        expr = request.url.params.get("filter", "")
        item_media = request.headers.get("Accept-Item", "")

        if "glossaryTermAsset" in expr:
            wanted = set(self._ids_in(expr, "endpoint1Id")) | set(self._ids_in(expr, "endpoint2Id"))
            hits = [
                rel
                for rel in self.relationships
                if {rel["endpoint1Id"], rel["endpoint2Id"]} & wanted
            ]
            # Endpoints only exist in the relationship representation; anything
            # else gets the stripped summary a real server returns.
            if "relationship+json" not in item_media:
                hits = [{k: v for k, v in h.items() if not k.startswith("endpoint")} for h in hits]
            # A real collection honours start/limit and reports the unpaged
            # total as ``count``; the tier pages on exactly that, so the fake
            # has to behave the same or the paging tests prove nothing.
            total = len(hits)
            offset = int(request.url.params.get("start", 0))
            hits = hits[offset : offset + int(request.url.params.get("limit", 100))]
            return self._json({"items": hits, "count": total})

        matched: list[dict[str, Any]] = []
        if expr.startswith("in(id,"):
            matched = [_ENTITIES[i] for i in self._ids_in(expr, "id") if i in _ENTITIES]
        elif expr.startswith("in(resourceId,"):
            wanted = set(self._ids_in(expr, "resourceId"))
            matched = [e for e in _ENTITIES.values() if e.get("resourceId") in wanted]
        elif expr.startswith("eq(resourceId,"):
            wanted = expr[len("eq(resourceId,'") : -2]
            matched = [e for e in _ENTITIES.values() if e.get("resourceId") == wanted]
        elif expr.startswith("startsWith(resourceId,"):
            prefix = expr[len("startsWith(resourceId,'") : -2]
            matched = [
                e for e in _ENTITIES.values() if (e.get("resourceId") or "").startswith(prefix)
            ]
        # resourceId lives only in the entity representation.
        if "entity+json" not in item_media:
            matched = [{k: v for k, v in m.items() if k != "resourceId"} for m in matched]
        return self._json({"items": matched, "count": len(matched)})

    # -- dispatch ---------------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method

        if (override := self.overrides.get(f"{method} {path}")) is not None:
            return override if isinstance(override, httpx.Response) else self._json(override)

        if method == "GET" and path == "/glossary/termTypes":
            return self._json({"items": [TERM_TYPE], "count": 1})
        if method == "GET" and path.startswith("/glossary/termTypes/"):
            return self._json(TERM_TYPE)
        if method == "GET" and path == "/glossary/terms":
            return self._json({"items": [TERM], "count": 1})
        if method == "GET" and path.startswith("/glossary/terms/"):
            return self._json(TERM)
        if method == "POST" and path == "/glossary/termTypes":
            body = json.loads(request.content)
            self.posted.append({"path": path, "body": body, "params": dict(request.url.params)})
            return self._json({**body, "id": "tt-new", "usageCount": 0}, 201)
        if method == "PUT" and path.startswith("/glossary/termTypes/"):
            body = json.loads(request.content)
            self.put_bodies.append(body)
            return self._json({**body, "version": 2})
        if method == "POST" and path == "/glossary/terms":
            body = json.loads(request.content)
            self.posted.append({"path": path, "body": body, "params": dict(request.url.params)})
            return self._json({**TERM, **body, "id": "gterm-new", "status": "Published"}, 201)
        if method == "PUT" and path.startswith("/glossary/terms/"):
            body = json.loads(request.content)
            self.put_bodies.append(body)
            return self._json({**body, "version": TERM["version"] + 1})
        if method == "DELETE":
            self.deleted.append(path)
            return httpx.Response(204)
        if method == "GET" and path == "/catalog/instances":
            return self._instances(request)
        if method == "POST" and path == "/catalog/instances":
            body = json.loads(request.content)
            self.posted.append({"path": path, "body": body, "params": dict(request.url.params)})
            created = {**body, "id": "rel-new"}
            self.relationships.append(created)
            return self._json(created, 201)
        if method == "GET" and path == "/catalog/search":
            return self._json(self.overrides.get("search", {"items": [], "count": 0}))
        return self._json({"message": f"unrouted {method} {path}"}, 404)


@asynccontextmanager
async def glossary_client(fake: FakeViya):
    """An MCP client whose glossary tools talk to *fake* instead of Viya."""
    transport = httpx.MockTransport(fake.handler)

    def make_client(token: str | None):  # noqa: ARG001 - signature parity
        return httpx.AsyncClient(transport=transport, base_url=VIYA)

    mcp = FastMCP("glossary-test")

    async def get_token(ctx):  # noqa: ARG001
        return "test-token"

    import sas_mcp_server.tools._common as common

    original = common.make_client
    common.make_client = make_client
    try:
        glossary.register(mcp, get_token)
        async with Client(mcp) as client:
            yield client
    finally:
        common.make_client = original


def result_of(call) -> dict[str, Any]:
    """The structured payload of a FastMCP tool result."""
    return call.data if call.data is not None else json.loads(call.content[0].text)


@pytest.fixture(autouse=True)
def _pin_endpoint(monkeypatch):
    """Pin VIYA_ENDPOINT so request URLs are predictable regardless of .env."""
    monkeypatch.setattr(glossary, "VIYA_ENDPOINT", VIYA)
    import sas_mcp_server.viya_client as viya_client

    monkeypatch.setattr(viya_client, "VIYA_ENDPOINT", VIYA)


# --- pure helpers (helpers/glossary_helpers.py) -------------------------------------------------------------


def test_quote_doubles_single_quotes():
    assert viya_client.filter_literal("O'Brien") == "O''Brien"
    assert viya_client.filter_literal("") == ""


def test_in_filter_escapes_each_value():
    assert viya_client.in_filter("id", ["a", "b'c"]) == "in(id,'a','b''c')"


def test_chunks_splits_at_the_configured_size():
    values = [str(i) for i in range(95)]
    chunks = gh.chunk_ids(values)
    assert [len(c) for c in chunks] == [40, 40, 15]
    assert [v for c in chunks for v in c] == values


def test_glossary_id_is_read_from_the_resource_id():
    assert gh.glossary_id_from_resource("/glossary/terms/abc-123") == "abc-123"
    # A table's resourceId must not be mistaken for a term's.
    assert gh.glossary_id_from_resource("/dataTables/x/tables/T") is None
    assert gh.glossary_id_from_resource(None) is None


def test_readable_attributes_names_keys_and_drops_empties():
    by_uuid, _ = gh.attribute_maps(TERM_TYPE)
    readable = gh.readable_attributes(TERM["attributes"], by_uuid)
    assert readable == {
        "Scope": "Group",
        "Used in Risk": True,
        # Stored comma-joined; handed back as the list the caller passed in.
        "Regions": ["EMEA", "APAC"],
    }


def test_readable_attributes_keeps_unknown_uuids():
    """A type edited after the term was written must not lose the term's data."""
    by_uuid, _ = gh.attribute_maps(TERM_TYPE)
    readable = gh.readable_attributes({"attr-gone": "value"}, by_uuid)
    assert readable == {"attr-gone": "value"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("FALSE", False)],
)
def test_boolean_attributes_are_encoded_as_json_booleans(value, expected):
    """Viya rejects the string "true": the value has to stay a real bool."""
    definition = {"label": "Used in Risk", "type": "boolean"}
    encoded = gh.encode_attribute(value, definition)
    assert encoded is expected
    assert isinstance(encoded, bool)


def test_boolean_attribute_rejects_a_non_boolean():
    with pytest.raises(ValueError, match="is a boolean"):
        gh.encode_attribute("yes", {"label": "Used in Risk", "type": "boolean"})


def test_single_select_rejects_a_value_outside_the_allowed_list():
    definition = {"label": "Scope", "type": "single-select", "items": ["Local", "Group"]}
    with pytest.raises(ValueError, match=r"only accepts \['Local', 'Group'\]"):
        gh.encode_attribute("Regional", definition)



# --- attribute wire formats ---------------------------------------------------
# Each format below was established against a live glossary; the API publishes no
# schema, and every one of these types rejects all but one spelling.


def test_multi_select_joins_a_list_the_way_viya_stores_it():
    """A JSON array and "a, b" are both rejected; only "a,b" is accepted."""
    definition = {"label": "Regions", "type": "multi-select", "items": ["EMEA", "APAC"]}
    assert gh.encode_attribute(["EMEA", "APAC"], definition) == "EMEA,APAC"


def test_multi_select_accepts_the_stored_string_and_strips_spaces():
    definition = {"label": "Regions", "type": "multi-select", "items": ["EMEA", "APAC"]}
    assert gh.encode_attribute("EMEA, APAC", definition) == "EMEA,APAC"


def test_multi_select_names_the_items_that_are_not_allowed():
    """Viya's own rejection names the attribute but never which item was wrong."""
    definition = {"label": "Regions", "type": "multi-select", "items": ["EMEA", "APAC"]}
    with pytest.raises(ValueError) as excinfo:
        gh.encode_attribute(["EMEA", "ANTARCTICA"], definition)
    assert "ANTARCTICA" in str(excinfo.value)


def test_multi_select_round_trips_through_decode():
    definition = {"label": "Regions", "type": "multi-select", "items": ["EMEA", "APAC"]}
    stored = gh.encode_attribute(["EMEA", "APAC"], definition)
    assert gh.decode_attribute(stored, definition) == ["EMEA", "APAC"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-04T13:41:24Z", "2026-09-04T13:41:24Z"),
        ("2026-09-04T13:41:24", "2026-09-04T13:41:24Z"),  # Z is mandatory
        ("2026-09-04T13:41", "2026-09-04T13:41:00Z"),  # seconds are mandatory
        ("2026-09-04", "2026-09-04T00:00:00Z"),  # a bare date is rejected as-is
        ("2026-09-04T15:41:24+02:00", "2026-09-04T13:41:24Z"),  # offsets are rejected
        ("2026-09-04T11:41:24-02:00", "2026-09-04T13:41:24Z"),
    ],
)
def test_date_time_is_normalised_to_utc(value, expected):
    assert gh.encode_attribute(value, {"label": "Last seen", "type": "date-time"}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("15:41:28Z", "15:41:28Z"), ("15:41:28", "15:41:28Z"), ("15:41", "15:41:00Z")],
)
def test_time_gets_its_seconds_and_z(value, expected):
    assert gh.encode_attribute(value, {"label": "Cutoff", "type": "time"}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2026-09-04", "2026-09-04"), ("2026-09-04T13:41:24Z", "2026-09-04")],
)
def test_date_keeps_only_the_date(value, expected):
    assert gh.encode_attribute(value, {"label": "As of", "type": "date"}) == expected


@pytest.mark.parametrize(
    ("attr_type", "value"),
    [("date", "31/12/2026"), ("time", "9am"), ("date-time", "next tuesday")],
)
def test_unparseable_dates_are_rejected_with_the_format_named(attr_type, value):
    """Viya answers these with a bare 400; the caller needs the format instead."""
    with pytest.raises(ValueError) as excinfo:
        gh.encode_attribute(value, {"label": "X", "type": attr_type})
    assert attr_type in str(excinfo.value)


def test_a_required_attribute_set_to_empty_counts_as_missing():
    """Viya rejects "" for a required attribute, so presence alone is not enough."""
    _, by_label = gh.attribute_maps(TERM_TYPE)
    with pytest.raises(ValueError, match="requires attribute"):
        gh.encode_attributes({"Scope": ""}, by_label, require_all=True)


def test_missing_required_finds_the_gap_in_a_merged_term():
    _, by_label = gh.attribute_maps(TERM_TYPE)
    assert gh.missing_required({"attr-scope": "Group"}, by_label) == []
    assert gh.missing_required({"attr-scope": ""}, by_label) == ["Scope"]
    assert gh.missing_required({}, by_label) == ["Scope"]

def test_encode_attributes_maps_labels_case_insensitively():
    _, by_label = gh.attribute_maps(TERM_TYPE)
    encoded = gh.encode_attributes(
        {"scope": "Local", "USED IN RISK": True}, by_label, require_all=False
    )
    assert encoded == {"attr-scope": "Local", "attr-risk": True}


def test_encode_attributes_names_the_valid_labels_for_an_unknown_one():
    _, by_label = gh.attribute_maps(TERM_TYPE)
    with pytest.raises(ValueError) as excinfo:
        gh.encode_attributes({"Scop": "Local"}, by_label, require_all=False)
    message = str(excinfo.value)
    assert "unknown attribute 'Scop'" in message
    assert "Scope" in message  # the correction is in the message


def test_encode_attributes_requires_mandatory_attributes_on_create():
    _, by_label = gh.attribute_maps(TERM_TYPE)
    with pytest.raises(ValueError, match=r"requires attribute\(s\) \['Scope'\]"):
        gh.encode_attributes({"Notes": "x"}, by_label, require_all=True)


def test_encode_attributes_does_not_require_them_on_update():
    """An update merges onto what exists, so a required attribute is already set."""
    _, by_label = gh.attribute_maps(TERM_TYPE)
    assert gh.encode_attributes({"Notes": "x"}, by_label, require_all=False) == {
        "attr-note": "x"
    }


# --- term types ---------------------------------------------------------------


async def test_list_glossary_term_types_flattens_the_summary():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("list_glossary_term_types", {}))
    assert result["count"] == 1
    assert result["items"][0] == {
        "term_type_id": TERM_TYPE_ID,
        "name": "BCBS239",
        "label": "BCBS239",
        "description": "Risk data aggregation terms.",
        "usage_count": 3,
        "attribute_count": 4,
    }


async def test_get_glossary_term_type_publishes_the_authoring_contract():
    """The attribute contract is what makes create_glossary_term usable."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool("get_glossary_term_type", {"term_type_id": TERM_TYPE_ID})
        )
    scope = next(a for a in result["attributes"] if a["label"] == "Scope")
    assert scope["required"] is True
    assert scope["allowed_values"] == ["Local", "Group"]
    assert scope["type"] == "single-select"
    risk = next(a for a in result["attributes"] if a["label"] == "Used in Risk")
    assert risk["required"] is False


async def test_term_type_is_resolvable_by_name():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool("get_glossary_term_type", {"term_type_id": "bcbs239"})
        )
    assert result["term_type_id"] == TERM_TYPE_ID


async def test_unknown_term_type_name_lists_the_available_ones():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        with pytest.raises(Exception, match="no term type named 'Nope'"):
            await client.call_tool("get_glossary_term_type", {"term_type_id": "Nope"})


# --- finding terms ------------------------------------------------------------


async def test_search_resolves_the_catalog_hit_to_a_glossary_term_id():
    """A search hit carries the *catalog* id; every other tool needs the glossary id."""
    fake = FakeViya(
        search={
            "count": 1,
            "start": 0,
            "items": [
                {
                    "id": TERM_ENTITY_ID,
                    "name": "Currency",
                    "typeLabel": "BCBS239",
                    "score": 12.5,
                    "attributes": {
                        "definition": TERM["definition"],
                        "reviewStatus": "Published",
                        "assignedAssets": 1,
                    },
                }
            ],
        }
    )
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("search_glossary_terms", {"query": "currency"}))
    item = result["items"][0]
    assert item["term_id"] == TERM_ID
    assert item["catalog_entity_id"] == TERM_ENTITY_ID
    assert item["assigned_asset_count"] == 1
    assert result["note"] == ""


async def test_search_uses_the_plural_terms_index():
    """The singular 'term' is rejected by the catalog with a 400."""
    fake = FakeViya(search={"count": 0, "items": []})
    async with glossary_client(fake) as client:
        await client.call_tool("search_glossary_terms", {"query": "*"})
    search = next(r for r in fake.requests if r.url.path == "/catalog/search")
    assert search.url.params["indices"] == "terms"


async def test_search_flags_a_count_that_exceeds_the_readable_items():
    """The index count is pre-authorization and can outrun what comes back."""
    fake = FakeViya(search={"count": 9, "start": 0, "items": []})
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("search_glossary_terms", {"query": "*"}))
    assert "may exceed the readable items" in result["note"]


async def test_get_glossary_term_names_its_attributes():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("get_glossary_term", {"term_id": TERM_ID}))
    assert result["attributes"] == {
        "Scope": "Group",
        "Used in Risk": True,
        "Regions": ["EMEA", "APAC"],
    }
    assert result["attribute_ids"] == TERM["attributes"]  # raw map preserved
    assert result["term_id"] == TERM_ID
    assert result["catalog_entity_id"] == TERM_ENTITY_ID


async def test_get_glossary_term_asks_for_the_representation_with_resource_id():
    """Without the entity representation the bridge silently yields no id."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool("get_glossary_term", {"term_id": TERM_ID})
    bridge = [
        r
        for r in fake.requests
        if r.url.path == "/catalog/instances" and "resourceId" in r.url.params.get("filter", "")
    ]
    assert bridge, "no bridge lookup was made"
    assert bridge[0].headers["Accept-Item"] == glossary._ENTITY_MEDIA


async def test_list_glossary_terms_builds_a_conjunction_of_filters():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "list_glossary_terms",
            {"term_type": "BCBS239", "parent_id": "p-1", "name_contains": "Cur"},
        )
    listing = [
        r
        for r in fake.requests
        if r.url.path == "/glossary/terms" and "parentId" in r.url.params.get("filter", "")
    ][0]
    expr = listing.url.params["filter"]
    assert expr.startswith("and(")
    assert f"eq(termTypeId,'{TERM_TYPE_ID}')" in expr
    assert "eq(parentId,'p-1')" in expr
    assert "contains(name,'Cur')" in expr


async def test_list_glossary_terms_excludes_drafts_by_default():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool("list_glossary_terms", {})
        await client.call_tool("list_glossary_terms", {"include_drafts": True})
    calls = [r for r in fake.requests if r.url.path == "/glossary/terms"]
    assert calls[0].url.params["allowDrafts"] == "none"
    assert calls[1].url.params["allowDrafts"] == "all"


async def test_a_single_filter_is_not_wrapped_in_and():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool("list_glossary_terms", {"parent_id": "p-1"})
    listing = [r for r in fake.requests if r.url.path == "/glossary/terms"][0]
    assert listing.url.params["filter"] == "eq(parentId,'p-1')"



async def test_term_type_publishes_the_wire_format_per_attribute():
    """The formats are undocumented upstream, so the contract has to carry them."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool("get_glossary_term_type", {"term_type_id": TERM_TYPE_ID})
        )
    formats = {a["label"]: a["value_format"] for a in result["attributes"]}
    assert "true/false" in formats["Used in Risk"]
    assert "list" in formats["Regions"]
    assert formats["As of"] == "yyyy-mm-dd"
    assert "Z" in formats["Cutoff"]


async def test_create_sends_a_real_boolean_not_the_string():
    """The bug this guards: Viya rejects "true" and stores only its own default."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "create_glossary_term",
            {
                "name": "Exposure",
                "term_type": "BCBS239",
                "attributes": {"Scope": "Group", "Used in Risk": True},
            },
        )
    body = next(p for p in fake.posted if p["path"] == "/glossary/terms")["body"]
    assert body["attributes"]["attr-risk"] is True
    # And it must survive serialisation as a JSON boolean, not "True".
    assert '"attr-risk": true' in json.dumps(body)


async def test_create_sends_multi_select_as_a_joined_string():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "create_glossary_term",
            {
                "name": "Exposure",
                "term_type": "BCBS239",
                "attributes": {"Scope": "Group", "Regions": ["EMEA", "APAC"]},
            },
        )
    body = next(p for p in fake.posted if p["path"] == "/glossary/terms")["body"]
    assert body["attributes"]["attr-region"] == "EMEA,APAC"


async def test_update_refuses_when_a_required_attribute_became_required_later():
    """The whole term is rewritten, so an unrelated edit fails on a field the
    caller never mentioned. Say which one, before calling Viya."""
    fake = FakeViya()
    stale = {**TERM, "attributes": {**TERM["attributes"], "attr-scope": ""}}
    fake.overrides[f"GET /glossary/terms/{TERM_ID}"] = stale
    async with glossary_client(fake) as client:
        with pytest.raises(Exception) as excinfo:
            await client.call_tool(
                "update_glossary_term", {"term_id": TERM_ID, "attributes": {"Notes": "x"}}
            )
    message = str(excinfo.value)
    assert "Scope" in message and "required" in message
    assert not fake.put_bodies, "must not reach Viya"

# --- term <-> asset linkage ---------------------------------------------------


async def test_list_term_assets_traverses_to_the_column_and_its_table():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("list_term_assets", {"term_id": TERM_ID}))
    assert result["asset_count"] == 1
    asset = result["assets"][0]
    assert asset["asset_name"] == "CURR_CD"
    assert asset["table_name"] == "BCBS_SOURCE"
    assert asset["table_resource_uri"] == TABLE_RESOURCE


async def test_relationship_lookup_requests_the_endpoints():
    """Without this Accept-Item the endpoints are stripped and nothing is found."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool("list_term_assets", {"term_id": TERM_ID})
    rel_calls = [
        r
        for r in fake.requests
        if "glossaryTermAsset" in r.url.params.get("filter", "")
    ]
    assert rel_calls
    assert all(r.headers["Accept-Item"] == glossary._RELATIONSHIP_MEDIA for r in rel_calls)


def _many_assets(count: int) -> list[dict[str, Any]]:
    """*count* relationships hanging off the same term, one per column."""
    return [
        {**RELATIONSHIP, "id": f"rel-{n}", "endpoint2Id": f"cent-col-{n}"} for n in range(count)
    ]


async def test_list_term_assets_reports_the_total_not_just_the_page():
    """A partial list must not read as the whole story: count is the term's total."""
    fake = FakeViya()
    fake.relationships = _many_assets(5)
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool("list_term_assets", {"term_id": TERM_ID, "limit": 2})
        )
    assert result["asset_count"] == 2
    assert result["count"] == 5
    assert result["truncated"] is True
    assert result["next_start"] == 2
    assert "2 of 5" in result["note"]
    rel_calls = [r for r in fake.requests if "glossaryTermAsset" in r.url.params.get("filter", "")]
    assert rel_calls[0].url.params["limit"] == "2"


async def test_paging_with_start_reaches_every_asset():
    """The answer to 'I want them all': page on next_start until truncated is false."""
    fake = FakeViya()
    fake.relationships = _many_assets(7)
    seen: list[str] = []
    start, pages = 0, 0
    async with glossary_client(fake) as client:
        while True:
            page = result_of(
                await client.call_tool(
                    "list_term_assets", {"term_id": TERM_ID, "limit": 3, "start": start}
                )
            )
            seen.extend(a["asset_id"] for a in page["assets"])
            pages += 1
            if not page["truncated"]:
                break
            start = page["next_start"]
    assert pages == 3
    assert len(seen) == 7
    assert len(set(seen)) == 7, "pages must not overlap"


async def test_list_term_assets_is_not_flagged_truncated_when_it_is_complete():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("list_term_assets", {"term_id": TERM_ID}))
    assert result["truncated"] is False
    assert result["count"] == result["asset_count"] == 1
    assert "note" not in result and "next_start" not in result


async def test_list_term_assets_limit_cannot_exceed_the_ceiling():
    """The ceiling is the catalog's page size, not a suggestion."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool("list_term_assets", {"term_id": TERM_ID, "limit": 100_000})
    rel_calls = [r for r in fake.requests if "glossaryTermAsset" in r.url.params.get("filter", "")]
    assert rel_calls[0].url.params["limit"] == str(glossary._RELATIONSHIP_PAGE)


async def test_list_term_assets_reports_a_term_with_no_catalog_entity():
    """A just-created term is mirrored asynchronously; that is not 'no assets'."""
    fake = FakeViya()
    fake.overrides["GET /glossary/terms/orphan"] = {**TERM, "id": "orphan"}
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("list_term_assets", {"term_id": "orphan"}))
    assert result["asset_count"] == 0
    assert "mirrored into the catalog" in result["note"]


async def test_list_table_terms_reports_the_governed_columns():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool("list_table_terms", {"resource_uri": TABLE_RESOURCE})
        )
    assert result["column_count"] == 2
    assert result["columns_with_terms"] == 1
    assert [c["column_name"] for c in result["columns"]] == ["CURR_CD"]
    assert result["columns"][0]["terms"][0]["term_id"] == TERM_ID


async def test_list_table_terms_can_show_the_ungoverned_columns_too():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "list_table_terms", {"resource_uri": TABLE_RESOURCE, "assigned_only": False}
            )
        )
    names = {c["column_name"]: c["terms"] for c in result["columns"]}
    assert set(names) == {"CURR_CD", "BAL_AMT"}
    assert names["BAL_AMT"] == []


async def test_list_table_terms_rejects_an_ambiguous_table_name():
    """Two libraries can hold the same table name; guessing would be wrong."""
    fake = FakeViya(
        search={
            "count": 2,
            "items": [
                {"id": "a", "name": "CARS", "attributes": {"library": "PUBLIC"}},
                {"id": "b", "name": "CARS", "attributes": {"library": "SASHELP"}},
            ],
        }
    )
    async with glossary_client(fake) as client:
        with pytest.raises(Exception, match="matches 2 tables"):
            await client.call_tool("list_table_terms", {"table_name": "CARS"})


async def test_unknown_resource_uri_says_how_to_find_the_right_one():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        with pytest.raises(Exception, match="catalog_search"):
            await client.call_tool("list_table_terms", {"resource_uri": "/nope"})


async def test_column_lookup_is_chunked_for_a_wide_table(monkeypatch):
    """A 200-column table must not build one filter the gateway would reject."""
    monkeypatch.setattr(gh, "ID_CHUNK", 5)
    fake = FakeViya()
    wide = {
        f"cent-w{i}": {
            "id": f"cent-w{i}",
            "name": f"C{i}",
            "type": "sasColumn",
            "resourceId": f"{TABLE_RESOURCE}/columns/C{i}",
            "attributes": {},
        }
        for i in range(12)
    }
    _ENTITIES.update(wide)
    try:
        async with glossary_client(fake) as client:
            await client.call_tool("list_table_terms", {"resource_uri": TABLE_RESOURCE})
        rel_calls = [
            r for r in fake.requests if "glossaryTermAsset" in r.url.params.get("filter", "")
        ]
        # 14 columns at 5 per chunk.
        assert len(rel_calls) == 3
    finally:
        for key in wide:
            _ENTITIES.pop(key, None)


# --- authoring ----------------------------------------------------------------


async def test_create_publishes_by_default():
    """The API defaults to a draft nobody can see; the tool must not inherit that."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "create_glossary_term",
            {"name": "Exposure", "term_type": "BCBS239", "attributes": {"Scope": "Group"}},
        )
    post = next(p for p in fake.posted if p["path"] == "/glossary/terms")
    assert post["params"]["publish"] == "true"


async def test_create_can_be_asked_for_a_draft():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "create_glossary_term",
            {
                "name": "Exposure",
                "term_type": "BCBS239",
                "attributes": {"Scope": "Group"},
                "publish": False,
            },
        )
    post = next(p for p in fake.posted if p["path"] == "/glossary/terms")
    assert post["params"]["publish"] == "false"


async def test_create_translates_labels_to_attribute_uuids():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "create_glossary_term",
            {
                "name": "Exposure",
                "term_type": "BCBS239",
                "definition": "Amount at risk.",
                "attributes": {"Scope": "Group", "Used in Risk": True},
            },
        )
    body = next(p for p in fake.posted if p["path"] == "/glossary/terms")["body"]
    assert body["attributes"] == {"attr-scope": "Group", "attr-risk": True}
    assert body["termTypeId"] == TERM_TYPE_ID
    assert body["definition"] == "Amount at risk."
    assert "parentId" not in body  # omitted rather than sent as null


async def test_create_rejects_a_missing_required_attribute_before_calling_viya():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        with pytest.raises(Exception, match=r"requires attribute\(s\) \['Scope'\]"):
            await client.call_tool(
                "create_glossary_term", {"name": "Exposure", "term_type": "BCBS239"}
            )
    assert not [p for p in fake.posted if p["path"] == "/glossary/terms"]


async def test_create_accepts_attributes_as_a_json_string():
    """Some MCP clients serialize object parameters as strings (see _common)."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "create_glossary_term",
            {
                "name": "Exposure",
                "term_type": "BCBS239",
                "attributes": json.dumps({"Scope": "Local"}),
            },
        )
    body = next(p for p in fake.posted if p["path"] == "/glossary/terms")["body"]
    assert body["attributes"] == {"attr-scope": "Local"}


async def test_update_merges_rather_than_replacing():
    """PUT replaces the whole term, so an omitted field must survive the call."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "update_glossary_term", {"term_id": TERM_ID, "description": "Updated."}
        )
    body = fake.put_bodies[0]
    assert body["description"] == "Updated."
    assert body["definition"] == TERM["definition"]  # untouched, not blanked
    assert body["name"] == "Currency"
    assert body["attributes"]["attr-scope"] == "Group"
    assert "links" not in body  # HATEOAS links are not part of the resource


async def test_update_merges_attributes_one_at_a_time():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "update_glossary_term", {"term_id": TERM_ID, "attributes": {"Notes": "checked"}}
        )
    attributes = fake.put_bodies[0]["attributes"]
    assert attributes["attr-note"] == "checked"
    assert attributes["attr-scope"] == "Group"  # the other attributes survive


async def test_update_can_clear_one_attribute():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "update_glossary_term", {"term_id": TERM_ID, "attributes": {"Notes": ""}}
        )
    assert fake.put_bodies[0]["attributes"]["attr-note"] == ""
    # The others survive the whole-resource PUT.
    assert fake.put_bodies[0]["attributes"]["attr-scope"] == "Group"


async def test_delete_calls_the_glossary_not_the_catalog():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(await client.call_tool("delete_glossary_term", {"term_id": TERM_ID}))
    assert result == {"status": "deleted", "term_id": TERM_ID}
    assert fake.deleted == [f"/glossary/terms/{TERM_ID}"]


async def test_assign_creates_the_relationship_with_the_term_on_endpoint1():
    """The relationship is not symmetric; every reader depends on this order."""
    fake = FakeViya()
    fake.relationships = []
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "assign_glossary_term",
                {"term_id": TERM_ID, "column_name": "CURR_CD", "resource_uri": TABLE_RESOURCE},
            )
        )
    body = next(p for p in fake.posted if p["path"] == "/catalog/instances")["body"]
    assert body["endpoint1Id"] == TERM_ENTITY_ID  # the term
    assert body["endpoint2Id"] == COLUMN_CURR["id"]  # the asset
    assert body["definition"] == "glossaryTermAsset"
    assert body["instanceType"] == "relationship"
    assert result["status"] == "assigned"


async def test_assign_is_case_insensitive_about_the_column():
    fake = FakeViya()
    fake.relationships = []
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "assign_glossary_term",
                {"term_id": TERM_ID, "column_name": "curr_cd", "resource_uri": TABLE_RESOURCE},
            )
        )
    assert result["column_name"] == "CURR_CD"


async def test_assign_reports_an_existing_link_instead_of_duplicating_it():
    fake = FakeViya()  # RELATIONSHIP already links this term and column
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "assign_glossary_term",
                {"term_id": TERM_ID, "column_name": "CURR_CD", "resource_uri": TABLE_RESOURCE},
            )
        )
    assert result["status"] == "already_assigned"
    assert result["relationship_id"] == RELATIONSHIP["id"]
    assert not [p for p in fake.posted if p["path"] == "/catalog/instances"]


async def test_assign_lists_the_columns_when_the_name_is_wrong():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        with pytest.raises(Exception, match="CURR_CD"):
            await client.call_tool(
                "assign_glossary_term",
                {"term_id": TERM_ID, "column_name": "NOPE", "resource_uri": TABLE_RESOURCE},
            )


async def test_unassign_deletes_only_the_relationship():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "unassign_glossary_term",
                {"term_id": TERM_ID, "column_name": "CURR_CD", "resource_uri": TABLE_RESOURCE},
            )
        )
    assert result["status"] == "unassigned"
    assert fake.deleted == [f"/catalog/instances/{RELATIONSHIP['id']}"]


async def test_unassign_is_a_no_op_when_nothing_is_linked():
    fake = FakeViya()
    fake.relationships = []
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "unassign_glossary_term",
                {"term_id": TERM_ID, "column_name": "BAL_AMT", "resource_uri": TABLE_RESOURCE},
            )
        )
    assert result["status"] == "not_assigned"
    assert fake.deleted == []


async def test_resolving_a_term_by_name_uses_an_exact_filter():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool("list_term_assets", {"term_name": "Currency"})
    lookup = [r for r in fake.requests if r.url.path == "/glossary/terms"][0]
    assert lookup.url.params["filter"] == "eq(name,'Currency')"


async def test_a_term_reference_is_required():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        with pytest.raises(Exception, match="provide term_id or term_name"):
            await client.call_tool("list_term_assets", {})


async def test_assign_reports_an_indexing_gap_rather_than_a_bad_column_name():
    """A table indexed without its columns rejects every column name; say why."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        with pytest.raises(Exception) as excinfo:
            await client.call_tool(
                "assign_glossary_term",
                {
                    "term_id": TERM_ID,
                    "resource_uri": EMPTY_TABLE_RESOURCE,
                    "column_name": "ANY_COLUMN",
                },
            )
    message = str(excinfo.value)
    assert "indexing gap" in message
    assert "catalog_run_agent" in message


# --- term type authoring ------------------------------------------------------


def test_attribute_definitions_get_generated_identifiers():
    """The API will not mint them and rejects the omission unhelpfully."""
    built = gh.build_attribute_definitions(
        [{"label": "Scope", "type": "single-select", "allowed_values": ["A", "B"]}]
    )
    assert len(built) == 1
    assert built[0]["label"] == "Scope"
    assert built[0]["items"] == ["A", "B"]
    assert len(built[0]["name"]) == 36  # a UUID


def test_editing_an_attribute_keeps_its_identifier():
    """Minting a new one orphans every stored value under the old key."""
    existing = [{"name": "attr-keep", "label": "Scope", "type": "single-select", "items": ["A"]}]
    built = gh.build_attribute_definitions(
        [{"label": "scope", "type": "single-select", "allowed_values": ["A", "B"], "required": True}],
        existing,
    )
    assert built[0]["name"] == "attr-keep", "the UUID must survive an edit"
    assert built[0]["items"] == ["A", "B"]
    assert built[0]["required"] is True


def test_unmentioned_attributes_survive_an_edit():
    existing = [
        {"name": "a1", "label": "Scope", "type": "single-line"},
        {"name": "a2", "label": "Notes", "type": "multi-line"},
    ]
    built = gh.build_attribute_definitions([{"label": "Scope", "type": "single-line"}], existing)
    assert {d["label"] for d in built} == {"Scope", "Notes"}


def test_attributes_can_be_removed_by_label():
    existing = [
        {"name": "a1", "label": "Scope", "type": "single-line"},
        {"name": "a2", "label": "Notes", "type": "multi-line"},
    ]
    built = gh.build_attribute_definitions(None, existing, remove=["notes"])
    assert [d["label"] for d in built] == ["Scope"]


def test_removing_an_attribute_that_does_not_exist_is_refused():
    existing = [{"name": "a1", "label": "Scope", "type": "single-line"}]
    with pytest.raises(ValueError, match="no such attribute"):
        gh.build_attribute_definitions(None, existing, remove=["Nope"])


def test_a_select_attribute_without_options_is_refused():
    """Nothing could ever be stored in it, and Viya accepts the definition."""
    with pytest.raises(ValueError, match="allowed_values"):
        gh.build_attribute_definitions([{"label": "Scope", "type": "single-select"}])


def test_an_unknown_attribute_type_lists_the_valid_ones():
    with pytest.raises(ValueError) as excinfo:
        gh.build_attribute_definitions([{"label": "Scope", "type": "dropdown"}])
    assert "single-select" in str(excinfo.value)


def test_duplicate_attribute_labels_are_refused():
    with pytest.raises(ValueError, match="twice"):
        gh.build_attribute_definitions(
            [{"label": "Scope", "type": "single-line"}, {"label": "scope", "type": "multi-line"}]
        )


async def test_create_term_type_defaults_the_label_and_sends_definitions():
    """The API leaves label empty rather than defaulting it, showing as a blank."""
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "create_glossary_term_type",
                {
                    "name": "Risk Terms",
                    "attributes": [
                        {"label": "Scope", "type": "single-select", "allowed_values": ["A", "B"]},
                        {"label": "Owned", "type": "boolean", "default": False},
                    ],
                },
            )
        )
    body = next(p for p in fake.posted if p["path"] == "/glossary/termTypes")["body"]
    assert body["label"] == "Risk Terms"
    assert [a["label"] for a in body["attributes"]] == ["Scope", "Owned"]
    # A term type's default is a string even for a boolean, unlike a term's value.
    assert body["attributes"][1]["defaultValue"] == "false"
    assert result["term_type_id"] == "tt-new"


async def test_update_term_type_preserves_identifiers_of_untouched_attributes():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        await client.call_tool(
            "update_glossary_term_type",
            {
                "term_type_id": TERM_TYPE_ID,
                "attributes": [{"label": "Scope", "type": "single-select",
                                "allowed_values": ["Local", "Group", "Regional"]}],
            },
        )
    sent = {a["label"]: a for a in fake.put_bodies[0]["attributes"]}
    assert sent["Scope"]["name"] == "attr-scope", "editing must not re-key the attribute"
    assert sent["Scope"]["items"] == ["Local", "Group", "Regional"]
    assert "Notes" in sent, "attributes the caller did not mention must survive"


async def test_delete_term_type_refuses_while_terms_use_it():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        with pytest.raises(Exception) as excinfo:
            await client.call_tool(
                "delete_glossary_term_type", {"term_type_id": TERM_TYPE_ID}
            )
    assert "used by 3 term(s)" in str(excinfo.value)
    assert not fake.deleted


async def test_delete_term_type_proceeds_with_force():
    fake = FakeViya()
    async with glossary_client(fake) as client:
        result = result_of(
            await client.call_tool(
                "delete_glossary_term_type", {"term_type_id": TERM_TYPE_ID, "force": True}
            )
        )
    assert result["status"] == "deleted"
    assert result["terms_affected"] == 3
    assert fake.deleted == [f"/glossary/termTypes/{TERM_TYPE_ID}"]
