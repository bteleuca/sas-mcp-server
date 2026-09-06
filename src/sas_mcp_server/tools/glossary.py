# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 9 — Business Glossary tools (SAS Data Governance).

Three properties of the glossary make a thin REST wrapper unusable by a model,
and this module exists to absorb all three.

**A term has two identities.** It is a Glossary object under ``/glossary/terms``
*and* a Catalog entity under ``/catalog/instances``, with **different ids**. The
glossary id is what you read, write and delete; the catalog entity id is what
asset relationships point at, and what the catalog's own search returns. The
bridge is the catalog entity's ``resourceId`` (``/glossary/terms/{glossary_id}``)
— which the default representation **omits**; see :data:`_ENTITY_MEDIA`. Every
tool here returns both ids under fixed names (``term_id`` and
``catalog_entity_id``) so a caller never has to know which one it is holding.

**Attributes are keyed by UUID.** A term's ``attributes`` map is
``{attribute-definition-uuid: value}``; the human label lives on the *term type*.
Returned raw it is unreadable, and unwriteable without a second lookup. The
translation both ways, and the validation against the type's declared
required-ness and allowed values, live in
:mod:`sas_mcp_server.helpers.glossary_helpers` — as do the other pure transforms
here, so the rules can be read without the request plumbing around them.

**Term↔asset links are catalog relationships**, not glossary objects. A
``glossaryTermAsset`` relationship joins the term entity (always ``endpoint1``)
to a column entity (always ``endpoint2``) — the direction held across every such
relationship in a live deployment, and the readers here rely on it. Reading the
endpoints needs :data:`_RELATIONSHIP_MEDIA` as the collection's ``Accept-Item``;
without it the server returns summaries with the endpoints stripped, so the
traversal silently finds nothing rather than failing.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import httpx
from fastmcp import Context, FastMCP
from pydantic import BeforeValidator

from ..config import VIYA_ENDPOINT
from ..helpers.glossary_helpers import (
    IMPORT_SYSTEM_COLUMNS,
    WIRE_FORMATS,
    attribute_maps,
    build_attribute_definitions,
    build_term_csv,
    chunk_ids,
    encode_attribute,
    encode_attributes,
    glossary_id_from_resource,
    import_lookup_names,
    match_imported_rows,
    matches_attribute_filter,
    missing_required,
    parse_import_log,
    readable_attributes,
    resolve_import_paths,
    unmet_required,
)
from ..viya_client import (
    JSONDict,
    delete_resource,
    filter_literal,
    get_json,
    in_filter,
    post_json,
    put_json,
    raise_for_viya_status,
)
from ._common import coerce_json_dict, coerce_json_list, make_session_helpers

# Tolerant alias for the attributes map, which some MCP clients deliver as a
# JSON-encoded string (see _common.coerce_json_dict). The schema is unchanged.
AttributeMap = Annotated[dict[str, Any], BeforeValidator(coerce_json_dict)]
# Same tolerance for the two list-shaped arguments the term-type tools take.
AttributeSpecList = Annotated[list[dict[str, Any]], BeforeValidator(coerce_json_list)]
TermRowList = Annotated[list[dict[str, Any]], BeforeValidator(coerce_json_list)]
StringList = Annotated[list[str], BeforeValidator(coerce_json_list)]

_GLOSSARY = "/glossary"
_CATALOG = "/catalog"

_COLLECTION_MEDIA = "application/vnd.sas.collection+json"
_TERM_MEDIA = "application/vnd.sas.glossary.term+json"
_TERM_TYPE_MEDIA = "application/vnd.sas.glossary.term.type+json"
_SEARCH_MEDIA = "application/vnd.sas.metadata.search.collection+json"
_JOB_MEDIA = "application/vnd.sas.job.execution.job+json"
# The catalog instance representation that carries ``resourceId``. The default
# (``application/json``) and the plain ``...metadata.instance+json`` both return
# the field as absent rather than as an error, so a term→glossary bridge built
# on either reads as "this term is not mirrored" instead of failing.
_ENTITY_MEDIA = "application/vnd.sas.metadata.instance.entity+json"
# Likewise for relationships: only this representation carries endpoint1Id and
# endpoint2Id.
_RELATIONSHIP_MEDIA = "application/vnd.sas.metadata.instance.relationship+json"

_TERM_ASSET_DEFINITION = "glossaryTermAsset"
# The catalog search index holding glossary terms. Plural — 'term' is rejected
# with "The indices \"term\" cannot be found."
_TERMS_INDEX = "terms"
_DATASETS_INDEX = "datasets"

# The glossary cannot filter on an attribute value — every spelling of
# ``eq(attributes.<uuid>,...)`` is rejected as an invalid filter — so
# ``attribute_filter`` is applied here, over pages the server *can* return.
# These bound that scan: enough to sweep a real dictionary (a live deployment
# held 1,720 terms) without turning one call into an unbounded crawl.
_SCAN_PAGE = 200
_SCAN_CAP = 2000
# How many valid labels to quote when rejecting a mistyped one. A glossary with
# many term types has more than a caller can read; enough to spot the typo is
# the useful amount.
_LABELS_IN_ERROR = 40

# How often to ask whether the bulk-import job has finished. A live import of
# 60 terms took 26s, most of it job start-up, so polling faster only adds calls.
_IMPORT_POLL_SECONDS = 2.0

# Ceiling on the relationships one call may read. ``list_term_assets`` takes a
# ``limit`` and this bounds it, the way fedsql_registry.MAX_LIMIT bounds
# ``query_data``'s.
_RELATIONSHIP_PAGE = 500


def register(mcp: FastMCP, get_token: Callable[[Context], Awaitable[str]]) -> None:
    """Register Tier 9 (Business Glossary) tools on *mcp*."""

    viya_session, _ = make_session_helpers(get_token)

    # --- shared lookups ------------------------------------------------------

    async def instance_collection(
        client: httpx.AsyncClient,
        filter_expr: str,
        limit: int,
        item_media: str,
        start: int = 0,
    ) -> tuple[list[JSONDict], int]:
        """GET a ``/catalog/instances`` collection with an explicit ``Accept-Item``.

        ``Accept-Item`` is what decides whether items come back complete or as
        summaries with ``resourceId`` and the relationship endpoints stripped.
        :func:`get_json` has no parameter for it, so this issues the request
        directly rather than widening a helper every other tier depends on.

        Returns ``(items, total)``, where *total* is the collection's own
        ``count`` — how many match the filter, not how many this page holds. A
        caller paging through needs that to know whether it has seen them all.
        """
        resp = await client.get(
            f"{VIYA_ENDPOINT}{_CATALOG}/instances",
            headers={"Accept": _COLLECTION_MEDIA, "Accept-Item": item_media},
            params={"filter": filter_expr, "start": start, "limit": limit},
        )
        raise_for_viya_status(resp)
        body = resp.json()
        items = body.get("items", []) or []
        return items, body.get("count", len(items))

    async def fetch_term_type(client: httpx.AsyncClient, term_type_id: str) -> JSONDict:
        if not term_type_id:
            return {"attributes": []}
        return await get_json(
            f"{_GLOSSARY}/termTypes/{term_type_id}", client, accept=_TERM_TYPE_MEDIA
        )

    async def all_term_types(client: httpx.AsyncClient) -> list[JSONDict]:
        data = await get_json(
            f"{_GLOSSARY}/termTypes",
            client,
            params={"start": 0, "limit": 500},
            accept=_COLLECTION_MEDIA,
        )
        return data.get("items", []) or []

    async def resolve_term_type_id(client: httpx.AsyncClient, term_type: str) -> str:
        """Accept a term-type UUID *or* its name/label, and return the UUID.

        Authors know their term type by name ("BCBS239"); demanding the UUID
        would force a list call before every create.
        """
        types = await all_term_types(client)
        if any(item.get("id") == term_type for item in types):
            return term_type
        wanted = term_type.strip().lower()
        matches = [
            item
            for item in types
            if (item.get("name") or "").strip().lower() == wanted
            or (item.get("label") or "").strip().lower() == wanted
        ]
        if len(matches) == 1:
            return matches[0]["id"]
        if not matches:
            names = sorted(item.get("name", "") for item in types)
            raise ValueError(
                f"no term type named '{term_type}'. Available: {names}. "
                "list_glossary_term_types returns their ids."
            )
        raise ValueError(
            f"term type name '{term_type}' is ambiguous ({len(matches)} matches); pass the "
            "id instead. list_glossary_term_types returns them."
        )

    async def entities_by_id(client: httpx.AsyncClient, ids: list[str]) -> dict[str, JSONDict]:
        """Batch-resolve catalog entity ids to their full entity representations."""
        found: dict[str, JSONDict] = {}
        for chunk in chunk_ids(sorted({i for i in ids if i})):
            page, _ = await instance_collection(
                client, in_filter("id", chunk), len(chunk) + 10, _ENTITY_MEDIA
            )
            for item in page:
                found[item["id"]] = item
        return found

    async def term_entities_for(
        client: httpx.AsyncClient, glossary_ids: list[str]
    ) -> dict[str, JSONDict]:
        """Batch-resolve glossary term ids to their catalog term entities."""
        found: dict[str, JSONDict] = {}
        for chunk in chunk_ids(sorted({i for i in glossary_ids if i})):
            resources = [f"/glossary/terms/{gid}" for gid in chunk]
            page, _ = await instance_collection(
                client, in_filter("resourceId", resources), len(chunk) + 10, _ENTITY_MEDIA
            )
            for item in page:
                gid = glossary_id_from_resource(item.get("resourceId"))
                if gid:
                    found[gid] = item
        return found

    async def term_asset_relationships(
        client: httpx.AsyncClient,
        entity_ids: list[str],
        limit: int = _RELATIONSHIP_PAGE,
        start: int = 0,
    ) -> tuple[list[JSONDict], int]:
        """Every ``glossaryTermAsset`` relationship touching any of *entity_ids*.

        One filtered call per chunk, endpoints included. The naive shape — filter
        for ids, then GET each relationship to read its endpoints — costs a round
        trip per link for the same answer.

        Returns ``(relationships, total)``, *total* being how many exist rather
        than how many this page holds, so a caller can page to the end.

        *start* offsets within a single chunk and is meaningful only for a
        one-entity lookup (``list_term_assets``). Paging across chunks would need
        a cursor per chunk; the multi-entity caller (``list_table_terms``) pages
        by column instead and always passes 0.
        """
        page = max(1, min(limit, _RELATIONSHIP_PAGE))
        chunks = chunk_ids(sorted({i for i in entity_ids if i}))
        if start and len(chunks) > 1:
            raise ValueError(
                "start is only supported for a single-entity lookup; this call spans "
                f"{len(chunks)} batches."
            )
        rels: list[JSONDict] = []
        total = 0
        for chunk in chunks:
            expr = (
                f"and(eq(definition,'{_TERM_ASSET_DEFINITION}'),"
                f"or({in_filter('endpoint1Id', chunk)},{in_filter('endpoint2Id', chunk)}))"
            )
            found, matched = await instance_collection(
                client, expr, page, _RELATIONSHIP_MEDIA, start
            )
            rels.extend(found)
            total += matched
        return rels, total

    async def table_entity(
        client: httpx.AsyncClient, resource_uri: str | None, table_name: str | None
    ) -> JSONDict:
        """Resolve a table to its catalog entity, by resource URI or by name."""
        if resource_uri:
            items, _ = await instance_collection(
                client, f"eq(resourceId,'{filter_literal(resource_uri)}')", 2, _ENTITY_MEDIA
            )
            if not items:
                raise ValueError(
                    f"no catalog instance indexes '{resource_uri}'. Confirm the URI with "
                    "catalog_search, or run catalog_run_agent to populate the catalog."
                )
            return items[0]
        if not table_name:
            raise ValueError("provide resource_uri (preferred) or table_name.")
        data = await get_json(
            f"{_CATALOG}/search",
            client,
            params={
                "q": f'Name:"{table_name}"',
                "indices": _DATASETS_INDEX,
                "start": 0,
                "limit": 5,
            },
            accept=_SEARCH_MEDIA,
        )
        hits = data.get("items", []) or []
        if not hits:
            raise ValueError(
                f"no catalog table matches '{table_name}'. Try catalog_search to find it."
            )
        if len(hits) > 1:
            libraries = [(hit.get("attributes") or {}).get("library") for hit in hits]
            raise ValueError(
                f"'{table_name}' matches {len(hits)} tables (libraries: {libraries}). Pass "
                "resource_uri instead — catalog_search returns it on every hit."
            )
        entity = (await entities_by_id(client, [hits[0]["id"]])).get(hits[0]["id"])
        if entity is None:
            raise ValueError(f"the catalog hit for '{table_name}' has no readable entity.")
        return entity

    async def column_entities(
        client: httpx.AsyncClient, table_resource: str, limit: int
    ) -> list[JSONDict]:
        """The column entities of a table, keyed off its resource URI."""
        columns, _ = await instance_collection(
            client,
            f"startsWith(resourceId,'{filter_literal(table_resource)}/columns/')",
            limit,
            _ENTITY_MEDIA,
        )
        return columns

    async def resolve_term(
        client: httpx.AsyncClient, term_id: str | None, term_name: str | None
    ) -> tuple[str, JSONDict]:
        """Resolve a term by id or exact name; return ``(glossary_id, term)``."""
        if term_id:
            return term_id, await get_json(
                f"{_GLOSSARY}/terms/{term_id}", client, accept=_TERM_MEDIA
            )
        if not term_name:
            raise ValueError("provide term_id or term_name.")
        data = await get_json(
            f"{_GLOSSARY}/terms",
            client,
            params={"filter": f"eq(name,'{filter_literal(term_name)}')", "start": 0, "limit": 5},
            accept=_COLLECTION_MEDIA,
        )
        items = data.get("items", []) or []
        if not items:
            raise ValueError(
                f"no term is named exactly '{term_name}'. search_glossary_terms matches loosely."
            )
        if len(items) > 1:
            types = [item.get("termTypeLabel") for item in items]
            raise ValueError(
                f"'{term_name}' matches {len(items)} terms (types: {types}); pass term_id. "
                "search_glossary_terms returns the ids."
            )
        return items[0]["id"], items[0]

    async def column_for(
        client: httpx.AsyncClient, table: JSONDict, column_name: str
    ) -> JSONDict:
        """Find one named column entity on *table*, or raise listing what exists."""
        columns = await column_entities(client, table.get("resourceId", ""), 500)
        wanted = column_name.strip().lower()
        for column in columns:
            if (column.get("name") or "").strip().lower() == wanted:
                return column
        available = sorted((column.get("name") or "") for column in columns)
        if not columns:
            # The table is indexed but its columns are not, which reads as
            # "no such column" for every column and is nothing the caller did
            # wrong. Existing assignments on this table stay visible, so it
            # looks even less like a discovery gap than it is.
            raise ValueError(
                f"the catalog holds table '{table.get('name')}' but none of its columns, so "
                "there is nothing to attach a term to. This is an indexing gap, not a wrong "
                "column name: run catalog_run_agent to discover the table's columns, then "
                f"retry. Table URI: {table.get('resourceId', '')}"
            )
        raise ValueError(
            f"table '{table.get('name')}' has no column '{column_name}'. Columns: {available}"
        )

    # --- term types ----------------------------------------------------------

    @mcp.tool()
    async def list_glossary_term_types(
        ctx: Context, limit: int = 50, start: int = 0
    ) -> dict[str, Any]:
        """List the term types defined in the SAS Business Glossary.

        A term type is the *template* a term is created from: it fixes which
        custom attributes the term carries and which of them are mandatory. Every
        term belongs to exactly one, and the choice is immutable after creation —
        so pick the type before calling ``create_glossary_term``, then read its
        attribute contract with ``get_glossary_term_type``.

        ``usage_count`` is how many terms already use the type, which is the
        quickest way to tell a deployment's working vocabulary from types that
        were created once and abandoned.

        Args:
            limit: Maximum term types to return (default 50).
            start: Offset of the first term type (default 0).
        """
        async with viya_session("list_glossary_term_types", ctx) as client:
            data = await get_json(
                f"{_GLOSSARY}/termTypes",
                client,
                params={"start": start, "limit": limit, "sortBy": "name:ascending"},
                accept=_COLLECTION_MEDIA,
            )
            items = [
                {
                    "term_type_id": item.get("id"),
                    "name": item.get("name"),
                    "label": item.get("label"),
                    "description": item.get("description", ""),
                    "usage_count": item.get("usageCount", 0),
                    "attribute_count": item.get("attributeCount", 0),
                }
                for item in data.get("items", []) or []
            ]
            return {"count": data.get("count", len(items)), "start": start, "items": items}

    @mcp.tool()
    async def get_glossary_term_type(term_type_id: str, ctx: Context) -> dict[str, Any]:
        """Get a term type and the attribute contract its terms must satisfy.

        Call this **before** creating or updating a term: it names every custom
        attribute, its data type, whether it is required, the exact values a
        single- or multi-select accepts, and ``value_format`` — the one spelling
        Viya takes for that type, which the API itself documents nowhere.
        ``create_glossary_term`` takes attributes keyed by the ``label`` shown
        here, so this is also the vocabulary to write in.

        Args:
            term_type_id: The term type UUID, or its name — list_glossary_term_types
                returns both.
        """
        async with viya_session("get_glossary_term_type", ctx) as client:
            term_type = await fetch_term_type(
                client, await resolve_term_type_id(client, term_type_id)
            )
            attributes = [
                {
                    "label": definition.get("label"),
                    "type": definition.get("type"),
                    "required": bool(definition.get("required", False)),
                    "description": definition.get("description", ""),
                    "allowed_values": definition.get("items", []),
                    "default": definition.get("defaultValue", ""),
                    "attribute_id": definition.get("name"),
                    # The glossary publishes no schema, and several of these
                    # types accept exactly one spelling. Saying which, here,
                    # is what stops the write being shaped by trial and error.
                    "value_format": WIRE_FORMATS.get(
                        (definition.get("type") or "").lower(), "a string"
                    ),
                }
                for definition in term_type.get("attributes", []) or []
            ]
            return {
                "term_type_id": term_type.get("id"),
                "name": term_type.get("name"),
                "label": term_type.get("label"),
                "description": term_type.get("description", ""),
                "usage_count": term_type.get("usageCount", 0),
                "allow_custom_attributes": term_type.get("allowCustomAttributes", False),
                "attributes": attributes,
            }

    # --- bulk import ---------------------------------------------------------

    async def await_import_job(
        client: httpx.AsyncClient, job_id: str, timeout_seconds: int
    ) -> JSONDict:
        """Poll a term-import job until it stops running.

        The import is asynchronous — the POST returns 202 and a job — so there
        is nothing to report until this finishes.
        """
        deadline = asyncio.get_running_loop().time() + max(1, timeout_seconds)
        while True:
            resp = await client.get(
                f"{VIYA_ENDPOINT}/jobExecution/jobs/{job_id}", headers={"Accept": _JOB_MEDIA}
            )
            raise_for_viya_status(resp)
            job = resp.json()
            if job.get("state") in ("completed", "failed", "cancelled", "timedOut"):
                return job
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"the term import was still running after {timeout_seconds}s. It may yet "
                    f"finish: check job {job_id} rather than importing again, or the terms "
                    "that did land will collide on a retry."
                )
            await asyncio.sleep(_IMPORT_POLL_SECONDS)

    def import_counts(job: JSONDict) -> dict[str, int]:
        """The per-row tallies, which the job reports as a JSON *string*."""
        raw = (job.get("results") or {}).get("counts")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            return {}
        return {k: v for k, v in parsed.items() if isinstance(v, int)}

    async def resolve_imported_terms(
        client: httpx.AsyncClient, ordered: list[dict[str, Any]], job: JSONDict
    ) -> tuple[list[dict[str, Any]], str]:
        """Find the term each imported row became, and whether it was new.

        The import reports names and tallies and no ids at all, so anything a
        caller wants to do next — assign an asset, re-parent, read the term back
        — needs a lookup per row. This does it in one filtered request per 40
        distinct names instead: sibling names are unique, so a ``(parentId,
        name)`` index over the candidates rebuilds the hierarchy exactly and
        every row's path walks straight down it.

        The job's ``creationTimeStamp`` then separates a term the job created
        from one that was already there, which the job itself does not — it
        counts a row it skipped as successful just the same.

        Returns the rows and a note, empty unless something could not be
        resolved. A failure here is never fatal: the import has already
        happened, and reporting it without ids beats losing the result.
        """
        names = import_lookup_names(ordered)
        if not names:
            return [], ""

        candidates: list[JSONDict] = []
        capped = False
        for chunk in chunk_ids(names):
            offset = 0
            while True:
                data = await get_json(
                    f"{_GLOSSARY}/terms",
                    client,
                    params={
                        "filter": in_filter("name", chunk),
                        "start": offset,
                        "limit": _SCAN_PAGE,
                        # A row may have landed under a parent that is still a
                        # draft, and an unresolved ancestor breaks the walk for
                        # everything below it.
                        "allowDrafts": "all",
                    },
                    accept=_COLLECTION_MEDIA,
                )
                page = data.get("items", []) or []
                candidates.extend(page)
                offset += len(page)
                if not page or offset >= data.get("count", 0):
                    break
                if len(candidates) >= _SCAN_CAP:
                    # A name common across a large glossary can match hundreds
                    # of terms that have nothing to do with this batch. Stopping
                    # loses ids rather than time, and the note says so.
                    capped = True
                    break
            if capped:
                break

        terms = match_imported_rows(
            ordered, candidates, started_at=str(job.get("creationTimeStamp") or "")
        )
        unresolved = [t["name"] for t in terms if not t["term_id"]]
        if not unresolved:
            return terms, ""
        shown = unresolved[:_LABELS_IN_ERROR]
        more = f" (+{len(unresolved) - len(shown)} more)" if len(unresolved) > len(shown) else ""
        reason = (
            f"the lookup stopped at {_SCAN_CAP} candidate terms"
            if capped
            else "they were not found under the path they were imported to"
        )
        return terms, (
            f"No id could be resolved for {len(unresolved)} row(s) — {reason}. "
            f"Unresolved: {shown}{more}. The import itself is unaffected; find those terms "
            "with list_glossary_terms or search_glossary_terms."
        )

    async def import_failures(client: httpx.AsyncClient, job: JSONDict) -> list[JSONDict]:
        """The per-row failures, read from the job's log.

        The job's ``state`` is ``completed`` even when every row failed, and the
        response carries no structured errors — the log is the only place that
        names the row and the reason.
        """
        location = job.get("logLocation")
        if not location:
            return []
        resp = await client.get(f"{VIYA_ENDPOINT}{location}/content")
        if resp.status_code >= 400:
            return [
                {
                    "term": None,
                    "row": None,
                    "error": f"the import log at {location} could not be read "
                    f"(HTTP {resp.status_code}), so per-row failures are unavailable.",
                }
            ]
        return parse_import_log(resp.text)

    # --- term type authoring -------------------------------------------------

    def term_type_summary(term_type: JSONDict) -> dict[str, Any]:
        """The shape every term-type tool returns, so they read alike."""
        return {
            "term_type_id": term_type.get("id"),
            "name": term_type.get("name"),
            "label": term_type.get("label"),
            "description": term_type.get("description", ""),
            "usage_count": term_type.get("usageCount", 0),
            "allow_custom_attributes": term_type.get("allowCustomAttributes", False),
            "attributes": [
                {
                    "label": definition.get("label"),
                    "type": definition.get("type"),
                    "required": bool(definition.get("required", False)),
                    "allowed_values": definition.get("items", []),
                    "default": definition.get("defaultValue", ""),
                    "attribute_id": definition.get("name"),
                    "value_format": WIRE_FORMATS.get(
                        (definition.get("type") or "").lower(), "a string"
                    ),
                }
                for definition in term_type.get("attributes", []) or []
            ],
        }

    @mcp.tool()
    async def create_glossary_term_type(
        name: str,
        ctx: Context,
        label: str | None = None,
        description: str | None = None,
        attributes: AttributeSpecList | None = None,
        allow_custom_attributes: bool = False,
    ) -> dict[str, Any]:
        """Create a term type — the template that fixes what a term must carry.

        A term type declares the custom attributes every term of that type
        holds, and which are mandatory. Without this tool a deployment's types
        can only be created in the SAS UI, so a glossary could be read and
        populated through MCP but never *designed* through it.

        Each attribute is ``{"label": ..., "type": ...}`` plus, optionally,
        ``required``, ``allowed_values``, ``default`` and ``description``:

        * ``single-line``, ``multi-line`` — free text
        * ``single-select``, ``multi-select`` — need ``allowed_values``
        * ``boolean``, ``date``, ``date-time``, ``time``

        The attribute identifiers the API demands are generated here, since it
        will not mint them itself and rejects the omission with a message that
        names neither the attribute nor the real problem.

        Args:
            name: Term type name, unique across the deployment.
            label: Display name. Defaults to *name* — the API leaves it empty
                rather than defaulting it, which shows as a blank in the UI.
            description: What terms of this type are for.
            attributes: The attribute definitions, e.g.
                ``[{"label": "Scope", "type": "single-select",
                "allowed_values": ["Local", "Group"], "required": true}]``.
            allow_custom_attributes: Let individual terms add attributes beyond
                these (default false).
        """
        async with viya_session("create_glossary_term_type", ctx) as client:
            body: dict[str, Any] = {
                "name": name,
                "label": label or name,
                "allowCustomAttributes": allow_custom_attributes,
                "attributes": build_attribute_definitions(attributes),
            }
            if description is not None:
                body["description"] = description
            created = await post_json(
                f"{_GLOSSARY}/termTypes",
                client,
                body,
                accept=_TERM_TYPE_MEDIA,
            )
            return term_type_summary(created)

    @mcp.tool()
    async def update_glossary_term_type(
        term_type_id: str,
        ctx: Context,
        name: str | None = None,
        label: str | None = None,
        description: str | None = None,
        attributes: AttributeSpecList | None = None,
        remove_attributes: StringList | None = None,
        allow_custom_attributes: bool | None = None,
    ) -> dict[str, Any]:
        """Change a term type: rename it, or add, edit and remove its attributes.

        Attributes are matched to the existing ones **by label**, and an edit
        keeps that attribute's identifier — which matters more than it looks,
        because every term's stored values are filed under it. An attribute the
        caller does not mention is left alone; a label that does not exist yet
        is added.

        **To rename one, give its ``attribute_id``** alongside the new
        ``label``. Matching by label alone cannot express a rename: the new
        label matches nothing, so the attribute is added afresh under a new
        identifier and every term's value stays behind under the old one, no
        longer readable as that attribute. ``get_glossary_term_type`` returns
        the id of each.

        Making an attribute ``required`` applies to terms created *afterwards*
        and to every later edit of the ones already there: an update rewrites
        the whole term, so older terms must be given a value for it before they
        can be saved again. ``update_glossary_term`` reports that by name.

        Args:
            term_type_id: The term type UUID, or its name.
            name: New name.
            label: New display name.
            description: New description.
            attributes: Attributes to add or change, same shape as
                ``create_glossary_term_type``, plus an optional
                ``attribute_id`` naming which existing attribute the spec is —
                required to rename one. Omitted attributes survive.
            remove_attributes: Labels to drop from the type. Terms keep the
                stored value, but under an identifier nothing names any more, so
                it stops being readable as that attribute.
            allow_custom_attributes: Whether terms may add their own.
        """
        async with viya_session("update_glossary_term_type", ctx) as client:
            resolved = await resolve_term_type_id(client, term_type_id)
            current = await fetch_term_type(client, resolved)
            body = {key: value for key, value in current.items() if key != "links"}
            if name is not None:
                body["name"] = name
            if label is not None:
                body["label"] = label
            if description is not None:
                body["description"] = description
            if allow_custom_attributes is not None:
                body["allowCustomAttributes"] = allow_custom_attributes
            body["attributes"] = build_attribute_definitions(
                attributes,
                current.get("attributes", []) or [],
                remove=remove_attributes,
            )
            updated = await put_json(
                f"{_GLOSSARY}/termTypes/{resolved}",
                client,
                body,
                content_type=_TERM_TYPE_MEDIA,
                accept=_TERM_TYPE_MEDIA,
            )
            return term_type_summary(updated or current)

    @mcp.tool()
    async def delete_glossary_term_type(
        term_type_id: str, ctx: Context, force: bool = False
    ) -> dict[str, Any]:
        """Delete a term type.

        Refuses while terms still use the type, because deleting it takes their
        attribute definitions with it. Check with ``list_glossary_terms``
        (``term_type=``) and move or delete those terms first — or pass
        ``force`` if you have already decided.

        Args:
            term_type_id: The term type UUID, or its name.
            force: Delete even though terms use this type (default false).
        """
        async with viya_session("delete_glossary_term_type", ctx) as client:
            resolved = await resolve_term_type_id(client, term_type_id)
            current = await fetch_term_type(client, resolved)
            in_use = current.get("usageCount", 0) or 0
            if in_use and not force:
                raise ValueError(
                    f"term type '{current.get('name')}' is used by {in_use} term(s), whose "
                    "attribute definitions go with it. List them with "
                    f"list_glossary_terms(term_type='{resolved}'), then delete or retype "
                    "them — or pass force=true to delete the type anyway."
                )
            await delete_resource(f"{_GLOSSARY}/termTypes/{resolved}", client)
            return {
                "status": "deleted",
                "term_type_id": resolved,
                "name": current.get("name"),
                "terms_affected": in_use,
            }

    # --- reading attributes off a list of terms -------------------------------

    async def type_maps_for(
        client: httpx.AsyncClient, type_id: str, cache: dict[str, dict[str, JSONDict]]
    ) -> dict[str, JSONDict]:
        """``uuid -> definition`` for one term type, fetched at most once."""
        if type_id not in cache:
            by_uuid, _ = attribute_maps(await fetch_term_type(client, type_id))
            cache[type_id] = by_uuid
        return cache[type_id]

    async def decode_for_terms(
        client: httpx.AsyncClient, raw_terms: list[JSONDict], cache: dict[str, dict[str, JSONDict]]
    ) -> list[dict[str, Any]]:
        """Label-keyed attributes for each term, in the order given.

        The list representation already carries the raw ``attributes`` map, so
        this costs one request per *distinct term type* on the page — normally
        one — and none per term.
        """
        decoded = []
        for term in raw_terms:
            by_uuid = await type_maps_for(client, term.get("termTypeId", ""), cache)
            decoded.append(readable_attributes(term.get("attributes"), by_uuid))
        return decoded

    async def known_attribute_labels(
        client: httpx.AsyncClient, type_id: str | None = None
    ) -> set[str]:
        """Attribute labels in play, lowercased, for validating a filter.

        Used to reject a mistyped filter label up front: filtering happens on
        this side, so a typo would otherwise return an empty list that reads
        like a real answer.

        Scoped to one term type when the caller named one — otherwise every type
        has to be read, which is a request each.
        """
        summaries = (
            [{"id": type_id}] if type_id else await all_term_types(client)
        )
        labels: set[str] = set()
        for summary in summaries:
            definition = await fetch_term_type(client, summary.get("id", ""))
            for attribute in definition.get("attributes", []) or []:
                if attribute.get("label"):
                    labels.add(attribute["label"].strip().lower())
        return labels

    # --- finding terms -------------------------------------------------------

    @mcp.tool()
    async def search_glossary_terms(
        query: str,
        ctx: Context,
        limit: int = 20,
        start: int = 0,
        include_attributes: bool = False,
    ) -> dict[str, Any]:
        """Free-text search of the business glossary — the way in when you know a word, not an id.

        Runs against the Information Catalog's ``terms`` index, so it is ranked
        and matches definitions as well as names, unlike ``list_glossary_terms``'
        exact structural filters. Supports the catalog grammar: wildcards
        (``rev*``), field constraints (``Name:revenue``, ``Status:Published``)
        and ``+`` to require a word.

        Each hit carries **both** identifiers — ``term_id`` for every other
        glossary tool, ``catalog_entity_id`` for catalog relationships — plus
        ``assigned_asset_count``, so you can tell whether a term is actually in
        use before spending a call on ``list_term_assets``. A term with a count
        of 0 exists in the dictionary and is attached to no data.

        Args:
            query: Search text. ``*`` matches every term.
            limit: Maximum hits to return (default 20).
            start: Offset of the first hit (default 0).
            include_attributes: Also return each hit's custom attributes, named
                (default false). The search index does not carry them, so this
                costs one extra batched call per 40 hits; leave it off when the
                names and definitions are all you need.
        """
        async with viya_session("search_glossary_terms", ctx) as client:
            data = await get_json(
                f"{_CATALOG}/search",
                client,
                params={"q": query, "indices": _TERMS_INDEX, "start": start, "limit": limit},
                accept=_SEARCH_MEDIA,
            )
            hits = data.get("items", []) or []
            entities = await entities_by_id(client, [hit.get("id", "") for hit in hits])
            items = []
            for hit in hits:
                attributes = hit.get("attributes") or {}
                entity = entities.get(hit.get("id", ""), {})
                items.append(
                    {
                        "term_id": glossary_id_from_resource(entity.get("resourceId")),
                        "catalog_entity_id": hit.get("id"),
                        "name": hit.get("name"),
                        "term_type": hit.get("typeLabel"),
                        "definition": attributes.get("definition", ""),
                        "status": attributes.get("reviewStatus", ""),
                        "assigned_asset_count": attributes.get("assignedAssets", 0),
                        "score": hit.get("score"),
                    }
                )
            if include_attributes:
                # The catalog index holds no custom attributes, so read them
                # from the glossary itself — batched by id rather than one call
                # per hit.
                wanted = [item["term_id"] for item in items if item.get("term_id")]
                by_id: dict[str, JSONDict] = {}
                for chunk in chunk_ids(wanted):
                    page = await get_json(
                        f"{_GLOSSARY}/terms",
                        client,
                        params={
                            "filter": in_filter("id", chunk),
                            "start": 0,
                            "limit": len(chunk),
                        },
                        accept=_COLLECTION_MEDIA,
                    )
                    for term in page.get("items", []) or []:
                        by_id[term.get("id", "")] = term
                cache: dict[str, dict[str, JSONDict]] = {}
                for item in items:
                    term = by_id.get(item.get("term_id") or "")
                    if term is None:
                        item["attributes"] = {}
                        continue
                    by_uuid = await type_maps_for(client, term.get("termTypeId", ""), cache)
                    item["attributes"] = readable_attributes(term.get("attributes"), by_uuid)

            total = data.get("count", len(items))
            return {
                "count": total,
                "start": data.get("start", start),
                "items": items,
                # The catalog's count is taken before authorization filtering, so
                # it can exceed the items actually returned when the index holds
                # terms this user cannot read. Saying so stops a caller paging
                # for hits that will never arrive.
                "note": (
                    "count comes from the search index and may exceed the readable items."
                    if total > start + len(items)
                    else ""
                ),
            }

    @mcp.tool()
    async def list_glossary_terms(
        ctx: Context,
        term_type: str | None = None,
        parent_id: str | None = None,
        name_contains: str | None = None,
        include_drafts: bool = False,
        limit: int = 20,
        start: int = 0,
        include_attributes: bool = False,
        attribute_filter: AttributeMap | None = None,
    ) -> dict[str, Any]:
        """List glossary terms by structure — term type, parent, or name fragment.

        The counterpart to ``search_glossary_terms``: exact filters instead of
        ranked text. Use it to walk the hierarchy (``parent_id`` returns a term's
        direct children, which is the authoritative parent/child relationship),
        to inventory one term type, or to page the whole dictionary with no
        arguments at all.

        Args:
            term_type: Restrict to one term type — its UUID or its name.
            parent_id: Return only the direct children of this term id.
            name_contains: Substring match on the term name.
            include_drafts: Include unpublished drafts (default false — published only).
            limit: Maximum terms to return (default 20).
            start: Offset of the first term (default 0).
            include_attributes: Also return each term's custom attributes,
                named (default false). Free — the listing already carries them.
            attribute_filter: Keep only terms whose attributes match, e.g.
                ``{"Needs masking": true}`` or ``{"Regions": "EMEA"}``. Clauses
                are ANDed; a multi-select matches when it *contains* the value.
                **The glossary cannot filter on attributes server-side**, so
                this is applied here over pages fetched for the purpose: the
                result reports how many terms were scanned and whether the scan
                reached the end. Narrow it with ``term_type`` or ``parent_id``
                where you can.
        """
        def project(item: JSONDict) -> dict[str, Any]:
            return {
                "term_id": item.get("id"),
                "name": item.get("name"),
                "term_type": item.get("termTypeLabel"),
                "term_type_id": item.get("termTypeId"),
                "definition": item.get("definition", ""),
                "description": item.get("description", ""),
                "parent_id": item.get("parentId"),
                "status": item.get("status"),
                "is_draft": item.get("isDraft", False),
                "assigned_asset_count": item.get("assetCount", 0),
            }

        async with viya_session("list_glossary_terms", ctx) as client:
            clauses = []
            if term_type:
                resolved = await resolve_term_type_id(client, term_type)
                clauses.append(f"eq(termTypeId,'{filter_literal(resolved)}')")
            if parent_id:
                clauses.append(f"eq(parentId,'{filter_literal(parent_id)}')")
            if name_contains:
                clauses.append(f"contains(name,'{filter_literal(name_contains)}')")
            params: dict[str, Any] = {
                "sortBy": "name:ascending",
                "allowDrafts": "all" if include_drafts else "none",
            }
            if clauses:
                params["filter"] = clauses[0] if len(clauses) == 1 else f"and({','.join(clauses)})"

            want_attributes = include_attributes or bool(attribute_filter)
            cache: dict[str, dict[str, JSONDict]] = {}

            if not attribute_filter:
                data = await get_json(
                    f"{_GLOSSARY}/terms",
                    client,
                    params={**params, "start": start, "limit": limit},
                    accept=_COLLECTION_MEDIA,
                )
                raw = data.get("items", []) or []
                items = [project(item) for item in raw]
                if want_attributes:
                    for item, decoded in zip(
                        items, await decode_for_terms(client, raw, cache), strict=True
                    ):
                        item["attributes"] = decoded
                return {"count": data.get("count", len(items)), "start": start, "items": items}

            # Filtering happens here, so a mistyped label would quietly return
            # nothing and read as a real answer. Reject it against the labels
            # that actually exist first.
            scope = await resolve_term_type_id(client, term_type) if term_type else None
            known = await known_attribute_labels(client, scope)
            unknown = sorted(
                label
                for label in attribute_filter
                if str(label).strip().lower() not in known
            )
            if unknown:
                where = f"term type '{term_type}'" if scope else "any term type in this glossary"
                # A glossary with many types has a long label list; enough of it
                # to spot the typo is the useful amount, not all of it.
                listed = sorted(known)
                shown = listed[:_LABELS_IN_ERROR]
                more = (
                    f" (+{len(listed) - len(shown)} more)" if len(listed) > len(shown) else ""
                )
                raise ValueError(
                    f"no attribute is named {unknown} on {where}. "
                    f"Defined attribute labels: {shown}{more}. "
                    "get_glossary_term_type lists them per type."
                )

            matched: list[dict[str, Any]] = []
            scanned = 0
            offset = start
            scan_complete = False
            while len(matched) < limit and scanned < _SCAN_CAP:
                data = await get_json(
                    f"{_GLOSSARY}/terms",
                    client,
                    params={**params, "start": offset, "limit": _SCAN_PAGE},
                    accept=_COLLECTION_MEDIA,
                )
                raw = data.get("items", []) or []
                if not raw:
                    scan_complete = True
                    break
                decoded = await decode_for_terms(client, raw, cache)
                for item, attributes in zip(raw, decoded, strict=True):
                    if matches_attribute_filter(attributes, attribute_filter):
                        hit = project(item)
                        hit["attributes"] = attributes
                        matched.append(hit)
                        if len(matched) == limit:
                            break
                scanned += len(raw)
                offset += len(raw)
                if offset >= data.get("count", 0):
                    scan_complete = True
                    break

            result: dict[str, Any] = {
                "count": len(matched),
                "start": start,
                "items": matched,
                # The glossary cannot filter on attributes, so these say how much
                # of the dictionary this answer actually covers. Without them a
                # short list is indistinguishable from a complete one.
                "scanned": scanned,
                "scan_complete": scan_complete,
            }
            if not scan_complete:
                result["next_start"] = offset
                result["note"] = (
                    f"Matched {len(matched)} term(s) in the first {scanned} scanned. "
                    "The glossary cannot filter on attribute values, so this scan is "
                    f"capped: call again with start={offset} to continue, or narrow it "
                    "with term_type or parent_id."
                )
            return result

    @mcp.tool()
    async def get_glossary_term(term_id: str, ctx: Context) -> dict[str, Any]:
        """Get one business term in full, with its custom attributes named rather than hashed.

        The raw API returns ``attributes`` keyed by attribute-definition UUID,
        which is unreadable on its own. This resolves each key to the label the
        glossary UI shows and drops the ones left empty, so what comes back is
        the term as a person would read it. ``attribute_ids`` is the raw map,
        unfiltered — so an attribute the term leaves unset is absent from
        ``attributes`` but present as ``""`` there. The two disagree by design:
        one says what the term holds, the other what was stored.

        Also returns ``catalog_entity_id`` — the *other* id this term has, the
        one asset relationships point at.

        Args:
            term_id: The glossary term UUID (not the catalog entity id —
                search_glossary_terms returns both).
        """
        async with viya_session("get_glossary_term", ctx) as client:
            term = await get_json(f"{_GLOSSARY}/terms/{term_id}", client, accept=_TERM_MEDIA)
            term_type, entities = await asyncio.gather(
                fetch_term_type(client, term.get("termTypeId", "")),
                term_entities_for(client, [term_id]),
            )
            by_uuid, _ = attribute_maps(term_type)
            raw_attributes = term.get("attributes") or {}
            return {
                "term_id": term.get("id"),
                "catalog_entity_id": (entities.get(term_id) or {}).get("id"),
                "name": term.get("name"),
                "label": term.get("label", ""),
                "definition": term.get("definition", ""),
                "description": term.get("description", ""),
                "term_type": term.get("termTypeLabel"),
                "term_type_id": term.get("termTypeId"),
                "parent_id": term.get("parentId"),
                "status": term.get("status"),
                "is_draft": term.get("isDraft", False),
                "assigned_asset_count": term.get("assetCount", 0),
                "attributes": readable_attributes(raw_attributes, by_uuid),
                "attribute_ids": raw_attributes,
                "created_by": term.get("createdBy"),
                "modified_by": term.get("modifiedBy"),
                "modified": term.get("modifiedTimeStamp"),
            }

    # --- term <-> asset linkage ---------------------------------------------

    @mcp.tool()
    async def list_term_assets(
        ctx: Context,
        term_id: str | None = None,
        term_name: str | None = None,
        limit: int = 100,
        start: int = 0,
    ) -> dict[str, Any]:
        """List the data assets a business term is attached to — the columns that mean it.

        The authoritative answer to "where is this term actually used?", read
        from the ``glossaryTermAsset`` relationships rather than inferred from
        names. Each entry names the column and the table it belongs to.

        An empty result means the term is **assigned** to nothing, which is not
        the same as no matching column existing — ``assign_glossary_term`` is
        what creates the link. For a looser, name-based sweep, ``catalog_search``
        accepts the ``Column.term:"<term name>"`` facet on the ``datasets``
        index, which returns matching tables without resolving columns.

        Args:
            term_id: The glossary term UUID.
            term_name: Exact term name, if the id is not known. One of the two is
                required.
            limit: Maximum assets to return in one call (default 100, ceiling
                500 — the catalog's page size).
            start: Offset of the first asset returned (default 0). ``count`` in
                the result is the term's **total** asset count, so to read every
                asset of a heavily used term, call again with
                ``start`` = ``next_start`` until ``truncated`` is false.
        """
        async with viya_session("list_term_assets", ctx) as client:
            glossary_id, term = await resolve_term(client, term_id, term_name)
            entity = (await term_entities_for(client, [glossary_id])).get(glossary_id)
            if entity is None:
                return {
                    "term_id": glossary_id,
                    "name": term.get("name"),
                    "asset_count": 0,
                    "assets": [],
                    "note": (
                        "This term has no catalog entity, so it cannot carry asset links "
                        "yet. Newly created terms are mirrored into the catalog "
                        "asynchronously."
                    ),
                }
            entity_id = entity["id"]
            rels, total = await term_asset_relationships(client, [entity_id], limit, start)
            asset_ids = [
                rel.get("endpoint2Id") if rel.get("endpoint1Id") == entity_id else rel.get("endpoint1Id")
                for rel in rels
            ]
            asset_ids = [asset_id for asset_id in asset_ids if asset_id]
            assets = await entities_by_id(client, asset_ids)
            resolved = []
            for asset_id in asset_ids:
                asset = assets.get(asset_id, {})
                resource = asset.get("resourceId", "") or ""
                # A column's resourceId is '<table resource>/columns/<name>', so
                # the owning table falls out of the id — no extra lookup.
                table_resource = resource.split("/columns/")[0] if "/columns/" in resource else ""
                resolved.append(
                    {
                        "asset_id": asset_id,
                        "asset_name": asset.get("name"),
                        "asset_type": asset.get("type"),
                        "resource_uri": resource,
                        "table_resource_uri": table_resource,
                        "table_name": table_resource.rsplit("/", 1)[-1] if table_resource else "",
                    }
                )
            seen = start + len(resolved)
            truncated = seen < total
            result: dict[str, Any] = {
                "term_id": glossary_id,
                "catalog_entity_id": entity_id,
                "name": term.get("name"),
                "count": total,
                "start": start,
                "asset_count": len(resolved),
                "assets": resolved,
                "truncated": truncated,
            }
            if truncated:
                result["next_start"] = seen
                result["note"] = (
                    f"Showing {len(resolved)} of {total} assets. Call again with "
                    f"start={seen} for the next page."
                )
            return result

    @mcp.tool()
    async def list_table_terms(
        ctx: Context,
        resource_uri: str | None = None,
        table_name: str | None = None,
        max_columns: int = 200,
        assigned_only: bool = True,
    ) -> dict[str, Any]:
        """List the business terms assigned to a table's columns.

        The reverse of ``list_term_assets``, and the fastest way to judge whether
        a table is governed: it reports each column's term together with the
        term's own definition, so a caller can read what a cryptically named
        column actually holds.

        Terms come from ``glossaryTermAsset`` relationships, so a column with no
        term here has genuinely never been assigned one — the catalog does not
        guess from column names.

        Args:
            resource_uri: The table's source URI (preferred) — catalog_search
                returns it on every hit.
            table_name: Table name, if the URI is not known. Rejected as ambiguous
                when more than one table matches.
            max_columns: Maximum columns to inspect (default 200).
            assigned_only: Return only columns that carry a term (default true).
                Set false to see the unassigned columns too.
        """
        async with viya_session("list_table_terms", ctx) as client:
            table = await table_entity(client, resource_uri, table_name)
            table_resource = table.get("resourceId", "")
            columns = await column_entities(client, table_resource, max_columns)
            by_entity = {column["id"]: column for column in columns if column.get("id")}
            rels, _ = await term_asset_relationships(client, list(by_entity))

            term_entity_ids = [
                rel.get("endpoint1Id") if rel.get("endpoint2Id") in by_entity else rel.get("endpoint2Id")
                for rel in rels
            ]
            term_entities = await entities_by_id(client, [tid for tid in term_entity_ids if tid])

            per_column: dict[str, list[dict[str, Any]]] = {cid: [] for cid in by_entity}
            for rel in rels:
                first, second = rel.get("endpoint1Id"), rel.get("endpoint2Id")
                column_id = second if second in by_entity else first
                term_entity_id = first if column_id == second else second
                if column_id not in per_column:
                    continue
                entity = term_entities.get(term_entity_id or "", {})
                per_column[column_id].append(
                    {
                        "term_id": glossary_id_from_resource(entity.get("resourceId")),
                        "catalog_entity_id": term_entity_id,
                        "name": entity.get("name"),
                        "definition": entity.get("description", ""),
                        "status": (entity.get("attributes") or {}).get("status"),
                    }
                )

            results = []
            for column_id, column in by_entity.items():
                terms = per_column.get(column_id, [])
                if assigned_only and not terms:
                    continue
                results.append(
                    {
                        "column_id": column_id,
                        "column_name": column.get("name"),
                        "data_type": (column.get("attributes") or {}).get("dataType"),
                        "terms": terms,
                    }
                )
            return {
                "table_name": table.get("name"),
                "resource_uri": table_resource,
                "column_count": len(by_entity),
                "columns_with_terms": sum(1 for column in per_column.values() if column),
                "columns": results,
            }

    # --- authoring -----------------------------------------------------------

    @mcp.tool()
    async def create_glossary_term(
        name: str,
        term_type: str,
        ctx: Context,
        definition: str | None = None,
        description: str | None = None,
        label: str | None = None,
        parent_id: str | None = None,
        attributes: AttributeMap | None = None,
        publish: bool = True,
    ) -> dict[str, Any]:
        """Create a business term in the SAS Business Glossary.

        ``attributes`` is keyed by the attribute **labels** from
        ``get_glossary_term_type`` — call that first, because a term type can
        make attributes mandatory and a term missing one is rejected. Pass each
        value in its natural Python form and it is converted to the one spelling
        the glossary accepts:

        * **boolean** — ``True`` / ``False`` (the *strings* ``"true"``/``"false"``
          are rejected by Viya; that conversion happens here)
        * **multi-select** — a list, e.g. ``["Retail", "Wholesale"]``
        * **date** — ``"2026-09-04"``
        * **date-time** — ``"2026-09-04T13:41:24Z"``; a bare date or a numeric
          offset is normalised to UTC rather than rejected
        * **time** — ``"15:41:28Z"``; seconds and the ``Z`` are required, and a
          numeric offset is converted to UTC rather than dropped

        Values are validated before the call, so a mistake comes back naming the
        attribute and what it expected, instead of as an opaque HTTP 400.

        **Terms are published by default.** The underlying API defaults to
        creating a *draft*, which nobody but its author can see; that is almost
        never what a caller asking to "create a term" means, so this publishes
        unless ``publish`` is set false. A draft is promoted afterwards with
        ``update_glossary_term(publish=true)``, and is visible to
        ``list_glossary_terms`` only under ``include_drafts``.

        A term's name must be unique among its siblings (case-insensitively) and
        differ from its parent's; a clash is rejected, not merged.

        Args:
            name: Term name, max 100 characters, no backslashes.
            term_type: The term type — its UUID or its name. Immutable afterwards.
            definition: What the term means. The field users read; worth filling in.
            description: Short overview, max 1000 characters.
            label: Display name, if it should differ from ``name``.
            parent_id: Parent term id, to nest this term in the hierarchy.
            attributes: Custom attributes keyed by label, e.g.
                ``{"Scope": "Group", "Used in Risk": True,
                "Regions": ["EMEA", "APAC"]}``.
            publish: Publish immediately (default true). False leaves a draft,
                which ``update_glossary_term(publish=true)`` promotes later.
        """
        async with viya_session("create_glossary_term", ctx) as client:
            term_type_id = await resolve_term_type_id(client, term_type)
            type_definition = await fetch_term_type(client, term_type_id)
            by_uuid, by_label = attribute_maps(type_definition)
            encoded = encode_attributes(attributes, by_label, require_all=True)

            body: dict[str, Any] = {"name": name, "termTypeId": term_type_id}
            if definition is not None:
                body["definition"] = definition
            if description is not None:
                body["description"] = description
            if label is not None:
                body["label"] = label
            if parent_id is not None:
                body["parentId"] = parent_id
            if encoded:
                body["attributes"] = encoded

            created = await post_json(
                f"{_GLOSSARY}/terms",
                client,
                body=body,
                params={"publish": "true" if publish else "false"},
                accept=_TERM_MEDIA,
            )
            return {
                "term_id": created.get("id"),
                "name": created.get("name"),
                "term_type": created.get("termTypeLabel"),
                "status": created.get("status"),
                "is_draft": created.get("isDraft", False),
                "parent_id": created.get("parentId"),
                "attributes": readable_attributes(created.get("attributes"), by_uuid),
                "next_step": (
                    "Attach it to data with assign_glossary_term — a term with no assigned "
                    "assets governs nothing."
                ),
            }

    @mcp.tool()
    async def update_glossary_term(
        term_id: str,
        ctx: Context,
        name: str | None = None,
        definition: str | None = None,
        description: str | None = None,
        label: str | None = None,
        parent_id: str | None = None,
        attributes: AttributeMap | None = None,
        publish: bool = False,
    ) -> dict[str, Any]:
        """Update a business term's text, parent or custom attributes — and publish a draft.

        The glossary API replaces the whole term on update, so this reads the
        current one first and merges your changes into it: omitting an argument
        leaves that field alone rather than blanking it. ``attributes`` merges
        the same way, per attribute — pass only the ones you are changing, and
        set one to ``""`` to clear it. A **boolean** is the exception: the
        glossary has no empty boolean and rejects ``""``, so set it to
        ``True``/``False`` rather than trying to clear it.

        Because the whole term is rewritten, every attribute the type marks
        **required** must hold a value — including ones made required after this
        term was created. That is checked before the call, and reported by name.

        A term's **type** cannot be changed after creation. Its **parent** can:
        pass ``parent_id`` to move it, or ``""`` to make it a root term.

        **A draft is a different resource.** A term left unpublished by
        ``create_glossary_term(publish=false)`` can be read and deleted at the
        ordinary path, but not written there — the service answers a plain
        ``PUT`` on a draft with a 404. This routes the write to the draft
        instead, so editing one works; and ``publish`` then promotes it to a
        published term, which nothing else here could do. Publishing a term that
        is already published is reported, not attempted: there is no draft to
        promote and the service answers that with a 404 too.

        Args:
            term_id: The glossary term UUID.
            name: New name (unique among siblings, max 100 characters).
            definition: New definition.
            description: New description, max 1000 characters.
            label: New display label.
            parent_id: Move the term under a different parent, or ``""`` to make
                it a root term. A term cannot be its own ancestor.
            attributes: Custom attributes to change, keyed by label. Same value
                forms as create_glossary_term — booleans as True/False,
                multi-select as a list.
            publish: Publish the term if it is still a draft (default false).
                Pass it on its own to publish without changing anything else.
        """
        async with viya_session("update_glossary_term", ctx) as client:
            current = await get_json(f"{_GLOSSARY}/terms/{term_id}", client, accept=_TERM_MEDIA)
            type_definition = await fetch_term_type(client, current.get("termTypeId", ""))
            by_uuid, by_label = attribute_maps(type_definition)

            merged = dict(current.get("attributes") or {})
            merged.update(encode_attributes(attributes, by_label, require_all=False))
            # An update is a whole-resource PUT, so it replays attributes the
            # caller never mentioned. If one of those was made required after
            # this term was written, Viya rejects the edit and names *that*
            # attribute — baffling when you were changing something else.
            unmet = missing_required(merged, by_label)
            if unmet:
                raise ValueError(
                    f"this term is missing required attribute(s) {unmet}, so it cannot be "
                    "saved. They were most likely made required after the term was created: "
                    "an update rewrites the whole term, so every required attribute must "
                    "hold a value. Supply them in this same call."
                )

            body = {key: value for key, value in current.items() if key != "links"}
            if name is not None:
                body["name"] = name
            if definition is not None:
                body["definition"] = definition
            if description is not None:
                body["description"] = description
            if label is not None:
                body["label"] = label
            if parent_id is not None:
                if parent_id == term_id:
                    raise ValueError("a term cannot be its own parent.")
                # "" clears the parent, which is how a term becomes a root.
                body["parentId"] = parent_id or None
            body["attributes"] = merged

            # A draft is a separate resource. The ordinary path answers GET and
            # DELETE for one, which is why this reads it there, but a PUT to it
            # is a 404 — the draft's own ``update`` link names ``/draft``, and
            # that is the only path that can write one.
            is_draft = bool(current.get("isDraft"))
            updated = await put_json(
                f"{_GLOSSARY}/terms/{term_id}/draft" if is_draft else f"{_GLOSSARY}/terms/{term_id}",
                client,
                body,
                content_type=_TERM_MEDIA,
                accept=_TERM_MEDIA,
            )

            note = ""
            if publish and not is_draft:
                # `/draft/state` answers a published term with a 404 (errorCode
                # 76900), which would read as "the term is gone" rather than
                # "there was nothing to publish".
                note = "already published; nothing to publish."
            elif publish:
                promoted = await client.put(
                    f"{VIYA_ENDPOINT}{_GLOSSARY}/terms/{term_id}/draft/state",
                    params={"action": "publish"},
                    headers={"Accept": _TERM_MEDIA},
                )
                raise_for_viya_status(promoted)
                updated = promoted.json() if promoted.content else updated
                is_draft = False

            result = {
                "term_id": updated.get("id", term_id),
                "name": updated.get("name"),
                "status": updated.get("status"),
                "version": updated.get("version"),
                # Returned whether or not it was the thing changed: a re-parent
                # is otherwise unconfirmable without a second call.
                "parent_id": updated.get("parentId"),
                "is_draft": is_draft,
                "attributes": readable_attributes(updated.get("attributes"), by_uuid),
            }
            if note:
                result["note"] = note
            return result

    @mcp.tool()
    async def import_glossary_terms(
        terms: TermRowList,
        ctx: Context,
        term_type: str | None = None,
        update_existing: bool = False,
        timeout_seconds: int = 600,
    ) -> dict[str, Any]:
        """Create many business terms, and their hierarchy, in one call.

        Building a hierarchy one term at a time means a call per term *and* a
        wait between levels, because a child needs the parent's id from the
        previous response. This uses the glossary's own bulk import instead: one
        request for the whole tree, with parents resolved by **name** rather than
        by id, so nothing has to be threaded through.

        Give each row a ``name`` and, for a child, a ``parent`` — the name of
        another row in the same batch, or the path of a term that already exists
        (levels separated by a backslash). Rows may be given in any order; they
        are sorted so every parent is created before its children.

        Attribute values take the same forms as ``create_glossary_term`` —
        ``True``/``False`` for a boolean, a list for a multi-select — and are
        validated here, per term type, before anything is sent. So are the term
        type's **required** attributes: a row missing one fails inside the job
        with a message naming only the field, so the batch is refused here
        instead, before any of it is committed.

        **The import runs as a job and reports rows individually.** A row can
        fail while the rest succeed, so the result carries ``created``,
        ``failed`` and a ``failures`` list naming each bad row and why. Treat a
        non-empty ``failures`` as a partial import: the successful rows are
        already committed.

        **The result names the term each row became.** ``terms`` carries
        ``{name, path, term_id, existed}`` per row, so the next step — assigning
        an asset, re-parenting, reading one back — needs no lookup. This costs
        one filtered request per 40 distinct names, not one per term.

        Two things the import does that a per-term create does not:

        * **A row is written whole.** With ``update_existing`` the term at that
          path is *replaced*, so an attribute the row omits is reset — not left
          as it was. Without it the existing term is left alone. Either way the
          job counts the row as successful, so ``created`` counts rows the job
          accepted; **``new`` is the count of terms that did not exist before**,
          with ``already_existed`` the rest and ``existed`` saying which is
          which per row.
        * **Omitted attributes take the term type's default**, on new rows as
          well as replaced ones — the same as creating a term through the API.

        Args:
            terms: The rows to create. Each is
                ``{"name": ..., "parent": ..., "definition": ...,
                "description": ..., "attributes": {...}}``; ``term_type`` may be
                given per row to mix types in one batch.
            term_type: The term type for rows that do not name one.
            update_existing: Replace a term that already exists at the same path
                (default false, which leaves it untouched).
            timeout_seconds: How long to wait for the import job (default 600).
        """
        if not terms:
            raise ValueError("terms is empty; there is nothing to import.")

        async with viya_session("import_glossary_terms", ctx) as client:
            # Resolve every term type named, so attributes can be validated and
            # the type's real name written into the CSV.
            type_names: dict[str, str] = {}
            label_maps: dict[str, dict[str, JSONDict]] = {}

            async def type_for(raw: str | None) -> tuple[str, dict[str, JSONDict]]:
                wanted = (raw or term_type or "").strip()
                if not wanted:
                    raise ValueError(
                        "no term type given. Set term_type for the batch, or a term_type "
                        "on each row."
                    )
                if wanted not in type_names:
                    resolved = await resolve_term_type_id(client, wanted)
                    definition = await fetch_term_type(client, resolved)
                    type_names[wanted] = definition.get("name", wanted)
                    _, by_label = attribute_maps(definition)
                    label_maps[wanted] = by_label
                return type_names[wanted], label_maps[wanted]

            rows: list[dict[str, Any]] = []
            used_labels: dict[str, str] = {}
            for row in terms:
                if not isinstance(row, dict):
                    # ValueError, not TypeError: it reaches the model as a plain
                    # message rather than as an internal error, and it is the
                    # caller's input that is wrong, not the code's.
                    raise ValueError(f"each term must be an object; got {row!r}.")  # noqa: TRY004
                name, by_label = await type_for(row.get("term_type"))
                encoded: dict[str, Any] = {}
                for label, value in (row.get("attributes") or {}).items():
                    definition = by_label.get(str(label).strip().lower())
                    if definition is None:
                        valid = sorted(d.get("label", "") for d in by_label.values())
                        raise ValueError(
                            f"term {row.get('name')!r}: unknown attribute {label!r} for term "
                            f"type {name!r}. Valid attributes: {valid}."
                        )
                    real = definition.get("label", "") or str(label)
                    if real.strip().lower() in IMPORT_SYSTEM_COLUMNS:
                        # The importer reads such a column as the term's own
                        # field, so the attribute would be silently skipped and
                        # the term rejected for missing it.
                        raise ValueError(
                            f"attribute {real!r} cannot be set by import: the CSV reads a "
                            f"column of that name as the term's own {real.lower()}. Create "
                            "these terms with create_glossary_term, or set this attribute "
                            "afterwards with update_glossary_term."
                        )
                    used_labels[real.strip().lower()] = real
                    value = encode_attribute(value, definition)
                    encoded[real] = "true" if value is True else "false" if value is False else value
                unmet = unmet_required(encoded, by_label, key="label")
                if unmet:
                    # create_glossary_term makes the same check. Without it here
                    # the row reaches the job and fails with Viya's ``The value
                    # "Owner" for the field "attributes" is invalid``, which
                    # names the attribute but not what is wrong with it — and by
                    # then the rest of the batch is already committed.
                    raise ValueError(
                        f"term {row.get('name')!r}: term type {name!r} requires attribute(s) "
                        f"{unmet}, which this row does not supply. "
                        "get_glossary_term_type lists each one's type and allowed values."
                    )
                rows.append(
                    {
                        "name": row.get("name"),
                        "term_type": name,
                        "parent": row.get("parent"),
                        # Definition and Description are separate columns and
                        # separate fields. With no Definition column the
                        # importer sets the definition to the term's own name.
                        "definition": row.get("definition") or "",
                        "description": row.get("description") or "",
                        "attributes": encoded,
                    }
                )

            ordered = resolve_import_paths(rows)
            columns = [used_labels[key] for key in sorted(used_labels)]
            payload = build_term_csv(ordered, columns)

            started = await client.post(
                f"{VIYA_ENDPOINT}{_GLOSSARY}/importTerms",
                headers={"Accept": _JOB_MEDIA},
                files={"termCSV": ("terms.csv", payload.encode(), "text/csv")},
                data={"updateExisting": "true" if update_existing else "false"},
            )
            raise_for_viya_status(started)
            job_id = started.json().get("id", "")

            job = await await_import_job(client, job_id, timeout_seconds)
            counts = import_counts(job)
            failures = await import_failures(client, job)
            created = counts.get("successful", len(ordered) - len(failures))
            terms, resolution_note = await resolve_imported_terms(client, ordered, job)
            existed = [t for t in terms if t["existed"]]
            result: dict[str, Any] = {
                "requested": len(ordered),
                "created": created,
                "failed": counts.get("errors", len(failures)),
                "job_id": job_id,
                "order": [row["name"] for row in ordered],
                "failures": failures,
                # Verbatim, because "successful" counts a row the job accepted —
                # including one whose term already existed and was left alone.
                # Anything else the job tallies is visible rather than dropped.
                "counts": counts,
                # What the job never reports: which term each row became, and
                # which rows were not new. Without the ids, the next step —
                # assigning an asset, re-parenting — needs a lookup per term.
                "terms": terms,
                "new": len(terms) - len(existed),
                "already_existed": len(existed),
            }
            notes = []
            if failures:
                # The job's own state is "completed" even when every row failed,
                # so saying this plainly is the only way a caller learns of it.
                notes.append(
                    f"{created} of {len(ordered)} term(s) were created; the rest failed and "
                    "are listed in 'failures'. The successful ones are already committed."
                )
            if existed:
                what = "replaced" if update_existing else "left untouched"
                notes.append(
                    f"{len(existed)} of {len(ordered)} row(s) named a term that already "
                    f"existed at that path and were {what}. The import counts them as "
                    "successful either way, so 'new' is the count that says what was created."
                )
            if resolution_note:
                notes.append(resolution_note)
            if notes:
                result["note"] = " ".join(notes)
            return result

    @mcp.tool()
    async def delete_glossary_term(term_id: str, ctx: Context) -> dict[str, str]:
        """Permanently delete a business term.

        The term goes, and with it every assignment to a column that referenced
        it — the data keeps its columns but loses the documented meaning. Check
        ``list_term_assets`` first: a term with assigned assets is in use.

        **There is no cascade.** A term that has children cannot be deleted at
        all: the glossary refuses with ``Cannot delete a term/draft with
        existing children``. Delete the subtree leaf-first — list a term's
        children with ``list_glossary_terms(parent_id=...)`` — or re-parent them
        with ``update_glossary_term`` before deleting this one.

        Args:
            term_id: The glossary term UUID.
        """
        async with viya_session("delete_glossary_term", ctx) as client:
            await delete_resource(f"{_GLOSSARY}/terms/{term_id}", client)
            return {"status": "deleted", "term_id": term_id}

    @mcp.tool()
    async def assign_glossary_term(
        ctx: Context,
        column_name: str,
        term_id: str | None = None,
        term_name: str | None = None,
        resource_uri: str | None = None,
        table_name: str | None = None,
    ) -> dict[str, Any]:
        """Assign a business term to a table column — the step that makes a term govern data.

        Creating a term only defines a word. This attaches it to the column that
        carries it, and it is what ``list_table_terms``, ``list_term_assets`` and
        the catalog's ``Column.term`` facet all read. Assigning the same term to
        the same column twice is reported, not duplicated.

        Args:
            column_name: The column to assign the term to (case-insensitive).
            term_id: The glossary term UUID.
            term_name: Exact term name, if the id is not known. One of the two is
                required.
            resource_uri: The table's source URI (preferred).
            table_name: Table name, if the URI is not known.
        """
        async with viya_session("assign_glossary_term", ctx) as client:
            glossary_id, term = await resolve_term(client, term_id, term_name)
            term_entity = (await term_entities_for(client, [glossary_id])).get(glossary_id)
            if term_entity is None:
                raise ValueError(
                    f"term '{term.get('name')}' has no catalog entity yet, so nothing can be "
                    "assigned to it. Newly created terms are mirrored into the catalog "
                    "asynchronously — retry shortly."
                )
            table = await table_entity(client, resource_uri, table_name)
            column = await column_for(client, table, column_name)

            column_rels, _ = await term_asset_relationships(client, [column["id"]])
            for rel in column_rels:
                if term_entity["id"] in (rel.get("endpoint1Id"), rel.get("endpoint2Id")):
                    return {
                        "status": "already_assigned",
                        "relationship_id": rel.get("id"),
                        "term_id": glossary_id,
                        "term_name": term.get("name"),
                        "column_name": column.get("name"),
                        "table_name": table.get("name"),
                    }

            # endpoint1 is the term and endpoint2 the asset. The relationship is
            # not symmetric and every reader above relies on that order.
            created = await post_json(
                f"{_CATALOG}/instances",
                client,
                body={
                    "version": 1,
                    "instanceType": "relationship",
                    "definition": _TERM_ASSET_DEFINITION,
                    "endpoint1Id": term_entity["id"],
                    "endpoint2Id": column["id"],
                },
                accept=_RELATIONSHIP_MEDIA,
            )
            return {
                "status": "assigned",
                "relationship_id": created.get("id"),
                "term_id": glossary_id,
                "term_name": term.get("name"),
                "column_id": column.get("id"),
                "column_name": column.get("name"),
                "table_name": table.get("name"),
            }

    @mcp.tool()
    async def unassign_glossary_term(
        ctx: Context,
        column_name: str,
        term_id: str | None = None,
        term_name: str | None = None,
        resource_uri: str | None = None,
        table_name: str | None = None,
    ) -> dict[str, Any]:
        """Remove a business term's assignment from a table column.

        Deletes only the link: the term and the column both survive. Use
        ``delete_glossary_term`` to remove the term from the dictionary itself.

        Args:
            column_name: The column to detach the term from (case-insensitive).
            term_id: The glossary term UUID.
            term_name: Exact term name, if the id is not known. One of the two is
                required.
            resource_uri: The table's source URI (preferred).
            table_name: Table name, if the URI is not known.
        """
        async with viya_session("unassign_glossary_term", ctx) as client:
            glossary_id, term = await resolve_term(client, term_id, term_name)
            term_entity = (await term_entities_for(client, [glossary_id])).get(glossary_id)
            table = await table_entity(client, resource_uri, table_name)
            column = await column_for(client, table, column_name)
            not_assigned = {
                "status": "not_assigned",
                "term_id": glossary_id,
                "term_name": term.get("name"),
                "column_name": column.get("name"),
                "table_name": table.get("name"),
            }
            if term_entity is None:
                return not_assigned
            column_rels, _ = await term_asset_relationships(client, [column["id"]])
            for rel in column_rels:
                if term_entity["id"] in (rel.get("endpoint1Id"), rel.get("endpoint2Id")):
                    await delete_resource(f"{_CATALOG}/instances/{rel['id']}", client)
                    return {
                        "status": "unassigned",
                        "relationship_id": rel.get("id"),
                        "term_id": glossary_id,
                        "term_name": term.get("name"),
                        "column_name": column.get("name"),
                        "table_name": table.get("name"),
                    }
            return not_assigned
