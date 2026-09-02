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
Returned raw it is unreadable, and unwriteable without a second lookup. These
tools resolve UUID→label on read and label→UUID on write, validating against the
type's declared required-ness and allowed values, so a caller works in the
vocabulary the glossary UI shows.

**Term↔asset links are catalog relationships**, not glossary objects. A
``glossaryTermAsset`` relationship joins the term entity (always ``endpoint1``)
to a column entity (always ``endpoint2``) — the direction held across every such
relationship in a live deployment, and the readers here rely on it. Reading the
endpoints needs :data:`_RELATIONSHIP_MEDIA` as the collection's ``Accept-Item``;
without it the server returns summaries with the endpoints stripped, so the
traversal silently finds nothing rather than failing.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import httpx
from fastmcp import Context, FastMCP
from pydantic import BeforeValidator

from ..config import VIYA_ENDPOINT
from ..viya_client import (
    JSONDict,
    delete_resource,
    get_json,
    post_json,
    put_json,
    raise_for_viya_status,
)
from ._common import coerce_json_dict, make_session_helpers

# Tolerant alias for the attributes map, which some MCP clients deliver as a
# JSON-encoded string (see _common.coerce_json_dict). The schema is unchanged.
AttributeMap = Annotated[dict[str, Any], BeforeValidator(coerce_json_dict)]

_GLOSSARY = "/glossary"
_CATALOG = "/catalog"

_COLLECTION_MEDIA = "application/vnd.sas.collection+json"
_TERM_MEDIA = "application/vnd.sas.glossary.term+json"
_TERM_TYPE_MEDIA = "application/vnd.sas.glossary.term.type+json"
_SEARCH_MEDIA = "application/vnd.sas.metadata.search.collection+json"
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

# Viya filters travel in the query string, so a few hundred UUIDs would build a
# URL the gateway rejects. Batched id lookups are chunked to stay inside that.
_ID_CHUNK = 40

_TERM_RESOURCE_RE = re.compile(r"/glossary/terms/([^/]+)$")


def _quote(value: str) -> str:
    """Escape a value for a Viya filter string literal (single quotes double)."""
    return (value or "").replace("'", "''")


def _in_filter(field: str, values: list[str]) -> str:
    """Build ``in(field,'a','b',...)`` for a batched lookup."""
    joined = "','".join(_quote(v) for v in values)
    return f"in({field},'{joined}')"


def _chunks(values: list[str], size: int | None = None) -> list[list[str]]:
    """Split *values* into batches small enough for one filter expression.

    The default is read at call time rather than bound into the signature, so
    :data:`_ID_CHUNK` stays the single place the batch size is defined.
    """
    size = size or _ID_CHUNK
    return [values[i : i + size] for i in range(0, len(values), size)]


def _glossary_id_from_resource(resource_id: str | None) -> str | None:
    """Pull the glossary term id out of a catalog entity's ``resourceId``."""
    match = _TERM_RESOURCE_RE.search(resource_id or "")
    return match.group(1) if match else None


def _attribute_maps(term_type: JSONDict) -> tuple[dict[str, str], dict[str, JSONDict]]:
    """Return ``(uuid -> label, lowercased label -> definition)`` for a term type.

    Labels are matched case-insensitively on write because they are display
    strings a caller reads off a screen, not identifiers.
    """
    by_uuid: dict[str, str] = {}
    by_label: dict[str, JSONDict] = {}
    for definition in term_type.get("attributes", []) or []:
        uuid = definition.get("name", "")
        label = definition.get("label", "") or uuid
        if uuid:
            by_uuid[uuid] = label
            by_label[label.strip().lower()] = definition
    return by_uuid, by_label


def _readable_attributes(raw: dict[str, Any] | None, by_uuid: dict[str, str]) -> dict[str, Any]:
    """Re-key a term's ``attributes`` map from UUIDs to their labels.

    Empty values are dropped: the glossary stores every declared attribute on
    every term, so keeping them would bury the two or three actually filled in.
    A UUID with no matching definition is kept under the UUID rather than
    discarded, so nothing is silently lost when a term type has been edited.
    """
    readable: dict[str, Any] = {}
    for uuid, value in (raw or {}).items():
        if value in (None, ""):
            continue
        readable[by_uuid.get(uuid, uuid)] = value
    return readable


def _encode_attribute(value: Any, definition: JSONDict) -> str:
    """Coerce one attribute value to the string form the glossary stores.

    Every attribute value is a string on the wire, booleans and dates included.
    Passing a real bool or number is accepted by the API but stores ``True`` or
    ``1``, which the glossary UI then displays verbatim.
    """
    # An empty value clears the attribute, which is how the glossary itself
    # stores an unset one. It has to bypass the checks below, or a required
    # single-select could be set once and never cleared again.
    if value is None or value == "":
        return ""
    attr_type = (definition.get("type") or "").lower()
    if attr_type == "boolean":
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).strip().lower()
        if text not in ("true", "false"):
            raise ValueError(
                f"attribute '{definition.get('label')}' is a boolean; got {value!r}. "
                "Pass true or false."
            )
        return text
    text = str(value)
    if attr_type == "single-select":
        allowed = definition.get("items") or []
        if allowed and text not in allowed:
            raise ValueError(
                f"attribute '{definition.get('label')}' only accepts {allowed}; got {text!r}."
            )
    return text


def _encode_attributes(
    supplied: dict[str, Any] | None,
    by_label: dict[str, JSONDict],
    *,
    require_all: bool,
) -> dict[str, str]:
    """Map a caller's label-keyed attributes onto the UUID keys the API wants.

    Raises :class:`ValueError` naming the valid labels for an unknown one and —
    when *require_all* (a create, where there is nothing to fall back on) —
    naming the required attributes that were not supplied. Both are mistakes a
    model can correct from the message alone, which a bare HTTP 400 does not
    allow.
    """
    encoded: dict[str, str] = {}
    for label, value in (supplied or {}).items():
        definition = by_label.get(label.strip().lower())
        if definition is None:
            valid = sorted(d.get("label", "") for d in by_label.values())
            raise ValueError(
                f"unknown attribute '{label}' for this term type. Valid attributes: "
                f"{valid or 'none — this term type declares no custom attributes'}. "
                "get_glossary_term_type lists each one's type and allowed values."
            )
        encoded[definition["name"]] = _encode_attribute(value, definition)
    if require_all:
        missing = sorted(
            d.get("label", "")
            for d in by_label.values()
            if d.get("required") and d["name"] not in encoded
        )
        if missing:
            raise ValueError(
                f"this term type requires attribute(s) {missing}, which were not supplied. "
                "get_glossary_term_type lists each one's type and allowed values."
            )
    return encoded


def register(mcp: FastMCP, get_token: Callable[[Context], Awaitable[str]]) -> None:
    """Register Tier 9 (Business Glossary) tools on *mcp*."""

    viya_session, _ = make_session_helpers(get_token)

    # --- shared lookups ------------------------------------------------------

    async def instance_collection(
        client: httpx.AsyncClient, filter_expr: str, limit: int, item_media: str
    ) -> list[JSONDict]:
        """GET a ``/catalog/instances`` collection with an explicit ``Accept-Item``.

        ``Accept-Item`` is what decides whether items come back complete or as
        summaries with ``resourceId`` and the relationship endpoints stripped.
        :func:`get_json` has no parameter for it, so this issues the request
        directly rather than widening a helper every other tier depends on.
        """
        resp = await client.get(
            f"{VIYA_ENDPOINT}{_CATALOG}/instances",
            headers={"Accept": _COLLECTION_MEDIA, "Accept-Item": item_media},
            params={"filter": filter_expr, "start": 0, "limit": limit},
        )
        raise_for_viya_status(resp)
        return resp.json().get("items", []) or []

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
        for chunk in _chunks(sorted({i for i in ids if i})):
            for item in await instance_collection(
                client, _in_filter("id", chunk), len(chunk) + 10, _ENTITY_MEDIA
            ):
                found[item["id"]] = item
        return found

    async def term_entities_for(
        client: httpx.AsyncClient, glossary_ids: list[str]
    ) -> dict[str, JSONDict]:
        """Batch-resolve glossary term ids to their catalog term entities."""
        found: dict[str, JSONDict] = {}
        for chunk in _chunks(sorted({i for i in glossary_ids if i})):
            resources = [f"/glossary/terms/{gid}" for gid in chunk]
            for item in await instance_collection(
                client, _in_filter("resourceId", resources), len(chunk) + 10, _ENTITY_MEDIA
            ):
                gid = _glossary_id_from_resource(item.get("resourceId"))
                if gid:
                    found[gid] = item
        return found

    async def term_asset_relationships(
        client: httpx.AsyncClient, entity_ids: list[str]
    ) -> list[JSONDict]:
        """Every ``glossaryTermAsset`` relationship touching any of *entity_ids*.

        One filtered call per chunk, endpoints included. The naive shape — filter
        for ids, then GET each relationship to read its endpoints — costs a round
        trip per link for the same answer.
        """
        rels: list[JSONDict] = []
        for chunk in _chunks(sorted({i for i in entity_ids if i})):
            expr = (
                f"and(eq(definition,'{_TERM_ASSET_DEFINITION}'),"
                f"or({_in_filter('endpoint1Id', chunk)},{_in_filter('endpoint2Id', chunk)}))"
            )
            rels.extend(await instance_collection(client, expr, 500, _RELATIONSHIP_MEDIA))
        return rels

    async def table_entity(
        client: httpx.AsyncClient, resource_uri: str | None, table_name: str | None
    ) -> JSONDict:
        """Resolve a table to its catalog entity, by resource URI or by name."""
        if resource_uri:
            items = await instance_collection(
                client, f"eq(resourceId,'{_quote(resource_uri)}')", 2, _ENTITY_MEDIA
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
        return await instance_collection(
            client,
            f"startsWith(resourceId,'{_quote(table_resource)}/columns/')",
            limit,
            _ENTITY_MEDIA,
        )

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
            params={"filter": f"eq(name,'{_quote(term_name)}')", "start": 0, "limit": 5},
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
        attribute, its data type, whether it is required, and — for a
        single-select — the exact values accepted. ``create_glossary_term`` takes
        attributes keyed by the ``label`` shown here, so this is also the
        vocabulary to write in.

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

    # --- finding terms -------------------------------------------------------

    @mcp.tool()
    async def search_glossary_terms(
        query: str, ctx: Context, limit: int = 20, start: int = 0
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
                        "term_id": _glossary_id_from_resource(entity.get("resourceId")),
                        "catalog_entity_id": hit.get("id"),
                        "name": hit.get("name"),
                        "term_type": hit.get("typeLabel"),
                        "definition": attributes.get("definition", ""),
                        "status": attributes.get("reviewStatus", ""),
                        "assigned_asset_count": attributes.get("assignedAssets", 0),
                        "score": hit.get("score"),
                    }
                )
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
        """
        async with viya_session("list_glossary_terms", ctx) as client:
            clauses = []
            if term_type:
                resolved = await resolve_term_type_id(client, term_type)
                clauses.append(f"eq(termTypeId,'{_quote(resolved)}')")
            if parent_id:
                clauses.append(f"eq(parentId,'{_quote(parent_id)}')")
            if name_contains:
                clauses.append(f"contains(name,'{_quote(name_contains)}')")
            params: dict[str, Any] = {
                "start": start,
                "limit": limit,
                "sortBy": "name:ascending",
                "allowDrafts": "all" if include_drafts else "none",
            }
            if clauses:
                params["filter"] = clauses[0] if len(clauses) == 1 else f"and({','.join(clauses)})"
            data = await get_json(
                f"{_GLOSSARY}/terms", client, params=params, accept=_COLLECTION_MEDIA
            )
            items = [
                {
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
                for item in data.get("items", []) or []
            ]
            return {"count": data.get("count", len(items)), "start": start, "items": items}

    @mcp.tool()
    async def get_glossary_term(term_id: str, ctx: Context) -> dict[str, Any]:
        """Get one business term in full, with its custom attributes named rather than hashed.

        The raw API returns ``attributes`` keyed by attribute-definition UUID,
        which is unreadable on its own. This resolves each key to the label the
        glossary UI shows and drops the ones left empty, so what comes back is
        the term as a person would read it; ``attribute_ids`` keeps the raw
        mapping for anything that needs it.

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
            by_uuid, _ = _attribute_maps(term_type)
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
                "attributes": _readable_attributes(raw_attributes, by_uuid),
                "attribute_ids": raw_attributes,
                "created_by": term.get("createdBy"),
                "modified_by": term.get("modifiedBy"),
                "modified": term.get("modifiedTimeStamp"),
            }

    # --- term <-> asset linkage ---------------------------------------------

    @mcp.tool()
    async def list_term_assets(
        ctx: Context, term_id: str | None = None, term_name: str | None = None
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
            rels = await term_asset_relationships(client, [entity_id])
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
            return {
                "term_id": glossary_id,
                "catalog_entity_id": entity_id,
                "name": term.get("name"),
                "asset_count": len(resolved),
                "assets": resolved,
            }

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
            rels = await term_asset_relationships(client, list(by_entity))

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
                        "term_id": _glossary_id_from_resource(entity.get("resourceId")),
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
        make attributes mandatory and a term missing one is rejected. Values are
        validated here (booleans, single-select options), so a mistake comes back
        naming the attribute instead of as an opaque HTTP 400.

        **Terms are published by default.** The underlying API defaults to
        creating a *draft*, which nobody but its author can see; that is almost
        never what a caller asking to "create a term" means, so this publishes
        unless ``publish`` is set false.

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
                ``{"Scope": "Group", "Used in Risk": true}``.
            publish: Publish immediately (default true). False leaves a draft.
        """
        async with viya_session("create_glossary_term", ctx) as client:
            term_type_id = await resolve_term_type_id(client, term_type)
            type_definition = await fetch_term_type(client, term_type_id)
            by_uuid, by_label = _attribute_maps(type_definition)
            encoded = _encode_attributes(attributes, by_label, require_all=True)

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
                "attributes": _readable_attributes(created.get("attributes"), by_uuid),
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
        attributes: AttributeMap | None = None,
    ) -> dict[str, Any]:
        """Update a business term's text or custom attributes.

        The glossary API replaces the whole term on update, so this reads the
        current one first and merges your changes into it: omitting an argument
        leaves that field alone rather than blanking it. ``attributes`` merges
        the same way, per attribute — pass only the ones you are changing, and
        set one to ``""`` to clear it.

        A term's type cannot be changed after creation, nor can a published
        term's parent.

        Args:
            term_id: The glossary term UUID.
            name: New name (unique among siblings, max 100 characters).
            definition: New definition.
            description: New description, max 1000 characters.
            label: New display label.
            attributes: Custom attributes to change, keyed by label.
        """
        async with viya_session("update_glossary_term", ctx) as client:
            current = await get_json(f"{_GLOSSARY}/terms/{term_id}", client, accept=_TERM_MEDIA)
            type_definition = await fetch_term_type(client, current.get("termTypeId", ""))
            by_uuid, by_label = _attribute_maps(type_definition)

            merged = dict(current.get("attributes") or {})
            merged.update(_encode_attributes(attributes, by_label, require_all=False))

            body = {key: value for key, value in current.items() if key != "links"}
            if name is not None:
                body["name"] = name
            if definition is not None:
                body["definition"] = definition
            if description is not None:
                body["description"] = description
            if label is not None:
                body["label"] = label
            body["attributes"] = merged

            updated = await put_json(
                f"{_GLOSSARY}/terms/{term_id}",
                client,
                body,
                content_type=_TERM_MEDIA,
                accept=_TERM_MEDIA,
            )
            return {
                "term_id": updated.get("id", term_id),
                "name": updated.get("name"),
                "status": updated.get("status"),
                "version": updated.get("version"),
                "attributes": _readable_attributes(updated.get("attributes"), by_uuid),
            }

    @mcp.tool()
    async def delete_glossary_term(term_id: str, ctx: Context) -> dict[str, str]:
        """Permanently delete a business term.

        The term goes, and with it every assignment to a column that referenced
        it — the data keeps its columns but loses the documented meaning. Check
        ``list_term_assets`` first: a term with assigned assets is in use.
        Deleting a parent term also affects its children.

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

            for rel in await term_asset_relationships(client, [column["id"]]):
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
            for rel in await term_asset_relationships(client, [column["id"]]):
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
