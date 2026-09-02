# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure transforms behind the Tier 9 Business Glossary tools.

Everything here is a plain function over data the tools have already fetched —
no HTTP, no MCP, no ``Context`` — so the rules that make the glossary usable can
be read and tested on their own, as with the other ``helpers`` modules. The
request-shaping (which representation carries which field, which ``Accept-Item``
returns relationship endpoints) stays in
:mod:`sas_mcp_server.tools.glossary`, because it is a property of the call
rather than of the data.

Two of the glossary's shapes need translating before a caller can work with them:

* **A term has two identities** — a Glossary object and an Information Catalog
  entity, with different ids, joined by the entity's ``resourceId``. See
  :func:`glossary_id_from_resource`.
* **Custom attributes are keyed by attribute-definition UUID**, with the human
  label held on the *term type*. :func:`readable_attributes` maps them to labels
  for reading and :func:`encode_attributes` maps them back for writing,
  validating against the type's declared required-ness and allowed values.
"""

import re
from typing import Any

JSONDict = dict[str, Any]

# Viya filters travel in the query string, so a few hundred UUIDs would build a
# URL the gateway rejects. Batched id lookups are chunked to stay inside that.
ID_CHUNK = 40

_TERM_RESOURCE_RE = re.compile(r"/glossary/terms/([^/]+)$")


def chunk_ids(values: list[str], size: int | None = None) -> list[list[str]]:
    """Split *values* into batches small enough for one filter expression.

    The default is read at call time rather than bound into the signature, so
    :data:`ID_CHUNK` stays the single place the batch size is defined.
    """
    size = size or ID_CHUNK
    return [values[i : i + size] for i in range(0, len(values), size)]


def glossary_id_from_resource(resource_id: str | None) -> str | None:
    """Pull the glossary term id out of a catalog entity's ``resourceId``.

    Returns ``None`` for a ``resourceId`` that points at something other than a
    term, so a table's or column's entity cannot be mistaken for one.
    """
    match = _TERM_RESOURCE_RE.search(resource_id or "")
    return match.group(1) if match else None


def attribute_maps(term_type: JSONDict) -> tuple[dict[str, str], dict[str, JSONDict]]:
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


def readable_attributes(raw: dict[str, Any] | None, by_uuid: dict[str, str]) -> dict[str, Any]:
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


def encode_attribute(value: Any, definition: JSONDict) -> str:
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


def encode_attributes(
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
        encoded[definition["name"]] = encode_attribute(value, definition)
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


__all__ = [
    "ID_CHUNK",
    "attribute_maps",
    "chunk_ids",
    "encode_attribute",
    "encode_attributes",
    "glossary_id_from_resource",
    "readable_attributes",
]
