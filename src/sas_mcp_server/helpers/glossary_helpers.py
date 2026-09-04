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
  label held on the *term type*, and each attribute type has its own wire format
  that the API documents nowhere. :func:`readable_attributes` maps them to labels
  for reading and :func:`encode_attributes` maps them back for writing,
  validating against the type's declared required-ness and allowed values.
  :data:`WIRE_FORMATS` states the formats; :func:`encode_attribute` enforces them.
"""

import re
from datetime import datetime, timedelta
from typing import Any

JSONDict = dict[str, Any]

# Viya filters travel in the query string, so a few hundred UUIDs would build a
# URL the gateway rejects. Batched id lookups are chunked to stay inside that.
#
# Deliberately not a tool argument, unlike the ``limit`` the list tools take.
# Those change *what the caller gets back*; this only changes how many requests
# it takes to fetch the same answer — chunking 100 ids as 40+40+20 or as 100
# returns identical results. So there is no value a caller could pick that
# improves the answer, and a large one silently reintroduces the rejected-URL
# failure it exists to prevent. Tune it here, where the reason lives.
ID_CHUNK = 40

_TERM_RESOURCE_RE = re.compile(r"/glossary/terms/([^/]+)$")

# What Viya accepts per attribute type, established by testing each one against a
# live glossary rather than from documentation — the service publishes no OpenAPI
# document. Surfaced to callers by ``get_glossary_term_type`` so the contract is
# visible before a write, not after a 400.
WIRE_FORMATS: dict[str, str] = {
    "boolean": "JSON true/false (not the strings 'true'/'false')",
    "multi-select": "one or more of the allowed values; pass a list",
    "date": "yyyy-mm-dd",
    "date-time": "yyyy-mm-ddThh:mm:ssZ (UTC; offsets are converted)",
    "time": "hh:mm:ssZ (UTC; seconds required)",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)(\.\d{1,3})?(.*)$")
_TIME_RE = re.compile(r"^(\d{2}):(\d{2})(?::(\d{2}))?(\.\d{1,3})?(.*)$")
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")

# Multi-select values are stored as one comma-joined string with no spaces. Viya
# rejects a JSON array, "a, b" and "a;b" alike, so the join happens here.
_MULTI_SELECT_SEPARATOR = ","


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


def attribute_maps(term_type: JSONDict) -> tuple[dict[str, JSONDict], dict[str, JSONDict]]:
    """Return ``(uuid -> definition, lowercased label -> definition)`` for a term type.

    Both maps carry the whole definition rather than just the label, because
    decoding a stored value needs its declared type as much as encoding one does
    — a multi-select comes back as a comma-joined string and is only splittable
    if you know that is what it is.

    Labels are matched case-insensitively on write because they are display
    strings a caller reads off a screen, not identifiers.
    """
    by_uuid: dict[str, JSONDict] = {}
    by_label: dict[str, JSONDict] = {}
    for definition in term_type.get("attributes", []) or []:
        uuid = definition.get("name", "")
        if uuid:
            label = definition.get("label", "") or uuid
            by_uuid[uuid] = definition
            by_label[label.strip().lower()] = definition
    return by_uuid, by_label


def decode_attribute(value: Any, definition: JSONDict) -> Any:
    """Turn one stored attribute value into the shape a caller works in.

    The inverse of :func:`encode_attribute`, and only multi-select actually
    needs it: the glossary stores the selected items as one comma-joined string,
    which reads as a single odd value rather than as the list it is.
    """
    if (definition.get("type") or "").lower() == "multi-select" and isinstance(value, str):
        return [part for part in value.split(_MULTI_SELECT_SEPARATOR) if part]
    return value


def readable_attributes(
    raw: dict[str, Any] | None, by_uuid: dict[str, JSONDict]
) -> dict[str, Any]:
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
        definition = by_uuid.get(uuid)
        if definition is None:
            readable[uuid] = value
            continue
        readable[definition.get("label", "") or uuid] = decode_attribute(value, definition)
    return readable


def _fail(definition: JSONDict, got: Any, expected: str) -> None:
    """Raise the same shape of message for every rejected attribute value."""
    attr_type = (definition.get("type") or "").lower()
    raise ValueError(
        f"attribute '{definition.get('label')}' is a {attr_type}; got {got!r}. "
        f"Expected {expected}."
    )


def _encode_boolean(value: Any, definition: JSONDict) -> bool:
    """Return a real JSON boolean, which is the only form Viya accepts.

    The string ``"true"`` is rejected with ``The value "true" for the field
    "<label>" is invalid`` — as are ``"True"``, ``"Yes"`` and ``1`` — so the
    value has to leave here as a ``bool`` and stay one through serialisation.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    _fail(definition, value, "true or false")
    raise AssertionError("unreachable")


def _encode_multi_select(value: Any, definition: JSONDict) -> str:
    """Join the selected items the way the glossary stores them.

    Viya rejects a JSON array, a space after the comma, and any other separator,
    so a caller's list is normalised to ``a,b`` here. Each item is checked
    against the type's ``items`` first, because Viya's own rejection names only
    the attribute and not which item was wrong.
    """
    if isinstance(value, str):
        # A caller may already have the stored form, or a spaced variant of it.
        chosen = [part.strip() for part in value.split(_MULTI_SELECT_SEPARATOR)]
    elif isinstance(value, (list, tuple, set)):
        chosen = [str(item).strip() for item in value]
    else:
        chosen = [str(value).strip()]
    chosen = [item for item in chosen if item]
    allowed = definition.get("items") or []
    if allowed:
        unknown = [item for item in chosen if item not in allowed]
        if unknown:
            raise ValueError(
                f"attribute '{definition.get('label')}' only accepts {allowed}; "
                f"{unknown} not among them."
            )
    return _MULTI_SELECT_SEPARATOR.join(chosen)


def _normalise_offset(rest: str, definition: JSONDict, value: Any) -> str:
    """Reduce a trailing timezone to the ``Z`` Viya insists on.

    An explicit offset is rejected outright by the service, so rather than
    passing on a 400 the offset is applied and the result expressed in UTC —
    the same instant, in the only spelling that is accepted.
    """
    rest = rest.strip()
    if rest in ("", "Z", "z"):
        return "Z"
    if _OFFSET_RE.match(rest):
        return rest  # applied by the caller, which has the hours and minutes
    _fail(definition, value, "a UTC time ending in Z")
    raise AssertionError("unreachable")


def _encode_datetime(value: Any, definition: JSONDict) -> str:
    """Normalise to ``yyyy-mm-ddThh:mm:ssZ``.

    Viya requires the ``T``, the seconds and the ``Z``; it rejects a date on its
    own and any numeric offset. Each of those is recoverable without guessing at
    intent, so they are fixed here instead of being forwarded to a 400.
    """
    text = str(value).strip()
    if _DATE_RE.match(text):
        return f"{text}T00:00:00Z"
    match = _DATETIME_RE.match(text)
    if not match:
        _fail(definition, value, "yyyy-mm-ddThh:mm:ssZ")
        raise AssertionError("unreachable")
    date_part, time_part, _millis, rest = match.groups()
    if len(time_part) == 5:  # hh:mm
        time_part = f"{time_part}:00"
    suffix = _normalise_offset(rest, definition, value)
    offset = _OFFSET_RE.match(suffix)
    if offset:
        sign, hours, minutes = offset.groups()
        try:
            moment = datetime.fromisoformat(f"{date_part}T{time_part}")
        except ValueError:
            _fail(definition, value, "yyyy-mm-ddThh:mm:ssZ")
            raise AssertionError("unreachable") from None
        delta = timedelta(hours=int(hours), minutes=int(minutes))
        moment = moment - delta if sign == "+" else moment + delta
        return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{date_part}T{time_part}Z"


def _encode_time(value: Any, definition: JSONDict) -> str:
    """Normalise to ``hh:mm:ssZ``, both the seconds and the ``Z`` being required."""
    text = str(value).strip()
    match = _TIME_RE.match(text)
    if not match:
        _fail(definition, value, "hh:mm:ssZ")
        raise AssertionError("unreachable")
    hours, minutes, seconds, _millis, rest = match.groups()
    _normalise_offset(rest, definition, value)
    return f"{hours}:{minutes}:{seconds or '00'}Z"


def _encode_date(value: Any, definition: JSONDict) -> str:
    """Normalise to ``yyyy-mm-dd``, accepting a date-time and keeping its date."""
    text = str(value).strip()
    if _DATE_RE.match(text):
        return text
    match = _DATETIME_RE.match(text)
    if match:
        return match.group(1)
    _fail(definition, value, "yyyy-mm-dd")
    raise AssertionError("unreachable")


def encode_attribute(value: Any, definition: JSONDict) -> Any:
    """Coerce one attribute value to the form the glossary actually accepts.

    Not every value is a string: a boolean must travel as a JSON boolean, and a
    multi-select as one comma-joined string. Both are rejected outright in any
    other form, with an error naming only the attribute — so the shaping and the
    validation happen here, where the caller's input can still be named.
    """
    # An empty value clears the attribute, which is how the glossary itself
    # stores an unset one. It has to bypass the checks below, or a required
    # single-select could be set once and never cleared again.
    if value is None or value == "":
        return ""
    attr_type = (definition.get("type") or "").lower()
    if attr_type == "boolean":
        return _encode_boolean(value, definition)
    if attr_type == "multi-select":
        return _encode_multi_select(value, definition)
    if attr_type == "date":
        return _encode_date(value, definition)
    if attr_type == "date-time":
        return _encode_datetime(value, definition)
    if attr_type == "time":
        return _encode_time(value, definition)
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
) -> dict[str, Any]:
    """Map a caller's label-keyed attributes onto the UUID keys the API wants.

    Raises :class:`ValueError` naming the valid labels for an unknown one and —
    when *require_all* (a create, where there is nothing to fall back on) —
    naming the required attributes that were not supplied. Both are mistakes a
    model can correct from the message alone, which a bare HTTP 400 does not
    allow.
    """
    encoded: dict[str, Any] = {}
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
        # An explicit "" counts as absent: it is what the glossary stores for an
        # unset attribute, so a required one set to it is rejected server-side.
        missing = sorted(
            d.get("label", "")
            for d in by_label.values()
            if d.get("required") and encoded.get(d["name"], "") == ""
        )
        if missing:
            raise ValueError(
                f"this term type requires attribute(s) {missing}, which were not supplied. "
                "get_glossary_term_type lists each one's type and allowed values."
            )
    return encoded


def missing_required(merged: dict[str, Any], by_label: dict[str, JSONDict]) -> list[str]:
    """Required attributes left empty in a term that is about to be written back.

    An update is a whole-resource PUT, so it replays every attribute the term
    already had. If an attribute was made required *after* the term was created,
    that replay carries an empty value for it and Viya rejects the write —
    naming an attribute the caller never mentioned, on an edit to something
    else. Checking the merged term first turns that into a message that says
    which attribute and why.
    """
    return sorted(
        d.get("label", "")
        for d in by_label.values()
        if d.get("required") and str(merged.get(d["name"], "") or "") == ""
    )


__all__ = [
    "ID_CHUNK",
    "WIRE_FORMATS",
    "attribute_maps",
    "chunk_ids",
    "decode_attribute",
    "encode_attribute",
    "encode_attributes",
    "glossary_id_from_resource",
    "missing_required",
    "readable_attributes",
]
