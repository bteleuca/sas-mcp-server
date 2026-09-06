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

import csv
import io
import re
import uuid
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
    "single-select": "exactly one of the allowed values",
    "multi-select": "one or more of the allowed values; pass a list",
    "date": "yyyy-mm-dd",
    "date-time": "yyyy-mm-ddThh:mm:ssZ (UTC; offsets are converted)",
    "time": "hh:mm:ssZ (UTC; seconds required; offsets are converted)",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)(\.\d{1,3})?(.*)$")
_TIME_RE = re.compile(r"^(\d{2}):(\d{2})(?::(\d{2}))?(\.\d{1,3})?(.*)$")
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")
_SECONDS_IN_A_DAY = 24 * 60 * 60

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
        attribute_id = definition.get("name", "")
        if attribute_id:
            label = definition.get("label", "") or attribute_id
            by_uuid[attribute_id] = definition
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
    for attribute_id, value in (raw or {}).items():
        if value in (None, ""):
            continue
        definition = by_uuid.get(attribute_id)
        if definition is None:
            readable[attribute_id] = value
            continue
        readable[definition.get("label", "") or attribute_id] = decode_attribute(value, definition)
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
    """Normalise to ``hh:mm:ssZ``, both the seconds and the ``Z`` being required.

    An offset is *converted*, as it is for a date-time, not dropped: storing
    ``09:15:00+02:00`` as ``09:15:00Z`` would be two hours wrong and say nothing
    about it, which is worse than either accepting or rejecting the value. With
    no date to carry, a conversion that crosses midnight wraps within the day —
    a time attribute is a time of day, not an instant.
    """
    text = str(value).strip()
    match = _TIME_RE.match(text)
    if not match:
        _fail(definition, value, "hh:mm:ssZ")
        raise AssertionError("unreachable")
    hours, minutes, seconds, _millis, rest = match.groups()
    if int(hours) > 23 or int(minutes) > 59 or int(seconds or 0) > 59:
        # Caught here because the wrap below would otherwise turn 25:00:00 into
        # a plausible-looking 01:00:00 rather than reporting the typo.
        _fail(definition, value, "hh:mm:ssZ with hh<24, mm<60 and ss<60")
    suffix = _normalise_offset(rest, definition, value)
    total = int(hours) * 3600 + int(minutes) * 60 + int(seconds or 0)
    offset = _OFFSET_RE.match(suffix)
    if offset:
        sign, off_hours, off_minutes = offset.groups()
        shift = int(off_hours) * 3600 + int(off_minutes) * 60
        total += -shift if sign == "+" else shift
    total %= _SECONDS_IN_A_DAY
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}Z"


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
    attr_type = (definition.get("type") or "").lower()
    # An empty value clears the attribute, which is how the glossary itself
    # stores an unset one. It has to bypass the checks below, or a required
    # single-select could be set once and never cleared again.
    if value is None or value == "":
        if attr_type == "boolean":
            # The one type with no empty form: Viya answers ``The value "" for
            # the field "<label>" is invalid``. Saying so here is the whole
            # point of encoding, and a boolean has a third state nowhere else.
            raise ValueError(
                f"attribute '{definition.get('label')}' is a boolean and cannot be cleared: "
                "the glossary has no empty boolean and rejects ''. Set it to true or false."
            )
        return ""
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
        missing = unmet_required(encoded, by_label, key="name")
        if missing:
            raise ValueError(
                f"this term type requires attribute(s) {missing}, which were not supplied. "
                "get_glossary_term_type lists each one's type and allowed values."
            )
    return encoded


def is_empty(value: Any) -> bool:
    """Is this attribute value the glossary's idea of unset?

    ``False`` is *not* empty. It reads as one under ``value or ""``, which is
    how a boolean set to false came to be reported as a missing required
    attribute — an edit refused over a value that was there all along.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return not value
    return False


def has_default(definition: JSONDict) -> bool:
    """Does this attribute declare a default the glossary fills in when omitted?

    A required attribute with a default is never actually missing: the service
    applies the default on create and on import — verified live, a required
    attribute omitted from an imported row came back holding its default — so
    refusing the call here would reject what the service would have accepted.
    """
    return not is_empty(definition.get("defaultValue"))


def unmet_required(
    values: dict[str, Any],
    by_label: dict[str, JSONDict],
    *,
    key: str = "name",
    defaults_apply: bool = True,
) -> list[str]:
    """Required attribute labels that *values* leaves empty.

    *key* says how *values* is keyed: ``"name"`` for the UUID-keyed map the API
    stores, ``"label"`` for the label-keyed rows an import is built from.

    *defaults_apply* covers a required attribute that declares a default, which
    the service fills in — but only where it does. It does so on create and on
    import; an update replays the term's stored attributes and has no such step,
    so a value the term never received is genuinely missing there.
    """
    lookup = (
        {str(k).strip().lower(): v for k, v in values.items()} if key == "label" else values
    )
    unmet = []
    for definition in by_label.values():
        if not definition.get("required"):
            continue
        if defaults_apply and has_default(definition):
            continue
        wanted = (
            (definition.get("label") or "").strip().lower()
            if key == "label"
            else definition.get("name", "")
        )
        if is_empty(lookup.get(wanted)):
            unmet.append(definition.get("label", ""))
    return sorted(unmet)


def missing_required(merged: dict[str, Any], by_label: dict[str, JSONDict]) -> list[str]:
    """Required attributes left empty in a term that is about to be written back.

    An update is a whole-resource PUT, so it replays every attribute the term
    already had. If an attribute was made required *after* the term was created,
    that replay carries an empty value for it and Viya rejects the write —
    naming an attribute the caller never mentioned, on an edit to something
    else. Checking the merged term first turns that into a message that says
    which attribute and why. A declared default does not save it: defaults are
    applied when a term is created, not when one is rewritten.
    """
    return unmet_required(merged, by_label, key="name", defaults_apply=False)



# The attribute types the glossary understands. A term type can declare no
# others, and the service rejects an unknown one with a message that does not
# say what the valid set is.
ATTRIBUTE_TYPES: tuple[str, ...] = (
    "single-line",
    "multi-line",
    "single-select",
    "multi-select",
    "boolean",
    "date",
    "date-time",
    "time",
)

# Types whose values are chosen from a fixed list, so a definition without one
# is a term type nobody can write to.
_CHOICE_TYPES = ("single-select", "multi-select")


def build_attribute_definitions(
    specs: list[dict[str, Any]] | None,
    existing: list[JSONDict] | None = None,
    *,
    remove: list[str] | None = None,
) -> list[JSONDict]:
    """Turn caller-friendly attribute specs into the definitions the API stores.

    Each definition is keyed by a UUID in its ``name`` field, and **that UUID is
    what every existing term's stored values are filed under**. So an edit
    matches an incoming spec to an existing definition and keeps its UUID: mint
    a fresh one and every term of that type silently loses the value, with the
    old key left orphaned in its attribute map.

    Matching is by label by default, which is enough to add, edit and reorder —
    but not to **rename**, because a new label matches nothing and so mints a
    new UUID. A spec may therefore give ``attribute_id`` (as
    ``get_glossary_term_type`` returns it) to say *which* attribute it is,
    leaving ``label`` free to be the new name.

    The API will not mint the UUIDs itself — omitting them fails with ``The
    value for field "name" must be unique``, which names neither the attribute
    nor the real problem.

    Args:
        specs: ``{label, type, required?, allowed_values?, default?,
            description?, attribute_id?}`` per attribute. ``None`` keeps
            *existing* untouched.
        existing: the type's current definitions, whose UUIDs are preserved.
        remove: labels to drop. Terms keep the stored value under its now
            unknown UUID rather than losing it outright.
    """
    by_label = {
        (d.get("label") or "").strip().lower(): d for d in (existing or []) if d.get("label")
    }
    by_id = {d["name"]: d for d in (existing or []) if d.get("name")}
    dropped = {label.strip().lower() for label in (remove or [])}
    unknown = dropped - by_label.keys()
    if unknown:
        raise ValueError(
            f"cannot remove attribute(s) {sorted(unknown)}: this term type has no such "
            f"attribute. It declares {sorted(by_label)}."
        )

    if specs is None:
        kept = [d for d in (existing or []) if (d.get("label") or "").strip().lower() not in dropped]
        return kept

    built: list[JSONDict] = []
    seen: set[str] = set()
    reused: set[str] = set()
    for spec in specs:
        label = str(spec.get("label", "")).strip()
        if not label:
            raise ValueError("every attribute needs a 'label'.")
        key = label.lower()
        if key in seen:
            raise ValueError(f"attribute label '{label}' is given twice; labels must be unique.")
        seen.add(key)

        attribute_id = str(spec.get("attribute_id", "") or "").strip()
        if attribute_id and attribute_id not in by_id:
            raise ValueError(
                f"attribute '{label}' names attribute_id {attribute_id!r}, which this term "
                f"type does not have. get_glossary_term_type returns the id of each one. "
                "Omit it to add a new attribute."
            )
        previous = by_id.get(attribute_id) if attribute_id else by_label.get(key)
        if previous is not None:
            reused.add(previous.get("name", ""))

        attr_type = str(spec.get("type", "")).strip().lower()
        if attr_type not in ATTRIBUTE_TYPES:
            raise ValueError(
                f"attribute '{label}' has type {attr_type or '(missing)'!r}; valid types are "
                f"{list(ATTRIBUTE_TYPES)}."
            )

        allowed = spec.get("allowed_values") or spec.get("items") or []
        if attr_type in _CHOICE_TYPES and not allowed:
            raise ValueError(
                f"attribute '{label}' is a {attr_type}, so it needs 'allowed_values' — "
                "without them no value can ever be stored in it."
            )
        if allowed and attr_type not in _CHOICE_TYPES:
            raise ValueError(
                f"attribute '{label}' is a {attr_type}, which takes no 'allowed_values'; "
                f"only {list(_CHOICE_TYPES)} do."
            )

        definition: JSONDict = {
            # Reuse the existing UUID so terms already carrying a value keep it.
            "name": (previous or {}).get("name") or str(uuid.uuid4()),
            "label": label,
            "type": attr_type,
        }
        if spec.get("required"):
            definition["required"] = True
        if allowed:
            definition["items"] = [str(item) for item in allowed]
        default = spec.get("default", spec.get("defaultValue"))
        if default not in (None, ""):
            # Stored as a string even for a boolean, unlike a term's own value.
            definition["defaultValue"] = (
                str(default).lower() if isinstance(default, bool) else str(default)
            )
        if spec.get("description"):
            definition["description"] = str(spec["description"])
        built.append(definition)

    # Anything the caller did not mention survives, unless explicitly removed —
    # the same merge rule update_glossary_term uses for a term's own fields.
    # Survival is decided by identity, not by label: an attribute a spec renamed
    # has already been rebuilt and must not reappear under its old label, while
    # one whose label a *different* spec has taken over is still its own
    # attribute and must not vanish because the name now reads as mentioned.
    for key, definition in by_label.items():
        if key in dropped or definition.get("name", "") in reused:
            continue
        built.append(definition)

    labels = [str(d.get("label", "")).strip().lower() for d in built]
    clashes = sorted({label for label in labels if labels.count(label) > 1})
    if clashes:
        raise ValueError(
            f"attribute label(s) {clashes} would appear twice on this term type. Renaming an "
            "attribute onto a label another one already uses needs that other one renamed or "
            "removed in the same call."
        )
    return built


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def attribute_matches(value: Any, wanted: Any) -> bool:
    """Does one decoded attribute value satisfy one filter value?

    A multi-select decodes to a list, where "contains" is the useful reading —
    asking for ``{"Regions": "EMEA"}`` should find a term tagged EMEA *and*
    APAC, not only one tagged EMEA alone. Passing a list asks for all of them.
    Everything else compares as a string, case-insensitively, because the caller
    is typing values they read off a screen.
    """
    if isinstance(value, list):
        wanted_items = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
        have = {str(item).strip().lower() for item in value}
        return all(str(item).strip().lower() in have for item in wanted_items)
    if isinstance(value, bool) or isinstance(wanted, bool):
        return _as_bool(value) == _as_bool(wanted)
    return str(value).strip().lower() == str(wanted).strip().lower()


def matches_attribute_filter(readable: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Does a term's label-keyed attributes satisfy every clause of *wanted*?

    Clauses are ANDed, and a term missing the attribute never matches — an
    absent value is not an empty one to compare against.
    """
    lowered = {str(label).strip().lower(): value for label, value in readable.items()}
    for label, expected in wanted.items():
        key = str(label).strip().lower()
        if key not in lowered:
            return False
        if not attribute_matches(lowered[key], expected):
            return False
    return True


# The columns the import reads as the term itself rather than as a custom
# attribute. The first occurrence of one of these names in the header wins, so
# an attribute sharing one of these labels cannot be set through an import.
#
# ``definition`` belongs here — verified live: a term type with an attribute
# labelled "Definition" imported through a column of that name left the
# attribute unset and wrote the value to the term's own definition instead.
IMPORT_SYSTEM_COLUMNS: tuple[str, ...] = (
    "name",
    "type",
    "path",
    "definition",
    "description",
    "requirements",
    "status",
)

# The import expresses the hierarchy as a path in the ``Path`` column, separated
# by a backslash — which is why a term name may not contain one, though a
# forward slash is fine. Verified against a live import building three levels.
PATH_SEPARATOR = "\\"

_IMPORT_FAILURE_RE = re.compile(
    r"Import failed for term \((?P<term>.*?)\) on row (?P<row>\d+)\. Error: (?P<error>.*)"
)


def resolve_import_paths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order rows parent-first and turn each ``parent`` into a full path.

    The import resolves a row's parent by looking up the ``Path`` column against
    terms that already exist *or* that earlier rows of the same file have just
    created. So a child must follow its parent, and must name the parent's whole
    path rather than only its name — get either wrong and the row fails with
    ``A parent term with the path "..." does not exist``.

    A caller should have to know neither. Each row names its ``parent`` — another
    row in the batch, or an existing term's path — and this produces the full
    path and an order that works.

    Raises :class:`ValueError` for a duplicate name or a circular parent chain,
    both of which would otherwise surface as a confusing per-row failure.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            raise ValueError("every term needs a 'name'.")
        if PATH_SEPARATOR in name:
            raise ValueError(
                f"term name {name!r} contains a backslash, which separates the levels of an "
                "import path and so cannot appear in a name. A forward slash is fine."
            )
        key = name.lower()
        if key in by_name:
            raise ValueError(f"term name {name!r} appears twice in this batch.")
        by_name[key] = row

    resolved: dict[str, str] = {}
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()

    def place(row: dict[str, Any]) -> str:
        """Return this row's parent path, having emitted its ancestors first."""
        name = str(row["name"]).strip()
        key = name.lower()
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ValueError(
                f"the parent chain through {name!r} is circular; a term cannot be its own "
                "ancestor."
            )
        visiting.add(key)
        parent = str(row.get("parent", "") or "").strip()
        if not parent:
            prefix = ""
        elif parent.lower() in by_name:
            parent_row = by_name[parent.lower()]
            grandparent = place(parent_row)
            prefix = (
                f"{grandparent}{PATH_SEPARATOR}{parent_row['name']}"
                if grandparent
                else str(parent_row["name"])
            )
        else:
            # Not in this batch, so it must already exist; take it as a path.
            prefix = parent
        visiting.discard(key)
        resolved[key] = prefix
        ordered.append({**row, "_path": prefix})
        return prefix

    for row in rows:
        place(row)
    return ordered


def build_term_csv(rows: list[dict[str, Any]], attribute_columns: list[str]) -> str:
    """Render ordered rows as the CSV the import expects.

    *rows* come from :func:`resolve_import_paths`, so each carries ``_path``.
    Values are already encoded — a boolean as ``true``/``false``, a multi-select
    as its comma-joined string — so this only quotes and joins them.

    ``Definition`` and ``Description`` are separate system columns and are *not*
    interchangeable: with no ``Definition`` column the importer sets the term's
    definition to its own name, which is what a whole hierarchy imported without
    one comes back holding. Both are always written, empty when unset, so the
    field a reader actually sees is the one the caller wrote.
    """
    header = ["Name", "Type", "Path", "Definition", "Description", *attribute_columns]
    buffer = io.StringIO()
    # Records are CRLF-terminated, as a CSV export from the UI produces.
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    for row in rows:
        attributes = row.get("attributes") or {}
        lowered = {str(key).strip().lower(): value for key, value in attributes.items()}
        writer.writerow(
            [
                row.get("name", ""),
                row.get("term_type", "") or "",
                row.get("_path", ""),
                row.get("definition", "") or "",
                row.get("description", "") or "",
                *[lowered.get(label.strip().lower(), "") for label in attribute_columns],
            ]
        )
    return buffer.getvalue()


def import_lookup_names(ordered: list[dict[str, Any]]) -> list[str]:
    """Every term name needed to resolve an import's rows to their ids.

    A row's own name, plus each level of its resolved ``_path`` — the levels
    matter because a path may lead through terms that already existed and are
    not rows of this batch, and the walk in :func:`match_imported_rows` starts
    from the root.

    De-duplicated case-insensitively but returned in their original spelling,
    since the filter matches on the stored value.
    """
    names: dict[str, str] = {}
    for row in ordered:
        parts = [str(row.get("name", ""))]
        parts += str(row.get("_path", "") or "").split(PATH_SEPARATOR)
        for part in parts:
            part = part.strip()
            if part:
                names.setdefault(part.lower(), part)
    return list(names.values())


def match_imported_rows(
    ordered: list[dict[str, Any]],
    candidates: list[JSONDict],
    *,
    started_at: str = "",
) -> list[dict[str, Any]]:
    """Match each import row to the term it became, by walking its path.

    The import reports names and tallies, never ids — so assigning an asset or
    re-parenting anything afterwards means finding each term again. *candidates*
    are the terms whose names appear anywhere in the batch, from which a
    ``(parentId, name)`` index rebuilds the hierarchy exactly: sibling names are
    unique, so that pair identifies a term even when the same name is used under
    several parents.

    *started_at* is the import job's own creation timestamp. A term created
    before the job began was already there and the job left it alone (or, with
    ``update_existing``, replaced it) — which is the only way to tell that from
    a term the job created, because the job counts both as successful.

    A row whose term cannot be found comes back with ``term_id`` of ``None``
    rather than being dropped, so a partial resolution stays visible.
    """
    index: dict[tuple[str | None, str], JSONDict] = {}
    for item in candidates:
        name = str(item.get("name", "")).strip().lower()
        if name:
            index[(item.get("parentId"), name)] = item

    matched: list[dict[str, Any]] = []
    for row in ordered:
        parent_id: str | None = None
        found = True
        for level in str(row.get("_path", "") or "").split(PATH_SEPARATOR):
            level = level.strip()
            if not level:
                continue
            ancestor = index.get((parent_id, level.lower()))
            if ancestor is None:
                found = False
                break
            parent_id = ancestor.get("id")

        name = str(row.get("name", "")).strip()
        term = index.get((parent_id, name.lower())) if found else None
        created = str((term or {}).get("creationTimeStamp") or "")
        matched.append(
            {
                "name": row.get("name"),
                "path": row.get("_path", ""),
                "term_id": (term or {}).get("id"),
                # Absent timestamps must not read as "already there": an
                # unresolved row would otherwise be reported as pre-existing.
                "existed": bool(term and started_at and created and created < started_at),
            }
        )
    return matched


def parse_import_log(text: str) -> list[dict[str, Any]]:
    """Pull the per-row failures out of an import job's log.

    The job reports ``completed`` even when every row failed, so the log is the
    only place that says what went wrong. Each failure names the term, the row
    and the reason; a line that does not parse is kept verbatim rather than
    dropped.
    """
    failures: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("Import failed"):
            continue
        match = _IMPORT_FAILURE_RE.match(line)
        if match:
            failures.append(
                {
                    "term": match.group("term"),
                    "row": int(match.group("row")),
                    "error": match.group("error").strip(),
                }
            )
        else:
            failures.append({"term": None, "row": None, "error": line})
    return failures


__all__ = [
    "resolve_import_paths",
    "parse_import_log",
    "build_term_csv",
    "PATH_SEPARATOR",
    "IMPORT_SYSTEM_COLUMNS",
    "ATTRIBUTE_TYPES",
    "ID_CHUNK",
    "WIRE_FORMATS",
    "attribute_matches",
    "attribute_maps",
    "build_attribute_definitions",
    "chunk_ids",
    "decode_attribute",
    "encode_attribute",
    "encode_attributes",
    "has_default",
    "is_empty",
    "matches_attribute_filter",
    "glossary_id_from_resource",
    "import_lookup_names",
    "match_imported_rows",
    "missing_required",
    "readable_attributes",
    "unmet_required",
]
