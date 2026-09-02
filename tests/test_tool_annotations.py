# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MCP tool annotations (read_only_hint / destructive_hint /
idempotent_hint / open_world_hint) — their derivation from the central
classification in ``tools/_access.py`` and their injection at registration."""

from typing import Any

import pytest
from fastmcp import Client, FastMCP
from mcp.types import ToolAnnotations

from sas_mcp_server import tools
from sas_mcp_server.tools import _access
from sas_mcp_server.tools._access import (
    DESTRUCTIVE_TOOLS,
    IDEMPOTENT_WRITE_TOOLS,
    OPEN_WORLD_TOOLS,
    READ_ONLY_TOOLS,
    WRITE_TOOLS,
    annotations_for,
)


async def _list(tiers=None, read_only=None):
    mcp = FastMCP("annotations-test")

    async def get_token(ctx):
        return "t"

    tools.register_tools(mcp, get_token, tiers=tiers, read_only=read_only)
    async with Client(mcp) as client:
        return await client.list_tools()


# --- the tables are internally consistent -------------------------------------


def test_hint_sets_are_subsets_of_the_write_side():
    """A read-only tool is non-destructive and idempotent by definition, so
    the finer sets may only name write tools (open-world may name either)."""
    assert DESTRUCTIVE_TOOLS <= WRITE_TOOLS, DESTRUCTIVE_TOOLS - WRITE_TOOLS
    assert IDEMPOTENT_WRITE_TOOLS <= WRITE_TOOLS, IDEMPOTENT_WRITE_TOOLS - WRITE_TOOLS
    assert OPEN_WORLD_TOOLS <= (READ_ONLY_TOOLS | WRITE_TOOLS), OPEN_WORLD_TOOLS - (READ_ONLY_TOOLS | WRITE_TOOLS)


def test_arbitrary_code_tools_are_destructive_and_open_world():
    for name in ("execute_sas_code", "submit_batch_job"):
        assert name in DESTRUCTIVE_TOOLS
        assert name in OPEN_WORLD_TOOLS
        assert name not in IDEMPOTENT_WRITE_TOOLS


# --- annotations_for ------------------------------------------------------------


def test_read_only_tool_hints():
    a = annotations_for("list_caslibs")
    assert isinstance(a, ToolAnnotations)
    assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (
        True,
        False,
        True,
        False,
    )


@pytest.mark.parametrize(
    "name,destructive,idempotent,open_world",
    [
        ("execute_sas_code", True, False, True),
        ("delete_report", True, True, False),
        ("update_business_rule", True, True, False),  # ETag-guarded PUT
        ("create_report", True, False, False),  # on_conflict="replace"
        ("upload_data", False, False, True),  # 409 on existing table; url source
        ("promote_table_to_memory", False, True, False),  # explicit already-loaded guard
        ("query_data", False, False, False),  # SELECT-only, but starts work
        ("catalog_run_agent", False, False, False),
        ("publish_ml_champion_model", True, False, False),
    ],
)
def test_write_tool_hints(name, destructive, idempotent, open_world):
    a = annotations_for(name)
    assert a.read_only_hint is False
    assert a.destructive_hint is destructive
    assert a.idempotent_hint is idempotent
    assert a.open_world_hint is open_world


def test_readonly_hint_mirrors_the_enforced_partition_exactly():
    """What clients are told and what MCP_READ_ONLY enforces come from one table."""
    for name in READ_ONLY_TOOLS:
        assert annotations_for(name).read_only_hint is True, name
    for name in WRITE_TOOLS:
        assert annotations_for(name).read_only_hint is False, name


def test_unknown_tool_gets_pessimistic_defaults_and_a_fresh_object():
    a = annotations_for("tool_nobody_classified")
    assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint, a.open_world_hint) == (
        False,
        True,
        False,
        True,
    )
    # A copy each time, so a caller mutating its annotations cannot poison the shared default.
    assert a is not annotations_for("tool_nobody_classified")
    assert a is not _access._PESSIMISTIC


# --- injection at registration, observed on the wire ----------------------------


async def test_every_registered_tool_carries_consistent_annotations():
    listed = await _list()
    assert listed, "no tools registered"
    for t in listed:
        a = t.annotations
        assert a is not None, f"{t.name} has no annotations"
        assert a.read_only_hint is (t.name in READ_ONLY_TOOLS), t.name
        assert a.destructive_hint is (t.name in DESTRUCTIVE_TOOLS), t.name
        assert a.open_world_hint is (t.name in OPEN_WORLD_TOOLS), t.name
        expected_idem = (t.name in READ_ONLY_TOOLS) or (t.name in IDEMPOTENT_WRITE_TOOLS)
        assert a.idempotent_hint is expected_idem, t.name


async def test_no_registered_tool_takes_the_pessimistic_path():
    """The drift guard, from the annotation side: registering an unclassified
    tool would advertise worst-case hints — make that a red build, not a
    silent downgrade."""
    listed = await _list()
    unclassified = {t.name for t in listed} - (READ_ONLY_TOOLS | WRITE_TOOLS)
    assert unclassified == set(), unclassified


async def test_read_only_mode_registers_only_read_only_annotated_tools():
    listed = await _list(read_only=True)
    assert listed
    assert all(t.annotations is not None and t.annotations.read_only_hint for t in listed)


async def test_annotation_counts_match_the_classification():
    listed = await _list()
    names = {t.name for t in listed}
    assert sum(1 for t in listed if t.annotations.read_only_hint) == len(READ_ONLY_TOOLS & names)
    assert sum(1 for t in listed if t.annotations.destructive_hint) == len(DESTRUCTIVE_TOOLS & names)


# --- the recorder honours every decorator form and never overrides a tier ---------


class _Spy:
    """Stand-in target that captures the kwargs each ``tool()`` call forwards."""

    def __init__(self):
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def tool(self, name_or_fn=None, **kwargs):
        if callable(name_or_fn):
            self.calls.append((kwargs.get("name") or name_or_fn.__name__, kwargs))
            return name_or_fn

        def decorator(fn):
            explicit = name_or_fn if isinstance(name_or_fn, str) else kwargs.get("name")
            self.calls.append((explicit or fn.__name__, kwargs))
            return fn

        return decorator


def test_tier_recorder_injects_annotations_for_every_calling_form():
    spy = _Spy()
    rec = tools._TierRecorder(spy, tier=1)

    @rec.tool
    def list_caslibs(): ...

    @rec.tool()
    def get_castable_info(): ...

    @rec.tool("list_castables")
    def _renamed(): ...

    @rec.tool(name="delete_report")
    def _kw(): ...

    seen = {name: kw["annotations"] for name, kw in spy.calls}
    assert set(seen) == {"list_caslibs", "get_castable_info", "list_castables", "delete_report"}
    assert seen["list_caslibs"].read_only_hint is True
    assert seen["list_castables"].read_only_hint is True  # keyed by the explicit name, not _renamed
    assert seen["delete_report"].destructive_hint is True
    assert all(name in tools.TOOL_TIERS for name in seen)


def test_tier_recorder_keeps_annotations_a_tier_passed_explicitly():
    spy = _Spy()
    rec = tools._TierRecorder(spy, tier=3)
    mine = ToolAnnotations(read_only_hint=False, destructive_hint=False, title="Custom")

    @rec.tool(annotations=mine)
    def delete_report(): ...

    ((_, kw),) = spy.calls
    assert kw["annotations"] is mine


def test_annotations_serialize_to_the_spec_wire_names():
    """Clients must receive the spec's camelCase keys, whatever the model calls them.

    MCP SDK v2 renamed the model fields to snake_case and kept camelCase as the
    *serialization* alias, so the field name and the wire name now differ. This
    asserts the wire half directly: on an SDK where that alias is missing the
    payload carries ``read_only_hint``, every client reads zero annotations, and
    all 43 read-only tools silently look writable. Nothing else in this file
    would notice, because the rest of it reads the model, not the payload.
    """
    wire = annotations_for("list_caslibs").model_dump(exclude_none=True, by_alias=True)

    assert wire["readOnlyHint"] is True
    assert set(wire) <= {
        "readOnlyHint",
        "destructiveHint",
        "idempotentHint",
        "openWorldHint",
        "title",
    }, f"non-spec annotation keys on the wire: {sorted(set(wire))}"


def test_every_read_only_tool_advertises_read_only_on_the_wire():
    for name in sorted(READ_ONLY_TOOLS):
        wire = annotations_for(name).model_dump(exclude_none=True, by_alias=True)
        assert wire.get("readOnlyHint") is True, name
    for name in sorted(DESTRUCTIVE_TOOLS):
        wire = annotations_for(name).model_dump(exclude_none=True, by_alias=True)
        assert wire.get("destructiveHint") is True, name
