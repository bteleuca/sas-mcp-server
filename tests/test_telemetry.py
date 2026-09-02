# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.tools.base import ToolResult

from sas_mcp_server.telemetry import (
    GOAL_SCHEMA,
    SUSPECT_DURATION_MS,
    TelemetryMiddleware,
    classify_error,
    install_telemetry,
)


class FakeLogger:
    """Captures records instead of writing to disk."""

    max_field_bytes = 4096
    max_result_bytes = 4096

    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


def _build_server(box):
    mcp = FastMCP("test")

    @mcp.tool()
    def echo(text: str, count: int = 1) -> dict:
        box["received"] = {"text": text, "count": count}
        return {"echoed": text * count, "count": count}

    return mcp


# ----------------------------- OFF by default ------------------------------ #


def test_install_off_by_default(monkeypatch, tmp_path):
    import sas_mcp_server.config as config

    monkeypatch.setattr(config, "COLLECTION_MODE", False, raising=False)
    mcp = FastMCP("test")
    before = list(mcp.middleware)
    assert install_telemetry(mcp, "stdio") is None
    # A fresh FastMCP already carries its own default middleware, so assert the
    # list is UNCHANGED (no telemetry added) rather than empty.
    assert mcp.middleware == before
    assert not any(isinstance(m, TelemetryMiddleware) for m in mcp.middleware)


# ----------------------------- schema injection ---------------------------- #


@pytest.mark.asyncio
async def test_on_list_tools_injects_goal_first_and_required():
    box = {}
    mcp = _build_server(box)
    mcp.add_middleware(
        TelemetryMiddleware(FakeLogger(), require_goal=True, transport="stdio")
    )
    async with Client(mcp) as client:
        tools = await client.list_tools()
        schema = tools[0].input_schema
        props = schema["properties"]
        assert list(props.keys())[0] == "goal"
        assert props["goal"] == GOAL_SCHEMA
        assert "goal" in schema["required"]
        assert "text" in props and "count" in props
        assert len(tools) == 1  # count unchanged -> coverage guard stays green


@pytest.mark.asyncio
async def test_on_list_tools_require_goal_false_omits_required():
    box = {}
    mcp = _build_server(box)
    mcp.add_middleware(
        TelemetryMiddleware(FakeLogger(), require_goal=False, transport="stdio")
    )
    async with Client(mcp) as client:
        schema = (await client.list_tools())[0].input_schema
        assert "goal" in schema["properties"]
        assert "goal" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_on_list_tools_idempotent():
    mw = TelemetryMiddleware(FakeLogger(), require_goal=True, transport="stdio")

    tool = SimpleNamespace(
        name="x",
        parameters={"type": "object", "properties": {"a": {"type": "string"}}},
    )

    def _copy(update):
        return SimpleNamespace(name="x", parameters=update["parameters"])

    tool.model_copy = _copy  # type: ignore[attr-defined]

    async def call_next(_):
        return [tool]

    out1 = await mw.on_list_tools(MagicMock(), call_next)
    assert list(out1[0].parameters["properties"].keys()) == ["goal", "a"]

    async def call_next2(_):
        return out1  # already injected

    out2 = await mw.on_list_tools(MagicMock(), call_next2)
    assert list(out2[0].parameters["properties"].keys()).count("goal") == 1


# ----------------------------- call + strip + log -------------------------- #


@pytest.mark.asyncio
async def test_on_call_tool_strips_goal_and_logs():
    box = {}
    logger = FakeLogger()
    mcp = _build_server(box)
    mcp.add_middleware(
        TelemetryMiddleware(logger, require_goal=True, transport="stdio")
    )
    async with Client(mcp) as client:
        res = await client.call_tool(
            "echo", {"goal": "user asked to echo", "text": "ab", "count": 2}
        )
    # underlying tool ran WITHOUT goal
    assert box["received"] == {"text": "ab", "count": 2}
    assert res.structured_content == {"echoed": "abab", "count": 2}
    # a run_start header precedes the first call record of the run
    header = logger.records[0]
    assert header["record"] == "run_start"
    assert header["transport"] == "stdio"
    assert header["result_mode"] == "always"
    assert header["run_id"]
    assert isinstance(header["pid"], int)
    # a well-formed call record was written
    rec = logger.records[-1]
    assert rec["tool"] == "echo"
    assert rec["goal"] == "user asked to echo"
    assert "goal" not in rec["arguments"]
    assert rec["arguments"]["text"] == "ab"
    assert rec["result"] is not None
    assert rec["result_logged"] is True
    assert rec["run_id"] == header["run_id"]
    assert "session_id" not in rec
    assert rec["ts"]
    assert rec["status"] == "success"
    assert isinstance(rec["duration_ms"], float)
    assert rec["seq"] == 1
    assert len(rec["args_hash"]) == 12
    assert "arguments_truncated" in rec and "result_truncated" in rec


@pytest.mark.asyncio
async def test_on_call_tool_records_redacted_and_truncated():
    logger = FakeLogger()
    logger.max_field_bytes = 50
    logger.max_result_bytes = 50
    mw = TelemetryMiddleware(logger, require_goal=True, transport="http")

    msg = SimpleNamespace(
        name="echo",
        arguments={"goal": "g", "password": "hunter2", "blob": "x" * 500},
    )
    msg.model_copy = lambda update: SimpleNamespace(
        name="echo", arguments=update["arguments"]
    )
    ctx = SimpleNamespace(message=msg, fastmcp_context=None)
    ctx.copy = lambda **kw: SimpleNamespace(
        message=kw["message"], fastmcp_context=None
    )

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(ctx, call_next)
    rec = logger.records[-1]
    # object type preserved even when truncated
    assert isinstance(rec["arguments"], dict)
    assert rec["arguments"]["password"] == "[REDACTED]"
    assert rec["arguments_truncated"] is True


@pytest.mark.asyncio
async def test_goal_is_redacted_and_bounded():
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")
    jwt = "eyJ" + "a" * 40

    msg = SimpleNamespace(
        name="echo", arguments={"goal": f"rerun with Bearer {jwt}", "text": "q"}
    )
    msg.model_copy = lambda update: SimpleNamespace(
        name="echo", arguments=update["arguments"]
    )
    ctx = SimpleNamespace(message=msg, fastmcp_context=None)
    ctx.copy = lambda **kw: SimpleNamespace(
        message=kw["message"], fastmcp_context=None
    )

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(ctx, call_next)
    rec = logger.records[-1]
    assert "[REDACTED]" in rec["goal"] and jwt not in rec["goal"]


@pytest.mark.asyncio
async def test_log_results_false_records_shape_only():
    logger = FakeLogger()
    mw = TelemetryMiddleware(
        logger, require_goal=True, transport="stdio", log_results=False
    )

    msg = SimpleNamespace(name="q", arguments={"goal": "g", "table": "t"})
    msg.model_copy = lambda update: SimpleNamespace(
        name="q", arguments=update["arguments"]
    )
    ctx = SimpleNamespace(message=msg, fastmcp_context=None)
    ctx.copy = lambda **kw: SimpleNamespace(
        message=kw["message"], fastmcp_context=None
    )

    async def call_next(_):
        # structured_content must be a dict for a real ToolResult; wrap the
        # PII-bearing rows so we can assert the shape summary omits them.
        return ToolResult(
            structured_content={"rows": [{"ssn": "123-45-6789", "email": "a@b.com"}]}
        )

    await mw.on_call_tool(ctx, call_next)
    rec = logger.records[-1]
    assert rec["result_logged"] is False
    assert "123-45-6789" not in json.dumps(rec["result"])
    assert rec["result"]["_type"] == "object"
    # v2 shape carries key NAMES (schema, not data) instead of a bare count.
    assert rec["result"]["_keys"] == ["rows"]


@pytest.mark.asyncio
async def test_on_call_tool_error_records_and_reraises():
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")

    msg = SimpleNamespace(name="boom", arguments={"goal": "g"})
    msg.model_copy = lambda update: SimpleNamespace(
        name="boom", arguments=update["arguments"]
    )
    ctx = SimpleNamespace(message=msg, fastmcp_context=None)
    ctx.copy = lambda **kw: SimpleNamespace(
        message=kw["message"], fastmcp_context=None
    )

    async def call_next(_):
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        await mw.on_call_tool(ctx, call_next)

    rec = logger.records[-1]
    assert rec["status"] == "error"
    assert rec["error"] == "ValueError: kaboom"


@pytest.mark.asyncio
async def test_logger_failure_never_breaks_call():
    class Exploding:
        max_field_bytes = 4096
        max_result_bytes = 4096

        def write(self, record):
            raise RuntimeError("disk full")

    mw = TelemetryMiddleware(Exploding(), require_goal=True, transport="stdio")
    msg = SimpleNamespace(name="echo", arguments={"goal": "g", "text": "q"})
    msg.model_copy = lambda update: SimpleNamespace(
        name="echo", arguments=update["arguments"]
    )
    ctx = SimpleNamespace(message=msg, fastmcp_context=None)
    seen = {}

    def _copy(**kw):
        return SimpleNamespace(message=kw["message"], fastmcp_context=None)

    ctx.copy = _copy

    async def call_next(c):
        seen["args"] = dict(c.message.arguments)
        return ToolResult(structured_content={"ok": True})

    res = await mw.on_call_tool(ctx, call_next)
    assert res.structured_content == {"ok": True}
    assert "goal" not in seen["args"]


def test_run_id_is_per_process_not_per_instance():
    """The grouping key is minted per PROCESS, not read off the transport.

    Two middlewares in one process (a composed server, a re-import) must not
    look like two server runs sharing a pid — the header stamps both.
    """
    a = TelemetryMiddleware(FakeLogger(), require_goal=True, transport="stdio")
    b = TelemetryMiddleware(FakeLogger(), require_goal=True, transport="http")
    assert a.run_id
    assert a.run_id == b.run_id


@pytest.mark.asyncio
async def test_run_id_is_stable_across_calls():
    """Pins the regression a `@property` returning a fresh uuid would cause."""
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(_ctx("t", {"goal": "a"}), call_next)
    await mw.on_call_tool(_ctx("t", {"goal": "b"}), call_next)
    ids = {r["run_id"] for r in logger.records}
    assert len(ids) == 1


def test_extract_output_prefers_structured_then_content():
    mw = TelemetryMiddleware(FakeLogger(), require_goal=True, transport="stdio")
    # structured_content wins
    r1 = SimpleNamespace(structured_content={"a": 1}, content=None)
    assert mw._extract_output(r1) == {"a": 1}
    # falls back to joined text of content blocks
    blocks = [SimpleNamespace(text="line1"), SimpleNamespace(text="line2")]
    r2 = SimpleNamespace(structured_content=None, content=blocks)
    assert mw._extract_output(r2) == "line1\nline2"
    # nothing extractable -> None
    r3 = SimpleNamespace(structured_content=None, content=None)
    assert mw._extract_output(r3) is None


# --------------------- v2: tool outcomes, failures mode, rescue ------------- #


def _ctx(name, arguments):
    msg = SimpleNamespace(name=name, arguments=arguments)
    msg.model_copy = lambda update: SimpleNamespace(name=name, arguments=update["arguments"])
    ctx = SimpleNamespace(message=msg, fastmcp_context=None)
    ctx.copy = lambda **kw: SimpleNamespace(message=kw["message"], fastmcp_context=None)
    return ctx


@pytest.mark.asyncio
async def test_tool_declared_failure_fields_extracted():
    """apply_failed & co. return as DATA (MCP success) — the record must say so."""
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio", log_results=False)

    async def call_next(_):
        return ToolResult(
            structured_content={
                "status": "apply_failed",
                "message": "Viya rejected the operations (HTTP 400).",
                "failed_operation_index": 2,
            }
        )

    await mw.on_call_tool(_ctx("apply_report_operations", {"goal": "g", "report_id": "r"}), call_next)
    rec = logger.records[-1]
    assert rec["status"] == "success"  # MCP layer saw a normal result...
    assert rec["tool_status"] == "apply_failed"  # ...but the tool declared failure
    assert rec["is_tool_error"] is True
    assert rec["failed_operation_index"] == 2
    assert "Viya rejected" in rec["tool_message"]


@pytest.mark.asyncio
async def test_success_tool_status_is_not_an_error():
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio", log_results=False)

    async def call_next(_):
        return ToolResult(structured_content={"status": "applied", "report_id": "r"})

    await mw.on_call_tool(_ctx("apply_report_operations", {"goal": "g"}), call_next)
    rec = logger.records[-1]
    assert rec["tool_status"] == "applied"
    assert rec["is_tool_error"] is False
    assert "tool_message" not in rec


@pytest.mark.asyncio
async def test_failures_mode_logs_full_result_only_on_failure():
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio", log_results="failures")

    async def ok(_):
        return ToolResult(structured_content={"status": "applied", "secret_rows": [1, 2, 3]})

    async def failed(_):
        return ToolResult(structured_content={"status": "apply_failed", "message": "boom"})

    await mw.on_call_tool(_ctx("t", {"goal": "g"}), ok)
    success_rec = logger.records[-1]
    assert success_rec["result_logged"] is False
    assert success_rec["result"]["_type"] == "object"  # shape only

    await mw.on_call_tool(_ctx("t", {"goal": "g"}), failed)
    failure_rec = logger.records[-1]
    assert failure_rec["result_logged"] is True
    assert failure_rec["result"]["status"] == "apply_failed"  # full content


@pytest.mark.asyncio
async def test_unparsed_tool_input_is_rescued():
    """The {'__unparsedToolInput': {'raw': ...}} client bug is unwrapped so the
    tool receives real arguments (and the record marks the rescue)."""
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")
    seen = {}

    async def call_next(c):
        seen["args"] = dict(c.message.arguments)
        return ToolResult(structured_content={"ok": True})

    blob = json.dumps({"goal": "build it", "report_id": "r1", "operations": [{"addPage": {"pageName": "P"}}]})
    await mw.on_call_tool(_ctx("apply_report_operations", {"__unparsedToolInput": {"raw": blob}}), call_next)
    assert seen["args"] == {"report_id": "r1", "operations": [{"addPage": {"pageName": "P"}}]}
    rec = logger.records[-1]
    assert rec["input_rescued"] is True
    assert rec["goal"] == "build it"


@pytest.mark.asyncio
async def test_goal_salvaged_from_unparseable_blob():
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    # Broken JSON that still contains a goal — coverage must not drop to null.
    blob = '{"goal": "check parsing", "operations": [{'
    await mw.on_call_tool(_ctx("t", {"__unparsedToolInput": {"raw": blob}}), call_next)
    rec = logger.records[-1]
    assert rec["goal"] == "check parsing"
    assert "input_rescued" not in rec


@pytest.mark.asyncio
async def test_seq_increments_and_args_hash_stable_per_arguments():
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(_ctx("t", {"goal": "a", "x": 1}), call_next)
    await mw.on_call_tool(_ctx("t", {"goal": "b", "x": 1}), call_next)
    await mw.on_call_tool(_ctx("t", {"goal": "c", "x": 2}), call_next)
    calls = [r for r in logger.records if r.get("record") != "run_start"]
    assert [r["seq"] for r in calls] == [1, 2, 3]
    # goal is excluded from the hash: identical arguments -> identical hash.
    assert calls[0]["args_hash"] == calls[1]["args_hash"]
    assert calls[0]["args_hash"] != calls[2]["args_hash"]


@pytest.mark.asyncio
async def test_grouping_ignores_transport_session_identity():
    """The whole point of run_id: one run, whatever the transport reports.

    A sessionless transport gives every call a fresh (or absent) session id.
    Grouping on that would emit a header per call and restart seq at 1 each
    time, shattering the trace; grouping on run_id must not.
    """
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="http")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    for n in range(3):
        ctx = _ctx("t", {"goal": "g"})
        # A per-request session id, as a sessionless transport would mint.
        ctx.fastmcp_context = SimpleNamespace(session_id=f"per-request-{n}")
        await mw.on_call_tool(ctx, call_next)

    headers = [r for r in logger.records if r.get("record") == "run_start"]
    calls = [r for r in logger.records if r.get("record") != "run_start"]
    assert len(headers) == 1  # ONE header for the run, not one per call
    assert [r["seq"] for r in calls] == [1, 2, 3]  # one continuous trace
    assert {r["run_id"] for r in calls} == {mw.run_id}
    assert all("session_id" not in r for r in logger.records)


def _client_ctx(name, version, goal="g"):
    ctx = _ctx("t", {"goal": goal})
    ctx.fastmcp_context = SimpleNamespace(
        request_context=SimpleNamespace(
            session=SimpleNamespace(
                client_params=SimpleNamespace(
                    clientInfo=SimpleNamespace(name=name, version=version)
                )
            )
        )
    )
    return ctx


@pytest.mark.asyncio
async def test_header_is_re_emitted_so_rotation_cannot_orphan_records(monkeypatch):
    """A header written once per process is simply gone after rotation."""
    import sas_mcp_server.telemetry as telemetry_mod

    monkeypatch.setattr(telemetry_mod, "HEADER_INTERVAL_RECORDS", 3)
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    for _ in range(7):
        await mw.on_call_tool(_ctx("t", {"goal": "g"}), call_next)

    headers = [r for r in logger.records if r.get("record") == "run_start"]
    calls = [r for r in logger.records if r.get("record") != "run_start"]
    assert len(calls) == 7
    assert len(headers) == 3  # records 1, 4, 7
    # Every emission is identical, so a consumer may take any one of them —
    # that is what makes re-emitting safe rather than a duplicate-key hazard.
    assert all(h == headers[0] for h in headers)
    # seq keeps running across headers; the run is one trace, not three.
    assert [r["seq"] for r in calls] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.asyncio
async def test_header_is_retried_when_the_first_write_fails():
    """The header must not be lost for the whole run because of one bad write."""

    class FlakyLogger(FakeLogger):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        def write(self, record):
            if self.fail_next:
                self.fail_next = False
                raise OSError("disk full")
            super().write(record)

    logger = FlakyLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(_ctx("t", {"goal": "a"}), call_next)  # write blows up
    assert logger.records == []
    await mw.on_call_tool(_ctx("t", {"goal": "b"}), call_next)
    assert logger.records[0].get("record") == "run_start"  # retried, not lost


@pytest.mark.asyncio
async def test_header_ts_is_the_run_start_not_the_first_call():
    """run_start.ts must anchor the run, not duplicate the first call's ts."""
    import sas_mcp_server.telemetry as telemetry_mod

    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(_ctx("t", {"goal": "g"}), call_next)
    header = logger.records[0]
    assert header["ts"] == telemetry_mod._RUN_STARTED_AT
    assert header["ts"] < logger.records[-1]["ts"]


@pytest.mark.asyncio
async def test_client_identity_is_per_record_and_omitted_when_unknown():
    """Client identity belongs to the caller, so it rides on each record."""
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="http")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(_client_ctx("claude-code", "2.1.0"), call_next)
    assert logger.records[-1]["client_name"] == "claude-code"
    assert logger.records[-1]["client_version"] == "2.1.0"

    # No handshake reachable (routine when sessionless): omitted, not null.
    await mw.on_call_tool(_ctx("t", {"goal": "g"}), call_next)
    assert "client_name" not in logger.records[-1]
    # ...and never in the header, which only holds per-run constants.
    header = next(r for r in logger.records if r.get("record") == "run_start")
    assert "client_name" not in header and "client" not in header


@pytest.mark.asyncio
async def test_client_identity_is_kept_as_two_fields_not_joined():
    """A name containing '/' must stay splittable from its version."""
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="http")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(_client_ctx("acme/agent-sdk", "1.0"), call_next)
    assert logger.records[-1]["client_name"] == "acme/agent-sdk"
    assert logger.records[-1]["client_version"] == "1.0"


@pytest.mark.asyncio
async def test_client_identity_is_redacted_and_bounded():
    """clientInfo is caller-supplied, so it gets the same treatment as any
    other logged field — .env.sample promises capped + Bearer/JWT-scrubbed."""
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="http")

    async def call_next(_):
        return ToolResult(structured_content={"ok": True})

    await mw.on_call_tool(
        _client_ctx("x" * 4000, "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def"),
        call_next,
    )
    rec = logger.records[-1]
    assert len(rec["client_name"]) <= 160  # capped well under a field budget
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in rec["client_version"]


@pytest.mark.asyncio
async def test_error_text_scrubs_viya_host():
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio")
    mw._viya_host = "viya.internal.example.com"

    async def call_next(_):
        raise RuntimeError("404 for url 'https://viya.internal.example.com/casManagement/x'")

    with pytest.raises(RuntimeError):
        await mw.on_call_tool(_ctx("t", {"goal": "g"}), call_next)
    rec = logger.records[-1]
    assert "viya.internal.example.com" not in rec["error"]
    assert "[viya-host]" in rec["error"]


@pytest.mark.asyncio
async def test_failures_mode_result_is_host_scrubbed():
    """Full results quote raw VA bodies and carry open_url — the host scrub
    must cover record['result'], not just error/tool_message."""
    logger = FakeLogger()
    mw = TelemetryMiddleware(logger, require_goal=True, transport="stdio", log_results="failures")
    mw._viya_host = "viya.internal.example.com"

    async def failed(_):
        return ToolResult(
            structured_content={
                "status": "apply_failed",
                "message": "Viya said: see https://viya.internal.example.com/reports/x",
                "detail": {"href": "https://viya.internal.example.com/visualAnalytics/y"},
            }
        )

    await mw.on_call_tool(_ctx("t", {"goal": "g"}), failed)
    rec = logger.records[-1]
    dumped = json.dumps(rec)
    assert "viya.internal.example.com" not in dumped
    assert "[viya-host]" in rec["result"]["message"]
    assert "[viya-host]" in rec["tool_message"]


def test_failure_status_heuristic_covers_repo_statuses():
    from sas_mcp_server.helpers.telemetry_helpers import is_tool_failure_status

    failures = [
        "apply_failed", "export_failed", "invalid_operation", "invalid_request",
        "unknown_object_type", "unknown_format", "not_found", "not_addable",
        "missing_identifier", "file_not_found", "table_not_found", "format_not_supported",
        "table_not_global", "export_too_large", "no_active_session", "error",
    ]
    successes = ["created", "applied", "copied", "deleted", "ok", "valid", "promoted", "already_global", "success"]
    assert all(is_tool_failure_status(s) for s in failures)
    assert not any(is_tool_failure_status(s) for s in successes)
    assert not is_tool_failure_status(None)


def test_parse_log_results_tristate():
    from sas_mcp_server.env import parse_log_results

    assert parse_log_results("failures") == "failures"
    assert parse_log_results("ALWAYS") == "always"
    assert parse_log_results("true") == "always"
    assert parse_log_results("false") == "never"
    assert parse_log_results("banana") == "never"  # warns, degrades safely
    # Unset defaults to the DIAGNOSTIC mode: with shape-only results a success
    # and a tool-declared failure look identical in the log.
    assert parse_log_results(None) == "failures"
    assert parse_log_results("") == "failures"


# --------------------- end-to-end install + on-disk write ------------------- #


@pytest.mark.asyncio
async def test_install_enabled_end_to_end_writes_jsonl(monkeypatch, tmp_path):
    """install_telemetry() with COLLECTION_MODE on wires a real logger and a
    call produces one JSONL record on disk (exercises the actual entry point)."""
    import sas_mcp_server.config as config

    log_path = tmp_path / "sub" / "tool-usage.log"
    monkeypatch.setattr(config, "COLLECTION_MODE", True, raising=False)
    monkeypatch.setattr(config, "COLLECTION_LOG_PATH", str(log_path), raising=False)
    monkeypatch.setattr(config, "COLLECTION_REQUIRE_GOAL", True, raising=False)
    monkeypatch.setattr(config, "COLLECTION_LOG_RESULTS", True, raising=False)

    box = {}
    mcp = _build_server(box)
    mw = install_telemetry(mcp, "stdio")
    assert mw is not None
    assert any(isinstance(m, TelemetryMiddleware) for m in mcp.middleware)

    async with Client(mcp) as client:
        await client.call_tool("echo", {"goal": "why echo", "text": "hi", "count": 1})

    # underlying tool ran without goal
    assert box["received"] == {"text": "hi", "count": 1}
    lines = [
        json.loads(x)
        for x in log_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    assert len(lines) == 2  # run_start header + one call record
    header, rec = lines
    assert header["record"] == "run_start"
    assert header["transport"] == "stdio"
    assert rec["tool"] == "echo"
    assert rec["goal"] == "why echo"
    assert "goal" not in rec["arguments"]
    assert rec["arguments"]["text"] == "hi"
    assert rec["result_logged"] is True
    assert rec["status"] == "success"
    assert rec["run_id"] == header["run_id"]


def test_install_disabled_log_path_unusable_returns_none(monkeypatch, tmp_path):
    """If the log path can't be opened, install_telemetry disables telemetry
    (returns None) rather than breaking the server."""
    import sas_mcp_server.config as config

    monkeypatch.setattr(config, "COLLECTION_MODE", True, raising=False)
    # Point at a path whose parent is a FILE, so mkdir/open fails with OSError.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setattr(
        config, "COLLECTION_LOG_PATH", str(blocker / "nested" / "x.log"), raising=False
    )
    mcp = FastMCP("test")
    before = list(mcp.middleware)
    assert install_telemetry(mcp, "stdio") is None
    assert mcp.middleware == before


# --------------------- error taxonomy / tier / duration --------------------- #


@pytest.mark.parametrize(
    ("text", "expected_type", "expected_status"),
    [
        ("ValidationError: 1 validation error for call[create_report]", "validation", None),
        ("ToolError: Error calling tool 'x': Client error '404 Not Found' for url", "client_error", 404),
        ("ToolError: Server error '500 Internal Server Error' for url", "server_error", 500),
        ("ToolError: [Errno 11001] getaddrinfo failed", "network", None),
        ("AuthenticationError: No auth header found", "auth", None),
        ("ToolError: something odd", "tool", None),
        ("completely unrecognised text", "unknown", None),
        (None, None, None),
        ("", None, None),
    ],
)
def test_classify_error(text, expected_type, expected_status):
    """Errors are free text; the taxonomy makes them groupable without regex
    at analysis time (the real log needed exactly these classes)."""
    assert classify_error(text) == (expected_type, expected_status)


@pytest.mark.asyncio
async def test_record_carries_error_taxonomy_and_always_present_is_error():
    box = FakeLogger()
    mw = TelemetryMiddleware(box, require_goal=True, transport="stdio")
    record = mw._build_record(
        "get_castable_columns",
        "goal",
        {},
        None,
        "error",
        True,
        "ToolError: Client error '404 Not Found' for url",
        dur_ms=12.0,
    )
    assert record["is_error"] is True  # present on EVERY record, both versions
    assert record["error_type"] == "client_error"
    assert record["http_status"] == 404
    assert "duration_suspect" not in record


@pytest.mark.asyncio
async def test_successful_record_has_null_taxonomy_and_false_is_error():
    box = FakeLogger()
    mw = TelemetryMiddleware(box, require_goal=True, transport="stdio")
    record = mw._build_record("echo", "g", {}, {"ok": 1}, "success", False, None, dur_ms=5.0)
    assert record["is_error"] is False
    assert record["error_type"] is None
    assert record["http_status"] is None
    # No handshake reachable -> the client keys are omitted, not null.
    assert "client_name" not in record and "client_version" not in record


@pytest.mark.asyncio
async def test_implausible_duration_is_flagged_not_dropped():
    """Real logs held an 18.8-HOUR call (hung, or the host slept mid-call);
    unflagged it silently poisons every latency percentile."""
    box = FakeLogger()
    mw = TelemetryMiddleware(box, require_goal=True, transport="stdio")
    ok = mw._build_record("echo", "g", {}, None, "success", False, None, dur_ms=SUSPECT_DURATION_MS - 1)
    bad = mw._build_record("echo", "g", {}, None, "success", False, None, dur_ms=SUSPECT_DURATION_MS + 1)
    assert "duration_suspect" not in ok
    assert bad["duration_suspect"] is True
    assert bad["duration_ms"] == SUSPECT_DURATION_MS + 1  # kept, not clamped


@pytest.mark.asyncio
async def test_tool_tier_is_stamped_from_the_registry(monkeypatch):
    box = FakeLogger()
    mw = TelemetryMiddleware(box, require_goal=True, transport="stdio")
    monkeypatch.setattr(mw, "_tool_tiers", {"execute_sas_code": 0, "query_data": 1})
    assert mw._build_record("execute_sas_code", "g", {}, None, "success", False, None, dur_ms=1.0)["tool_tier"] == 0
    assert mw._build_record("query_data", "g", {}, None, "success", False, None, dur_ms=1.0)["tool_tier"] == 1
    # A tool registered outside register_tools simply has no tier.
    assert mw._build_record("mystery", "g", {}, None, "success", False, None, dur_ms=1.0)["tool_tier"] is None


def test_server_version_prefers_the_source_checkout():
    """importlib.metadata reports the INSTALLED dist, which goes stale under an
    editable install — real logs said 1.2.0 while the checkout was 1.7.x."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        expected = tomllib.load(fh)["project"]["version"]
    from sas_mcp_server.helpers.telemetry_helpers import server_version

    assert server_version() == expected
