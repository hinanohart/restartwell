from __future__ import annotations

import json

from restartwell.adapters import from_langfuse_jsonl, from_otel_jsonl
from restartwell.adapters.langfuse import traces_to_records
from restartwell.adapters.otel import spans_to_records


def test_otel_spans_to_records():
    spans = [
        {"attributes": {"gen_ai.usage.total_tokens": 1000}, "status": {"status_code": "OK"}},
        {
            "attributes": {"gen_ai.usage.total_tokens": 5000, "restartwell.timeout": True},
            "status": {"status_code": "OK"},
        },
        {"attributes": {"gen_ai.usage.total_tokens": 300}, "status": {"status_code": "ERROR"}},
    ]
    recs = spans_to_records(spans)
    assert [r.label for r in recs] == ["success", "timeout", "fail"]
    assert recs[0].cost == 1000.0


def test_otel_duration_fallback():
    spans = [{"duration_ns": 2_000_000_000, "status": {"status_code": "OK"}}]
    recs = spans_to_records(spans)
    assert recs[0].cost == 2.0


def test_langfuse_traces_to_records():
    traces = [
        {"totalTokens": 800, "level": "DEFAULT"},
        {"totalTokens": 9000, "metadata": {"timeout": True}},
        {"totalTokens": 200, "level": "ERROR"},
    ]
    recs = traces_to_records(traces)
    assert [r.label for r in recs] == ["success", "timeout", "fail"]


def test_from_otel_jsonl(tmp_path):
    p = tmp_path / "spans.jsonl"
    spans = [{"attributes": {"gen_ai.usage.total_tokens": 100}, "status": {"status_code": "OK"}}]
    p.write_text("\n".join(json.dumps(s) for s in spans) + "\n", encoding="utf-8")
    recs = from_otel_jsonl(str(p))
    assert len(recs) == 1


def test_from_langfuse_jsonl(tmp_path):
    p = tmp_path / "traces.jsonl"
    traces = [{"totalTokens": 100, "level": "DEFAULT"}]
    p.write_text("\n".join(json.dumps(t) for t in traces) + "\n", encoding="utf-8")
    recs = from_langfuse_jsonl(str(p))
    assert len(recs) == 1
