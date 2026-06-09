"""Langfuse-trace JSON-Lines adapter.

Treats each trace as one attempt. Cost is read from a usage field (default ``totalTokens``,
falling back to ``latency`` seconds); the outcome is derived from the trace ``level`` /
``statusMessage`` and an optional timeout marker in ``metadata``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from restartwell.types import AttemptRecord, Label


def _label_from_trace(trace: Mapping[str, Any]) -> Label:
    meta = trace.get("metadata", {}) or {}
    if meta.get("timeout") in (True, "true", 1, "1") or meta.get("restartwell_timeout"):
        return "timeout"
    level = str(trace.get("level", "")).upper()
    status = str(trace.get("statusMessage", "")).lower()
    if "timeout" in status or "timed out" in status:
        return "timeout"
    if level in ("ERROR", "WARNING") or "error" in status or "fail" in status:
        return "fail"
    return "success"


def traces_to_records(
    traces: Iterable[Mapping[str, Any]],
    *,
    cost_field: str = "totalTokens",
    cohort_field: str | None = None,
) -> list[AttemptRecord]:
    out: list[AttemptRecord] = []
    for tr in traces:
        usage = tr.get("usage", {}) or {}
        if cost_field in tr:
            cost = float(tr[cost_field])
        elif cost_field in usage:
            cost = float(usage[cost_field])
        elif "latency" in tr:
            cost = float(tr["latency"])
        else:
            continue
        cohort = (
            str(tr[cohort_field]) if cohort_field and tr.get(cohort_field) is not None else None
        )
        out.append(
            AttemptRecord(
                cost=cost,
                label=_label_from_trace(tr),
                cohort=cohort,
                attempt_id=str(tr.get("id") or tr.get("traceId") or "") or None,
            )
        )
    return out


def from_langfuse_jsonl(path: str | Path, **kwargs: Any) -> list[AttemptRecord]:
    """Read a Langfuse-trace JSON-Lines export into attempt records."""
    traces: list[Mapping[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces_to_records(traces, **kwargs)
