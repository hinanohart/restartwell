"""OpenTelemetry-span JSON-Lines adapter.

Treats each root span as one attempt. Cost is read from a usage attribute (default
``gen_ai.usage.total_tokens``) or the span duration; the outcome is derived from the span
status and an optional timeout marker attribute.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from restartwell.types import AttemptRecord, Label


def _label_from_span(span: Mapping[str, Any], timeout_attr: str) -> Label:
    attrs = span.get("attributes", {}) or {}
    if attrs.get(timeout_attr) in (True, "true", 1, "1"):
        return "timeout"
    status = span.get("status", {})
    code = status.get("status_code") or status.get("code") or span.get("status_code")
    if str(code).upper() in ("OK", "STATUS_CODE_OK", "0"):
        return "success"
    if str(code).upper() in ("ERROR", "STATUS_CODE_ERROR", "2"):
        return "fail"
    return "fail"


def spans_to_records(
    spans: Iterable[Mapping[str, Any]],
    *,
    cost_attr: str = "gen_ai.usage.total_tokens",
    timeout_attr: str = "restartwell.timeout",
    cohort_attr: str | None = None,
) -> list[AttemptRecord]:
    out: list[AttemptRecord] = []
    for span in spans:
        attrs = span.get("attributes", {}) or {}
        if cost_attr in attrs:
            cost = float(attrs[cost_attr])
        elif "duration_ns" in span:
            cost = float(span["duration_ns"]) / 1e9
        elif "duration" in span:
            cost = float(span["duration"])
        else:
            continue
        cohort = (
            str(attrs[cohort_attr]) if cohort_attr and attrs.get(cohort_attr) is not None else None
        )
        out.append(
            AttemptRecord(
                cost=cost,
                label=_label_from_span(span, timeout_attr),
                cohort=cohort,
                attempt_id=str(span.get("span_id") or span.get("spanId") or "") or None,
            )
        )
    return out


def from_otel_jsonl(path: str | Path, **kwargs: Any) -> list[AttemptRecord]:
    """Read an OpenTelemetry-span JSON-Lines export into attempt records."""
    spans: list[Mapping[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans_to_records(spans, **kwargs)
