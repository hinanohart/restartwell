"""Parse raw log rows into :class:`~restartwell.types.AttemptRecord`.

Generic, framework-agnostic intake: a row is a mapping with a cost field and an outcome
field. Outcome strings are normalised to ``success`` / ``timeout`` / ``fail`` via a small,
explicit synonym table (unknown non-success outcomes map to ``fail``, never silently to
``success``). Framework-specific adapters live in :mod:`restartwell.adapters`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from restartwell.types import AttemptRecord, Label

_SUCCESS = {
    "success",
    "solved",
    "ok",
    "pass",
    "passed",
    "resolved",
    "done",
    "completed",
    "true",
    "1",
}
_TIMEOUT = {"timeout", "timed_out", "time_out", "deadline", "budget_exhausted", "budget", "cutoff"}
_FAIL = {
    "fail",
    "failed",
    "failure",
    "error",
    "errored",
    "wrong",
    "wrong_patch",
    "abort",
    "aborted",
    "false",
    "0",
}


def normalize_label(raw: Any) -> Label:
    """Map a raw outcome value to ``success`` / ``timeout`` / ``fail`` (fail-safe default)."""
    s = str(raw).strip().lower()
    if s in _SUCCESS:
        return "success"
    if s in _TIMEOUT:
        return "timeout"
    if s in _FAIL:
        return "fail"
    # Unknown non-success outcome: never invent success; treat as a (non-timeout) failure.
    return "fail"


def parse_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    cost_key: str = "cost",
    label_key: str = "label",
    cohort_key: str | None = None,
    id_key: str | None = None,
) -> list[AttemptRecord]:
    """Parse an iterable of mappings into validated attempt records."""
    out: list[AttemptRecord] = []
    for i, row in enumerate(rows):
        if cost_key not in row:
            raise KeyError(f"row {i}: missing cost key {cost_key!r}")
        if label_key not in row:
            raise KeyError(f"row {i}: missing label key {label_key!r}")
        out.append(
            AttemptRecord(
                cost=float(row[cost_key]),
                label=normalize_label(row[label_key]),
                cohort=str(row[cohort_key])
                if cohort_key and row.get(cohort_key) is not None
                else None,
                attempt_id=str(row[id_key]) if id_key and row.get(id_key) is not None else None,
            )
        )
    return out


def from_jsonl(
    path: str | Path,
    *,
    cost_key: str = "cost",
    label_key: str = "label",
    cohort_key: str | None = None,
    id_key: str | None = None,
) -> list[AttemptRecord]:
    """Read a JSON-Lines file (one object per line) into attempt records."""
    rows: list[Mapping[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return parse_records(
        rows, cost_key=cost_key, label_key=label_key, cohort_key=cohort_key, id_key=id_key
    )
