"""Optional log adapters: map common agent-observability exports to AttemptRecord.

These are thin, dependency-free best-effort mappers for OpenTelemetry-span and Langfuse-trace
JSON-Lines exports. They reuse :func:`restartwell.intake.parse_records` after extracting a
cost field and deriving a success/timeout/fail label from the export's status fields. Adapt
the field names to your own schema where needed.
"""

from __future__ import annotations

from restartwell.adapters.langfuse import from_langfuse_jsonl
from restartwell.adapters.otel import from_otel_jsonl

__all__ = ["from_langfuse_jsonl", "from_otel_jsonl"]
