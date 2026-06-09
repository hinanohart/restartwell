"""restartwell — offline renewal-reward restart-cutoff instrument for agent cost logs.

Reads attempt costs + outcome labels and emits a fail-closed restart verdict, a
renewal-reward optimal cutoff tau* with a bootstrap CI, expected savings, and a paste-able
config. hazardloop estimates the hazard shape; restartwell ships the restart decision layer
on top. CPU-only; see docs/CLAIMS.md for the CLAIM / NON-CLAIM boundary.
"""

from __future__ import annotations

from restartwell.cutoff import e_total, expected_cost_per_success, optimal_cutoff
from restartwell.effectiveness import concavity, restart_effectiveness
from restartwell.emit import emit_config
from restartwell.instrument import analyze, analyze_by_cohort
from restartwell.intake import from_jsonl, normalize_label, parse_records
from restartwell.luby import luby, luby_schedule
from restartwell.savings import expected_savings
from restartwell.survival import (
    backend,
    cost_hazard,
    cost_survival,
    weibull_shape,
)
from restartwell.types import (
    AttemptRecord,
    BootstrapCI,
    CostHazard,
    CostSurvival,
    CutoffResult,
    LubySchedule,
    RestartReport,
    RestartVerdict,
    SavingsReport,
)

__version__ = "0.1.0a1"

__all__ = [
    "AttemptRecord",
    "BootstrapCI",
    "CostHazard",
    "CostSurvival",
    "CutoffResult",
    "LubySchedule",
    "RestartReport",
    "RestartVerdict",
    "SavingsReport",
    "__version__",
    "analyze",
    "analyze_by_cohort",
    "backend",
    "concavity",
    "cost_hazard",
    "cost_survival",
    "e_total",
    "emit_config",
    "expected_cost_per_success",
    "expected_savings",
    "from_jsonl",
    "luby",
    "luby_schedule",
    "normalize_label",
    "optimal_cutoff",
    "parse_records",
    "restart_effectiveness",
    "weibull_shape",
]
