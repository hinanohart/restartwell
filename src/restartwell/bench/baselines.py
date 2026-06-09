"""Baseline cutoffs to compare tau* against (the anti-theater comparison set)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from restartwell.cutoff import default_grid, e_total
from restartwell.survival import cost_survival
from restartwell.types import AttemptRecord


def adhoc_cutoff(user_timeout: float) -> float:
    """The user's current fixed timeout."""
    return float(user_timeout)


def p90_cutoff(attempts: Sequence[AttemptRecord]) -> float:
    """Naive p90-of-success-cost cutoff."""
    sc = [a.cost for a in attempts if a.label == "success"]
    base = sc if sc else [a.cost for a in attempts]
    return float(np.percentile(base, 90))


def best_fixed_cutoff(
    attempts: Sequence[AttemptRecord], r: float, grid: np.ndarray | None = None
) -> float:
    """Oracle: the grid cutoff minimising E_total on *this* sample (in-sample optimum)."""
    surv = cost_survival(attempts)
    g = default_grid(attempts) if grid is None else np.asarray(grid, dtype=np.float64)
    curve = np.asarray([e_total(float(t), surv, r) for t in g], dtype=np.float64)
    return float(g[int(np.argmin(curve))])
