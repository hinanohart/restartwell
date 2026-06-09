"""Renewal-reward optimal restart cutoff tau*.

Under a fixed cutoff ``tau`` the expected total cost to first success is

    E_total(tau) = (E[min(A, tau)] + r * P(A > tau)) / P(A <= tau)

where ``A`` is cost-to-success, ``r`` is the per-restart overhead, and ``E[min(A, tau)] =
integral_0^tau S(u) du``. The numerator is the expected cost paid per attempt (run cost plus
a restart-overhead ``r`` charged on each failed attempt); ``1 / P(A <= tau)`` is the expected
number of attempts to first success (geometric). The minimiser tau* is a hazard-weighted
balance point, **not** a fixed quantile: a naive p90 cutoff ignores both ``r`` and the
``1 / P(A <= tau)`` renewal multiplier. The gap between tau* and p90 is what restartwell
measures, and it depends on ``r`` (hence r is a required, swept input).

See Luby, Sinclair & Zuckerman (1993), "Optimal speedup of Las Vegas algorithms"; and the
renewal-reward / restart-cutoff treatment in timvieira's restart-strategy notes and
arXiv:1709.10405.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from restartwell.survival import bootstrap_survival_ci, cost_survival, n_successes
from restartwell.types import AttemptRecord, CostSurvival, CutoffResult


def e_total(tau: float, surv: CostSurvival, r: float) -> float:
    """Expected total cost-per-success under a restart cutoff at ``tau``."""
    if r < 0:
        raise ValueError(f"restart overhead r must be >= 0, got {r}")
    if tau <= 0:
        return float("inf")
    s_tau = surv.survival_at(tau)
    p_le = 1.0 - s_tau
    if p_le <= 0.0:
        return float("inf")
    return (surv.e_min(tau) + r * s_tau) / p_le


def _success_costs(attempts: Sequence[AttemptRecord]) -> np.ndarray:
    return np.asarray([a.cost for a in attempts if a.label == "success"], dtype=np.float64)


def default_grid(attempts: Sequence[AttemptRecord], *, n: int = 96) -> np.ndarray:
    """A cutoff-candidate grid spanning the observed success-cost range."""
    sc = _success_costs(attempts)
    if sc.size == 0:
        allc = np.asarray([a.cost for a in attempts], dtype=np.float64)
        lo, hi = float(np.quantile(allc, 0.05)), float(allc.max())
    else:
        lo = float(np.quantile(sc, 0.02))
        hi = float(sc.max())
    lo = max(lo, hi / 1e6) if hi > 0 else 1.0
    if hi <= lo:
        hi = lo * 2.0
    return np.unique(np.linspace(lo, hi, n))


def _tau_star_on_grid(surv: CostSurvival, r: float, grid: np.ndarray) -> float:
    curve = np.asarray([e_total(float(t), surv, r) for t in grid], dtype=np.float64)
    if not np.any(np.isfinite(curve)):
        raise ValueError("E_total is infinite across the whole grid (no successes by any cutoff)")
    return float(grid[int(np.argmin(curve))])


def optimal_cutoff(
    attempts: Sequence[AttemptRecord],
    *,
    r: float,
    grid: np.ndarray | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CutoffResult:
    """Minimise E_total over a cutoff grid; return tau* with a cluster-bootstrap CI.

    ``r`` (restart overhead, same unit as cost) is required: it has no default because it is
    precisely what moves tau* off a naive percentile.
    """
    if not attempts:
        raise ValueError("optimal_cutoff requires at least one attempt")
    if r < 0:
        raise ValueError(f"restart overhead r must be >= 0, got {r}")
    surv = cost_survival(attempts)
    g = default_grid(attempts) if grid is None else np.asarray(grid, dtype=np.float64)
    curve = np.asarray([e_total(float(t), surv, r) for t in g], dtype=np.float64)
    if not np.any(np.isfinite(curve)):
        raise ValueError("E_total is infinite across the whole grid (no successes by any cutoff)")
    star_idx = int(np.argmin(curve))
    tau_star = float(g[star_idx])

    sc = _success_costs(attempts)
    base = sc if sc.size else np.asarray([a.cost for a in attempts], dtype=np.float64)
    p90 = float(np.percentile(base, 90))
    p95 = float(np.percentile(base, 95))

    ci = bootstrap_survival_ci(
        attempts,
        lambda cs: _tau_star_on_grid(cs, r, g),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )

    return CutoffResult(
        tau_star=tau_star,
        ci=ci,
        e_total_at_star=float(curve[star_idx]),
        grid=g,
        e_total_curve=curve,
        r=r,
        p90=p90,
        e_total_at_p90=e_total(p90, surv, r),
        p95=p95,
        e_total_at_p95=e_total(p95, surv, r),
        backend=surv.backend,
    )


def expected_cost_per_success(attempts: Sequence[AttemptRecord], cutoff: float, r: float) -> float:
    """Convenience: E_total at an arbitrary cutoff on the full sample."""
    if not attempts:
        raise ValueError("expected_cost_per_success requires at least one attempt")
    if n_successes(attempts) == 0:
        return float("inf")
    return e_total(cutoff, cost_survival(attempts), r)
