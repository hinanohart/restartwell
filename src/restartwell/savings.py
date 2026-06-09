"""Expected savings of the optimal cutoff vs the current timeout and vs a naive p90 cutoff.

All three baselines are reported as expected cost-per-success (the renewal-reward objective):
the user's current cutoff, tau*, and the p90-of-success-cost cutoff. The headline savings
fraction is ``(E_total(current) - E_total(tau*)) / E_total(current)``, with a cluster-bootstrap
CI. If that CI straddles zero, or the current cutoff lies outside the observed cost range, the
headline is suppressed (an explicit anti-theater guard) rather than reported as a win.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from restartwell.cutoff import e_total
from restartwell.survival import bootstrap_survival_ci, cost_survival, n_successes
from restartwell.types import AttemptRecord, BootstrapCI, CostSurvival, SavingsReport


def _savings_fraction(
    surv: CostSurvival, current_cutoff: float, tau_star: float, r: float
) -> float:
    cur = e_total(current_cutoff, surv, r)
    opt = e_total(tau_star, surv, r)
    if not np.isfinite(cur) or cur <= 0.0:
        raise ValueError("current-cutoff expected cost is not finite/positive")
    return (cur - opt) / cur


def expected_savings(
    attempts: Sequence[AttemptRecord],
    *,
    current_cutoff: float,
    tau_star: float,
    r: float,
    held_out: bool = False,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> SavingsReport:
    """Savings of tau* vs ``current_cutoff`` (and a p90 baseline), with a bootstrap CI."""
    if not attempts:
        raise ValueError("expected_savings requires at least one attempt")
    surv = cost_survival(attempts)
    cur = e_total(current_cutoff, surv, r)
    opt = e_total(tau_star, surv, r)

    sc = np.asarray([a.cost for a in attempts if a.label == "success"], dtype=np.float64)
    base = sc if sc.size else np.asarray([a.cost for a in attempts], dtype=np.float64)
    p90 = float(np.percentile(base, 90))
    p90_cost = e_total(p90, surv, r)

    costs = np.asarray([a.cost for a in attempts], dtype=np.float64)
    in_range = float(costs.min()) <= current_cutoff <= float(costs.max())

    if not np.isfinite(cur) or cur <= 0.0:
        return SavingsReport(
            current_cost=cur,
            optimal_cost=opt,
            p90_cost=p90_cost,
            savings_fraction=0.0,
            ci=BootstrapCI(0.0, 0.0, 0.0, alpha, "degenerate", 0, 0),
            held_out=held_out,
            suppressed=True,
            reason="current-cutoff expected cost is not finite/positive (no successes by cutoff)",
        )

    frac = (cur - opt) / cur
    ci = bootstrap_survival_ci(
        attempts,
        lambda cs: _savings_fraction(cs, current_cutoff, tau_star, r),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    crosses_zero = ci.lower <= 0.0 <= ci.upper
    suppressed = crosses_zero or not in_range
    if not in_range:
        reason = (
            f"current cutoff {current_cutoff:g} is outside the observed cost range "
            f"[{costs.min():g}, {costs.max():g}]; savings not reported as a win."
        )
    elif crosses_zero:
        reason = (
            f"savings CI [{ci.lower:.3f}, {ci.upper:.3f}] straddles 0; not a statistically "
            "separated win."
        )
    else:
        reason = (
            f"tau* reduces expected cost-per-success by {frac:.1%} vs the current cutoff "
            f"(95% CI [{ci.lower:.1%}, {ci.upper:.1%}])."
        )

    return SavingsReport(
        current_cost=cur,
        optimal_cost=opt,
        p90_cost=p90_cost,
        savings_fraction=frac,
        ci=ci,
        held_out=held_out,
        suppressed=suppressed,
        reason=reason,
    )


def has_successes(attempts: Sequence[AttemptRecord]) -> bool:
    return n_successes(attempts) > 0
