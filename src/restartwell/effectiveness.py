"""Restart-effectiveness: the fail-closed 3-way verdict that gates the cutoff.

A Las-Vegas restart helps exactly when the cost-to-success hazard is *decreasing* (DFR):
``E[A] < E[A - t | A > t]`` for some t, i.e. waiting longer buys you less. We test this
non-parametrically from the Nelson-Aalen cumulative hazard H(t): under DFR the hazard
h = H' decreases, so H is **concave** (bulges above the chord from the origin to the last
event); under wear-out (IFR) H is **convex**. The signed, scale-free area between H and that
chord is the test statistic; its sign and a cluster-bootstrap CI give the verdict.
hazardloop's Weibull-AFT shape beta is a secondary consistency anchor (beta<1 DFR / beta>1
IFR), surfaced but not authoritative. When successes are too few or the trend is not
CI-separated from zero, the verdict is INCONCLUSIVE and the caller falls back to Luby rather
than fabricate a precise cutoff.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

from restartwell.survival import bootstrap_hazard_ci, cost_hazard, n_successes, weibull_shape
from restartwell.types import AttemptRecord, CostHazard, Decision, HazardTrend, RestartVerdict


def concavity(hazard: CostHazard) -> float:
    """Signed, scale-free concavity of the cumulative hazard H(t), in roughly [-0.5, 0.5].

    Positive => H concave => decreasing hazard (DFR, restart can help). Negative => convex
    => increasing hazard (IFR, raising the timeout beats restarting). Zero => ~constant.

    Measured over the **event support** [t_first, t_last], anchoring the chord at the first
    and last success times. Anchoring at the origin would inject a spurious convex bias from
    the dead zone before the first possible success (where H is flat at 0).
    """
    t = hazard.times.astype(np.float64)
    h = hazard.cumulative_hazard.astype(np.float64)
    if t.size < 3:
        return 0.0
    t0, t1 = float(t[0]), float(t[-1])
    h0, h1 = float(h[0]), float(h[-1])
    span_t = t1 - t0
    span_h = h1 - h0
    if span_t <= 0.0 or span_h <= 0.0:
        return 0.0
    chord = h0 + span_h * (t - t0) / span_t
    area = float(np.trapezoid(h - chord, t))
    return area / (span_t * span_h)


def restart_effectiveness(
    attempts: Sequence[AttemptRecord],
    *,
    n_min_success: int = 20,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> RestartVerdict:
    """Decide whether restarting helps, from the cost-to-success hazard shape."""
    if not attempts:
        raise ValueError("restart_effectiveness requires at least one attempt")
    ns = n_successes(attempts)
    hazard = cost_hazard(attempts)
    point = concavity(hazard)
    ci = bootstrap_hazard_ci(attempts, concavity, n_boot=n_boot, alpha=alpha, seed=seed)
    wb = weibull_shape(attempts)

    half = (ci.upper - ci.lower) / 2.0
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    if half > 0.0 and z_crit > 0.0:
        se = half / z_crit
        z = point / se if se > 0.0 else 0.0
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))
    else:
        p_value = 1.0

    decision: Decision
    trend: HazardTrend
    if ns < n_min_success:
        trend = "decreasing" if point > 0 else "increasing" if point < 0 else "flat"
        return RestartVerdict(
            decision="inconclusive",
            hazard_trend=trend,
            concavity=point,
            concavity_ci=ci,
            p_value=p_value,
            weibull_shape=wb,
            weibull_agrees=None,
            n_success=ns,
            reason=(
                f"only {ns} successes (< n_min_success={n_min_success}); too few to resolve a "
                "restart cutoff — fall back to a Luby schedule."
            ),
        )

    if ci.lower > 0.0:
        decision = "restart_helps"
        trend = "decreasing"
        reason = (
            f"cost-to-success hazard is decreasing (concavity {point:.3f}, 95% CI "
            f"[{ci.lower:.3f}, {ci.upper:.3f}] > 0): restarting can beat waiting."
        )
    elif ci.upper < 0.0:
        decision = "do_not_restart"
        trend = "increasing"
        reason = (
            f"cost-to-success hazard is increasing (concavity {point:.3f}, 95% CI "
            f"[{ci.lower:.3f}, {ci.upper:.3f}] < 0): do not restart — raise the timeout instead."
        )
    else:
        decision = "inconclusive"
        trend = "flat"
        reason = (
            f"hazard trend not separated from zero (concavity {point:.3f}, 95% CI "
            f"[{ci.lower:.3f}, {ci.upper:.3f}] straddles 0): fall back to a Luby schedule."
        )

    if wb is None or decision == "inconclusive":
        weibull_agrees: bool | None = None
    else:
        weibull_agrees = (wb < 1.0) == (decision == "restart_helps")

    return RestartVerdict(
        decision=decision,
        hazard_trend=trend,
        concavity=point,
        concavity_ci=ci,
        p_value=p_value,
        weibull_shape=wb,
        weibull_agrees=weibull_agrees,
        n_success=ns,
        reason=reason,
    )
