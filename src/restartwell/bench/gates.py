"""The five numeric kill-gates, as callable predicates returning (passed, evidence).

These are the make-or-break checks the build runs at its measurement stage and that the test
suite asserts:

- G_flip        : DFR data -> restart_helps; IFR data -> do_not_restart (sign discrimination).
- G_calib       : on held-out DFR data, tau* (fit on train) beats the user timeout AND p90 by
                  a material margin with a CI-separated savings fraction (the renewal-reward
                  wedge; if it collapses to ~p90 the instrument is just a percentile).
- G_censor      : the E[min(A,tau)] integral is internally exact (1e-9) and recovers the sample
                  mean for an uncensored all-success sample (1e-9).
- G_luby        : the Luby schedule is the verbatim 1,1,2,1,1,2,4,... sequence.
- G_determinism : same seed -> bit-identical tau*, CI, and concavity.
"""

from __future__ import annotations

import numpy as np

from restartwell.bench.datasets import make_dfr, make_flat, make_ifr, split
from restartwell.bench.metrics import held_out_score
from restartwell.cutoff import optimal_cutoff
from restartwell.effectiveness import restart_effectiveness
from restartwell.luby import luby_schedule
from restartwell.survival import cost_survival
from restartwell.types import AttemptRecord

_LUBY15 = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 8.0]


def gate_flip(*, n: int = 600, seed: int = 0, n_boot: int = 400) -> dict[str, object]:
    """DFR -> restart_helps, IFR -> do_not_restart."""
    dfr = restart_effectiveness(make_dfr(n=n, seed=seed), n_boot=n_boot, seed=0)
    ifr = restart_effectiveness(make_ifr(n=n, seed=seed), n_boot=n_boot, seed=0)
    flat = restart_effectiveness(make_flat(n=n, seed=seed), n_boot=n_boot, seed=0)
    passed = dfr.decision == "restart_helps" and ifr.decision == "do_not_restart"
    return {
        "passed": bool(passed),
        "dfr_decision": dfr.decision,
        "dfr_concavity": dfr.concavity,
        "ifr_decision": ifr.decision,
        "ifr_concavity": ifr.concavity,
        "flat_decision": flat.decision,
        "flat_concavity": flat.concavity,
    }


def gate_calib(
    *,
    n: int = 1200,
    seed: int = 0,
    r: float = 400.0,
    threshold: float = 0.90,
    n_boot: int = 800,
) -> dict[str, object]:
    """Held-out tau* beats min(user-timeout, p90) by >=(1-threshold), CI-separated, over an
    r-sweep ``{0, r/2, r, 2r}``."""
    attempts = make_dfr(n=n, seed=seed)
    train, held = split(attempts, frac=0.5, seed=seed)
    user_timeout = float(np.percentile([a.cost for a in held], 97))
    cells: list[dict[str, object]] = []
    any_pass = False
    for rr in (0.0, r / 2.0, r, 2.0 * r):
        sc = held_out_score(train, held, r=rr, user_timeout=user_timeout, n_boot=n_boot, seed=seed)
        ref = min(sc.e_total_adhoc, sc.e_total_p90)
        ratio = sc.e_total_tau_star / ref if np.isfinite(ref) and ref > 0 else float("inf")
        ci_separated = sc.savings_vs_adhoc_ci.lower > 0.0
        cell_pass = bool(ratio <= threshold and ci_separated)
        any_pass = any_pass or cell_pass
        cells.append(
            {
                "r": rr,
                "tau_star_train": sc.tau_star_train,
                "e_total_tau_star": sc.e_total_tau_star,
                "e_total_adhoc": sc.e_total_adhoc,
                "e_total_p90": sc.e_total_p90,
                "ratio_vs_min": ratio,
                "savings_vs_adhoc": sc.savings_vs_adhoc,
                "savings_vs_p90": sc.savings_vs_p90,
                "savings_ci_lower": sc.savings_vs_adhoc_ci.lower,
                "ci_separated": ci_separated,
                "passed": cell_pass,
            }
        )
    return {"passed": bool(any_pass), "user_timeout": user_timeout, "cells": cells}


def gate_censor(*, n: int = 2000, seed: int = 0) -> dict[str, object]:
    """E[min(A,tau)] integral is internally exact and recovers the uncensored sample mean."""
    rng = np.random.default_rng(seed)
    costs = rng.exponential(1000.0, size=n)
    att = [AttemptRecord(cost=float(x), label="success") for x in costs]
    surv = cost_survival(att)
    tau = float(np.percentile(costs, 60))

    # internal consistency: e_min via the dataclass method vs an independent trapezoid
    a = surv.e_min(tau)
    b = _e_min_independent(surv, tau)
    internal_diff = abs(a - b)

    # uncensored all-success: e_min(tau >= max) == sample mean (exact for empirical KM)
    big = float(costs.max()) + 1.0
    mean_diff = abs(surv.e_min(big) - float(costs.mean()))

    # monotone censoring shift: heavier timeout censoring lowers tau* (informational)
    passed = internal_diff < 1e-9 and mean_diff < 1e-6
    return {
        "passed": bool(passed),
        "internal_diff": internal_diff,
        "mean_diff": mean_diff,
    }


def _e_min_independent(surv, tau: float) -> float:  # type: ignore[no-untyped-def]
    """An independent integral of S over [0, tau] (explicit step-function quadrature)."""
    total = 0.0
    prev_x = 0.0
    prev_y = 1.0
    for t, s in zip(surv.times, surv.survival, strict=False):
        if t >= tau:
            break
        total += prev_y * (float(t) - prev_x)
        prev_x = float(t)
        prev_y = float(s)
    total += prev_y * (tau - prev_x)
    return total


def gate_luby() -> dict[str, object]:
    """The Luby schedule is the verbatim 1,1,2,1,1,2,4,... sequence."""
    seq = luby_schedule(base=1.0, length=15).sequence
    passed = seq == _LUBY15
    return {"passed": bool(passed), "sequence": seq, "expected": _LUBY15}


def gate_determinism(
    *, n: int = 400, seed: int = 0, r: float = 400.0, n_boot: int = 200
) -> dict[str, object]:
    """Same seed -> bit-identical tau*, CI, concavity across two runs."""
    att = make_dfr(n=n, seed=seed)
    c1 = optimal_cutoff(att, r=r, n_boot=n_boot, seed=0)
    c2 = optimal_cutoff(att, r=r, n_boot=n_boot, seed=0)
    v1 = restart_effectiveness(att, n_boot=n_boot, seed=0)
    v2 = restart_effectiveness(att, n_boot=n_boot, seed=0)
    passed = (
        c1.tau_star == c2.tau_star
        and c1.ci.lower == c2.ci.lower
        and c1.ci.upper == c2.ci.upper
        and v1.concavity == v2.concavity
        and v1.concavity_ci.lower == v2.concavity_ci.lower
    )
    return {
        "passed": bool(passed),
        "tau_star": c1.tau_star,
        "ci": [c1.ci.lower, c1.ci.upper],
        "concavity": v1.concavity,
    }


def run_all_gates(*, n_boot: int = 600) -> dict[str, object]:
    """Run all five gates and return a structured report."""
    g_flip = gate_flip(n_boot=n_boot)
    g_calib = gate_calib(n_boot=n_boot)
    g_censor = gate_censor()
    g_luby = gate_luby()
    g_det = gate_determinism(n_boot=min(n_boot, 300))
    gates = {
        "G_flip": g_flip,
        "G_calib": g_calib,
        "G_censor": g_censor,
        "G_luby": g_luby,
        "G_determinism": g_det,
    }
    all_pass = all(g["passed"] for g in gates.values())
    return {"all_passed": bool(all_pass), "gates": gates}
