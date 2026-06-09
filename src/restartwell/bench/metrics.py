"""Held-out evaluation metrics: train tau* on one split, score expected cost on the other."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from restartwell.bench.baselines import adhoc_cutoff, best_fixed_cutoff, p90_cutoff
from restartwell.cutoff import e_total, optimal_cutoff
from restartwell.survival import bootstrap_survival_ci, cost_survival
from restartwell.types import AttemptRecord, BootstrapCI


@dataclass(frozen=True)
class HeldOutScore:
    r: float
    user_timeout: float
    tau_star_train: float
    e_total_tau_star: float
    e_total_adhoc: float
    e_total_p90: float
    e_total_best_fixed_heldout: float
    savings_vs_adhoc: float
    savings_vs_p90: float
    savings_vs_adhoc_ci: BootstrapCI
    n_train: int
    n_held: int


def held_out_score(
    train: Sequence[AttemptRecord],
    held: Sequence[AttemptRecord],
    *,
    r: float,
    user_timeout: float,
    grid: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> HeldOutScore:
    """Fit tau* on ``train``; score expected cost-per-success on ``held``."""
    # n_boot=1: only the point tau* (the grid argmin) is needed here; the headline CI is
    # computed separately below via bootstrap_survival_ci on the held split.
    tau_star = optimal_cutoff(train, r=r, grid=grid, n_boot=1, alpha=alpha, seed=seed).tau_star
    surv_h = cost_survival(held)
    e_star = e_total(tau_star, surv_h, r)
    e_adhoc = e_total(adhoc_cutoff(user_timeout), surv_h, r)
    e_p90 = e_total(p90_cutoff(held), surv_h, r)
    e_best = e_total(best_fixed_cutoff(held, r, grid), surv_h, r)

    def _sv_adhoc(cs) -> float:  # type: ignore[no-untyped-def]
        ea = e_total(user_timeout, cs, r)
        es = e_total(tau_star, cs, r)
        if not np.isfinite(ea) or ea <= 0:
            raise ValueError("adhoc cost not finite")
        return (ea - es) / ea

    ci = bootstrap_survival_ci(held, _sv_adhoc, n_boot=n_boot, alpha=alpha, seed=seed)
    sv_adhoc = (e_adhoc - e_star) / e_adhoc if np.isfinite(e_adhoc) and e_adhoc > 0 else 0.0
    sv_p90 = (e_p90 - e_star) / e_p90 if np.isfinite(e_p90) and e_p90 > 0 else 0.0
    return HeldOutScore(
        r=r,
        user_timeout=user_timeout,
        tau_star_train=tau_star,
        e_total_tau_star=e_star,
        e_total_adhoc=e_adhoc,
        e_total_p90=e_p90,
        e_total_best_fixed_heldout=e_best,
        savings_vs_adhoc=sv_adhoc,
        savings_vs_p90=sv_p90,
        savings_vs_adhoc_ci=ci,
        n_train=len(train),
        n_held=len(held),
    )
