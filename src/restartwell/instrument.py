"""High-level orchestration: attempts -> verdict -> cutoff/luby/savings -> RestartReport.

This wires the fail-closed verdict to what restartwell emits:
- ``restart_helps``  -> renewal-reward cutoff tau* (+ savings vs current cutoff if given).
- ``do_not_restart`` -> no cutoff; advise raising the timeout.
- ``inconclusive``   -> a Luby universal schedule instead of a fabricated cutoff.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from restartwell.cutoff import optimal_cutoff
from restartwell.effectiveness import restart_effectiveness
from restartwell.luby import luby_schedule
from restartwell.savings import expected_savings
from restartwell.survival import backend, n_successes
from restartwell.types import AttemptRecord, RestartReport


def analyze(
    attempts: Sequence[AttemptRecord],
    *,
    r: float,
    unit: str = "tokens",
    current_cutoff: float | None = None,
    cohort: str | None = None,
    n_min_success: int = 20,
    alpha: float = 0.05,
    n_boot: int = 2000,
    grid: np.ndarray | None = None,
    seed: int = 0,
) -> RestartReport:
    """Produce the full restart report for one cohort of attempts."""
    if not attempts:
        raise ValueError("analyze requires at least one attempt")
    if r < 0:
        raise ValueError(f"restart overhead r must be >= 0, got {r}")

    verdict = restart_effectiveness(
        attempts, n_min_success=n_min_success, alpha=alpha, n_boot=n_boot, seed=seed
    )
    notes: list[str] = []
    cutoff = None
    luby = None
    savings = None

    if verdict.decision == "restart_helps":
        cutoff = optimal_cutoff(attempts, r=r, grid=grid, n_boot=n_boot, alpha=alpha, seed=seed)
        if cutoff.tau_star >= cutoff.p90:
            notes.append(
                f"tau* ({cutoff.tau_star:g}) is not below the p90 cutoff ({cutoff.p90:g}); "
                "the renewal-reward cutoff offers little over a naive percentile here."
            )
        if current_cutoff is not None:
            savings = expected_savings(
                attempts,
                current_cutoff=current_cutoff,
                tau_star=cutoff.tau_star,
                r=r,
                n_boot=n_boot,
                alpha=alpha,
                seed=seed,
            )
        if verdict.weibull_agrees is False:
            notes.append(
                "Weibull-AFT shape disagrees with the non-parametric hazard trend; treat tau* "
                "as provisional."
            )
    elif verdict.decision == "do_not_restart":
        notes.append("Hazard is increasing: raise the per-attempt timeout rather than restarting.")
    else:  # inconclusive
        sc = [a.cost for a in attempts if a.label == "success"]
        base = float(np.median(sc)) if sc else float(np.median([a.cost for a in attempts]))
        luby = luby_schedule(base=max(base, 1.0))
        notes.append("Evidence is insufficient for a precise cutoff; use the Luby schedule below.")

    return RestartReport(
        verdict=verdict,
        cutoff=cutoff,
        luby=luby,
        savings=savings,
        unit=unit,
        r=r,
        n_attempts=len(attempts),
        backend=backend(),
        cohort=cohort,
        notes=tuple(notes),
    )


def analyze_by_cohort(
    attempts: Sequence[AttemptRecord],
    *,
    r: float,
    unit: str = "tokens",
    current_cutoff: float | None = None,
    n_min_success: int = 20,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, RestartReport]:
    """Per-cohort reports (cohort = AttemptRecord.cohort; None -> '_all')."""
    buckets: dict[str, list[AttemptRecord]] = {}
    for a in attempts:
        buckets.setdefault(a.cohort or "_all", []).append(a)
    reports: dict[str, RestartReport] = {}
    for name, recs in buckets.items():
        if n_successes(recs) == 0:
            continue
        reports[name] = analyze(
            recs,
            r=r,
            unit=unit,
            current_cutoff=current_cutoff,
            cohort=name,
            n_min_success=n_min_success,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
    return reports
