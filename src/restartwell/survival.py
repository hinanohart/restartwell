"""Cost-to-success survival adapter — the ONLY module that imports hazardloop.

restartwell reuses hazardloop's Kaplan-Meier / Nelson-Aalen / cluster-bootstrap estimators
on the *cost* axis (``SurvivalRecord.duration`` is unit-free) under the
``completion_as_event`` model: success is the event, timeout/fail are right-censored. This
module converts hazardloop's result objects into restartwell's neutral
:class:`~restartwell.types.CostSurvival` / :class:`~restartwell.types.CostHazard` /
:class:`~restartwell.types.BootstrapCI`, so the rest of the package never touches
hazardloop. If hazardloop cannot be imported, the bundled :mod:`restartwell._shim`
product-limit estimator activates automatically (with a percentile bootstrap).

restartwell does NOT estimate hazard shape itself — that is hazardloop's job. This adapter
only consumes KM/NA so the renewal-reward decision layer can sit on top.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from restartwell._shim import _km_na
from restartwell.types import AttemptRecord, BootstrapCI, CostHazard, CostSurvival

try:  # pragma: no cover - the import path taken depends on the install
    from hazardloop import (
        EventModel,
        SurvivalRecord,
        TerminationMode,
        cluster_bootstrap_ci,
        kaplan_meier,
        nelson_aalen,
        weibull_aft,
    )

    _HAZARDLOOP_AVAILABLE = True
    _BACKEND = "hazardloop"
    _EVENT_MODEL = EventModel.completion_as_event()
    _LABEL_TO_MODE = {
        "success": TerminationMode.SOLVED,
        "timeout": TerminationMode.TIMEOUT,
        "fail": TerminationMode.UNLABELED,
    }
except ImportError:  # pragma: no cover - the standalone shim path
    _HAZARDLOOP_AVAILABLE = False
    _BACKEND = "shim"


def backend() -> str:
    """Which survival backend is active: ``hazardloop`` or ``shim``."""
    return _BACKEND


def n_successes(attempts: Sequence[AttemptRecord]) -> int:
    return sum(1 for a in attempts if a.label == "success")


def _arrays(attempts: Sequence[AttemptRecord]) -> tuple[np.ndarray, np.ndarray]:
    dur = np.fromiter((a.cost for a in attempts), dtype=np.float64, count=len(attempts))
    ev = np.fromiter((a.label == "success" for a in attempts), dtype=bool, count=len(attempts))
    return dur, ev


def _to_survival_records(attempts: Sequence[AttemptRecord]) -> list[SurvivalRecord]:
    return [
        SurvivalRecord(
            duration=float(a.cost),
            terminal_mode=_LABEL_TO_MODE[a.label],
            cluster=a.cohort,
            run_id=a.attempt_id,
        )
        for a in attempts
    ]


def _cost_survival_from_records(records: Sequence[SurvivalRecord]) -> CostSurvival:
    km = kaplan_meier(records, _EVENT_MODEL)
    n_success = sum(1 for r in records if r.terminal_mode == TerminationMode.SOLVED)
    return CostSurvival(
        times=np.asarray(km.times, dtype=np.float64),
        survival=np.asarray(km.survival, dtype=np.float64),
        n_at_risk=np.asarray(km.n_at_risk, dtype=np.int64),
        n_events=np.asarray(km.n_events, dtype=np.int64),
        n_success=n_success,
        n_censored=len(records) - n_success,
        backend="hazardloop",
    )


def _cost_hazard_from_records(records: Sequence[SurvivalRecord]) -> CostHazard:
    na = nelson_aalen(records, _EVENT_MODEL)
    n_success = sum(1 for r in records if r.terminal_mode == TerminationMode.SOLVED)
    return CostHazard(
        times=np.asarray(na.times, dtype=np.float64),
        cumulative_hazard=np.asarray(na.cumulative_hazard, dtype=np.float64),
        hazard_increment=np.asarray(na.hazard_increment, dtype=np.float64),
        n_at_risk=np.asarray(na.n_at_risk, dtype=np.int64),
        n_events=np.asarray(na.n_events, dtype=np.int64),
        n_success=n_success,
    )


def cost_survival(attempts: Sequence[AttemptRecord]) -> CostSurvival:
    """Kaplan-Meier survival of cost-to-success."""
    if not attempts:
        raise ValueError("cost_survival requires at least one attempt")
    if _HAZARDLOOP_AVAILABLE:
        return _cost_survival_from_records(_to_survival_records(attempts))
    cs, _ = _km_na(*_arrays(attempts))
    return cs


def cost_hazard(attempts: Sequence[AttemptRecord]) -> CostHazard:
    """Nelson-Aalen cumulative hazard / increments of cost-to-success."""
    if not attempts:
        raise ValueError("cost_hazard requires at least one attempt")
    if _HAZARDLOOP_AVAILABLE:
        return _cost_hazard_from_records(_to_survival_records(attempts))
    _, ch = _km_na(*_arrays(attempts))
    return ch


def weibull_shape(attempts: Sequence[AttemptRecord]) -> float | None:
    """Weibull-AFT shape beta of cost-to-success (secondary DFR/IFR anchor).

    beta < 1 = decreasing hazard (restart can help); beta > 1 = wear-out. Returns None
    when hazardloop is unavailable (the shim does not re-implement Weibull-AFT) or the fit
    is not estimable.
    """
    if not _HAZARDLOOP_AVAILABLE:
        return None
    try:
        fit = weibull_aft(_to_survival_records(attempts), _EVENT_MODEL)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    shape = float(fit.shape)
    return shape if np.isfinite(shape) else None


_Kind = Literal["survival", "hazard"]


def bootstrap_survival_ci(
    attempts: Sequence[AttemptRecord],
    statistic: Callable[[CostSurvival], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    seed: int = 0,
) -> BootstrapCI:
    """Cluster bootstrap CI for a statistic of the cost-to-success survival curve."""
    return _bootstrap(
        attempts, "survival", statistic, n_boot=n_boot, alpha=alpha, method=method, seed=seed
    )


def bootstrap_hazard_ci(
    attempts: Sequence[AttemptRecord],
    statistic: Callable[[CostHazard], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    seed: int = 0,
) -> BootstrapCI:
    """Cluster bootstrap CI for a statistic of the cost-to-success hazard curve."""
    return _bootstrap(
        attempts, "hazard", statistic, n_boot=n_boot, alpha=alpha, method=method, seed=seed
    )


def _bootstrap(
    attempts: Sequence[AttemptRecord],
    kind: _Kind,
    statistic: Callable,  # type: ignore[type-arg]
    *,
    n_boot: int,
    alpha: float,
    method: str,
    seed: int,
) -> BootstrapCI:
    if not attempts:
        raise ValueError("bootstrap requires at least one attempt")
    if _HAZARDLOOP_AVAILABLE:
        records = _to_survival_records(attempts)

        def _stat(recs: Sequence[SurvivalRecord]) -> float:
            obj = (
                _cost_survival_from_records(recs)
                if kind == "survival"
                else _cost_hazard_from_records(recs)
            )
            return float(statistic(obj))

        hl = cluster_bootstrap_ci(
            records, _stat, n_boot=n_boot, alpha=alpha, method=method, seed=seed
        )
        return BootstrapCI(
            hl.point, hl.lower, hl.upper, hl.alpha, hl.method, hl.n_boot, hl.n_effective
        )

    def _obj(att: Sequence[AttemptRecord]) -> CostSurvival | CostHazard:
        cs, ch = _km_na(*_arrays(att))
        return cs if kind == "survival" else ch

    point = float(statistic(_obj(attempts)))
    clusters = _group_clusters(attempts)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(clusters))
    reps: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(idx, size=len(clusters), replace=True)
        resampled: list[AttemptRecord] = []
        for ci in chosen:
            resampled.extend(clusters[ci])
        try:
            val = float(statistic(_obj(resampled)))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(val):
            reps.append(val)
    boot = np.asarray(reps, dtype=np.float64)
    if boot.size == 0:
        return BootstrapCI(point, point, point, alpha, "percentile", n_boot, 0)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return BootstrapCI(point, lo, hi, alpha, "percentile", n_boot, int(boot.size))


def _group_clusters(attempts: Sequence[AttemptRecord]) -> list[list[AttemptRecord]]:
    named: dict[str, list[AttemptRecord]] = {}
    singletons: list[list[AttemptRecord]] = []
    for a in attempts:
        if a.cohort is None:
            singletons.append([a])
        else:
            named.setdefault(a.cohort, []).append(a)
    return list(named.values()) + singletons
