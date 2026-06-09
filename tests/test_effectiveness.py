from __future__ import annotations

import numpy as np

from restartwell.bench.datasets import make_dfr, make_ifr
from restartwell.effectiveness import concavity, restart_effectiveness
from restartwell.survival import cost_hazard
from restartwell.types import AttemptRecord, CostHazard


def _concave_hazard() -> CostHazard:
    # cumulative hazard that bulges above the chord (decreasing hazard)
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    h = np.array([0.4, 0.7, 0.9, 1.0, 1.05])  # increments shrink -> concave
    return CostHazard(
        times=t,
        cumulative_hazard=h,
        hazard_increment=np.diff(h, prepend=0.0),
        n_at_risk=np.array([10, 8, 6, 4, 2]),
        n_events=np.ones(5, dtype=np.int64),
        n_success=5,
    )


def _convex_hazard() -> CostHazard:
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    h = np.array([0.05, 0.15, 0.35, 0.7, 1.3])  # increments grow -> convex
    return CostHazard(
        times=t,
        cumulative_hazard=h,
        hazard_increment=np.diff(h, prepend=0.0),
        n_at_risk=np.array([10, 8, 6, 4, 2]),
        n_events=np.ones(5, dtype=np.int64),
        n_success=5,
    )


def test_concavity_sign():
    assert concavity(_concave_hazard()) > 0
    assert concavity(_convex_hazard()) < 0


def test_concavity_degenerate_returns_zero():
    ch = CostHazard(
        times=np.array([1.0]),
        cumulative_hazard=np.array([0.5]),
        hazard_increment=np.array([0.5]),
        n_at_risk=np.array([2]),
        n_events=np.array([1]),
        n_success=1,
    )
    assert concavity(ch) == 0.0


def test_restart_effectiveness_flip():
    dfr = restart_effectiveness(make_dfr(n=500, seed=1), n_boot=150, seed=0)
    ifr = restart_effectiveness(make_ifr(n=500, seed=1), n_boot=150, seed=0)
    assert dfr.decision == "restart_helps"
    assert dfr.hazard_trend == "decreasing"
    assert ifr.decision == "do_not_restart"
    assert ifr.hazard_trend == "increasing"


def test_restart_effectiveness_inconclusive_when_few_successes():
    att = [AttemptRecord(cost=float(i + 1), label="success") for i in range(5)]
    att += [AttemptRecord(cost=100.0, label="timeout") for _ in range(20)]
    v = restart_effectiveness(att, n_min_success=20, n_boot=100, seed=0)
    assert v.decision == "inconclusive"
    assert v.n_success == 5


def test_cost_hazard_runs():
    ch = cost_hazard(make_dfr(n=200, seed=0))
    assert ch.times.size >= 2
    assert ch.n_success > 0
