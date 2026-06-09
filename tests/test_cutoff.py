from __future__ import annotations

import numpy as np
import pytest

from restartwell.bench.datasets import make_dfr
from restartwell.cutoff import default_grid, e_total, expected_cost_per_success, optimal_cutoff
from restartwell.types import AttemptRecord, CostSurvival


def _surv() -> CostSurvival:
    return CostSurvival(
        times=np.array([10.0, 20.0]),
        survival=np.array([0.5, 0.0]),
        n_at_risk=np.array([2, 1]),
        n_events=np.array([1, 1]),
        n_success=2,
        n_censored=0,
        backend="shim",
    )


def test_e_total_formula_hand_values():
    surv = _surv()
    # tau=20: S=0, e_min=15, P(A<=tau)=1 -> 15
    assert e_total(20.0, surv, r=5.0) == pytest.approx(15.0)
    # tau=10: S=0.5, e_min=10, P(A<=tau)=0.5 -> (10 + 5*0.5)/0.5 = 25
    assert e_total(10.0, surv, r=5.0) == pytest.approx(25.0)


def test_e_total_infinite_when_no_success_by_cutoff():
    surv = _surv()
    # before first success time, P(A<=tau)=0 -> inf
    assert e_total(5.0, surv, r=1.0) == float("inf")


def test_e_total_rejects_negative_r():
    with pytest.raises(ValueError):
        e_total(10.0, _surv(), r=-1.0)


def test_default_grid_spans_range():
    att = make_dfr(n=200, seed=0)
    g = default_grid(att, n=50)
    assert g.ndim == 1
    assert g[0] < g[-1]


def test_optimal_cutoff_beats_p90_on_dfr():
    att = make_dfr(n=600, seed=3)
    c = optimal_cutoff(att, r=400.0, n_boot=1, seed=0)
    assert np.isfinite(c.tau_star)
    assert c.e_total_at_star <= c.e_total_at_p90
    assert c.tau_star < c.p90  # restart cutoff sits below the naive percentile here


def test_optimal_cutoff_requires_r():
    att = make_dfr(n=100, seed=0)
    with pytest.raises(ValueError):
        optimal_cutoff(att, r=-1.0, n_boot=1)


def test_optimal_cutoff_deterministic():
    att = make_dfr(n=300, seed=5)
    c1 = optimal_cutoff(att, r=400.0, n_boot=100, seed=0)
    c2 = optimal_cutoff(att, r=400.0, n_boot=100, seed=0)
    assert c1.tau_star == c2.tau_star
    assert c1.ci.lower == c2.ci.lower
    assert c1.ci.upper == c2.ci.upper


def test_expected_cost_per_success_no_success_is_inf():
    att = [AttemptRecord(cost=1.0, label="timeout"), AttemptRecord(cost=2.0, label="fail")]
    assert expected_cost_per_success(att, 5.0, 1.0) == float("inf")
