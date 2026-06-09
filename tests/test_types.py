from __future__ import annotations

import numpy as np
import pytest

from restartwell.types import AttemptRecord, BootstrapCI, CostSurvival


def test_attempt_record_validates_label():
    with pytest.raises(ValueError):
        AttemptRecord(cost=1.0, label="nope")  # type: ignore[arg-type]


def test_attempt_record_rejects_negative_cost():
    with pytest.raises(ValueError):
        AttemptRecord(cost=-1.0, label="success")


def test_attempt_record_rejects_nonfinite_cost():
    with pytest.raises(ValueError):
        AttemptRecord(cost=float("inf"), label="success")


def test_bootstrap_ci_excludes():
    ci = BootstrapCI(0.5, 0.2, 0.8, 0.05, "bca", 100, 100)
    assert ci.excludes(0.0)
    assert ci.excludes(1.0)
    assert not ci.excludes(0.5)


def test_cost_survival_survival_at_before_first_is_one():
    cs = CostSurvival(
        times=np.array([10.0, 20.0]),
        survival=np.array([0.5, 0.25]),
        n_at_risk=np.array([4, 2]),
        n_events=np.array([2, 1]),
        n_success=3,
        n_censored=1,
        backend="shim",
    )
    assert cs.survival_at(0.0) == 1.0
    assert cs.survival_at(5.0) == 1.0
    assert cs.survival_at(10.0) == 0.5
    assert cs.survival_at(15.0) == 0.5
    assert cs.survival_at(100.0) == 0.25


def test_cost_survival_e_min_step_integral():
    cs = CostSurvival(
        times=np.array([10.0, 20.0]),
        survival=np.array([0.5, 0.0]),
        n_at_risk=np.array([2, 1]),
        n_events=np.array([1, 1]),
        n_success=2,
        n_censored=0,
        backend="shim",
    )
    # integral of S over [0,20]: 1*10 + 0.5*10 = 15
    assert cs.e_min(20.0) == pytest.approx(15.0)
    # over [0,5]: 1*5 = 5
    assert cs.e_min(5.0) == pytest.approx(5.0)
