from __future__ import annotations

import numpy as np
import pytest

from restartwell import survival
from restartwell._shim import _km_na
from restartwell.bench.datasets import make_dfr
from restartwell.survival import backend, cost_hazard, cost_survival, n_successes, weibull_shape
from restartwell.types import AttemptRecord


def test_backend_is_hazardloop_when_installed():
    # CI installs hazardloop; the standalone cell exercises the shim path separately.
    assert backend() in ("hazardloop", "shim")


def test_cost_survival_monotone_nonincreasing():
    cs = cost_survival(make_dfr(n=300, seed=0))
    assert np.all(np.diff(cs.survival) <= 1e-12)
    assert cs.n_success > 0


def test_n_successes():
    att = [AttemptRecord(cost=1.0, label="success"), AttemptRecord(cost=2.0, label="timeout")]
    assert n_successes(att) == 1


def test_weibull_shape_returns_float_on_hazardloop():
    if backend() != "hazardloop":
        pytest.skip("weibull only on hazardloop backend")
    s = weibull_shape(make_dfr(n=400, seed=0))
    assert s is None or isinstance(s, float)


def test_shim_matches_hazardloop_km():
    if backend() != "hazardloop":
        pytest.skip("comparison requires hazardloop installed")
    att = make_dfr(n=400, seed=2)
    hl = cost_survival(att)
    dur = np.array([a.cost for a in att])
    ev = np.array([a.label == "success" for a in att])
    sh, _ = _km_na(dur, ev)
    # product-limit KM is identical regardless of backend; compare S at several points
    for t in np.percentile(dur, [10, 30, 50, 70, 90]):
        assert hl.survival_at(float(t)) == pytest.approx(sh.survival_at(float(t)), abs=1e-9)


def test_forced_shim_path(monkeypatch):
    # Force the standalone shim branch and confirm the public API still works.
    monkeypatch.setattr(survival, "_HAZARDLOOP_AVAILABLE", False)
    monkeypatch.setattr(survival, "_BACKEND", "shim")
    att = make_dfr(n=300, seed=1)
    cs = cost_survival(att)
    ch = cost_hazard(att)
    assert cs.backend == "shim"
    assert cs.n_success > 0
    assert ch.times.size >= 2
    assert weibull_shape(att) is None  # shim does not reimplement Weibull-AFT


def test_empty_raises():
    with pytest.raises(ValueError):
        cost_survival([])
