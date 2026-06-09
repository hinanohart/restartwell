from __future__ import annotations

import numpy as np

from restartwell.bench.datasets import make_dfr
from restartwell.cutoff import optimal_cutoff
from restartwell.savings import expected_savings


def test_savings_positive_on_dfr():
    att = make_dfr(n=600, seed=4)
    c = optimal_cutoff(att, r=400.0, n_boot=1, seed=0)
    user_timeout = float(np.percentile([a.cost for a in att], 97))
    s = expected_savings(
        att, current_cutoff=user_timeout, tau_star=c.tau_star, r=400.0, n_boot=200, seed=0
    )
    assert s.savings_fraction > 0
    assert not s.suppressed
    assert s.ci.lower > 0


def test_savings_suppressed_when_cutoff_out_of_range():
    att = make_dfr(n=400, seed=0)
    c = optimal_cutoff(att, r=400.0, n_boot=1, seed=0)
    huge = max(a.cost for a in att) * 100.0
    s = expected_savings(att, current_cutoff=huge, tau_star=c.tau_star, r=400.0, n_boot=50, seed=0)
    assert s.suppressed


def test_savings_reports_p90_cost():
    att = make_dfr(n=400, seed=1)
    c = optimal_cutoff(att, r=400.0, n_boot=1, seed=0)
    ut = float(np.percentile([a.cost for a in att], 95))
    s = expected_savings(att, current_cutoff=ut, tau_star=c.tau_star, r=400.0, n_boot=50, seed=0)
    assert np.isfinite(s.p90_cost)
