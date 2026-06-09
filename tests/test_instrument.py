from __future__ import annotations

import numpy as np

from restartwell.bench.datasets import make_dfr, make_flat, make_ifr
from restartwell.instrument import analyze, analyze_by_cohort


def test_analyze_restart_helps_has_cutoff_and_savings():
    att = make_dfr(n=600, seed=2)
    ut = float(np.percentile([a.cost for a in att], 97))
    rep = analyze(att, r=400.0, unit="tokens", current_cutoff=ut, n_boot=150, seed=0)
    assert rep.verdict.decision == "restart_helps"
    assert rep.cutoff is not None
    assert rep.savings is not None
    assert rep.luby is None


def test_analyze_do_not_restart_no_cutoff():
    att = make_ifr(n=500, seed=2)
    rep = analyze(att, r=400.0, n_boot=150, seed=0)
    assert rep.verdict.decision == "do_not_restart"
    assert rep.cutoff is None
    assert rep.luby is None


def test_analyze_inconclusive_gives_luby():
    att = [
        *(make_flat(n=10, seed=1)[:5]),
    ]
    # few successes -> inconclusive -> luby
    rep = analyze(att, r=100.0, n_min_success=20, n_boot=50, seed=0)
    assert rep.verdict.decision == "inconclusive"
    assert rep.luby is not None
    assert rep.cutoff is None


def test_analyze_by_cohort():
    a = make_dfr(n=300, seed=1, cohort="A")
    b = make_ifr(n=300, seed=2, cohort="B")
    reports = analyze_by_cohort(a + b, r=400.0, n_boot=120, seed=0)
    assert set(reports.keys()) == {"A", "B"}
    assert reports["A"].verdict.decision == "restart_helps"
    assert reports["B"].verdict.decision == "do_not_restart"


def test_analyze_rejects_negative_r():
    import pytest

    with pytest.raises(ValueError):
        analyze(make_dfr(n=50, seed=0), r=-1.0)
