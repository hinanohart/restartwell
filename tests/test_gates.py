"""The five numeric kill-gates, asserted at reduced scale for CI.

The full-scale run (used for the committed results JSON) lives in
``restartwell.bench.harness``; these tests assert the same gates pass on smaller samples so
CI stays fast.
"""

from __future__ import annotations

from restartwell.bench import gates


def test_gate_flip():
    g = gates.gate_flip(n=500, seed=0, n_boot=150)
    assert g["passed"], g


def test_gate_censor():
    g = gates.gate_censor(n=2000, seed=0)
    assert g["passed"], g
    assert g["internal_diff"] < 1e-9


def test_gate_luby():
    assert gates.gate_luby()["passed"]


def test_gate_determinism():
    g = gates.gate_determinism(n=300, seed=0, n_boot=120)
    assert g["passed"], g


def test_gate_calib():
    g = gates.gate_calib(n=600, seed=0, r=400.0, n_boot=120)
    assert g["passed"], g
    # at least the canonical r cell must show a material, CI-separated win
    canonical = next(c for c in g["cells"] if c["r"] == 400.0)
    assert canonical["ratio_vs_min"] <= 0.90
    assert canonical["ci_separated"]
