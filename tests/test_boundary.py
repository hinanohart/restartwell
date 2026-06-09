"""Boundary tests: restartwell must not re-implement hazardloop's survival scope (Aalen-Johansen
/ cumulative-incidence), and only the survival adapter may import hazardloop directly.
"""

from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "restartwell"

# Identifiers from hazardloop's own scope that restartwell must never define/call.
FORBIDDEN = [
    r"aalen[_\-]?johansen",
    r"cumulative_incidence",
    r"\bcif_at\b",
    r"\bcif_by_cause\b",
]

# An actual hazardloop import *statement* (line-anchored), not a mention in prose.
_IMPORT_RE = re.compile(r"^\s*(?:import\s+hazardloop\b|from\s+hazardloop\b)", re.MULTILINE)


def _py_files():
    return list(SRC.rglob("*.py"))


def test_no_aalen_johansen_or_cif_reimplementation():
    pat = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
    offenders = [p.name for p in _py_files() if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"AJ/CIF vocabulary found in: {offenders}"


def test_hazardloop_imported_only_in_survival():
    hits = [p.name for p in _py_files() if _IMPORT_RE.search(p.read_text(encoding="utf-8"))]
    assert hits == ["survival.py"], f"hazardloop imported outside survival.py: {hits}"


def test_shim_has_no_hazardloop_import_statement():
    shim = (SRC / "_shim.py").read_text(encoding="utf-8")
    assert not _IMPORT_RE.search(shim), "shim must not import hazardloop"
