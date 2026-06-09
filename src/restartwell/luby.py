"""Luby universal restart schedule (the INCONCLUSIVE / unknown-distribution fallback).

When the cost-to-success distribution is unknown or there is too little evidence to fit a
precise cutoff, restartwell does not fabricate one: it returns the Luby universal sequence
``1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...`` scaled by a unit. Luby, Sinclair &
Zuckerman (1993) proved this schedule is O(log) competitive with the optimal fixed cutoff in
the worst case for *discrete* Las Vegas algorithms. That guarantee is the classical discrete
result; in a continuous-cost setting it is a heuristic (its worst-case optimality does not
transfer verbatim — see "Restart Strategies in a Continuous Setting", 2021).
"""

from __future__ import annotations

from restartwell.types import LubySchedule

_COMPETITIVE_NOTE = (
    "Luby universal sequence (1,1,2,1,1,2,4,...). O(log) competitive with the optimal fixed "
    "cutoff in the worst case for discrete Las Vegas algorithms (Luby-Sinclair-Zuckerman 1993). "
    "Used here as a fail-closed fallback when a precise cutoff is not estimable; in a "
    "continuous-cost setting the worst-case optimality is heuristic, not proven."
)


def luby(i: int) -> int:
    """The i-th term (1-indexed) of the Luby sequence 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,..."""
    if i < 1:
        raise ValueError(f"Luby index must be >= 1, got {i}")
    k = 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        if (1 << (k - 1)) <= i < (1 << k) - 1:
            return luby(i - (1 << (k - 1)) + 1)
        k += 1


def luby_schedule(base: float, length: int = 64) -> LubySchedule:
    """Return the first ``length`` Luby cutoffs scaled by ``base`` (the unit cutoff)."""
    if base <= 0:
        raise ValueError(f"base unit must be > 0, got {base}")
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    sequence = [base * float(luby(i)) for i in range(1, length + 1)]
    return LubySchedule(unit=float(base), sequence=sequence, competitive_note=_COMPETITIVE_NOTE)
