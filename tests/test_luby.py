from __future__ import annotations

import pytest

from restartwell.luby import luby, luby_schedule

LUBY15 = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]


def test_luby_sequence_verbatim():
    assert [luby(i) for i in range(1, 16)] == LUBY15


def test_luby_schedule_scaled():
    sched = luby_schedule(base=100.0, length=15)
    assert sched.sequence == [100.0 * x for x in LUBY15]
    assert sched.unit == 100.0


def test_luby_schedule_has_competitive_note():
    sched = luby_schedule(base=1.0)
    assert "competitive" in sched.competitive_note.lower()
    assert "luby" in sched.competitive_note.lower()


def test_luby_rejects_bad_index():
    with pytest.raises(ValueError):
        luby(0)


def test_luby_schedule_rejects_bad_base():
    with pytest.raises(ValueError):
        luby_schedule(base=0.0)
