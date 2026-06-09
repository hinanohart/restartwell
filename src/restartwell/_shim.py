"""Standalone Kaplan-Meier / Nelson-Aalen availability fallback.

This is a ~40-line product-limit KM + Nelson-Aalen increment estimator used **only** when
the survival-backend dependency is unavailable (see :mod:`restartwell.survival`). It is a
fallback, not a competitor: it implements just the cost-to-success KM/NA primitives
restartwell needs and nothing more. The richer survival scope (multi-cause incidence curves
and parametric fits) is intentionally left to the backend and never re-implemented here.
"""

from __future__ import annotations

import numpy as np

from restartwell.types import CostHazard, CostSurvival


def _km_na(durations: np.ndarray, is_success: np.ndarray) -> tuple[CostSurvival, CostHazard]:
    dur = np.asarray(durations, dtype=np.float64)
    ev = np.asarray(is_success, dtype=bool)
    n = dur.size
    n_success = int(np.count_nonzero(ev))
    event_times = np.unique(dur[ev]) if n_success else np.empty(0, dtype=np.float64)

    times: list[float] = []
    surv: list[float] = []
    hinc: list[float] = []
    cumh: list[float] = []
    nrisk: list[int] = []
    nev: list[int] = []
    s = 1.0
    h_cum = 0.0
    for t in event_times:
        y = int(np.count_nonzero(dur >= t))  # at risk (censored at exactly t still at risk)
        di = int(np.count_nonzero((dur == t) & ev))
        if y == 0 or di == 0:
            continue
        h = di / y
        s *= 1.0 - h
        h_cum += h
        times.append(float(t))
        surv.append(s)
        hinc.append(h)
        cumh.append(h_cum)
        nrisk.append(y)
        nev.append(di)

    t_arr = np.asarray(times, dtype=np.float64)
    cs = CostSurvival(
        times=t_arr,
        survival=np.asarray(surv, dtype=np.float64),
        n_at_risk=np.asarray(nrisk, dtype=np.int64),
        n_events=np.asarray(nev, dtype=np.int64),
        n_success=n_success,
        n_censored=n - n_success,
        backend="shim",
    )
    ch = CostHazard(
        times=t_arr,
        cumulative_hazard=np.asarray(cumh, dtype=np.float64),
        hazard_increment=np.asarray(hinc, dtype=np.float64),
        n_at_risk=np.asarray(nrisk, dtype=np.int64),
        n_events=np.asarray(nev, dtype=np.int64),
        n_success=n_success,
    )
    return cs, ch
