"""Public types for restartwell.

These dataclasses are the contract between the restartwell modules. They are
hazardloop-free on purpose: the survival adapter (:mod:`restartwell.survival`) is the
*only* module allowed to import hazardloop, and it converts hazardloop's result objects
into the neutral :class:`CostSurvival` / :class:`CostHazard` / :class:`BootstrapCI`
defined here. That keeps the rest of the package (cutoff, effectiveness, savings, luby,
emit, cli) decoupled from the survival backend and lets the bundled shim stand in when
hazardloop is unavailable. The boundary is enforced by import-linter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

Label = Literal["success", "timeout", "fail"]
Decision = Literal["restart_helps", "do_not_restart", "inconclusive"]
HazardTrend = Literal["decreasing", "increasing", "flat"]


@dataclass(frozen=True)
class AttemptRecord:
    """One agent attempt.

    ``cost`` is the attempt's cost on a single chosen axis (tokens or wall-clock seconds),
    finite and non-negative. ``label`` is the terminal outcome: ``success`` means the
    attempt reached its goal; ``timeout`` means it hit a wall (budget/time) and was cut
    off; ``fail`` means it terminated unsuccessfully for another reason. ``cohort`` groups
    attempts that should be resampled together by the cluster bootstrap (e.g. model x
    harness x task-bucket). ``attempt_id`` is an optional opaque identifier.
    """

    cost: float
    label: Label
    cohort: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        if self.label not in ("success", "timeout", "fail"):
            raise ValueError(f"label must be success|timeout|fail, got {self.label!r}")
        if not np.isfinite(self.cost):
            raise ValueError(f"cost must be finite, got {self.cost}")
        if self.cost < 0:
            raise ValueError(f"cost must be >= 0, got {self.cost}")


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap confidence interval (neutral mirror of hazardloop's BootstrapCI)."""

    point: float
    lower: float
    upper: float
    alpha: float
    method: str
    n_boot: int
    n_effective: int

    def excludes(self, value: float) -> bool:
        """True iff ``value`` lies strictly outside [lower, upper]."""
        return value < self.lower or value > self.upper


@dataclass(frozen=True)
class CostSurvival:
    """Kaplan-Meier survival of cost-to-success: S(t) = P(success-cost > t).

    Built by the survival adapter under hazardloop's ``completion_as_event`` model
    (success is the event; timeout/fail are right-censored). ``backend`` records whether
    the curve came from ``hazardloop`` or the bundled ``shim``.
    """

    times: FloatArray
    survival: FloatArray
    n_at_risk: IntArray
    n_events: IntArray
    n_success: int
    n_censored: int
    backend: str

    def survival_at(self, t: float) -> float:
        """Right-continuous S(t); returns 1.0 before the first success time."""
        if self.times.size == 0:
            return 1.0
        idx = int(np.searchsorted(self.times, t, side="right")) - 1
        if idx < 0:
            return 1.0
        return float(self.survival[idx])

    def e_min(self, tau: float) -> float:
        """E[min(A, tau)] = integral_0^tau S(u) du, with S the cost-to-success survival.

        This is the expected per-attempt cost under a cutoff at ``tau``: you pay the
        success cost if you succeed by ``tau``, otherwise you pay ``tau``.
        """
        if tau <= 0:
            return 0.0
        total = 0.0
        prev_t = 0.0
        prev_s = 1.0
        for t, s in zip(self.times, self.survival):
            if t >= tau:
                break
            total += prev_s * (t - prev_t)
            prev_t = t
            prev_s = float(s)
        total += prev_s * (tau - prev_t)
        return total


@dataclass(frozen=True)
class CostHazard:
    """Nelson-Aalen cumulative hazard / per-time hazard increments of cost-to-success."""

    times: FloatArray
    cumulative_hazard: FloatArray
    hazard_increment: FloatArray
    n_at_risk: IntArray
    n_events: IntArray
    n_success: int


@dataclass(frozen=True)
class RestartVerdict:
    """The fail-closed restart decision that gates the cutoff.

    ``decision`` is the headline gate: ``restart_helps`` (decreasing cost-to-success
    hazard -> a finite cutoff can beat waiting), ``do_not_restart`` (increasing/wear-out
    hazard -> raising the timeout is better than restarting), or ``inconclusive`` (too few
    successes or no resolvable trend -> fall back to a Luby schedule rather than fabricate
    a precise cutoff).
    """

    decision: Decision
    hazard_trend: HazardTrend
    concavity: float
    concavity_ci: BootstrapCI
    p_value: float
    weibull_shape: float | None
    weibull_agrees: bool | None
    n_success: int
    reason: str


@dataclass(frozen=True)
class CutoffResult:
    """The renewal-reward optimal restart cutoff and its percentile baselines."""

    tau_star: float
    ci: BootstrapCI
    e_total_at_star: float
    grid: FloatArray
    e_total_curve: FloatArray
    r: float
    p90: float
    e_total_at_p90: float
    p95: float
    e_total_at_p95: float
    backend: str


@dataclass(frozen=True)
class LubySchedule:
    """A Luby universal restart schedule, unit-scaled."""

    unit: float
    sequence: list[float]
    competitive_note: str


@dataclass(frozen=True)
class SavingsReport:
    """Expected cost-per-success at the current cutoff vs tau* vs the p90 cutoff."""

    current_cost: float
    optimal_cost: float
    p90_cost: float
    savings_fraction: float
    ci: BootstrapCI
    held_out: bool
    suppressed: bool
    reason: str


@dataclass(frozen=True)
class RestartReport:
    """The full bundle emitted by the high-level instrument."""

    verdict: RestartVerdict
    cutoff: CutoffResult | None
    luby: LubySchedule | None
    savings: SavingsReport | None
    unit: str
    r: float
    n_attempts: int
    backend: str
    cohort: str | None = None
    notes: Sequence[str] = field(default_factory=tuple)
