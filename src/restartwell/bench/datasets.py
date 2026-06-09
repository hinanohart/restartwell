"""Synthetic-faithful agent-cost-log generators with known ground truth.

Each generator draws a would-be cost-to-success ``A`` from a distribution with a known
hazard shape, applies a data-generating timeout ``T`` (attempts with ``A > T`` are recorded
as ``timeout`` at cost ``T`` — right-censored), and sprinkles a small fraction of non-timeout
``fail`` outcomes. The hazard shape is the ground truth the kill-gates check against:

- ``make_dfr``  : Pareto cost-to-success -> DECREASING hazard -> restart SHOULD help.
- ``make_ifr``  : Weibull(k>1) cost-to-success -> INCREASING hazard -> restart should NOT help.
- ``make_flat`` : Exponential cost-to-success -> CONSTANT hazard -> borderline / inconclusive.

These are deliberately faithful to the heavy-tailed runtime distributions reported for
randomized search and LLM-agent runs; they are not a real log.
"""

from __future__ import annotations

import numpy as np

from restartwell.types import AttemptRecord


def _assemble(
    a: np.ndarray,
    timeout: float,
    fail_frac: float,
    rng: np.random.Generator,
    cohort: str | None,
) -> list[AttemptRecord]:
    out: list[AttemptRecord] = []
    for i, cost in enumerate(a):
        u = rng.random()
        if u < fail_frac:
            # a non-timeout failure at a fraction of its would-be cost (right-censored)
            out.append(
                AttemptRecord(
                    cost=float(min(cost, timeout) * rng.uniform(0.2, 0.9)),
                    label="fail",
                    cohort=cohort,
                    attempt_id=f"a{i}",
                )
            )
        elif cost <= timeout:
            out.append(
                AttemptRecord(cost=float(cost), label="success", cohort=cohort, attempt_id=f"a{i}")
            )
        else:
            out.append(
                AttemptRecord(
                    cost=float(timeout), label="timeout", cohort=cohort, attempt_id=f"a{i}"
                )
            )
    return out


def make_dfr(
    n: int = 600,
    *,
    seed: int = 0,
    fast_frac: float = 0.65,
    fast_scale: float = 600.0,
    slow_scale: float = 9000.0,
    timeout_q: float = 0.92,
    fail_frac: float = 0.05,
    cohort: str | None = None,
) -> list[AttemptRecord]:
    """Bimodal 'fast-success vs hang' cost-to-success (decreasing hazard): restarting helps.

    A fraction ``fast_frac`` of attempts finish quickly (lognormal around ``fast_scale``); the
    rest hang in a heavy slow mode (lognormal around ``slow_scale``). Once the fast successes
    are harvested, the at-risk population is dominated by slow hangs whose instantaneous
    success rate is low — i.e. the hazard decreases — so cutting a hung attempt and restarting
    gives a fresh shot at a fast success. This is the canonical regime where restart wins.
    """
    rng = np.random.default_rng(seed)
    is_fast = rng.random(n) < fast_frac
    a = np.empty(n, dtype=np.float64)
    a[is_fast] = rng.lognormal(np.log(fast_scale), 0.5, size=int(is_fast.sum()))
    a[~is_fast] = rng.lognormal(np.log(slow_scale), 0.6, size=int((~is_fast).sum()))
    timeout = float(np.quantile(a, timeout_q))
    return _assemble(a, timeout, fail_frac, rng, cohort)


def make_ifr(
    n: int = 600,
    *,
    seed: int = 0,
    k: float = 2.5,
    scale: float = 1000.0,
    timeout_q: float = 0.95,
    fail_frac: float = 0.05,
    cohort: str | None = None,
) -> list[AttemptRecord]:
    """Weibull(k>1) cost-to-success (increasing hazard): restarting does not help."""
    rng = np.random.default_rng(seed)
    a = scale * rng.weibull(k, size=n)
    timeout = float(np.quantile(a, timeout_q))
    return _assemble(a, timeout, fail_frac, rng, cohort)


def make_flat(
    n: int = 600,
    *,
    seed: int = 0,
    scale: float = 1000.0,
    timeout_q: float = 0.95,
    fail_frac: float = 0.05,
    cohort: str | None = None,
) -> list[AttemptRecord]:
    """Exponential cost-to-success (constant hazard): borderline.

    A true constant hazard sits on the restart_helps/do_not_restart boundary; on a finite
    sample the verdict reads either ``inconclusive`` or (with a small negative concavity bias)
    ``do_not_restart`` — never ``restart_helps``. Used by G_flip as the fail-closed guard.
    """
    rng = np.random.default_rng(seed)
    a = rng.exponential(scale, size=n)
    timeout = float(np.quantile(a, timeout_q))
    return _assemble(a, timeout, fail_frac, rng, cohort)


def split(
    attempts: list[AttemptRecord], *, frac: float = 0.5, seed: int = 0
) -> tuple[list[AttemptRecord], list[AttemptRecord]]:
    """Deterministic train/held-out split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(attempts))
    rng.shuffle(idx)
    cut = int(len(attempts) * frac)
    train = [attempts[i] for i in idx[:cut]]
    held = [attempts[i] for i in idx[cut:]]
    return train, held
