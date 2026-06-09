"""S4 measurement harness: run the gates + held-out headline and emit an env-stamped JSON.

This is the single source of every numeric claim in the README. It is deterministic given
the seed and substrate. The substrate is synthetic-faithful (no redistributable real agent
log); that fact is recorded verbatim in the output so the README can disclose it.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import restartwell
from restartwell.bench.datasets import make_dfr, split
from restartwell.bench.gates import run_all_gates
from restartwell.bench.metrics import held_out_score
from restartwell.survival import backend


def _env(seed: int, n_boot: int) -> dict[str, object]:
    return {
        "hw": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "seed": seed,
        "version": restartwell.__version__,
        "backend": backend(),
        "n_boot": n_boot,
        "substrate": "synthetic-faithful",
        "substrate_note": (
            "validated on a faithful heavy-tail synthetic mixture; no redistributable real "
            "agent cost log was available. Numbers are illustrative, not a benchmark claim."
        ),
    }


def headline(
    *, n: int = 1200, seed: int = 0, r: float = 400.0, n_boot: int = 1000
) -> dict[str, object]:
    """The held-out DFR headline: tau* savings vs the user timeout and vs p90."""
    attempts = make_dfr(n=n, seed=seed)
    train, held = split(attempts, frac=0.5, seed=seed)
    user_timeout = float(np.percentile([a.cost for a in held], 97))
    sc = held_out_score(train, held, r=r, user_timeout=user_timeout, n_boot=n_boot, seed=seed)
    return {
        "n": n,
        "r": r,
        "user_timeout": user_timeout,
        "tau_star_train": sc.tau_star_train,
        "e_total_tau_star": sc.e_total_tau_star,
        "e_total_adhoc": sc.e_total_adhoc,
        "e_total_p90": sc.e_total_p90,
        "e_total_best_fixed_heldout": sc.e_total_best_fixed_heldout,
        "savings_vs_adhoc": sc.savings_vs_adhoc,
        "savings_vs_p90": sc.savings_vs_p90,
        "savings_vs_adhoc_ci": [sc.savings_vs_adhoc_ci.lower, sc.savings_vs_adhoc_ci.upper],
        "n_train": sc.n_train,
        "n_held": sc.n_held,
    }


def run(
    *, seed: int = 0, n_boot: int = 1000, out_path: str | Path | None = None
) -> dict[str, object]:
    """Run gates + headline; write the env-stamped results JSON."""
    gates = run_all_gates(n_boot=n_boot)
    head = headline(seed=seed, n_boot=n_boot)
    results = {
        "env": _env(seed, n_boot),
        "headline": head,
        "gates": gates,
        "all_gates_passed": gates["all_passed"],
    }
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
            fh.write("\n")
    return results


if __name__ == "__main__":  # pragma: no cover
    out = sys.argv[1] if len(sys.argv) > 1 else "results/v0.1.0a1_metrics.json"
    res = run(out_path=out)
    print(json.dumps(res["env"], indent=2))
    print("all_gates_passed:", res["all_gates_passed"])
