"""restartwell command-line interface.

restartwell report  --input logs.jsonl --unit tokens --r 1500 --current 8000
restartwell verdict --input logs.jsonl
restartwell cutoff  --input logs.jsonl --r 1500
restartwell savings --input logs.jsonl --r 1500 --current 8000
restartwell emit    --input logs.jsonl --r 1500 --framework swe-agent --unit tokens
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar, cast

import typer

from restartwell.cutoff import optimal_cutoff
from restartwell.effectiveness import restart_effectiveness
from restartwell.emit import Framework, Unit, emit_config
from restartwell.instrument import analyze, analyze_by_cohort
from restartwell.intake import from_jsonl
from restartwell.savings import expected_savings
from restartwell.types import AttemptRecord

app = typer.Typer(add_completion=False, help="Offline renewal-reward restart-cutoff instrument.")


def _load(input: str, cost_key: str, label_key: str, cohort_key: str | None) -> list[AttemptRecord]:
    attempts = from_jsonl(input, cost_key=cost_key, label_key=label_key, cohort_key=cohort_key)
    if not attempts:
        typer.echo("no attempts parsed from input", err=True)
        raise typer.Exit(code=2)
    return attempts


_T = TypeVar("_T")


def _compute(fn: Callable[[], _T]) -> _T:
    """Run a domain computation, turning a domain ``ValueError`` (e.g. no successes by any
    cutoff) into a clean CLI error with exit code 2 instead of a raw traceback."""
    try:
        return fn()
    except ValueError as exc:
        typer.echo(f"cannot compute cutoff: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def verdict(
    input: str = typer.Option(..., "--input", "-i", help="JSON-Lines log file."),
    cost_key: str = typer.Option("cost"),
    label_key: str = typer.Option("label"),
    cohort_key: str | None = typer.Option(None),
    n_min_success: int = typer.Option(20),
    alpha: float = typer.Option(0.05),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(0),
) -> None:
    """Print the fail-closed restart verdict."""
    attempts = _load(input, cost_key, label_key, cohort_key)
    v = restart_effectiveness(
        attempts, n_min_success=n_min_success, alpha=alpha, n_boot=n_boot, seed=seed
    )
    typer.echo(f"decision: {v.decision}")
    typer.echo(f"hazard_trend: {v.hazard_trend}")
    lo, hi = v.concavity_ci.lower, v.concavity_ci.upper
    typer.echo(f"concavity: {v.concavity:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    typer.echo(f"weibull_shape: {v.weibull_shape}  agrees: {v.weibull_agrees}")
    typer.echo(f"n_success: {v.n_success}")
    typer.echo(f"reason: {v.reason}")


@app.command()
def cutoff(
    input: str = typer.Option(..., "--input", "-i"),
    r: float = typer.Option(..., "--r", help="Per-restart overhead (same unit as cost)."),
    cost_key: str = typer.Option("cost"),
    label_key: str = typer.Option("label"),
    cohort_key: str | None = typer.Option(None),
    alpha: float = typer.Option(0.05),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(0),
) -> None:
    """Print the renewal-reward optimal cutoff tau* and its percentile baselines."""
    attempts = _load(input, cost_key, label_key, cohort_key)
    c = _compute(lambda: optimal_cutoff(attempts, r=r, n_boot=n_boot, alpha=alpha, seed=seed))
    typer.echo(f"tau_star: {c.tau_star:g}  95% CI [{c.ci.lower:g}, {c.ci.upper:g}]")
    typer.echo(f"E_total(tau*): {c.e_total_at_star:g}")
    typer.echo(f"p90 cutoff: {c.p90:g}  E_total(p90): {c.e_total_at_p90:g}")
    typer.echo(f"p95 cutoff: {c.p95:g}  E_total(p95): {c.e_total_at_p95:g}")
    typer.echo(f"r: {c.r:g}  backend: {c.backend}")


@app.command()
def savings(
    input: str = typer.Option(..., "--input", "-i"),
    r: float = typer.Option(..., "--r"),
    current: float = typer.Option(..., "--current", help="Your current per-attempt cutoff."),
    cost_key: str = typer.Option("cost"),
    label_key: str = typer.Option("label"),
    cohort_key: str | None = typer.Option(None),
    alpha: float = typer.Option(0.05),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(0),
) -> None:
    """Print expected savings of tau* vs your current cutoff."""
    attempts = _load(input, cost_key, label_key, cohort_key)
    c = _compute(lambda: optimal_cutoff(attempts, r=r, n_boot=n_boot, alpha=alpha, seed=seed))
    s = expected_savings(
        attempts,
        current_cutoff=current,
        tau_star=c.tau_star,
        r=r,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    typer.echo(f"current E_total: {s.current_cost:g}")
    typer.echo(f"tau* E_total: {s.optimal_cost:g}")
    typer.echo(f"p90 E_total: {s.p90_cost:g}")
    typer.echo(
        f"savings_fraction: {s.savings_fraction:.4f}  95% CI [{s.ci.lower:.4f}, {s.ci.upper:.4f}]"
    )
    typer.echo(f"suppressed: {s.suppressed}")
    typer.echo(f"reason: {s.reason}")


@app.command()
def emit(
    input: str = typer.Option(..., "--input", "-i"),
    r: float = typer.Option(..., "--r"),
    framework: str = typer.Option(..., "--framework", help="langgraph|crewai|swe-agent|opencode"),
    unit: str = typer.Option("tokens", "--unit", help="tokens|seconds"),
    cost_key: str = typer.Option("cost"),
    label_key: str = typer.Option("label"),
    cohort_key: str | None = typer.Option(None),
    seed: int = typer.Option(0),
) -> None:
    """Emit a paste-able cutoff config snippet for a framework."""
    if framework not in ("langgraph", "crewai", "swe-agent", "opencode"):
        typer.echo(f"unknown framework {framework!r}", err=True)
        raise typer.Exit(code=2)
    if unit not in ("tokens", "seconds"):
        typer.echo(f"unknown unit {unit!r}", err=True)
        raise typer.Exit(code=2)
    attempts = _load(input, cost_key, label_key, cohort_key)
    v = restart_effectiveness(attempts, seed=seed)
    if v.decision != "restart_helps":
        typer.echo(f"# verdict is '{v.decision}', not 'restart_helps' — no cutoff config emitted.")
        typer.echo(f"# {v.reason}")
        raise typer.Exit(code=0)
    c = _compute(lambda: optimal_cutoff(attempts, r=r, seed=seed))
    # framework/unit are validated above, so the casts are runtime-safe narrowings to the
    # Literal types emit_config expects (satisfies both mypy and ty without an ignore).
    snippet, faithful = emit_config(
        c.tau_star,
        framework=cast(Framework, framework),
        unit=cast(Unit, unit),
    )
    typer.echo(snippet)
    if not faithful:
        typer.echo("# NOTE: approximate mapping — adapt the budget knob to your runner.")


@app.command()
def report(
    input: str = typer.Option(..., "--input", "-i"),
    r: float = typer.Option(..., "--r"),
    unit: str = typer.Option("tokens", "--unit"),
    current: float | None = typer.Option(None, "--current"),
    cost_key: str = typer.Option("cost"),
    label_key: str = typer.Option("label"),
    cohort_key: str | None = typer.Option(None),
    by_cohort: bool = typer.Option(False, "--by-cohort"),
    as_json: bool = typer.Option(False, "--json"),
    alpha: float = typer.Option(0.05),
    n_boot: int = typer.Option(2000),
    seed: int = typer.Option(0),
) -> None:
    """Full report: verdict + cutoff/luby + savings (+ optional per-cohort)."""
    attempts = _load(input, cost_key, label_key, cohort_key)
    if by_cohort:
        reports = analyze_by_cohort(
            attempts,
            r=r,
            unit=unit,
            current_cutoff=current,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        if as_json:
            typer.echo(json.dumps({k: _report_dict(v) for k, v in reports.items()}, indent=2))
            return
        for name, rp in reports.items():
            line = f"[{name}] {rp.verdict.decision}"
            if rp.cutoff is not None:
                line += f"  tau*={rp.cutoff.tau_star:g} (p90={rp.cutoff.p90:g})"
            typer.echo(line)
        return
    rep = analyze(
        attempts,
        r=r,
        unit=unit,
        current_cutoff=current,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
    )
    if as_json:
        typer.echo(json.dumps(_report_dict(rep), indent=2))
        return
    typer.echo(
        f"backend: {rep.backend}   n_attempts: {rep.n_attempts}   unit: {rep.unit}   r: {rep.r:g}"
    )
    typer.echo(f"VERDICT: {rep.verdict.decision}  ({rep.verdict.reason})")
    if rep.cutoff is not None:
        c = rep.cutoff
        typer.echo(
            f"tau*: {c.tau_star:g}  95% CI [{c.ci.lower:g}, {c.ci.upper:g}]   (p90={c.p90:g})"
        )
    if rep.savings is not None and not rep.savings.suppressed:
        typer.echo(f"savings: {rep.savings.savings_fraction:.1%}  ({rep.savings.reason})")
    elif rep.savings is not None:
        typer.echo(f"savings: suppressed — {rep.savings.reason}")
    if rep.luby is not None:
        head = ", ".join(f"{x:g}" for x in rep.luby.sequence[:8])
        typer.echo(f"luby schedule (unit={rep.luby.unit:g}): {head}, ...")
    for note in rep.notes:
        typer.echo(f"note: {note}")


def _report_dict(rep) -> dict[str, object]:  # type: ignore[no-untyped-def]
    d: dict[str, object] = {
        "backend": rep.backend,
        "n_attempts": rep.n_attempts,
        "unit": rep.unit,
        "r": rep.r,
        "verdict": {
            "decision": rep.verdict.decision,
            "hazard_trend": rep.verdict.hazard_trend,
            "concavity": rep.verdict.concavity,
            "concavity_ci": [rep.verdict.concavity_ci.lower, rep.verdict.concavity_ci.upper],
            "p_value": rep.verdict.p_value,
            "weibull_shape": rep.verdict.weibull_shape,
            "n_success": rep.verdict.n_success,
        },
    }
    if rep.cutoff is not None:
        d["cutoff"] = {
            "tau_star": rep.cutoff.tau_star,
            "ci": [rep.cutoff.ci.lower, rep.cutoff.ci.upper],
            "e_total_at_star": rep.cutoff.e_total_at_star,
            "p90": rep.cutoff.p90,
            "e_total_at_p90": rep.cutoff.e_total_at_p90,
        }
    if rep.savings is not None:
        d["savings"] = {
            "savings_fraction": rep.savings.savings_fraction,
            "ci": [rep.savings.ci.lower, rep.savings.ci.upper],
            "suppressed": rep.savings.suppressed,
        }
    if rep.luby is not None:
        d["luby"] = {"unit": rep.luby.unit, "sequence": rep.luby.sequence[:16]}
    return d


if __name__ == "__main__":  # pragma: no cover
    app()
