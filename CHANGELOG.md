# Changelog

All notable changes to restartwell are documented here. This project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: minor/patch may break).

## [0.1.0a2] — 2026-06-09 (pre-alpha)

Polish-only release; no change to any headline number or gate result (the measurement core
is untouched). A post-release audit (two fidelity/quality monitors + one meta verifier)
found no defects, only minor refinements.

### Changed
- `restart_effectiveness`: the inconclusive (too-few-successes) branch now reports a neutral
  `hazard_trend="flat"` instead of an over-read raw point sign, matching the CI-straddle
  inconclusive branch.
- `concavity` docstring now states that only the **sign** (and its CI) is load-bearing; the
  magnitude is censoring-dependent and is not an effect size.
- CLI `cutoff` / `savings` / `emit`: a domain error on no-success input now exits cleanly
  with code 2 and a message instead of a raw traceback.
- `gate_flip` additionally asserts a constant-hazard (flat) sample never reads
  `restart_helps` (fail-closed guard); docstrings clarified for the flat case and for the
  intentionally-independent `E[min]` re-derivation in `gate_censor`.

## [0.1.0a1] — 2026-06-09 (pre-alpha)

Initial pre-alpha. An offline renewal-reward restart-cutoff instrument for agent
attempt-cost logs.

### Added
- `restart_effectiveness` — fail-closed 3-way verdict (`restart_helps` /
  `do_not_restart` / `inconclusive`) from the cost-to-success hazard concavity, with a
  cluster-bootstrap CI and a Weibull-shape secondary anchor.
- `optimal_cutoff` — renewal-reward optimal restart cutoff τ\* (`E_total(tau) =
  (E[min(A,tau)] + r·P(A>tau)) / P(A<=tau)`) with a BCa cluster-bootstrap CI, alongside p90
  / p95 percentile baselines.
- `expected_savings` — expected cost-per-success of τ\* vs the current cutoff and vs p90,
  with headline suppression when the CI straddles zero.
- `luby_schedule` — Luby universal fallback schedule for the inconclusive verdict.
- `emit_config` — paste-able cutoff config snippets for LangGraph / CrewAI / SWE-agent /
  opencode.
- `analyze` / `analyze_by_cohort` — high-level orchestration into a `RestartReport`.
- `restartwell` CLI: `verdict`, `cutoff`, `savings`, `emit`, `report`.
- Log adapters: generic JSONL, OpenTelemetry spans, Langfuse traces.
- Survival backend via hazardloop (pinned commit) with a bundled standalone KM/NA shim
  fallback (`restartwell._shim`).
- Synthetic-faithful benchmark substrate and five numeric kill-gates (G_flip, G_calib,
  G_censor, G_luby, G_determinism).

### Boundary
- hazardloop estimates the hazard **shape**; restartwell ships only the restart
  **decision + cutoff config** layer. Enforced by import-linter and an AJ/CIF grep guard.
