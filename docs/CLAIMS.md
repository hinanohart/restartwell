# CLAIM / NON-CLAIM boundary

restartwell is an **offline measurement instrument**. This file is the contract for what it
does and does not claim.

## What restartwell CLAIMS

- Given a 1-D array of agent attempt **costs** (tokens or wall-clock seconds) and outcome
  **labels** (`success` / `timeout` / `fail`), it computes the **renewal-reward optimal
  restart cutoff** τ\* that minimises expected cost-per-success under a fixed-cutoff restart
  model, with a cluster-bootstrap confidence interval.
- It reports a fail-closed **3-way verdict** — `restart_helps`, `do_not_restart` (raise the
  timeout), or `inconclusive` (use a Luby schedule) — from the sign of the cost-to-success
  hazard trend, with a CI.
- It reports **expected savings** of τ\* versus your current cutoff and versus a naive p90
  cutoff, suppressing the headline when the savings CI straddles zero.
- It emits a paste-able cutoff **config snippet** for common agent frameworks.
- All of the above is **deterministic** given the input and seed, runs on **CPU**, and uses
  no model weights.

## What restartwell does NOT claim (NON-CLAIMS)

- It does **not** estimate the hazard **shape** itself — it consumes Kaplan-Meier /
  Nelson-Aalen from [hinanohart/hazardloop](https://github.com/hinanohart/hazardloop). Use
  hazardloop for survival/hazard estimation.
- It does **not** compute pass@k / pass^k reliability (use hinanohart/passwedge).
- It does **not** run, retrain, or live-control your agent. τ\* and savings are
  **counterfactual** estimates on logged cost, not a live trial.
- The renewal-reward / Luby optimality results hold for **serial** restart-or-wait only.
  Parallel best-of-n and per-step early-abort are explicitly **out of scope** for v0.1.
- τ\* numbers on synthetic data are **illustrative**. Results on a real agent log are
  **exploratory**, not a benchmark claim about any system.
- It makes **no** SOTA / first / best / production-ready / permanent claims. <!-- banned-word list in a NON-CLAIM negation context. # honest:ok -->
- When the hazard is increasing/flat or evidence is insufficient, restartwell **refuses** to
  emit a precise cutoff and recommends a Luby schedule or a higher timeout instead — it does
  not fabricate a cutoff.

## Validation substrate

No redistributable real agent-cost log is bundled. The committed numbers are measured on a
**synthetic-faithful** heavy-tail mixture with known DFR/IFR ground truth, on a held-out
split. This is disclosed verbatim in `results/v0.1.0a1_metrics.json` and the README.

## Prior art (honestly cited)

The restart theory restartwell packages is classical:

- M. Luby, A. Sinclair, D. Zuckerman, "Optimal speedup of Las Vegas algorithms" (1993) — the
  fixed-cutoff and universal (Luby) restart strategies.
- M. Gagliolo, J. Schmidhuber, "Learning restart strategies" (IJCAI 2007).
- The renewal-reward / restart-cutoff treatment in T. Vieira's restart-strategy notes and
  arXiv:1709.10405.
- "Restart Strategies in a Continuous Setting" (2021) — note that Luby's worst-case
  optimality is a **discrete** result and does not transfer verbatim to continuous cost.

restartwell's contribution is the **packaging**: applying serial restart theory to agent
attempt-cost logs as an offline decision + cutoff-config instrument, on top of hazardloop's
survival primitives.
