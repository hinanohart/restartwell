"""Emit a paste-able cutoff config snippet for common agent frameworks.

These are illustrative templates that wire the measured tau* into a framework's
per-attempt budget knob. ``faithful`` is True when the snippet maps to a documented,
direct configuration field for the given unit, and False when the mapping is approximate
and the user must adapt it. restartwell does not execute these configs.
"""

from __future__ import annotations

from typing import Literal

Framework = Literal["langgraph", "crewai", "swe-agent", "opencode"]
Unit = Literal["tokens", "seconds"]


def emit_config(tau_star: float, *, framework: Framework, unit: Unit) -> tuple[str, bool]:
    """Return ``(snippet, faithful)`` wiring ``tau_star`` into a framework's cutoff knob."""
    cutoff = round(tau_star) if unit == "tokens" else round(float(tau_star), 3)
    if framework == "swe-agent":
        key = "per_instance_cost_limit" if unit == "tokens" else "per_instance_call_timeout"
        snippet = (
            "# sweagent config (YAML) — restart each instance once this per-attempt budget is hit\n"
            "agent:\n"
            f"  {key}: {cutoff}\n"
        )
        return snippet, unit in ("tokens", "seconds")
    if framework == "langgraph":
        if unit == "seconds":
            snippet = (
                "# LangGraph — cap each attempt's wall-clock, then restart the run\n"
                "# your compiled graph runs each attempt; cap its wall-clock then restart\n"
                f"ATTEMPT_TIMEOUT_S = {cutoff}\n"
                "# enforce ATTEMPT_TIMEOUT_S in your node/runner and restart on TimeoutError\n"
            )
            return snippet, False
        snippet = (
            "# LangGraph — cap each attempt's token budget, then restart the run\n"
            f"ATTEMPT_TOKEN_CUTOFF = {cutoff}\n"
            "# track per-attempt tokens; raise/restart once ATTEMPT_TOKEN_CUTOFF is exceeded\n"
        )
        return snippet, False
    if framework == "crewai":
        if unit == "seconds":
            snippet = (
                "# CrewAI — bound each attempt, then restart\n"
                f"crew = Crew(agents=[...], tasks=[...], max_execution_time={cutoff})  # seconds\n"
            )
            return snippet, True
        snippet = (
            "# CrewAI — token cutoff per attempt (enforce in a callback), then restart\n"
            f"ATTEMPT_TOKEN_CUTOFF = {cutoff}\n"
        )
        return snippet, False
    if framework == "opencode":
        snippet = (
            "# opencode — restart the session once this per-attempt budget is reached\n"
            f"# unit={unit}\n"
            f"attempt_cutoff = {cutoff}\n"
        )
        return snippet, False
    raise ValueError(f"unknown framework {framework!r}")
