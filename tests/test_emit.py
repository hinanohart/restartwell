from __future__ import annotations

import pytest

from restartwell.emit import emit_config


@pytest.mark.parametrize("framework", ["langgraph", "crewai", "swe-agent", "opencode"])
@pytest.mark.parametrize("unit", ["tokens", "seconds"])
def test_emit_config_returns_snippet_and_flag(framework, unit):
    snippet, faithful = emit_config(1234.5, framework=framework, unit=unit)
    assert isinstance(snippet, str)
    assert isinstance(faithful, bool)
    assert len(snippet) > 0


def test_emit_tokens_rounds_int():
    snippet, _ = emit_config(1234.7, framework="swe-agent", unit="tokens")
    assert "1235" in snippet


def test_emit_unknown_framework():
    with pytest.raises(ValueError):
        emit_config(100.0, framework="nope", unit="tokens")  # type: ignore[arg-type]
