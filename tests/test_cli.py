from __future__ import annotations

import json

from typer.testing import CliRunner

from restartwell.bench.datasets import make_dfr
from restartwell.cli import app

runner = CliRunner()


def _write_logs(path, attempts):
    rows = [{"cost": a.cost, "label": a.label} for a in attempts]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_cli_verdict(tmp_path):
    p = tmp_path / "logs.jsonl"
    _write_logs(p, make_dfr(n=400, seed=1))
    res = runner.invoke(app, ["verdict", "-i", str(p), "--n-boot", "100"])
    assert res.exit_code == 0
    assert "decision:" in res.stdout


def test_cli_cutoff(tmp_path):
    p = tmp_path / "logs.jsonl"
    _write_logs(p, make_dfr(n=400, seed=1))
    res = runner.invoke(app, ["cutoff", "-i", str(p), "--r", "400", "--n-boot", "80"])
    assert res.exit_code == 0
    assert "tau_star:" in res.stdout


def test_cli_report_json(tmp_path):
    p = tmp_path / "logs.jsonl"
    _write_logs(p, make_dfr(n=400, seed=1))
    res = runner.invoke(app, ["report", "-i", str(p), "--r", "400", "--json", "--n-boot", "80"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data["verdict"]["decision"] == "restart_helps"


def test_cli_emit(tmp_path):
    p = tmp_path / "logs.jsonl"
    _write_logs(p, make_dfr(n=400, seed=1))
    res = runner.invoke(
        app, ["emit", "-i", str(p), "--r", "400", "--framework", "swe-agent", "--unit", "tokens"]
    )
    assert res.exit_code == 0


def test_cli_emit_unknown_framework(tmp_path):
    p = tmp_path / "logs.jsonl"
    _write_logs(p, make_dfr(n=100, seed=1))
    res = runner.invoke(app, ["emit", "-i", str(p), "--r", "400", "--framework", "nope"])
    assert res.exit_code == 2
