from __future__ import annotations

import json

import pytest

from restartwell.intake import from_jsonl, normalize_label, parse_records


def test_normalize_label_synonyms():
    assert normalize_label("solved") == "success"
    assert normalize_label("PASS") == "success"
    assert normalize_label("timeout") == "timeout"
    assert normalize_label("budget_exhausted") == "timeout"
    assert normalize_label("error") == "fail"
    assert normalize_label("wrong_patch") == "fail"


def test_normalize_label_unknown_is_fail_not_success():
    assert normalize_label("???unknown???") == "fail"


def test_parse_records_basic():
    rows = [
        {"cost": 100, "label": "success"},
        {"cost": 200, "label": "timeout"},
        {"cost": 50, "label": "error"},
    ]
    recs = parse_records(rows)
    assert [r.label for r in recs] == ["success", "timeout", "fail"]
    assert recs[0].cost == 100.0


def test_parse_records_with_cohort_and_id():
    rows = [{"c": 1.0, "out": "ok", "model": "gpt", "rid": "x1"}]
    recs = parse_records(rows, cost_key="c", label_key="out", cohort_key="model", id_key="rid")
    assert recs[0].cohort == "gpt"
    assert recs[0].attempt_id == "x1"


def test_parse_records_missing_key():
    with pytest.raises(KeyError):
        parse_records([{"label": "success"}])


def test_from_jsonl(tmp_path):
    p = tmp_path / "logs.jsonl"
    rows = [{"cost": 1.0, "label": "success"}, {"cost": 2.0, "label": "fail"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    recs = from_jsonl(str(p))
    assert len(recs) == 2
    assert recs[0].label == "success"
