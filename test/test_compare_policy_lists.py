"""Tests for agent-redact/eval/compare_policy_lists.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from compare_policy_lists import (  # noqa: E402
    compare_phrase_lists,
    compare_policy_extraction,
    extract_lists_from_tool_calls,
    normalize_phrase,
    policy_args_guardrail,
)
from compare_policy_lists import (
    main as compare_main,
)

_FIXTURE = _EVAL / "fixtures" / "example_smoke"


def test_normalize_phrase():
    assert normalize_phrase(r"Alice\s+Example") == "alice example"
    assert normalize_phrase('  "Lambeth"  ') == "lambeth"


def test_compare_phrase_lists_exact():
    report = compare_phrase_lists(
        ["Alice", "Bob"],
        ["alice", "Bob"],
        list_name="must_redact",
    )
    assert report.f1 == 1.0
    assert report.missing_gold == []


def test_compare_phrase_lists_soft_substring():
    hard = compare_phrase_lists(
        ["Lambeth"],
        ["Lambeth Council"],
        list_name="must_redact",
        soft_match=False,
    )
    assert hard.recall == 0.0
    soft = compare_phrase_lists(
        ["Lambeth"],
        ["Lambeth Council"],
        list_name="must_redact",
        soft_match=True,
    )
    assert soft.recall == 1.0


def test_extract_lists_from_tool_calls_union():
    payload = {
        "tool_calls": [
            {"tool": "doc_redact", "args": {"deny_list": ["Alice"]}},
            {
                "tool": "verify_coverage",
                "args": {"must_redact": ["Alice"], "must_not_redact": ["Council"]},
            },
            {
                "tool": "verify_coverage",
                "args": {"must_redact": ["Bob"], "must_not_redact": []},
            },
        ]
    }
    lists = extract_lists_from_tool_calls(payload, aggregate="union")
    assert set(normalize_phrase(x) for x in lists["must_redact"]) == {"alice", "bob"}
    assert lists["deny_list"] == ["Alice"]
    last = extract_lists_from_tool_calls(payload, aggregate="last")
    assert last["must_redact"] == ["Bob"]


def test_guardrail_empty_must_redact():
    g = policy_args_guardrail(
        {"must_redact": [], "must_not_redact": [], "deny_list": []},
        expect_must_redact=True,
    )
    assert g["pass"] is False
    assert g["issues"]


def test_fixture_extraction_cli():
    code = compare_main(
        [
            "--fixture",
            str(_FIXTURE),
            "--min-must-redact-recall",
            "0.9",
            "--min-must-not-redact-recall",
            "0.9",
            "--require-guardrail",
            "--expect-must-not-redact",
            "--expect-deny-list",
            "--min-deny-list-f1",
            "0.9",
        ]
    )
    assert code == 0


def test_compare_policy_extraction_report():
    report = compare_policy_extraction(
        gold_must_redact=["Alice"],
        gold_must_not_redact=["Example Council"],
        observed_must_redact=["Alice"],
        observed_must_not_redact=["Example Council"],
        gold_deny_list=["Alice"],
        observed_deny_list=["Alice"],
        expect_must_redact=True,
        expect_deny_list=True,
    )
    assert report.must_redact.f1 == 1.0
    assert report.guardrail["pass"] is True
    assert "must_redact" in report.to_dict()


def test_cli_fails_on_missing_extraction(tmp_path: Path):
    observed = tmp_path / "empty.json"
    observed.write_text(
        json.dumps({"tool_calls": [{"tool": "verify_coverage", "args": {}}]}),
        encoding="utf-8",
    )
    gold = tmp_path / "must.txt"
    gold.write_text("Alice\n", encoding="utf-8")
    code = compare_main(
        [
            "--gold-must-redact",
            str(gold),
            "--observed",
            str(observed),
            "--min-must-redact-recall",
            "0.9",
            "--require-guardrail",
        ]
    )
    assert code == 1
