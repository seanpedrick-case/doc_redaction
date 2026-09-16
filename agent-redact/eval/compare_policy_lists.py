#!/usr/bin/env python3
"""Compare gold vs agent policy phrase lists (must_redact / must_not_redact / deny_list).

Scores **policy extraction**: did the agent derive the right phrases from free-text
instructions? Separate from box-coverage ([`run_policy_gate.py`](run_policy_gate.py)).

Observed lists usually come from recorded tool args::

    {
      "tool_calls": [
        {"tool": "doc_redact", "args": {"deny_list": ["Alice"]}},
        {"tool": "verify_coverage", "args": {"must_redact": ["Alice"], "must_not_redact": ["Council"]}}
      ]
    }

Example::

    python agent-redact/eval/compare_policy_lists.py \\
      --fixture agent-redact/eval/fixtures/example_smoke \\
      --observed agent-redact/eval/fixtures/example_smoke/tool_args.json \\
      --min-must-redact-recall 0.9
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


def normalize_phrase(raw: str) -> str:
    """Normalize a policy phrase for set comparison."""
    text = (raw or "").strip()
    if not text:
        return ""
    # Drop surrounding quotes; collapse whitespace; casefold.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    # Unescape common trivial regex spaces so Alice\s+Example ≈ alice example
    text = text.replace(r"\s+", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def load_phrase_file(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Pipe-separated single string (tool parity).
        if "|" in text and "\n" not in text:
            return [p.strip() for p in text.split("|") if p.strip()]
        return [text]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_as_str_list(item))
        return out
    return [str(value)]


def load_observed_tool_args(path: Path) -> dict[str, Any]:
    """Load a tool-args JSON recording (dict with tool_calls or top-level lists)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Observed tool args must be a JSON object: {path}")
    return data


def extract_lists_from_tool_calls(
    payload: dict[str, Any],
    *,
    aggregate: str = "union",
) -> dict[str, list[str]]:
    """
    Pull must_redact / must_not_redact / deny_list from recorded tool calls.

    ``aggregate``:
      - ``union`` — merge all matching tool calls (default)
      - ``last`` — last verify_coverage (must_*) / last doc_redact (deny_list)
    """
    calls = payload.get("tool_calls")
    if calls is None and any(
        k in payload for k in ("must_redact", "must_not_redact", "deny_list")
    ):
        return {
            "must_redact": _as_str_list(payload.get("must_redact")),
            "must_not_redact": _as_str_list(payload.get("must_not_redact")),
            "deny_list": _as_str_list(payload.get("deny_list")),
        }
    if not isinstance(calls, list):
        return {"must_redact": [], "must_not_redact": [], "deny_list": []}

    must_redact: list[str] = []
    must_not: list[str] = []
    deny: list[str] = []
    last_must: list[str] = []
    last_must_not: list[str] = []
    last_deny: list[str] = []

    for call in calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool") or call.get("name") or "").strip().lower()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name in {"verify_coverage", "run_verify_coverage"}:
            mr = _as_str_list(args.get("must_redact"))
            mn = _as_str_list(args.get("must_not_redact"))
            must_redact.extend(mr)
            must_not.extend(mn)
            last_must = mr
            last_must_not = mn
        elif name in {"doc_redact", "run_doc_redact"}:
            d = _as_str_list(args.get("deny_list") or args.get("deny"))
            deny.extend(d)
            last_deny = d

    if aggregate == "last":
        return {
            "must_redact": last_must,
            "must_not_redact": last_must_not,
            "deny_list": last_deny,
        }
    # union + stable unique by normalized form (keep first surface form)
    return {
        "must_redact": _unique_preserve(must_redact),
        "must_not_redact": _unique_preserve(must_not),
        "deny_list": _unique_preserve(deny),
    }


def _unique_preserve(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        key = normalize_phrase(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
    return out


@dataclass
class ListCompareReport:
    """Set precision / recall / F1 for one policy list type."""

    list_name: str
    gold_count: int = 0
    observed_count: int = 0
    matched: int = 0
    missing_gold: list[str] = field(default_factory=list)
    extra_observed: list[str] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    soft_match: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionReport:
    must_redact: ListCompareReport
    must_not_redact: ListCompareReport
    deny_list: ListCompareReport | None = None
    guardrail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "must_redact": self.must_redact.to_dict(),
            "must_not_redact": self.must_not_redact.to_dict(),
            "deny_list": self.deny_list.to_dict() if self.deny_list else None,
            "guardrail": self.guardrail,
        }


def _phrases_match(gold: str, observed: str, *, soft: bool) -> bool:
    g = normalize_phrase(gold)
    o = normalize_phrase(observed)
    if not g or not o:
        return False
    if g == o:
        return True
    if soft and (g in o or o in g):
        return True
    return False


def compare_phrase_lists(
    gold: list[str],
    observed: list[str],
    *,
    list_name: str,
    soft_match: bool = False,
) -> ListCompareReport:
    """Greedy one-to-one match of gold phrases to observed phrases."""
    gold_items = list(gold)
    obs_items = list(observed)
    matched = 0
    used_obs: set[int] = set()
    missing: list[str] = []

    for g in gold_items:
        found = False
        for oi, o in enumerate(obs_items):
            if oi in used_obs:
                continue
            if _phrases_match(g, o, soft=soft_match):
                used_obs.add(oi)
                matched += 1
                found = True
                break
        if not found:
            missing.append(g)

    extras = [o for oi, o in enumerate(obs_items) if oi not in used_obs]
    g_n = len(gold_items)
    o_n = len(obs_items)
    recall = (matched / g_n) if g_n else 1.0
    precision = (matched / o_n) if o_n else (1.0 if g_n == 0 else 0.0)
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return ListCompareReport(
        list_name=list_name,
        gold_count=g_n,
        observed_count=o_n,
        matched=matched,
        missing_gold=missing,
        extra_observed=extras,
        precision=precision,
        recall=recall,
        f1=f1,
        soft_match=soft_match,
    )


def policy_args_guardrail(
    observed: dict[str, list[str]],
    *,
    expect_must_redact: bool = True,
    expect_deny_list: bool = False,
    expect_must_not_redact: bool = False,
) -> dict[str, Any]:
    """
    Completeness checks: non-empty policy tool args when expected.

    Does not prove phrases are correct — only that extraction was attempted.
    """
    must = observed.get("must_redact") or []
    must_not = observed.get("must_not_redact") or []
    deny = observed.get("deny_list") or []
    issues: list[str] = []
    if expect_must_redact and not must:
        issues.append("verify_coverage must_redact is empty")
    if expect_must_not_redact and not must_not:
        issues.append("verify_coverage must_not_redact is empty")
    if expect_deny_list and not deny:
        issues.append("doc_redact deny_list is empty")
    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "must_redact_count": len(must),
        "must_not_redact_count": len(must_not),
        "deny_list_count": len(deny),
    }


def compare_policy_extraction(
    *,
    gold_must_redact: list[str],
    gold_must_not_redact: list[str],
    observed_must_redact: list[str],
    observed_must_not_redact: list[str],
    gold_deny_list: list[str] | None = None,
    observed_deny_list: list[str] | None = None,
    soft_match: bool = False,
    expect_must_redact: bool = True,
    expect_deny_list: bool = False,
    expect_must_not_redact: bool = False,
) -> ExtractionReport:
    observed_bundle = {
        "must_redact": observed_must_redact,
        "must_not_redact": observed_must_not_redact,
        "deny_list": observed_deny_list or [],
    }
    deny_report = None
    if gold_deny_list is not None:
        deny_report = compare_phrase_lists(
            gold_deny_list,
            observed_deny_list or [],
            list_name="deny_list",
            soft_match=soft_match,
        )
    return ExtractionReport(
        must_redact=compare_phrase_lists(
            gold_must_redact,
            observed_must_redact,
            list_name="must_redact",
            soft_match=soft_match,
        ),
        must_not_redact=compare_phrase_lists(
            gold_must_not_redact,
            observed_must_not_redact,
            list_name="must_not_redact",
            soft_match=soft_match,
        ),
        deny_list=deny_report,
        guardrail=policy_args_guardrail(
            observed_bundle,
            expect_must_redact=expect_must_redact,
            expect_deny_list=expect_deny_list,
            expect_must_not_redact=expect_must_not_redact,
        ),
    )


def _discover_fixture(fixture_dir: Path) -> dict[str, Path | None]:
    root = fixture_dir.resolve()

    def first(*names: str) -> Path | None:
        for name in names:
            p = root / name
            if p.is_file():
                return p
        return None

    return {
        "must_redact": first("must_redact.txt"),
        "must_not_redact": first("must_not_redact.txt"),
        "deny_list": first("expected_deny_list.txt", "deny_list.txt"),
        "tool_args": first("tool_args.json", "observed_tool_args.json"),
        "instructions": first("instructions.txt"),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare gold vs observed policy phrase lists (extraction F1)."
    )
    p.add_argument("--fixture", type=Path, help="Fixture directory")
    p.add_argument("--gold-must-redact", type=Path)
    p.add_argument("--gold-must-not-redact", type=Path)
    p.add_argument("--gold-deny-list", type=Path)
    p.add_argument(
        "--observed",
        type=Path,
        help="JSON with tool_calls or top-level must_redact/must_not_redact/deny_list",
    )
    p.add_argument(
        "--aggregate",
        choices=("union", "last"),
        default="union",
        help="How to combine multiple tool calls (default: union)",
    )
    p.add_argument(
        "--soft-match",
        action="store_true",
        help="Allow substring matches between gold and observed phrases",
    )
    p.add_argument("--min-must-redact-f1", type=float, default=None)
    p.add_argument("--min-must-redact-recall", type=float, default=None)
    p.add_argument("--min-must-not-redact-f1", type=float, default=None)
    p.add_argument("--min-must-not-redact-recall", type=float, default=None)
    p.add_argument("--min-deny-list-f1", type=float, default=None)
    p.add_argument(
        "--require-guardrail",
        action="store_true",
        help="Fail if completeness guardrail fails (empty must_redact when expected)",
    )
    p.add_argument(
        "--expect-deny-list",
        action="store_true",
        help="Guardrail also requires non-empty doc_redact deny_list",
    )
    p.add_argument(
        "--expect-must-not-redact",
        action="store_true",
        help="Guardrail requires non-empty must_not_redact",
    )
    p.add_argument(
        "--no-expect-must-redact",
        action="store_true",
        help="Do not require non-empty must_redact in guardrail",
    )
    p.add_argument("--json-out", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    gold_must_path = args.gold_must_redact
    gold_not_path = args.gold_must_not_redact
    gold_deny_path = args.gold_deny_list
    observed_path = args.observed

    if args.fixture:
        disc = _discover_fixture(args.fixture)
        gold_must_path = gold_must_path or disc["must_redact"]
        gold_not_path = gold_not_path or disc["must_not_redact"]
        gold_deny_path = gold_deny_path or disc["deny_list"]
        observed_path = observed_path or disc["tool_args"]

    if observed_path is None or not Path(observed_path).is_file():
        print(
            "Observed tool args JSON required (--observed or fixture tool_args.json)",
            file=sys.stderr,
        )
        return 2

    payload = load_observed_tool_args(Path(observed_path))
    observed = extract_lists_from_tool_calls(payload, aggregate=args.aggregate)
    gold_must = load_phrase_file(gold_must_path)
    gold_not = load_phrase_file(gold_not_path)
    gold_deny = load_phrase_file(gold_deny_path) if gold_deny_path else None

    report = compare_policy_extraction(
        gold_must_redact=gold_must,
        gold_must_not_redact=gold_not,
        observed_must_redact=observed["must_redact"],
        observed_must_not_redact=observed["must_not_redact"],
        gold_deny_list=gold_deny,
        observed_deny_list=observed["deny_list"],
        soft_match=args.soft_match,
        expect_must_redact=not args.no_expect_must_redact,
        expect_deny_list=args.expect_deny_list,
        expect_must_not_redact=args.expect_must_not_redact,
    )

    summary = {
        "must_redact_f1": round(report.must_redact.f1, 4),
        "must_redact_recall": round(report.must_redact.recall, 4),
        "must_not_redact_f1": round(report.must_not_redact.f1, 4),
        "must_not_redact_recall": round(report.must_not_redact.recall, 4),
        "deny_list_f1": (round(report.deny_list.f1, 4) if report.deny_list else None),
        "guardrail_pass": report.guardrail.get("pass"),
        "guardrail_issues": report.guardrail.get("issues"),
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

    failed = False
    checks = [
        (args.min_must_redact_f1, report.must_redact.f1, "must_redact F1"),
        (args.min_must_redact_recall, report.must_redact.recall, "must_redact recall"),
        (args.min_must_not_redact_f1, report.must_not_redact.f1, "must_not_redact F1"),
        (
            args.min_must_not_redact_recall,
            report.must_not_redact.recall,
            "must_not_redact recall",
        ),
    ]
    if report.deny_list is not None:
        checks.append(
            (args.min_deny_list_f1, report.deny_list.f1, "deny_list F1"),
        )
    for threshold, value, label in checks:
        if threshold is not None and value < threshold:
            print(f"{label} {value:.4f} below {threshold}", file=sys.stderr)
            failed = True
    if args.require_guardrail and not report.guardrail.get("pass"):
        print(
            f"guardrail failed: {report.guardrail.get('issues')}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
