"""Discover gold cases eligible for the standardized benchmark suite."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gold_match import iter_gold_case_dirs
from run_agent_regression import _ensure_agent_imports, resolve_gold_case_pdf


@dataclass
class CaseReadiness:
    case_id: str
    case_dir: str
    benchmark_eligible: bool
    has_ideal_review: bool
    has_ocr_words: bool
    has_must_redact: bool
    has_app_eval_settings: bool
    agent_surface_ready: bool
    harness_surface_ready: bool
    main_surface_ready: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _case_id(case_dir: Path) -> str:
    manifest = case_dir / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            cid = (data.get("case_id") or "").strip()
            if cid:
                return cid
        except (OSError, json.JSONDecodeError):
            pass
    return case_dir.name


def case_readiness(case_dir: Path) -> CaseReadiness:
    """Report artifacts present for agent vs main benchmark legs."""
    case_dir = case_dir.resolve()
    notes: list[str] = []
    has_ideal = (case_dir / "ideal_review_file.csv").is_file()
    has_ocr = (case_dir / "ocr_results_with_words.csv").is_file()
    has_must = (case_dir / "must_redact.txt").is_file()
    has_settings = (case_dir / "app_eval_settings.json").is_file()

    pdf_ok = True
    try:
        resolve_gold_case_pdf(case_dir)
    except FileNotFoundError:
        pdf_ok = False
        notes.append("missing input PDF (input.pdf or manifest input_pdf)")

    if not has_ideal:
        notes.append("missing ideal_review_file.csv (required for benchmark suite)")
    if not has_ocr:
        notes.append("missing ocr_results_with_words.csv (OCR gold tier skipped)")
    if not has_must:
        notes.append("missing must_redact.txt (policy gate may be weak)")

    agent_ready = has_ideal and pdf_ok and (case_dir / "instructions.txt").is_file()
    if not (case_dir / "instructions.txt").is_file():
        notes.append("missing instructions.txt (agent leg)")

    main_ready = has_settings and has_ideal and pdf_ok
    if not has_settings:
        notes.append("missing app_eval_settings.json (main-app leg skipped)")

    eligible = has_ideal and pdf_ok

    return CaseReadiness(
        case_id=_case_id(case_dir),
        case_dir=str(case_dir),
        benchmark_eligible=eligible,
        has_ideal_review=has_ideal,
        has_ocr_words=has_ocr,
        has_must_redact=has_must,
        has_app_eval_settings=has_settings,
        agent_surface_ready=agent_ready,
        harness_surface_ready=agent_ready,
        main_surface_ready=main_ready,
        notes=notes,
    )


def discover_benchmark_gold_cases(gold_root: Path) -> list[Path]:
    """
    Gold directories with ``ideal_review_file.csv`` and a resolvable input PDF.

    Uses ``iter_gold_case_dirs`` then filters; cases without ideals are omitted.
    """
    gold_root = gold_root.resolve()
    out: list[Path] = []
    for case_dir in iter_gold_case_dirs(gold_root):
        ideal = case_dir / "ideal_review_file.csv"
        if not ideal.is_file():
            continue
        try:
            resolve_gold_case_pdf(case_dir)
        except FileNotFoundError:
            continue
        out.append(case_dir)
    return out


def audit_benchmark_gold_cases(gold_root: Path) -> list[CaseReadiness]:
    """Full audit of gold case folders under *gold_root* (including non-benchmark)."""
    gold_root = gold_root.resolve()
    if not gold_root.is_dir():
        return []
    reports: list[CaseReadiness] = []
    seen: set[str] = set()
    for case_dir in iter_gold_case_dirs(gold_root):
        reports.append(case_readiness(case_dir))
        seen.add(str(case_dir.resolve()))
    for case_dir in sorted(gold_root.iterdir()):
        if not case_dir.is_dir():
            continue
        key = str(case_dir.resolve())
        if key in seen:
            continue
        if (case_dir / "instructions.txt").is_file() or (
            case_dir / "manifest.json"
        ).is_file():
            reports.append(case_readiness(case_dir))
    return reports


def default_gold_root() -> Path:
    repo_root, _, eval_dir = _ensure_agent_imports()
    try:
        from app_eval_env import resolve_app_eval_gold_root

        return resolve_app_eval_gold_root(repo_root)
    except ImportError:
        from agent_eval_env import resolve_gold_root

        return resolve_gold_root(repo_root)
