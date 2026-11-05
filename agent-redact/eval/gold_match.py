#!/usr/bin/env python3
"""Match a Gradio redaction task to exactly one gold case (strict / soft tiers)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent_eval_env import gold_match_mode
from run_agent_regression import resolve_gold_case_pdf

MatchLevel = Literal["strict", "soft", "none"]


def normalize_instructions_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def collapse_instruction_whitespace(raw: str) -> str:
    return re.sub(r"\s+", " ", normalize_instructions_text(raw)).strip().casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(case_dir: Path) -> dict[str, Any]:
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _case_id(case_dir: Path, manifest: dict[str, Any]) -> str:
    cid = str(manifest.get("case_id") or "").strip()
    return cid or case_dir.name


def _pdf_basename(case_dir: Path, manifest: dict[str, Any]) -> str:
    rel = str(manifest.get("input_pdf") or "input.pdf").strip()
    return Path(rel).name.casefold()


def _instruction_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        ln.strip()
        for ln in normalize_instructions_text(path.read_text(encoding="utf-8")).split(
            "\n"
        )
        if ln.strip()
    ]


def pdf_matches_strict(
    upload_pdf: Path,
    case_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    upload_hash = sha256_file(upload_pdf)
    pinned = str(manifest.get("input_pdf_sha256") or "").strip().lower()
    if pinned:
        return upload_hash.lower() == pinned
    try:
        gold_pdf = resolve_gold_case_pdf(case_dir)
    except FileNotFoundError:
        return False
    return sha256_file(gold_pdf) == upload_hash


def pdf_matches_soft(
    document_name: str,
    case_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    upload_base = Path(document_name or "").name.casefold()
    if not upload_base:
        return False
    return upload_base == _pdf_basename(case_dir, manifest)


def instructions_match_strict(
    case_dir: Path, task_instructions: str, manifest: dict[str, Any]
) -> bool:
    pinned = str(manifest.get("instructions_sha256") or "").strip().lower()
    if pinned:
        user_bytes = normalize_instructions_text(task_instructions).encode("utf-8")
        return hashlib.sha256(user_bytes).hexdigest().lower() == pinned
    gold_path = case_dir / "instructions.txt"
    if not gold_path.is_file():
        return False
    gold = normalize_instructions_text(gold_path.read_text(encoding="utf-8"))
    user = normalize_instructions_text(task_instructions)
    return gold == user


def instructions_match_soft(case_dir: Path, task_instructions: str) -> bool:
    gold_path = case_dir / "instructions.txt"
    if not gold_path.is_file():
        return False
    user_norm = normalize_instructions_text(task_instructions)
    user_fold = user_norm.casefold()
    if collapse_instruction_whitespace(
        gold_path.read_text(encoding="utf-8")
    ) == collapse_instruction_whitespace(task_instructions):
        return True
    for line in _instruction_lines(gold_path):
        if line.casefold() not in user_fold:
            return False
    return bool(_instruction_lines(gold_path))


def iter_gold_case_dirs(gold_root: Path) -> list[Path]:
    root = gold_root.resolve()
    if not root.is_dir():
        return []
    cases: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "instructions.txt").is_file():
            continue
        try:
            resolve_gold_case_pdf(child)
        except FileNotFoundError:
            continue
        cases.append(child)
    return cases


@dataclass
class GoldMatchResult:
    gold_dir: Path | None = None
    match_level: MatchLevel = "none"
    case_id: str | None = None
    reason: str = ""
    candidates_strict: list[str] = field(default_factory=list)
    candidates_soft: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.gold_dir is not None:
            data["gold_dir"] = str(self.gold_dir)
        else:
            data["gold_dir"] = None
        return data


def _match_tier(
    *,
    cases: list[Path],
    upload_pdf: Path,
    document_name: str,
    task_instructions: str,
    strict: bool,
) -> list[tuple[Path, str]]:
    matched: list[tuple[Path, str]] = []
    for case_dir in cases:
        manifest = _read_manifest(case_dir)
        cid = _case_id(case_dir, manifest)
        if strict:
            pdf_ok = pdf_matches_strict(upload_pdf, case_dir, manifest)
            inst_ok = instructions_match_strict(case_dir, task_instructions, manifest)
        else:
            pdf_ok = pdf_matches_soft(document_name, case_dir, manifest)
            inst_ok = instructions_match_soft(case_dir, task_instructions)
        if pdf_ok and inst_ok:
            matched.append((case_dir, cid))
    return matched


def resolve_gold_case_for_task(
    *,
    pdf_path: Path,
    document_name: str,
    task_instructions: str,
    gold_root: Path,
    match_mode: str | None = None,
) -> GoldMatchResult:
    mode = (match_mode or gold_match_mode()).strip().lower()
    if not pdf_path.is_file():
        return GoldMatchResult(reason="upload PDF not found for gold match")

    cases = iter_gold_case_dirs(gold_root)
    if not cases:
        return GoldMatchResult(reason=f"no gold cases under {gold_root}")

    strict_hits = _match_tier(
        cases=cases,
        upload_pdf=pdf_path,
        document_name=document_name,
        task_instructions=task_instructions,
        strict=True,
    )
    if mode == "strict_only":
        soft_hits: list[tuple[Path, str]] = []
    elif mode == "soft_only":
        strict_hits = []
        soft_hits = _match_tier(
            cases=cases,
            upload_pdf=pdf_path,
            document_name=document_name,
            task_instructions=task_instructions,
            strict=False,
        )
    else:
        soft_hits = (
            []
            if strict_hits
            else _match_tier(
                cases=cases,
                upload_pdf=pdf_path,
                document_name=document_name,
                task_instructions=task_instructions,
                strict=False,
            )
        )

    if len(strict_hits) == 1:
        d, cid = strict_hits[0]
        return GoldMatchResult(
            gold_dir=d,
            match_level="strict",
            case_id=cid,
            reason="strict PDF + instructions match",
            candidates_strict=[cid],
        )
    if len(strict_hits) > 1:
        ids = [c[1] for c in strict_hits]
        return GoldMatchResult(
            reason=f"ambiguous strict gold match: {ids}",
            candidates_strict=ids,
        )

    if len(soft_hits) == 1:
        d, cid = soft_hits[0]
        return GoldMatchResult(
            gold_dir=d,
            match_level="soft",
            case_id=cid,
            reason="soft PDF + instructions match",
            candidates_soft=[cid],
        )
    if len(soft_hits) > 1:
        ids = [c[1] for c in soft_hits]
        return GoldMatchResult(
            reason=f"ambiguous soft gold match: {ids}",
            candidates_soft=ids,
        )

    return GoldMatchResult(
        reason="no gold case matched PDF and instructions",
        candidates_strict=[c[1] for c in strict_hits],
        candidates_soft=[c[1] for c in soft_hits],
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Match upload + instructions to a gold case."
    )
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--document-name", default="")
    p.add_argument("--instructions", type=str, default="")
    p.add_argument("--instructions-file", type=Path, default=None)
    p.add_argument("--gold-root", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    from agent_eval_env import resolve_gold_root

    root = args.gold_root or resolve_gold_root(repo_root)
    instructions = args.instructions
    if args.instructions_file and args.instructions_file.is_file():
        instructions = args.instructions_file.read_text(encoding="utf-8")
    doc_name = args.document_name or args.pdf.name
    result = resolve_gold_case_for_task(
        pdf_path=args.pdf,
        document_name=doc_name,
        task_instructions=instructions,
        gold_root=root,
    )
    payload = result.to_dict()
    text = json.dumps(payload, indent=2)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if result.gold_dir is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
