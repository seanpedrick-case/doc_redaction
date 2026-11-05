"""Deterministic task text from main-app redaction UI settings (gold auto-match)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from gold_match import normalize_instructions_text

DERIVED_INSTRUCTIONS_VERSION = 1


def _pii_method_includes_llm_entities(pii_method: str) -> bool:
    """LLM entity checkboxes apply only when PII is not plain Local (matches SHOW_EXAMPLES)."""
    key = (pii_method or "").strip().casefold()
    if not key or key == "local":
        return False
    if "llm" in key:
        return True
    if "inference server" in key or "transformers" in key:
        return True
    if "bedrock" in key and "vlm" not in key:
        return True
    return False


def _quote_term(term: str) -> str:
    text = (term or "").strip()
    if not text:
        return "''"
    if "'" in text:
        return f'"{text}"'
    return f"'{text}'"


def _clean_terms(items: Sequence[str] | None) -> list[str]:
    if not items:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _merge_entities(
    *lists: Sequence[str] | None,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for lst in lists:
        for raw in lst or []:
            ent = str(raw).strip()
            if not ent:
                continue
            key = ent.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(ent)
    return sorted(merged, key=lambda x: x.casefold())


@dataclass
class AppEvalRedactionSettings:
    chosen_redact_entities: list[str] = field(default_factory=list)
    chosen_redact_comprehend_entities: list[str] = field(default_factory=list)
    chosen_llm_entities: list[str] = field(default_factory=list)
    deny_list: list[str] = field(default_factory=list)
    allow_list: list[str] = field(default_factory=list)
    text_extraction_method: str = ""
    pii_identification_method: str = ""
    page_min: int = 0
    page_max: int = 0
    text_extraction_only: bool = False
    handwrite_signature_checkbox: list[str] = field(default_factory=list)
    custom_llm_instructions: str = ""

    def merged_entities(self) -> list[str]:
        """Entity labels that affect derived instructions / gold strict match."""
        parts: list[Sequence[str] | None] = [
            self.chosen_redact_entities,
            self.chosen_redact_comprehend_entities,
        ]
        if _pii_method_includes_llm_entities(self.pii_identification_method):
            parts.append(self.chosen_llm_entities)
        return _merge_entities(*parts)

    def to_settings_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["derived_instructions_version"] = DERIVED_INSTRUCTIONS_VERSION
        data["merged_entities"] = self.merged_entities()
        return data


def derive_app_eval_instructions(settings: AppEvalRedactionSettings) -> str:
    """
    Build canonical bullet text for gold ``instructions.txt`` matching (v1).

    Only bullet lines are included (no OCR/PII footer) so strict hash match stays stable.
    """
    lines: list[str] = []
    entities = settings.merged_entities()

    page_min = int(settings.page_min or 0)
    page_max = int(settings.page_max or 0)
    if page_max > 0:
        if page_min == page_max:
            lines.append(f"- Process page {page_min} only")
        else:
            lines.append(f"- Process pages {page_min} through {page_max}")

    if settings.text_extraction_only:
        lines.append("- Extract text only (no PII redaction pass)")
    elif entities:
        joined = ", ".join(entities)
        lines.append(f"- Redact any instance of {joined}")

    for term in _clean_terms(settings.deny_list):
        lines.append(f"- Redact any instances of the term {_quote_term(term)}")

    allow = _clean_terms(settings.allow_list)
    if allow:
        quoted = ", ".join(_quote_term(t) for t in allow)
        lines.append(f"- Do not redact instances of {quoted}")

    sig = _clean_terms(settings.handwrite_signature_checkbox)
    if sig and any("signature" in s.casefold() for s in sig):
        lines.append(
            "- When using AWS Textract, include signature extraction in handwriting options"
        )

    custom = (settings.custom_llm_instructions or "").strip()
    if custom:
        for ln in normalize_instructions_text(custom).split("\n"):
            stripped = ln.strip()
            if not stripped:
                continue
            if stripped.startswith("-"):
                lines.append(stripped)
            else:
                lines.append(f"- {stripped}")

    return normalize_instructions_text("\n".join(lines))


def settings_from_legacy_redactor_kwargs(
    flat: dict[str, Any],
) -> AppEvalRedactionSettings:
    """Build settings from ``choose_and_run_redactor`` / ``RedactionOptions`` kwargs."""
    sig = flat.get("handwrite_signature_checkbox")
    if not isinstance(sig, list):
        sig = list(sig) if sig else []

    def _list(key: str) -> list[str]:
        val = flat.get(key)
        if val is None:
            return []
        if isinstance(val, list):
            return [str(x) for x in val]
        return [str(val)]

    return AppEvalRedactionSettings(
        chosen_redact_entities=_list("chosen_redact_entities"),
        chosen_redact_comprehend_entities=_list("chosen_redact_comprehend_entities"),
        chosen_llm_entities=_list("chosen_llm_entities"),
        deny_list=_list("in_deny_list"),
        allow_list=_list("in_allow_list"),
        text_extraction_method=str(flat.get("text_extraction_method") or "").strip(),
        pii_identification_method=str(
            flat.get("pii_identification_method") or ""
        ).strip(),
        page_min=int(flat.get("page_min") or 0),
        page_max=int(flat.get("page_max") or 0),
        text_extraction_only=bool(flat.get("text_extraction_only")),
        handwrite_signature_checkbox=[str(x) for x in sig],
        custom_llm_instructions=str(flat.get("custom_llm_instructions") or "").strip(),
    )


def synthetic_doc_redact_tool_args(
    settings: AppEvalRedactionSettings,
) -> dict[str, Any]:
    """Observed tool args shape for policy extraction compare (no LLM)."""
    entities = settings.merged_entities()
    deny = _clean_terms(settings.deny_list)
    return {
        "tool_calls": [
            {
                "tool": "doc_redact",
                "args": {
                    "redact_entities": entities,
                    "deny_list": deny,
                    "ocr_method": settings.text_extraction_method,
                    "pii_method": settings.pii_identification_method,
                },
            }
        ]
    }


def settings_from_json(path: str) -> AppEvalRedactionSettings:
    data = json.loads(open(path, encoding="utf-8").read())
    if not isinstance(data, dict):
        raise ValueError("Settings JSON must be an object")
    return AppEvalRedactionSettings(
        chosen_redact_entities=list(data.get("chosen_redact_entities") or []),
        chosen_redact_comprehend_entities=list(
            data.get("chosen_redact_comprehend_entities") or []
        ),
        chosen_llm_entities=list(data.get("chosen_llm_entities") or []),
        deny_list=list(data.get("deny_list") or data.get("in_deny_list") or []),
        allow_list=list(data.get("allow_list") or data.get("in_allow_list") or []),
        text_extraction_method=str(data.get("text_extraction_method") or ""),
        pii_identification_method=str(data.get("pii_identification_method") or ""),
        page_min=int(data.get("page_min") or 0),
        page_max=int(data.get("page_max") or 0),
        text_extraction_only=bool(data.get("text_extraction_only")),
        handwrite_signature_checkbox=list(
            data.get("handwrite_signature_checkbox") or []
        ),
        custom_llm_instructions=str(data.get("custom_llm_instructions") or ""),
    )
