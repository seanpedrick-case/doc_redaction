"""
AgentCore Harness-driven redaction inside the main doc_redaction app.

The harness (an AWS-managed agent loop, see ``AWS::BedrockAgentCore::Harness``) plans the
task and follows the redaction skills. Redaction actions are *inline function tools*: the
harness pauses with ``stopReason=tool_use``, this module runs the tool in-process against a
per-run session folder under ``OUTPUT_FOLDER``, and returns a ``toolResult`` on the same
``runtimeSessionId``. The harness never needs network access to this app.

Page review subagents cannot call inline tools, so ``stage_page_review_bundles`` publishes
per-page slices to S3 with presigned GET/PUT URLs and ``merge_page_reviews`` folds the
uploaded page CSVs back into the master review CSV.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from tools.agentcore_harness_stream import (
    HarnessTokenUsage,
    HarnessTurnCollector,
    harness_invoke_allowed_tools,
    harness_runtime_session_id,
    inline_tool_names,
    load_inline_tool_specs,
    region_from_harness_arn,
    tool_result_message,
)
from tools.config import (
    AGENTCORE_HARNESS_ARN,
    AGENTCORE_HARNESS_ENABLE_PAGE_REVIEW,
    AGENTCORE_HARNESS_ENDPOINT,
    AGENTCORE_HARNESS_MAX_READ_CHARS,
    AGENTCORE_HARNESS_MAX_TOOL_ROUNDS,
    AGENTCORE_HARNESS_PRESIGN_SECONDS,
    AGENTCORE_HARNESS_S3_STAGING_PREFIX,
    AGENTCORE_REGION,
    OUTPUT_FOLDER,
)
from tools.secure_path_utils import secure_path_join

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTNERSHIP_TEMPLATE_PATH = REPO_ROOT / "skills" / "Example prompt partnership.txt"

TEXT_SUFFIXES = {".csv", ".json", ".txt", ".md", ".log"}
MAX_WRITE_CHARS = 20_000_000
MAX_TOOL_RESULT_CHARS = 40_000
DEFAULT_PREVIEW_PAGE_LIMIT = 10
REDACT_OUTPUT_DIR = "output_redact"
REVIEW_OUTPUT_DIR = "review/output_review_final"


class HarnessToolError(Exception):
    """Raised by tool handlers; the message is returned to the harness as an error result."""


@dataclass
class HarnessRunDefaults:
    """App-side defaults suggested to the agent and merged into ``doc_redact`` calls."""

    ocr_method: str = "Local text"
    pii_method: str = "Local"
    redact_entities: list[str] = field(default_factory=list)
    allow_list: list[str] = field(default_factory=list)
    deny_list: list[str] = field(default_factory=list)
    handwrite_signature_checkbox: list[str] | None = None
    page_min: int = 0
    page_max: int = 0


@dataclass
class HarnessProgress:
    activity: str
    output_files: list[str]
    summary: str
    done: bool = False
    token_usage: HarnessTokenUsage | None = None
    elapsed_seconds: float = 0.0


def harness_feature_ready() -> tuple[bool, str]:
    """Return ``(ready, reason)`` for showing / running the agent route."""
    if not (AGENTCORE_HARNESS_ARN or "").strip():
        return False, "AGENTCORE_HARNESS_ARN is not set."
    try:
        region_from_harness_arn(AGENTCORE_HARNESS_ARN)
    except ValueError as exc:
        return False, str(exc)
    return True, ""


# ---------------------------------------------------------------------------
# Review CSV helpers
# ---------------------------------------------------------------------------

_DEFAULT_REVIEW_COLOR = "(0, 0, 0)"
_PLACEHOLDER_TOKENS = {"", "placeholder", "n/a", "na", "none", "null", "tbd", "?"}


def _normalize_review_color_cell(raw: Any) -> str:
    import ast
    import re

    s = str(raw if raw is not None else "").strip()
    if not s or s.lower() in _PLACEHOLDER_TOKENS | {"nan", "black"}:
        return _DEFAULT_REVIEW_COLOR
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (tuple, list)) and len(parsed) == 3:
            rgb = tuple(int(p) for p in parsed)
            if all(0 <= c <= 255 for c in rgb):
                return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
    except (SyntaxError, ValueError, TypeError):
        pass
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", s)
    if match:
        h = match.group(1)
        rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        return f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
    return _DEFAULT_REVIEW_COLOR


def _read_csv_rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HarnessToolError("CSV is missing a header row.")
    return list(reader.fieldnames), list(reader)


def _rows_to_csv(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(
        out, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _field_lookup(fieldnames: list[str]) -> dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames if name}


def validate_and_repair_review_csv(text: str) -> tuple[str, int]:
    """Validate bbox columns and repair colour cells. Returns ``(csv_text, repaired)``."""
    fieldnames, rows = _read_csv_rows(text)
    fields = _field_lookup(fieldnames)
    bbox_keys = [fields[k] for k in ("xmin", "xmax", "ymin", "ymax") if k in fields]
    if len(bbox_keys) == 4:
        for row_idx, row in enumerate(rows, start=2):
            for key in bbox_keys:
                raw = str(row.get(key, "") or "").strip()
                if raw.lower() in _PLACEHOLDER_TOKENS:
                    raise HarnessToolError(
                        f"Invalid review CSV at row {row_idx}: column {key!r} is {raw!r}. "
                        "Bbox values must be floats in [0, 1]; take them from the word OCR CSV."
                    )
                try:
                    val = float(raw)
                except ValueError as exc:
                    raise HarnessToolError(
                        f"Invalid review CSV at row {row_idx}: {key}={raw!r} is not a number."
                    ) from exc
                if val < -0.05 or val > 1.05:
                    raise HarnessToolError(
                        f"Invalid review CSV at row {row_idx}: {key}={val} is outside [0, 1]."
                    )
    repaired = 0
    color_key = fields.get("color")
    if color_key:
        for row in rows:
            fixed = _normalize_review_color_cell(row.get(color_key))
            if str(row.get(color_key, "")).strip() != fixed:
                repaired += 1
            row[color_key] = fixed
    return _rows_to_csv(fieldnames, rows), repaired


def _page_of(row: dict[str, Any], page_key: str | None) -> int | None:
    if not page_key:
        return None
    try:
        return int(float(str(row.get(page_key, "") or "").strip()))
    except ValueError:
        return None


def _sort_key(
    row: dict[str, Any], fields: dict[str, str]
) -> tuple[float, float, float]:
    def num(name: str) -> float:
        key = fields.get(name)
        try:
            return float(str(row.get(key, "") or 0)) if key else 0.0
        except ValueError:
            return 0.0

    return num("page"), num("ymin"), num("xmin")


# ---------------------------------------------------------------------------
# S3 staging (page-review bundles)
# ---------------------------------------------------------------------------


def _parse_s3_prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse((uri or "").strip())
    if parsed.scheme != "s3" or not parsed.netloc:
        raise HarnessToolError(f"Invalid S3 staging prefix: {uri!r}")
    prefix = (parsed.path or "").lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def _default_s3_client():
    import boto3

    return boto3.client("s3", region_name=AGENTCORE_REGION or None)


def _default_agentcore_client(region: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-agentcore",
        region_name=region,
        config=Config(connect_timeout=30, read_timeout=1800, tcp_keepalive=True),
    )


# ---------------------------------------------------------------------------
# Session folder and tool handlers
# ---------------------------------------------------------------------------


class HarnessSession:
    """Per-run session folder plus the inline tool implementations."""

    def __init__(
        self,
        source_document: str | Path,
        *,
        defaults: HarnessRunDefaults | None = None,
        output_root: str | Path = OUTPUT_FOLDER,
        run_id: str | None = None,
        staging_prefix: str | None = None,
        s3_client: Any = None,
        enable_page_review: bool = AGENTCORE_HARNESS_ENABLE_PAGE_REVIEW,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.defaults = defaults or HarnessRunDefaults()
        self.output_root = Path(output_root)
        self.session_dir = secure_path_join(self.output_root, "agentcore", self.run_id)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        src = Path(source_document)
        if not src.is_file():
            raise FileNotFoundError(f"Document not found: {source_document}")
        dest = secure_path_join(self.session_dir, "source", src.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        self.source_pdf = dest

        self.staging_prefix = (
            AGENTCORE_HARNESS_S3_STAGING_PREFIX
            if staging_prefix is None
            else staging_prefix
        ).strip()
        self._s3_client = s3_client
        self.enable_page_review = bool(enable_page_review and self.staging_prefix)
        self.staged_pages: dict[int, dict[str, Any]] = {}
        self.staged_fieldnames: list[str] = []
        self.finished = False
        self.finish_summary = ""
        self.final_files: list[str] = []
        self.review_apply_outputs: list[str] = []

        self.handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "list_session_files": self.list_session_files,
            "doc_redact": self.doc_redact,
            "read_session_text": self.read_session_text,
            "write_session_text": self.write_session_text,
            "verify_redaction_coverage": self.verify_redaction_coverage,
            "preview_redaction_boxes": self.preview_redaction_boxes,
            "stage_page_review_bundles": self.stage_page_review_bundles,
            "merge_page_reviews": self.merge_page_reviews,
            "review_apply": self.review_apply,
            "finish_task": self.finish_task,
        }

    # -- paths ------------------------------------------------------------

    def rel(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.session_dir.resolve()).as_posix()

    def resolve(self, relative_path: Any, *, must_exist: bool = True) -> Path:
        raw = str(relative_path or "").strip().replace("\\", "/")
        if not raw:
            raise HarnessToolError("A path relative to the session folder is required.")
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                raw = (
                    candidate.resolve()
                    .relative_to(self.session_dir.resolve())
                    .as_posix()
                )
            except ValueError as exc:
                raise HarnessToolError(
                    f"Path {raw!r} is outside the session folder; use relative paths."
                ) from exc
        try:
            resolved = secure_path_join(self.session_dir, raw)
        except (PermissionError, ValueError) as exc:
            raise HarnessToolError(str(exc)) from exc
        if must_exist and not resolved.is_file():
            raise HarnessToolError(
                f"File not found in session folder: {raw!r}. Call list_session_files."
            )
        return resolved

    def _pdf_arg(self, args: dict[str, Any]) -> Path:
        value = args.get("pdf_path")
        return self.resolve(value) if value else self.source_pdf

    def _files(self) -> list[str]:
        return sorted(self.rel(p) for p in self.session_dir.rglob("*") if p.is_file())

    def _find(self, *, contains: str, suffix: str = ".csv") -> list[str]:
        return [
            f
            for f in self._files()
            if contains in Path(f).name.lower() and f.lower().endswith(suffix)
        ]

    def _newest_rel(
        self,
        *,
        name_contains: str,
        suffixes: tuple[str, ...] = (".pdf", ".csv"),
        under: str | None = None,
    ) -> str | None:
        """Newest session file whose basename contains *name_contains* (case-insensitive)."""
        matches: list[tuple[float, str]] = []
        prefix = (under or "").strip().rstrip("/")
        for rel in self._files():
            if prefix and not rel.replace("\\", "/").startswith(f"{prefix}/"):
                continue
            path = Path(rel)
            if path.suffix.lower() not in suffixes:
                continue
            if name_contains.lower() not in path.name.lower():
                continue
            try:
                resolved = self.resolve(rel)
            except HarnessToolError:
                continue
            matches.append((resolved.stat().st_mtime, rel))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    def standard_ui_output_rels(self) -> list[str]:
        """Pass 1 artifacts to show in the Gradio Output files box."""
        rels: list[str] = []
        seen: set[str] = set()

        def push(rel: str | None) -> None:
            if not rel or rel in seen:
                return
            seen.add(rel)
            rels.append(rel)

        push(
            self._newest_rel(
                name_contains="_redacted",
                suffixes=(".pdf",),
                under=REVIEW_OUTPUT_DIR,
            )
        )
        push(
            self._newest_rel(
                name_contains="_redacted",
                suffixes=(".pdf",),
                under=REDACT_OUTPUT_DIR,
            )
        )
        push(
            self._newest_rel(
                name_contains="redactions_for_review",
                suffixes=(".pdf",),
            )
        )
        push(self._newest_rel(name_contains="_review_file", suffixes=(".csv",)))
        push(
            self._newest_rel(name_contains="ocr_results_with_words", suffixes=(".csv",))
        )
        return rels

    def _ocr_words_for(self, args: dict[str, Any]) -> Path:
        value = args.get("ocr_words_csv_path")
        if value:
            return self.resolve(value)
        found = self._find(contains="ocr_results_with_words")
        if not found:
            raise HarnessToolError(
                "No *_ocr_results_with_words*.csv in the session folder; run doc_redact first."
            )
        return self.resolve(found[-1])

    def _review_csv(self, args: dict[str, Any]) -> Path:
        path = self.resolve(args.get("review_csv_path"))
        if "_review_file" not in path.stem.lower():
            raise HarnessToolError(
                "review_csv_path basename must contain '_review_file'."
            )
        return path

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        handler = self.handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool {name!r}"}), False
        try:
            payload = handler(arguments or {})
            ok = True
        except HarnessToolError as exc:
            payload, ok = {"error": str(exc)}, False
        except Exception as exc:  # noqa: BLE001 - report tool failures to the agent
            payload, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
        text = json.dumps(payload, indent=2, default=str)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = (
                text[:MAX_TOOL_RESULT_CHARS]
                + f"\n... [truncated {len(text) - MAX_TOOL_RESULT_CHARS} chars]"
            )
        return text, ok

    # -- tools ------------------------------------------------------------

    def list_session_files(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_document": self.rel(self.source_pdf),
            "files": self._files()[:500],
        }

    def doc_redact(self, args: dict[str, Any]) -> dict[str, Any]:
        from tools.simplified_api import doc_redact_api

        pdf = self._pdf_arg(args)
        out_dir = secure_path_join(self.session_dir, REDACT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        d = self.defaults

        def merged(key: str, base: list[str]) -> list[str] | None:
            extra = [str(x) for x in (args.get(key) or []) if str(x).strip()]
            combined = list(dict.fromkeys([*base, *extra]))
            return combined or None

        paths, message = doc_redact_api(
            document_file=str(pdf),
            redact_entities=merged("redact_entities", d.redact_entities),
            output_dir=str(out_dir),
            ocr_method=str(args.get("ocr_method") or d.ocr_method),
            pii_method=str(args.get("pii_method") or d.pii_method),
            allow_list=merged("allow_list", d.allow_list),
            deny_list=merged("deny_list", d.deny_list),
            page_min=int(args.get("page_min") or d.page_min or 0),
            page_max=int(args.get("page_max") or d.page_max or 0),
            handwrite_signature_checkbox=(
                args.get("handwrite_signature_checkbox")
                or d.handwrite_signature_checkbox
            ),
        )
        rels = [self.rel(p) for p in paths if Path(p).is_file()]
        payload: dict[str, Any] = {"message": message, "output_files": rels}
        for rel in rels:
            name = Path(rel).name.lower()
            if "_review_file" in name and name.endswith(".csv"):
                payload["review_csv_path"] = rel
            elif "ocr_results_with_words" in name and name.endswith(".csv"):
                payload["ocr_words_csv_path"] = rel
            elif "_ocr_output_" in name and name.endswith(".csv"):
                payload["ocr_lines_csv_path"] = rel
        payload["note"] = (
            "The *_redacted.pdf from doc_redact is a draft. Review the CSV, verify, then "
            "review_apply once to produce the deliverable."
        )
        return payload

    def read_session_text(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.resolve(args.get("path"))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise HarnessToolError(
                f"Only text files can be read ({sorted(TEXT_SUFFIXES)})."
            )
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        offset = max(0, int(args.get("offset") or 0))
        limit = int(args.get("max_chars") or AGENTCORE_HARNESS_MAX_READ_CHARS)
        limit = max(1, min(limit, AGENTCORE_HARNESS_MAX_READ_CHARS))
        chunk = text[offset : offset + limit]
        end = offset + len(chunk)
        return {
            "path": self.rel(path),
            "total_chars": len(text),
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "content": chunk,
        }

    def write_session_text(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.resolve(args.get("path"), must_exist=False)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise HarnessToolError(
                f"Only text files can be written ({sorted(TEXT_SUFFIXES)})."
            )
        if path.resolve() == self.source_pdf.resolve():
            raise HarnessToolError("The source document cannot be overwritten.")
        content = args.get("content")
        if not isinstance(content, str):
            raise HarnessToolError("content must be a string.")
        if len(content) > MAX_WRITE_CHARS:
            raise HarnessToolError("content is too large.")
        repaired = 0
        if path.suffix.lower() == ".csv" and "_review_file" in path.stem.lower():
            content, repaired = validate_and_repair_review_csv(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8-sig")
        result: dict[str, Any] = {"path": self.rel(path), "chars": len(content)}
        if repaired:
            result["repaired_color_cells"] = repaired
        return result

    def verify_redaction_coverage(self, args: dict[str, Any]) -> dict[str, Any]:
        from tools.simplified_api import run_verify_redaction_coverage

        review = self._review_csv(args)
        ocr_words = self._ocr_words_for(args)
        redacted = args.get("redacted_pdf_path")
        report, pruned, prune_log = run_verify_redaction_coverage(
            str(review),
            str(ocr_words),
            must_redact=list(args.get("must_redact") or []) or None,
            must_not_redact=list(args.get("must_not_redact") or []) or None,
            redacted_pdf_path=str(self.resolve(redacted)) if redacted else None,
            auto_prune_suspicious=bool(args.get("auto_prune_suspicious")),
        )
        result: dict[str, Any] = {"report": report}
        if pruned:
            result["pruned_review_csv_path"] = self.rel(pruned)
        if prune_log:
            result["prune_log"] = prune_log
        return result

    def preview_redaction_boxes(self, args: dict[str, Any]) -> dict[str, Any]:
        from tools.preview_redaction_boxes import preview_redaction_boxes

        review = self._review_csv(args)
        pdf = self._pdf_arg(args)
        pages = [int(p) for p in (args.get("pages") or [])]
        if not pages:
            fieldnames, rows = _read_csv_rows(review.read_text(encoding="utf-8-sig"))
            page_key = _field_lookup(fieldnames).get("page")
            pages = sorted({p for r in rows if (p := _page_of(r, page_key))})
            pages = pages[:DEFAULT_PREVIEW_PAGE_LIMIT]
        out_dir = secure_path_join(self.session_dir, "previews", review.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = preview_redaction_boxes(
            str(pdf), str(review), out_dir=str(out_dir), pages=pages or None
        )
        rels = [self.rel(p) for p in out_paths]
        result: dict[str, Any] = {"preview_pngs": rels}
        if self.staging_prefix:
            urls = {}
            for rel in rels:
                key = self._staging_key(rel)
                self._upload_file(self.resolve(rel), key, "image/png")
                urls[rel] = self._presign("get_object", key)
            result["download_urls"] = urls
        return result

    def stage_page_review_bundles(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.enable_page_review:
            raise HarnessToolError(
                "Parallel page review is not available in this deployment (no S3 staging "
                "prefix or AGENTCORE_HARNESS_ENABLE_PAGE_REVIEW=False). Review pages yourself "
                "with read_session_text / write_session_text."
            )
        review = self._review_csv(args)
        review_fields, review_rows = _read_csv_rows(
            review.read_text(encoding="utf-8-sig")
        )
        review_page_key = _field_lookup(review_fields).get("page")
        word_path = self._ocr_words_for(args)
        word_fields, word_rows = _read_csv_rows(
            word_path.read_text(encoding="utf-8-sig")
        )
        word_page_key = _field_lookup(word_fields).get("page")
        line_csvs = self._find(contains="_ocr_output_")
        line_fields: list[str] = []
        line_rows: list[dict[str, str]] = []
        if line_csvs:
            line_fields, line_rows = _read_csv_rows(
                self.resolve(line_csvs[-1]).read_text(encoding="utf-8-sig")
            )
        line_page_key = _field_lookup(line_fields).get("page") if line_fields else None

        pages = [int(p) for p in (args.get("pages") or [])]
        if not pages:
            pages = sorted(
                {p for r in review_rows if (p := _page_of(r, review_page_key))}
                | {p for r in word_rows if (p := _page_of(r, word_page_key))}
            )
        if not pages:
            raise HarnessToolError("No pages found to stage.")

        self.staged_fieldnames = review_fields
        bundles = []
        for page in pages:
            files = {
                "review_rows": (
                    review_fields,
                    [r for r in review_rows if _page_of(r, review_page_key) == page],
                ),
                "ocr_words": (
                    word_fields,
                    [r for r in word_rows if _page_of(r, word_page_key) == page],
                ),
            }
            if line_fields:
                files["ocr_lines"] = (
                    line_fields,
                    [r for r in line_rows if _page_of(r, line_page_key) == page],
                )
            urls: dict[str, str] = {}
            for label, (fields, rows) in files.items():
                local = secure_path_join(
                    self.session_dir, "page_review", f"page_{page}", f"{label}.csv"
                )
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(_rows_to_csv(fields, rows), encoding="utf-8")
                key = self._staging_key(self.rel(local))
                self._upload_file(local, key, "text/csv")
                urls[f"{label}_url"] = self._presign("get_object", key)
            result_key = self._staging_key(f"page_review/page_{page}/result_review.csv")
            self.staged_pages[page] = {"result_key": result_key}
            bundles.append(
                {
                    "page": page,
                    "review_row_count": len(files["review_rows"][1]),
                    **urls,
                    "result_put_url": self._presign(
                        "put_object", result_key, content_type="text/csv"
                    ),
                }
            )
        return {
            "review_csv_columns": review_fields,
            "pages": bundles,
            "upload_instructions": (
                "Each subagent downloads its page files with curl -fsSL '<url>' -o <file>, "
                "writes the final rows for that page only (same columns, utf-8, bbox floats "
                "in [0, 1], reuse that page's 'image' value), then uploads with: "
                "curl -fsS -X PUT -H 'Content-Type: text/csv' --upload-file result_review.csv "
                "'<result_put_url>'. Then call merge_page_reviews."
            ),
            "expires_in_seconds": AGENTCORE_HARNESS_PRESIGN_SECONDS,
        }

    def merge_page_reviews(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.staged_pages:
            raise HarnessToolError(
                "No staged pages; call stage_page_review_bundles first."
            )
        review = self._review_csv(args)
        fieldnames, master_rows = _read_csv_rows(review.read_text(encoding="utf-8-sig"))
        fields = _field_lookup(fieldnames)
        page_key = fields.get("page")
        id_key = fields.get("id")
        pages = [int(p) for p in (args.get("pages") or [])] or sorted(self.staged_pages)

        replaced: dict[int, int] = {}
        missing: list[int] = []
        dropped_other_pages = 0
        new_page_rows: list[dict[str, Any]] = []
        for page in pages:
            staged = self.staged_pages.get(page)
            if staged is None:
                missing.append(page)
                continue
            body = self._download_text(staged["result_key"])
            if body is None:
                missing.append(page)
                continue
            checked, _ = validate_and_repair_review_csv(body)
            _, page_rows = _read_csv_rows(checked)
            kept = []
            for row in page_rows:
                if _page_of(row, page_key) != page:
                    dropped_other_pages += 1
                    continue
                if id_key and not str(row.get(id_key) or "").strip():
                    row[id_key] = uuid.uuid4().hex[:12]
                kept.append({k: row.get(k, "") for k in fieldnames})
            replaced[page] = len(kept)
            new_page_rows.extend(kept)

        merged = [r for r in master_rows if _page_of(r, page_key) not in replaced]
        merged.extend(new_page_rows)
        merged.sort(key=lambda r: _sort_key(r, fields))
        if id_key:
            seen: set[str] = set()
            for row in merged:
                rid = str(row.get(id_key) or "")
                if rid in seen:
                    row[id_key] = uuid.uuid4().hex[:12]
                seen.add(str(row[id_key]))

        stem = review.stem
        if stem.endswith("_merged"):
            stem = stem[: -len("_merged")]
        out = secure_path_join(review.parent, f"{stem}_merged.csv")
        out.write_text(_rows_to_csv(fieldnames, merged), encoding="utf-8-sig")
        return {
            "merged_review_csv_path": self.rel(out),
            "pages_replaced": replaced,
            "pages_without_upload_kept_master_rows": missing,
            "rows_dropped_for_wrong_page": dropped_other_pages,
            "total_rows": len(merged),
        }

    def review_apply(self, args: dict[str, Any]) -> dict[str, Any]:
        from tools.simplified_api import run_apply_review_redactions

        review = self._review_csv(args)
        pdf = self._pdf_arg(args)
        out_dir = secure_path_join(self.session_dir, REVIEW_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_apply_review_redactions(
            pdf_path=str(pdf), review_csv_path=str(review), output_dir=str(out_dir)
        )
        rels = []
        for p in result.get("output_paths") or []:
            try:
                rels.append(self.rel(p))
            except ValueError:
                continue
        self.review_apply_outputs = rels
        redacted = [r for r in rels if r.lower().endswith("_redacted.pdf")]
        return {
            "message": result.get("message", ""),
            "output_files": rels,
            "redacted_pdf_path": redacted[-1] if redacted else None,
        }

    def finish_task(self, args: dict[str, Any]) -> dict[str, Any]:
        self.finish_summary = str(args.get("summary") or "").strip()
        finals: list[str] = []
        seen: set[str] = set()
        for value in args.get("final_files") or []:
            try:
                rel = self.rel(self.resolve(value))
            except HarnessToolError:
                continue
            if rel not in seen:
                seen.add(rel)
                finals.append(rel)
        for rel in self.standard_ui_output_rels():
            if rel not in seen:
                seen.add(rel)
                finals.append(rel)
        self.final_files = finals
        self.finished = True
        return {"status": "finished", "final_files": finals}

    # -- S3 ---------------------------------------------------------------

    @property
    def s3(self):
        if self._s3_client is None:
            self._s3_client = _default_s3_client()
        return self._s3_client

    def _staging_key(self, relative: str) -> str:
        _, prefix = _parse_s3_prefix(self.staging_prefix)
        return f"{prefix}{self.run_id}/{relative.lstrip('/')}"

    @property
    def _bucket(self) -> str:
        return _parse_s3_prefix(self.staging_prefix)[0]

    def _upload_file(self, path: Path, key: str, content_type: str) -> None:
        self.s3.upload_file(
            str(path), self._bucket, key, ExtraArgs={"ContentType": content_type}
        )

    def _presign(
        self, method: str, key: str, *, content_type: str | None = None
    ) -> str:
        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        return self.s3.generate_presigned_url(
            method, Params=params, ExpiresIn=AGENTCORE_HARNESS_PRESIGN_SECONDS
        )

    def _download_text(self, key: str) -> str | None:
        try:
            obj = self.s3.get_object(Bucket=self._bucket, Key=key)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - missing upload means "keep master rows"
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        return obj["Body"].read().decode("utf-8-sig")

    # -- outputs ----------------------------------------------------------

    def output_files(self) -> list[str]:
        """Absolute paths to show in the app's Output files box."""
        rels: list[str] = []
        seen: set[str] = set()
        for rel in (
            *self.final_files,
            *self.review_apply_outputs,
            *self.standard_ui_output_rels(),
        ):
            if rel in seen:
                continue
            seen.add(rel)
            rels.append(rel)
        if not rels:
            rels = [
                f
                for f in self._files()
                if f.startswith(f"{REDACT_OUTPUT_DIR}/")
                and (f.endswith(".pdf") or "_review_file" in f)
            ]
        paths: list[str] = []
        for rel in rels:
            try:
                abs_path = str(self.resolve(rel))
            except HarnessToolError:
                continue
            if abs_path not in paths:
                paths.append(abs_path)
        return paths


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_HARNESS_SKILLS_SECTION = """### Harness tool workflow (doc_redaction app inline tools)

You are running in an AgentCore Harness invoked by the doc_redaction app. The app's redaction
functions are **inline tools that run inside the app** (`list_session_files`, `doc_redact`, …).
Call those tools by name on every step — they are registered for this session. The harness VM's
built-in `shell`, `read`, and `write` cannot see the app's session folder. There is no Gradio URL:
do not use `gradio_client`, curl to the app, or `/gradio_api/*`.

If the `doc-redaction-harness` skill is available, read it first; it supersedes the
Gradio-client mechanics in the other redaction skills. The other skills (`doc-redaction-app`,
`doc-redaction-modifications`, `doc-redact-page-review`) remain the reference for *what* good
Pass 1 review looks like (CSV rules, coverage gates, single apply).

When this prompt or a skill names an endpoint, use the matching tool:

| Endpoint / script in prompt or skills | Inline tool |
|---|---|
| `/doc_redact`, `/redact_document`, `cli_redact.py` | `doc_redact` |
| Reading / editing `*_review_file.csv`, OCR CSVs, summary markdown | `read_session_text`, `write_session_text` |
| `verify_redaction_coverage` (CLI or `POST /agent/verify_redaction_coverage`) | `verify_redaction_coverage` |
| `/preview_boxes`, `preview_redaction_boxes` | `preview_redaction_boxes` |
| Parallel page review (`doc-redact-page-review`) | `stage_page_review_bundles` → `subagent` per page → `merge_page_reviews` |
| `/review_apply`, `POST /agent/apply_review_redactions` | `review_apply` |
| Delivering the outputs | `finish_task` (exactly once, last) |

- All paths are **relative to the session folder** (see `list_session_files`). The source
  document is `{INPUT_PATH}`.
- `doc_redact` writes to `output_redact/` (including `*_redactions_for_review*.pdf` and
  `*_ocr_results_with_words*.csv`); `review_apply` writes to `review/output_review_final/`.
- Pass 1 review aids are included in the Gradio Output files list when present (automatically and
  via `finish_task` `final_files`).
- Your own shell/filesystem is a separate sandbox: the app's files are only reachable through
  the inline tools (or the presigned URLs they return).
- {PAGE_REVIEW_NOTE}

"""


def _format_user_requirements(instructions: str) -> str:
    lines = []
    for raw in instructions.strip().splitlines():
        line = raw.strip()
        if line:
            lines.append(line if line.startswith("-") else f"- {line}")
    return "\n".join(lines)


def _replace_section(text: str, start_marker: str, end_marker: str, new: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        return new + text
    return text[:start] + new + text[end:]


def _strip_long_document_section(text: str) -> str:
    start = text.find("## Specific rules for long documents")
    if start == -1:
        return text
    end = text.find("## User redaction requirements", start)
    if end == -1:
        return text[:start].rstrip() + "\n\n"
    return text[:start].rstrip() + "\n\n" + text[end:]


def _pdf_page_count(path: Path) -> int:
    try:
        import pymupdf

        with pymupdf.open(str(path)) as doc:
            return int(doc.page_count)
    except Exception:  # noqa: BLE001 - images / unreadable docs
        return 0


def build_harness_task_prompt(
    session: HarnessSession,
    instructions: str,
    *,
    page_range: str = "all",
    template_path: Path = PARTNERSHIP_TEMPLATE_PATH,
) -> str:
    """Partnership task prompt adapted for inline tools; user requirements go last."""
    if not instructions.strip():
        raise ValueError("Redaction instructions are required.")
    template = template_path.read_text(encoding="utf-8")
    page_review_note = (
        "Parallel page review is available: for long documents or many flagged pages, "
        "call `stage_page_review_bundles`, give each `subagent` one page's URLs plus the "
        "rules (subagents cannot call the app's tools), then `merge_page_reviews` and one "
        "`review_apply`."
        if session.enable_page_review
        else "Parallel page review is not available in this deployment: review pages "
        "yourself with `read_session_text` / `write_session_text`."
    )
    text = _replace_section(
        template,
        "### Required skills",
        "### Agent anti-confusion rules",
        _HARNESS_SKILLS_SECTION.replace("{PAGE_REVIEW_NOTE}", page_review_note),
    )
    for obsolete in (
        "Do **not** skip reading `doc-redaction-modifications`",
        "**Before any API calls**, read the repo skills below in order.",
    ):
        text = text.replace(obsolete, "")
    text = text.replace("- {REMOTE_BACKEND_GUIDANCE}\n", "")
    d = session.defaults
    replacements = {
        "{FILE_NAME}": session.source_pdf.name,
        "{INPUT_PATH}": session.rel(session.source_pdf),
        "{OUTPUT_BASE}": "",
        "{GRADIO_URL}": "not used (inline tools)",
        "{PAGE_RANGE}": (page_range or "all").strip() or "all",
        "{VLM_BASE_URL}": "N/A — no VLM endpoint for the agent",
        "{VLM_MODEL}": "N/A",
        "{DEFAULT_OCR_METHOD}": d.ocr_method,
        "{DEFAULT_PII_METHOD}": d.pii_method,
        "{VLM_FACES_GUIDANCE}": (
            "Add `CUSTOM_VLM_FACES` to `redact_entities` only if the user asks to redact faces."
        ),
        "{VLM_SIGNATURE_GUIDANCE}": (
            "Add `CUSTOM_VLM_SIGNATURE` to `redact_entities` (or use AWS Textract with "
            "'Extract signatures') only if the user asks to redact signatures."
        ),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    if _pdf_page_count(session.source_pdf) < 100:
        text = _strip_long_document_section(text)

    marker = "## User redaction requirements"
    idx = text.find(marker)
    head = text[:idx] if idx != -1 else text.rstrip() + "\n\n"
    return (
        f"{head}{marker} (authoritative for this task)\n\n"
        f"{_format_user_requirements(instructions)}\n"
    )


# ---------------------------------------------------------------------------
# Handoff loop
# ---------------------------------------------------------------------------


class _Activity:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.lines.append(f"- `{stamp}` {line}")

    def markdown(self) -> str:
        return "\n".join(self.lines[-200:])


def _short(text: str, limit: int = 400) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def run_agentic_redaction(
    document_path: str | Path,
    instructions: str,
    *,
    defaults: HarnessRunDefaults | None = None,
    page_range: str = "all",
    output_root: str | Path = OUTPUT_FOLDER,
    harness_arn: str | None = None,
    endpoint: str | None = None,
    client: Any = None,
    session: HarnessSession | None = None,
    max_rounds: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    eval_recorder: Any | None = None,
    eval_session_hash: str = "",
) -> Iterator[HarnessProgress]:
    """
    Drive one harness redaction task; yields progress for streaming UIs.

    The final yielded item has ``done=True`` and the output files to show the user.
    """
    arn = (harness_arn if harness_arn is not None else AGENTCORE_HARNESS_ARN).strip()
    region = region_from_harness_arn(arn)
    qualifier = (
        endpoint if endpoint is not None else AGENTCORE_HARNESS_ENDPOINT
    ).strip()
    rounds = int(max_rounds or AGENTCORE_HARNESS_MAX_TOOL_ROUNDS)
    activity = _Activity()

    if session is None:
        session = HarnessSession(
            document_path, defaults=defaults, output_root=output_root
        )
    specs = load_inline_tool_specs()
    names = inline_tool_names(specs)
    prompt = build_harness_task_prompt(session, instructions, page_range=page_range)
    if client is None:
        client = _default_agentcore_client(region)

    runtime_session_id = harness_runtime_session_id(f"doc-redaction:{session.run_id}")
    run_started = time.time()
    token_usage = HarnessTokenUsage()
    activity.add(
        f"Started agent run `{session.run_id}` on `{session.source_pdf.name}`."
    )
    yield HarnessProgress(activity.markdown(), [], "Agent run started…")
    activity.add(f"Inline tools on each InvokeHarness: {', '.join(sorted(names))}.")

    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": prompt}]}]
    summary = ""
    for round_no in range(1, rounds + 1):
        request: dict[str, Any] = {
            "harnessArn": arn,
            "runtimeSessionId": runtime_session_id,
            "messages": messages,
            "tools": specs,
            "allowedTools": harness_invoke_allowed_tools(),
        }
        if qualifier and qualifier.upper() != "DEFAULT":
            request["qualifier"] = qualifier
        try:
            response = client.invoke_harness(**request)
        except Exception as exc:  # noqa: BLE001 - surface AWS errors in the UI
            activity.add(f"**Harness invoke failed:** {_short(str(exc))}")
            summary = f"Agent run failed: {exc}"
            break

        collector = HarnessTurnCollector()
        stopped = False
        try:
            event_iter = response.get("stream") or []
            for raw in event_iter:
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if not isinstance(raw, dict):
                    continue
                token_usage.absorb_stream_event(raw)
                for event in collector.feed(raw, inline_names=names):
                    if event.kind == "tool_start" and event.tool_name not in names:
                        activity.add(f"Harness tool: `{event.tool_name}`")
                    elif event.kind == "error":
                        activity.add(f"**Harness error:** {_short(event.text)}")
        except Exception as exc:  # noqa: BLE001 — EventStreamError from botocore
            err_text = str(exc)
            activity.add(f"**Harness stream error:** {_short(err_text)}")
            if "no pending handoff" in err_text.lower():
                activity.add(
                    "The harness session lost the inline-tool handoff (often after a "
                    "long local tool run or a stale ``toolResult``). Restart the agent run."
                )
            summary = f"Agent run failed: {err_text}"
            break
        if stopped:
            activity.add("Stopped by user.")
            summary = "Agent run stopped."
            break

        turn = collector.turn
        if turn.text.strip():
            activity.add(f"Agent: {_short(turn.text)}")
            summary = turn.text.strip()
        orphan = turn.pending_inline_calls(names)
        pending = turn.handoff_inline_calls(names)
        if pending:
            activity.add(
                f"Client inline dispatch: {', '.join(f'`{c.name}`' for c in pending)} "
                f"(stop `{turn.stop_reason or 'unknown'}`)."
            )
        elif orphan and turn.stop_reason != "tool_use":
            activity.add(
                f"Ignored {len(orphan)} inline tool use(s) without harness handoff "
                f"(stop `{turn.stop_reason}`)."
            )
        elif turn.tool_uses and turn.stop_reason in ("tool_use", "tool_result", ""):
            activity.add(
                "**No client inline dispatch** — harness tool uses: "
                f"{[t.name for t in turn.tool_uses]} "
                f"(resolved {sorted(turn.resolved_tool_use_ids)})."
            )
        if not pending:
            if turn.stop_reason and turn.stop_reason != "end_turn":
                activity.add(f"Harness stopped: `{turn.stop_reason}`.")
            break

        results: list[tuple[str, str, bool]] = []
        for call in pending:
            try:
                arguments = call.arguments()
            except ValueError as exc:
                results.append(
                    (call.tool_use_id, json.dumps({"error": str(exc)}), False)
                )
                continue
            if eval_recorder is not None:
                eval_recorder.record_tool(call.name, arguments)
            activity.add(f"Running `{call.name}`…")
            yield HarnessProgress(
                activity.markdown(), [], f"Running {call.name} (round {round_no})…"
            )
            text, ok = session.dispatch(call.name, arguments)
            activity.add(
                f"`{call.name}` {'done' if ok else 'failed'}"
                + ("" if ok else f": {_short(text, 300)}")
            )
            results.append((call.tool_use_id, text, ok))
        yield HarnessProgress(activity.markdown(), [], f"Round {round_no} complete.")
        if session.finished:
            break
        messages = [tool_result_message(results)]
    else:
        activity.add(
            f"**Stopped after {rounds} tool rounds** (AGENTCORE_HARNESS_MAX_TOOL_ROUNDS)."
        )

    if session.finished and session.finish_summary:
        summary = session.finish_summary
    outputs = session.output_files()
    if eval_recorder is not None and getattr(eval_recorder, "started", False):
        eval_path = eval_recorder.write_post_run(
            workspace=session.session_dir,
            document_name=session.source_pdf.name,
            instructions=instructions,
            assistant_output=summary,
            workflow_incomplete=not session.finished,
            session_hash=eval_session_hash,
        )
        if eval_path is not None:
            activity.add(
                f"**Post-run eval:** `{eval_path.name}` → `{eval_path.parent}`"
            )
    elapsed = max(0.0, time.time() - run_started)
    if token_usage.metadata_events:
        activity.add(
            "Harness LLM tokens (stream metadata): "
            f"in={token_usage.input_tokens}, out={token_usage.output_tokens}"
        )
    activity.add(f"Finished with {len(outputs)} output file(s).")
    yield HarnessProgress(
        activity.markdown(),
        outputs,
        summary or "Agent run ended.",
        True,
        token_usage=token_usage,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Gradio adapters
# ---------------------------------------------------------------------------


def ui_ocr_method_to_tool_value(text_extract_method: str, local_ocr_model: str) -> str:
    """Map the Redact tab radio labels to ``doc_redact`` ``ocr_method`` values."""
    from tools.config import (
        AZURE_OPENAI_VLM_TEXT_EXTRACT_OPTION,
        BEDROCK_VLM_TEXT_EXTRACT_OPTION,
        GEMINI_VLM_TEXT_EXTRACT_OPTION,
        LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION,
        SELECTABLE_TEXT_EXTRACT_OPTION,
        TEXTRACT_TEXT_EXTRACT_OPTION,
    )

    mapping = {
        SELECTABLE_TEXT_EXTRACT_OPTION: "Local text",
        TEXTRACT_TEXT_EXTRACT_OPTION: "AWS Textract",
        BEDROCK_VLM_TEXT_EXTRACT_OPTION: "bedrock-vlm",
        GEMINI_VLM_TEXT_EXTRACT_OPTION: "gemini-vlm",
        AZURE_OPENAI_VLM_TEXT_EXTRACT_OPTION: "azure-openai-vlm",
    }
    if text_extract_method == LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION:
        return (local_ocr_model or "Local OCR").strip() or "Local OCR"
    return mapping.get(text_extract_method, text_extract_method or "Local text")


def ui_entities_for_pii_method(
    pii_method: str,
    local_entities: list[str] | None,
    comprehend_entities: list[str] | None,
    llm_entities: list[str] | None,
) -> list[str]:
    from tools.config import (
        AWS_LLM_PII_OPTION,
        AWS_PII_OPTION,
        INFERENCE_SERVER_PII_OPTION,
        LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    )

    if pii_method == AWS_PII_OPTION:
        chosen = comprehend_entities
    elif pii_method in {
        AWS_LLM_PII_OPTION,
        INFERENCE_SERVER_PII_OPTION,
        LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    }:
        chosen = llm_entities
    else:
        chosen = local_entities
    return [str(e) for e in (chosen or []) if str(e).strip()]


def _first_document(files: Any) -> str:
    from tools.simplified_api import normalize_gradio_file_to_path

    items = files if isinstance(files, (list, tuple)) else [files]
    for item in items:
        path = normalize_gradio_file_to_path(item)
        if path and Path(path).suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}:
            return path
    raise ValueError("Upload a PDF or image document before running the agent.")


def _page_range_text(page_min: Any, page_max: Any) -> str:
    lo, hi = int(page_min or 0), int(page_max or 0)
    if not lo and not hi:
        return "all"
    return f"{lo or 1}-{hi or 'end'}"


def resolve_harness_instructions(
    inline: str | None,
    instructions_file: str | Path | None,
) -> str:
    """Resolve agent instructions from inline text or a UTF-8 file (headless / CLI)."""
    text = (inline or "").strip()
    if text:
        return text
    path = str(instructions_file or "").strip()
    if not path:
        return ""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Agent instructions file not found: {path}")
    return resolved.read_text(encoding="utf-8").strip()


def _terms_from_path_or_list(value: Any) -> list[str]:
    """Load allow/deny terms from a CSV path (first column) or a string list."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value and isinstance(value, str):
        path = Path(value.strip())
        if path.is_file():
            import pandas as pd

            frame = pd.read_csv(path, header=None, low_memory=False)
            if frame.empty:
                return []
            return frame.iloc[:, 0].dropna().astype(str).tolist()
    return []


def build_harness_run_defaults_from_cli_args(args: Any) -> HarnessRunDefaults:
    """Map ``cli_redact`` / direct-mode namespace fields to harness ``doc_redact`` defaults."""
    from tools.config import DEFAULT_LOCAL_OCR_MODEL

    allow_raw = getattr(args, "allow_list_file", None) or getattr(
        args, "allow_list", None
    )
    deny_raw = getattr(args, "deny_list_file", None) or getattr(args, "deny_list", None)
    return HarnessRunDefaults(
        ocr_method=ui_ocr_method_to_tool_value(
            str(getattr(args, "ocr_method", "") or "Local text"),
            str(getattr(args, "chosen_local_ocr_model", "") or DEFAULT_LOCAL_OCR_MODEL),
        ),
        pii_method=str(getattr(args, "pii_detector", "") or "Local"),
        redact_entities=ui_entities_for_pii_method(
            str(getattr(args, "pii_detector", "") or "Local"),
            getattr(args, "local_redact_entities", None),
            getattr(args, "aws_redact_entities", None),
            getattr(args, "llm_redact_entities", None),
        ),
        allow_list=_terms_from_path_or_list(allow_raw),
        deny_list=_terms_from_path_or_list(deny_raw),
        handwrite_signature_checkbox=list(
            getattr(args, "handwrite_signature_extraction", None) or []
        )
        or None,
        page_min=int(getattr(args, "page_min", 0) or 0),
        page_max=int(getattr(args, "page_max", 0) or 0),
    )


def agentic_redact_for_gradio(
    files: Any,
    instructions: str,
    text_extract_method: str,
    local_ocr_model: str,
    pii_method: str,
    local_entities: list[str] | None,
    comprehend_entities: list[str] | None,
    llm_entities: list[str] | None,
    allow_list: list[str] | None,
    deny_list: list[str] | None,
    handwrite_signature: list[str] | None,
    page_min: Any,
    page_max: Any,
    output_folder: str | None = None,
    session_hash: str = "",
    cost_code: str = "",
) -> Iterator[tuple[str, str, list[str], list[str]]]:
    """Gradio streaming handler: yields ``(summary, activity, output_files, output_file_list_state)``."""
    import gradio as gr

    ready, reason = harness_feature_ready()
    if not ready:
        raise gr.Error(f"AgentCore agent redaction is not configured: {reason}")
    if not (instructions or "").strip():
        raise gr.Error("Enter redaction instructions for the agent.")
    try:
        document = _first_document(files)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    defaults = HarnessRunDefaults(
        ocr_method=ui_ocr_method_to_tool_value(text_extract_method, local_ocr_model),
        pii_method=pii_method or "Local",
        redact_entities=ui_entities_for_pii_method(
            pii_method, local_entities, comprehend_entities, llm_entities
        ),
        allow_list=[str(x) for x in (allow_list or []) if str(x).strip()],
        deny_list=[str(x) for x in (deny_list or []) if str(x).strip()],
        handwrite_signature_checkbox=list(handwrite_signature or []) or None,
        page_min=int(page_min or 0),
        page_max=int(page_max or 0),
    )
    from tools.harness_eval_support import (
        HarnessEvalRecorder,
        harness_post_run_eval_enabled,
    )

    eval_recorder: HarnessEvalRecorder | None = None
    session = HarnessSession(
        document, defaults=defaults, output_root=output_folder or OUTPUT_FOLDER
    )
    if harness_post_run_eval_enabled():
        eval_recorder = HarnessEvalRecorder(session_hash=(session_hash or "").strip())
        gold_line = eval_recorder.begin(
            workspace=session.session_dir,
            pdf_path=session.source_pdf,
            document_name=session.source_pdf.name,
            instructions=instructions,
            text_extract_method=text_extract_method,
            pii_method=pii_method or "Local",
            ocr_method_short=defaults.ocr_method,
        )
        if gold_line:
            yield "Starting agent run…", gold_line, [], []
    final: HarnessProgress | None = None
    for progress in run_agentic_redaction(
        document,
        instructions,
        defaults=defaults,
        page_range=_page_range_text(page_min, page_max),
        output_root=output_folder or OUTPUT_FOLDER,
        session=session,
        eval_recorder=eval_recorder,
        eval_session_hash=(session_hash or "").strip(),
    ):
        final = progress
        files = progress.output_files
        yield progress.summary, progress.activity, files, files
    if final is not None and final.done:
        from tools.harness_usage_log import record_harness_agent_usage

        record_harness_agent_usage(
            session_hash=(session_hash or "").strip() or session.run_id,
            document_name=session.source_pdf.name,
            time_taken=final.elapsed_seconds,
            token_usage=final.token_usage,
            pii_method=pii_method or "Local",
            text_extraction_method=text_extract_method or defaults.ocr_method,
            cost_code=(cost_code or "").strip(),
            total_pages=_pdf_page_count(session.source_pdf),
        )


def upload_agentic_outputs_to_s3(
    output_paths: list[str] | None,
    *,
    s3_output_folder: str,
    save_outputs_to_s3: bool,
    base_document: str | Path | None = None,
    s3_bucket: str | None = None,
) -> str | None:
    """Upload harness deliverables using the same helper as the main redaction tab."""
    from tools.aws_functions import export_outputs_to_s3
    from tools.config import S3_OUTPUTS_BUCKET

    paths = [p for p in (output_paths or []) if p]
    if not paths or not (s3_output_folder or "").strip():
        return None
    return export_outputs_to_s3(
        file_list_state=paths,
        s3_output_folder_state_value=s3_output_folder.strip(),
        save_outputs_to_s3_flag=save_outputs_to_s3,
        base_file_state=str(base_document) if base_document else None,
        s3_bucket=s3_bucket or S3_OUTPUTS_BUCKET,
    )


def agentic_redact_api(
    document_file: Any,
    instructions: str,
    ocr_method: str | None = "Local text",
    pii_method: str | None = "Local",
    redact_entities: list[str] | None = None,
    allow_list: list[str] | None = None,
    deny_list: list[str] | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
) -> tuple[list[str], str]:
    """Short ``gr.api`` wrapper: run the harness agent to completion on one document."""
    from tools.simplified_api import (
        normalize_gradio_file_to_path,
        stage_gradio_upload_if_ephemeral,
    )

    ready, reason = harness_feature_ready()
    if not ready:
        raise ValueError(f"AgentCore agent redaction is not configured: {reason}")
    document = normalize_gradio_file_to_path(document_file)
    if not document or not Path(document).is_file():
        raise ValueError("document_file is missing or could not be resolved to a path.")
    document = stage_gradio_upload_if_ephemeral(document)
    defaults = HarnessRunDefaults(
        ocr_method=ocr_method or "Local text",
        pii_method=pii_method or "Local",
        redact_entities=list(redact_entities or []),
        allow_list=list(allow_list or []),
        deny_list=list(deny_list or []),
        page_min=int(page_min or 0),
        page_max=int(page_max or 0),
    )
    final: HarnessProgress | None = None
    for progress in run_agentic_redaction(
        document,
        instructions,
        defaults=defaults,
        page_range=_page_range_text(page_min, page_max),
    ):
        final = progress
    if final is None:
        return [], "Agent run produced no output."
    if final.done:
        from tools.harness_usage_log import record_harness_agent_usage

        record_harness_agent_usage(
            session_hash="api",
            document_name=Path(document).name,
            time_taken=final.elapsed_seconds,
            token_usage=final.token_usage,
            pii_method=pii_method or "Local",
            text_extraction_method=ocr_method or "Local text",
            total_pages=_pdf_page_count(Path(document)),
        )
    return final.output_files, final.summary


__all__ = [
    "HarnessProgress",
    "agentic_redact_api",
    "agentic_redact_for_gradio",
    "upload_agentic_outputs_to_s3",
    "HarnessRunDefaults",
    "HarnessSession",
    "HarnessToolError",
    "build_harness_run_defaults_from_cli_args",
    "build_harness_task_prompt",
    "harness_feature_ready",
    "resolve_harness_instructions",
    "run_agentic_redaction",
    "validate_and_repair_review_csv",
]
