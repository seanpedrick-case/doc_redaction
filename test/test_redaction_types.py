"""Unit tests for tools.redaction_types (options/context split and legacy bridging)."""

from __future__ import annotations

from dataclasses import fields

import pandas as pd

from tools.redaction_types import (
    RedactionContext,
    RedactionOptions,
    from_legacy_dict,
    to_legacy_kwargs,
)


def test_to_legacy_kwargs_contains_all_option_and_context_keys():
    opts = RedactionOptions()
    ctx = RedactionContext()
    merged = to_legacy_kwargs(opts, ctx)
    opt_names = {f.name for f in fields(RedactionOptions)}
    ctx_names = {f.name for f in fields(RedactionContext)}
    assert set(merged.keys()) == opt_names | ctx_names
    assert len(merged) == len(opt_names) + len(ctx_names)


def test_from_legacy_dict_ignores_file_paths_and_progress():
    flat = {
        "file_paths": ["/x.pdf"],
        "progress": object(),
        "language": "fr",
        "review_file_path": "/out/review.csv",
        "page_min": 2,
    }
    opts, ctx = from_legacy_dict(flat)
    assert opts.language == "fr"
    assert opts.page_min == 2
    assert ctx.review_file_path == "/out/review.csv"


def test_from_legacy_roundtrip_preserves_values():
    opts_in = RedactionOptions(
        language="de",
        text_extraction_method="Local text",
        page_min=1,
        page_max=5,
        text_extraction_only=True,
        chosen_redact_entities=["PERSON"],
    )
    ctx_in = RedactionContext(
        latest_file_completed=2,
        review_file_path="/tmp/r.csv",
        all_request_metadata_str="meta",
        all_page_line_level_ocr_results_df=pd.DataFrame({"a": [1]}),
    )
    flat = to_legacy_kwargs(opts_in, ctx_in)
    flat["file_paths"] = ["/a.pdf"]
    flat["progress"] = None
    opts_out, ctx_out = from_legacy_dict(flat)
    assert opts_out == opts_in
    assert ctx_out.latest_file_completed == ctx_in.latest_file_completed
    assert ctx_out.review_file_path == ctx_in.review_file_path
    assert ctx_out.all_request_metadata_str == ctx_in.all_request_metadata_str
    pd.testing.assert_frame_equal(
        ctx_out.all_page_line_level_ocr_results_df,
        ctx_in.all_page_line_level_ocr_results_df,
    )


def test_merge_defaults_for_missing_keys():
    flat = {"language": "es"}
    opts, ctx = from_legacy_dict(flat)
    assert opts.language == "es"
    assert opts.page_min == 0
    assert isinstance(ctx.out_file_paths, list)
    assert ctx.out_file_paths == []


def test_explicit_none_optional_entities():
    flat = {"chosen_llm_entities": None, "chosen_redact_entities": ["X"]}
    opts, _ = from_legacy_dict(flat)
    assert opts.chosen_llm_entities is None
    assert opts.chosen_redact_entities == ["X"]
