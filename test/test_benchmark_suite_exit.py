"""Benchmark suite exit codes: technical vs QA."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from run_benchmark_suite import (  # noqa: E402
    _accumulate_surface_outcome,
    _apply_gradio_url_env,
    _apply_llm_base_url_env,
    _apply_qa_outcome,
    _env_default,
)


def test_apply_qa_outcome_poor_score_technical_zero():
    rec: dict = {"exit_code": 1}
    _apply_qa_outcome(
        rec, technical_exit=0, qa_pass=False, message="overall_pass=False"
    )
    assert rec["exit_code"] == 0
    assert rec["qa_pass"] is False
    assert rec["qa_completed"] is True


def test_accumulate_surface_qa_fail_does_not_raise_technical():
    rec = {"skipped": False, "exit_code": 0, "qa_completed": True, "qa_pass": False}
    max_t, any_qa = _accumulate_surface_outcome(rec, 0, False)
    assert max_t == 0
    assert any_qa is True


def test_env_default_prefers_pristine_env_then_env_file(monkeypatch) -> None:
    import run_benchmark_suite as rbs

    # Replace the whole snapshot: in a pytest process, tools.config may already
    # have written code defaults into os.environ before run_benchmark_suite was
    # imported, and those must not leak into the fallback.
    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {"BENCHMARK_OCR_METHOD": "from-env"})
    monkeypatch.setattr(
        rbs,
        "_env_file_values",
        lambda: {"BENCHMARK_OCR_METHOD": "from-file", "OTHER_VAR": "file-only"},
    )
    assert (
        _env_default("BENCHMARK_OCR_METHOD", "DEFAULT_TEXT_EXTRACTION_MODEL")
        == "from-env"
    )
    assert _env_default("OTHER_VAR") == "file-only"
    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {})
    assert _env_default("BENCHMARK_OCR_METHOD") == "from-file"
    assert _env_default("NO_SUCH_VAR_ANYWHERE") is None


def test_parser_stack_flags_fall_back_to_env(monkeypatch) -> None:
    import run_benchmark_suite as rbs

    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {"BENCHMARK_LOCAL_OCR_MODEL": "paddle"})
    monkeypatch.setattr(
        rbs,
        "_env_file_values",
        lambda: {"DEFAULT_TEXT_EXTRACTION_MODEL": "Local model - selectable text"},
    )
    args = rbs._build_parser().parse_args([])
    assert args.chosen_local_ocr_model == "paddle"  # BENCHMARK_* env var
    assert args.ocr_method == "Local model - selectable text"  # app config env file
    # explicit flag beats env
    args2 = rbs._build_parser().parse_args(["--chosen-local-ocr-model", "tesseract"])
    assert args2.chosen_local_ocr_model == "tesseract"


def test_gradio_url_flag_and_env_override(monkeypatch) -> None:
    import os

    import run_benchmark_suite as rbs

    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {})
    monkeypatch.setattr(
        rbs,
        "_env_file_values",
        lambda: {"DOC_REDACTION_GRADIO_URL": "http://env-file:7860"},
    )
    # env-file fallback becomes the flag default
    args = rbs._build_parser().parse_args([])
    assert args.gradio_url == "http://env-file:7860"
    # explicit flag wins and normalizes the trailing slash when applied
    args2 = rbs._build_parser().parse_args(["--gradio-url", "http://flag:9999/"])
    assert args2.gradio_url == "http://flag:9999/"
    monkeypatch.delenv("DOC_REDACTION_GRADIO_URL", raising=False)
    _apply_gradio_url_env(args2.gradio_url)
    assert os.environ["DOC_REDACTION_GRADIO_URL"] == "http://flag:9999"
    # None/empty is a no-op
    _apply_gradio_url_env(None)
    _apply_gradio_url_env("")
    assert os.environ["DOC_REDACTION_GRADIO_URL"] == "http://flag:9999"


def test_llm_base_url_flag_and_env_override(monkeypatch) -> None:
    import os

    import run_benchmark_suite as rbs

    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {})
    monkeypatch.setattr(
        rbs,
        "_env_file_values",
        lambda: {"AGENT_LLAMA_BASE_URL": "http://agent-env:8080/v1"},
    )
    # env-file fallback becomes the flag default
    args = rbs._build_parser().parse_args([])
    assert args.llm_base_url == "http://agent-env:8080/v1"
    # explicit flag wins
    args2 = rbs._build_parser().parse_args(["--llm-base-url", "http://flag:9999"])
    assert args2.llm_base_url == "http://flag:9999"
    monkeypatch.delenv("AGENT_LLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _apply_llm_base_url_env(args2.llm_base_url)
    assert os.environ["AGENT_LLAMA_BASE_URL"] == "http://flag:9999"
    assert os.environ["OPENAI_BASE_URL"] == "http://flag:9999"
    # None/empty is a no-op
    _apply_llm_base_url_env(None)
    assert os.environ["AGENT_LLAMA_BASE_URL"] == "http://flag:9999"
    # an existing OPENAI_BASE_URL is not clobbered
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai:4000/v1")
    _apply_llm_base_url_env("http://other:1")
    assert os.environ["AGENT_LLAMA_BASE_URL"] == "http://other:1"
    assert os.environ["OPENAI_BASE_URL"] == "http://openai:4000/v1"


def test_parser_records_explicit_flags(monkeypatch) -> None:
    import run_benchmark_suite as rbs

    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {})
    monkeypatch.setattr(
        rbs,
        "_env_file_values",
        lambda: {
            "DEFAULT_TEXT_EXTRACTION_MODEL": "Local model - selectable text",
            "DEFAULT_LOCAL_OCR_MODEL": "paddle",
        },
    )
    args = rbs._build_parser().parse_args([])
    assert args.ocr_method == "Local model - selectable text"
    assert args.chosen_local_ocr_model == "paddle"
    assert getattr(args, "_explicit_flags", set()) == set()
    args2 = rbs._build_parser().parse_args(["--chosen-local-ocr-model", "tesseract"])
    assert args2.chosen_local_ocr_model == "tesseract"
    assert "--chosen-local-ocr-model" in args2._explicit_flags
    assert "--ocr-method" not in args2._explicit_flags


def test_validate_llm_base_url() -> None:
    import socket

    import run_benchmark_suite as rbs

    err = rbs._validate_llm_base_url("http://nonexistent-host.invalid:8080/v1")
    assert err is not None and "does not resolve" in err
    assert "no host" in (rbs._validate_llm_base_url("http://") or "")
    # closed port on localhost: resolves but refuses the connection
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    err2 = rbs._validate_llm_base_url(f"http://127.0.0.1:{closed_port}/v1")
    assert err2 is not None and "cannot connect" in err2
    # live listener is accepted
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    live_port = srv.getsockname()[1]
    try:
        assert rbs._validate_llm_base_url(f"http://127.0.0.1:{live_port}/v1") is None
    finally:
        srv.close()
    # IPv6 literal host (getaddrinfo returns 4-tuple sockaddrs on Windows):
    # must return an error message, never raise
    err3 = rbs._validate_llm_base_url("http://[::1]:1/v1")
    assert err3 is not None and ("cannot connect" in err3 or "does not resolve" in err3)


def test_main_fails_fast_on_unreachable_llm_base_url(monkeypatch, capsys) -> None:
    import socket

    import run_benchmark_suite as rbs

    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {})
    monkeypatch.setattr(rbs, "_env_file_values", lambda: {})
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    rc = rbs.main(["--llm-base-url", f"http://127.0.0.1:{closed_port}/v1"])
    assert rc == 2
    assert "cannot connect" in capsys.readouterr().err


def test_main_fails_fast_on_unresolvable_llm_base_url(monkeypatch, capsys) -> None:
    import run_benchmark_suite as rbs

    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {})
    monkeypatch.setattr(rbs, "_env_file_values", lambda: {})
    rc = rbs.main(["--llm-base-url", "http://nonexistent-host.invalid:8080/v1"])
    assert rc == 2
    assert "does not resolve" in capsys.readouterr().err


def test_find_case_eval_summary_scoped_and_no_cross_case_fallback(tmp_path) -> None:
    import json as jsonlib

    import run_benchmark_suite as rbs

    runs_root = tmp_path / "runs"
    other = runs_root / "main" / "other_case" / "eval" / "run_1"
    other.mkdir(parents=True)
    (other / "eval_summary.json").write_text(
        jsonlib.dumps({"gold_case": str(tmp_path / "gold" / "other_case")}),
        encoding="utf-8",
    )
    case_dir = tmp_path / "gold" / "my_case"
    case_dir.mkdir(parents=True)
    eval_root = tmp_path / "evalroot"

    # No summary for my_case anywhere: must NOT fall back to other_case's summary.
    assert rbs._find_case_eval_summary(runs_root, case_dir, eval_root=eval_root) is None

    # A summary scoped to the case folder wins.
    mine = runs_root / "main" / "my_case" / "eval" / "run_2"
    mine.mkdir(parents=True)
    scoped_path = mine / "eval_summary.json"
    scoped_path.write_text(
        jsonlib.dumps({"gold_case": str(case_dir.resolve())}), encoding="utf-8"
    )
    assert (
        rbs._find_case_eval_summary(runs_root, case_dir, eval_root=eval_root)
        == scoped_path
    )

    # Mirrored app_* runs are matched on the gold_case recorded inside the summary.
    mirror_root = eval_root / "runs"
    app_dir = mirror_root / "app_doc_123"
    app_dir.mkdir(parents=True)
    mirror_path = app_dir / "eval_summary.json"
    mirror_path.write_text(
        jsonlib.dumps({"gold_case": str(case_dir.resolve())}), encoding="utf-8"
    )
    empty_runs = tmp_path / "runs_empty"
    assert (
        rbs._find_case_eval_summary(empty_runs, case_dir, eval_root=eval_root)
        == mirror_path
    )


def test_apply_vlm_endpoint_env(monkeypatch) -> None:
    import os

    import run_benchmark_suite as rbs

    for key in (
        "INFERENCE_SERVER_API_URL",
        "DEFAULT_INFERENCE_SERVER_VLM_MODEL",
        "INFERENCE_SERVER_MODEL_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    # no llm-base-url: no-op
    rbs._apply_vlm_endpoint_env(None, "some-model")
    assert "INFERENCE_SERVER_API_URL" not in os.environ
    # /v1 suffix is stripped (the engine appends /v1 itself); model name applied
    rbs._apply_vlm_endpoint_env(
        "http://localhost:8004/v1", "unsloth/Inkling-2-Mini-GGUF"
    )
    assert os.environ["INFERENCE_SERVER_API_URL"] == "http://localhost:8004"
    assert (
        os.environ["DEFAULT_INFERENCE_SERVER_VLM_MODEL"]
        == "unsloth/Inkling-2-Mini-GGUF"
    )
    assert os.environ["INFERENCE_SERVER_MODEL_NAME"] == "unsloth/Inkling-2-Mini-GGUF"
    # URL without /v1 is kept as-is
    rbs._apply_vlm_endpoint_env("http://localhost:8004/", None)
    assert os.environ["INFERENCE_SERVER_API_URL"] == "http://localhost:8004"


def test_main_applies_vlm_endpoint_env(monkeypatch) -> None:
    import os

    import run_benchmark_suite as rbs

    monkeypatch.setattr(rbs, "_PRISTINE_ENV", {})
    monkeypatch.setattr(rbs, "_env_file_values", lambda: {})
    monkeypatch.setattr(rbs, "_validate_llm_base_url", lambda url, **kw: None)
    for key in (
        "INFERENCE_SERVER_API_URL",
        "DEFAULT_INFERENCE_SERVER_VLM_MODEL",
        "INFERENCE_SERVER_MODEL_NAME",
    ):
        monkeypatch.delenv(key, raising=False)

    def fake_suite(**kwargs):
        raise FileNotFoundError("stop before any case runs")

    monkeypatch.setattr(rbs, "run_benchmark_suite", fake_suite)
    rc = rbs.main(
        [
            "--llm-base-url",
            "http://localhost:8004/v1",
            "--model",
            "unsloth/Inkling-2-Mini-GGUF",
        ]
    )
    assert rc == 2
    assert os.environ["INFERENCE_SERVER_API_URL"] == "http://localhost:8004"
    assert (
        os.environ["DEFAULT_INFERENCE_SERVER_VLM_MODEL"]
        == "unsloth/Inkling-2-Mini-GGUF"
    )
    assert os.environ["INFERENCE_SERVER_MODEL_NAME"] == "unsloth/Inkling-2-Mini-GGUF"


def test_settings_to_cli_args_picks_up_vlm_endpoint(monkeypatch, tmp_path) -> None:
    import run_benchmark_suite as rbs
    from derive_app_eval_instructions import AppEvalRedactionSettings

    monkeypatch.setenv("INFERENCE_SERVER_API_URL", "http://localhost:8004")
    monkeypatch.setenv(
        "DEFAULT_INFERENCE_SERVER_VLM_MODEL", "unsloth/Inkling-2-Mini-GGUF"
    )
    settings = AppEvalRedactionSettings()
    args = rbs._settings_to_cli_args(
        settings, pdf_path=tmp_path / "in.pdf", output_dir=tmp_path / "out"
    )
    assert args["inference_server_api_url"] == "http://localhost:8004"
    assert args["inference_server_vlm_model"] == "unsloth/Inkling-2-Mini-GGUF"


def test_apply_provider_env_aligns_llama_model_id(monkeypatch) -> None:
    import os
    import types

    import run_agent_regression as rar

    fake = types.ModuleType("pi_agent_config")
    fake.PROVIDER_LLAMA = "llama-cpp"
    fake.normalize_provider = lambda p: (p or "").strip().lower()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pi_agent_config", fake)
    monkeypatch.delenv("AGENT_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("AGENT_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("AGENT_LLAMA_MODEL_ID", raising=False)

    rar._apply_provider_env("llama-cpp", "unsloth/Inkling-2-Mini-GGUF")
    assert os.environ["AGENT_DEFAULT_MODEL"] == "unsloth/Inkling-2-Mini-GGUF"
    assert os.environ["AGENT_LLAMA_MODEL_ID"] == "unsloth/Inkling-2-Mini-GGUF"

    # A non-llama provider must not set the llama model id.
    monkeypatch.delenv("AGENT_LLAMA_MODEL_ID", raising=False)
    rar._apply_provider_env("amazon-bedrock", "amazon.nova-pro-v1:0")
    assert "AGENT_LLAMA_MODEL_ID" not in os.environ
    assert os.environ["AGENT_DEFAULT_MODEL"] == "amazon.nova-pro-v1:0"
