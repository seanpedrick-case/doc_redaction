#!/usr/bin/env python3
"""Run standardized benchmark: gold cases on LangGraph agent, main app, and/or AgentCore harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

# Snapshot the process environment before any project import can populate config
# defaults into os.environ (tools.config's get_or_create_env_var writes defaults
# back). Stack-flag fallbacks read this snapshot plus the app config env file, so an
# unset knob never becomes a code-default "override" of the per-case gold settings.
_PRISTINE_ENV = dict(os.environ)

SurfaceChoice = Literal["agent", "main", "harness"]

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from benchmark_cases import (  # noqa: E402
    audit_benchmark_gold_cases,
    case_readiness,
    default_gold_root,
    discover_benchmark_gold_cases,
)
from benchmark_stack import (  # noqa: E402
    BenchmarkStackOverrides,
    apply_benchmark_overrides,
    effective_ocr_method_label,
    overrides_from_argv,
)
from derive_app_eval_instructions import (  # noqa: E402
    AppEvalRedactionSettings,
    settings_from_json,
)
from run_agent_regression import (  # noqa: E402
    _ensure_agent_imports,
    resolve_gold_case_pdf,
    run_regression,
)


def _repo_root() -> Path:
    repo, _, _ = _ensure_agent_imports()
    return repo


def _app_config_env_file() -> Path:
    raw = (_PRISTINE_ENV.get("APP_CONFIG_PATH") or "").strip()
    repo = Path(__file__).resolve().parents[2]
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else (repo / candidate)
    return repo / "config" / "app_config.env"


def _agent_config_env_file() -> Path:
    """Agent config file (config/agent.env, legacy pi_agent.env)."""
    config_dir = Path(__file__).resolve().parents[2] / "config"
    for name in ("agent.env", "pi_agent.env"):
        candidate = config_dir / name
        if candidate.is_file():
            return candidate
    return config_dir / "agent.env"


def _env_file_values() -> dict[str, str]:
    """
    Values from the config env files: the app config file (APP_CONFIG_PATH or
    config/app_config.env) first, then config/agent.env (legacy pi_agent.env) as a
    lower-priority layer for agent-side variables like DOC_REDACTION_GRADIO_URL.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    values: dict[str, str] = {}
    for path in (_app_config_env_file(), _agent_config_env_file()):
        try:
            if not path.is_file():
                continue
            for key, value in dotenv_values(str(path)).items():
                if value is not None and key not in values:
                    values[key] = value or ""
        except Exception:
            continue
    return values


def _env_default(*names: str) -> str | None:
    """
    Fallback value for a CLI flag: the first non-empty of ``names`` found in the
    real process environment (snapshot taken before project imports), then in the
    config env files (app config, then agent.env). Returns None when unset, so
    per-case gold settings stay in charge unless the user explicitly sets a flag
    or env var.
    """
    for name in names:
        value = (_PRISTINE_ENV.get(name) or "").strip()
        if value:
            return value
    file_values = _env_file_values()
    for name in names:
        value = (file_values.get(name) or "").strip()
        if value:
            return value
    return None


def _apply_gradio_url_env(url: str | None) -> None:
    """
    Point the agent surface at a given main-app URL by setting
    ``DOC_REDACTION_GRADIO_URL`` (read on each call by the agent tools, so setting
    it here — before ``run_regression`` imports the agent stack — takes effect).
    None/empty is a no-op.
    """
    value = (url or "").strip().rstrip("/")
    if value:
        os.environ["DOC_REDACTION_GRADIO_URL"] = value


def _apply_llm_base_url_env(
    url: str | None, *, force_openai_base_url: bool = False
) -> None:
    """
    Point the OpenAI-compatible agent orchestrator LLM (Pi ``llama-cpp`` provider,
    LangGraph ``ChatOpenAI``, DeepEval judge) at a given inference-server URL by
    setting ``AGENT_LLAMA_BASE_URL`` (``/v1`` is appended by the consumers when
    missing). ``OPENAI_BASE_URL`` is set too unless already configured — or is
    forced when the flag was given explicitly on the command line
    (*force_openai_base_url*), so a stale ``OPENAI_BASE_URL`` from the shell/config
    (e.g. an in-network ``llama-inference`` alias) cannot send the DeepEval judge
    to an unreachable host. This runs in ``main()`` before ``run_regression``
    imports the agent stack; if ``pi_agent_config`` was already imported (it
    resolves the URL at import time), its constant is updated as well.
    None/empty is a no-op.
    """
    value = (url or "").strip().rstrip("/")
    if not value:
        return
    os.environ["AGENT_LLAMA_BASE_URL"] = value
    if force_openai_base_url:
        os.environ["OPENAI_BASE_URL"] = value
    else:
        os.environ.setdefault("OPENAI_BASE_URL", value)
    try:
        import pi_agent_config

        pi_agent_config.LLAMA_BASE_URL = (
            value if value.endswith("/v1") else f"{value}/v1"
        )
    except ImportError:
        pass


def _validate_llm_base_url(url: str, *, timeout_s: float = 5.0) -> str | None:
    """
    Fail-fast check for ``--llm-base-url``: the host must resolve and accept a TCP
    connection, because the agent orchestrator LLM and the DeepEval judge both use
    this URL. Returns an error message, or None when reachable.
    """
    import socket
    from urllib.parse import urlparse

    raw = (url or "").strip()
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = parsed.hostname
    if not host:
        return f"--llm-base-url {raw!r} has no host"
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        return f"--llm-base-url {raw!r} has an invalid port ({exc})"
    # create_connection resolves host+port itself and accepts IPv4/IPv6 alike
    # (passing a raw getaddrinfo sockaddr breaks on 4-tuple IPv6 addresses).
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return None
    except socket.gaierror as exc:
        return (
            f"--llm-base-url {raw!r}: host {host!r} does not resolve ({exc}); "
            "the agent LLM and DeepEval judge cannot use this URL"
        )
    except OSError as exc:
        return (
            f"--llm-base-url {raw!r}: cannot connect to {host}:{port} ({exc}); "
            "the agent LLM and DeepEval judge cannot use this URL"
        )


def _apply_vlm_endpoint_env(llm_base_url: str | None, model: str | None) -> None:
    """
    When ``--llm-base-url`` points at an OpenAI-compatible inference server, also aim
    the main app's inference-server VLM (custom ``CUSTOM_VLM_*`` passes and hybrid OCR
    second pass) at the same server: ``INFERENCE_SERVER_API_URL`` gets the base URL
    with any trailing ``/v1`` stripped (the engine appends ``/v1/...`` itself), and the
    VLM model name gets ``--model``. Must run before ``tools.config`` is first imported
    (it reads ``os.environ`` via ``get_or_create_env_var``); the subprocess main leg
    inherits these through the copied environment and ``load_dotenv(override=False)``.
    """
    value = (llm_base_url or "").strip().rstrip("/")
    if not value:
        return
    base = value[:-3] if value.endswith("/v1") else value
    os.environ["INFERENCE_SERVER_API_URL"] = base
    if (model or "").strip():
        os.environ["DEFAULT_INFERENCE_SERVER_VLM_MODEL"] = model.strip()
        os.environ["INFERENCE_SERVER_MODEL_NAME"] = model.strip()


def _git_commit_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def _new_suite_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sha = _git_commit_short()
    return f"bench_{stamp}_{sha}" if sha else f"bench_{stamp}"


def _parse_surfaces(raw: str) -> set[SurfaceChoice]:
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    allowed = {"agent", "main", "harness"}
    bad = parts - allowed
    if bad:
        raise ValueError(f"Unknown surfaces: {bad}; use agent, main, harness")
    if not parts:
        parts = allowed
    return parts  # type: ignore[return-value]


def resolve_benchmark_runs_root(eval_dir: Path, suite_id: str) -> Path:
    """
    Writable directory for per-suite artifacts (``eval/runs`` in dev).

    Docker ``redaction-app-llama`` mounts ``eval/outputs`` but often not
    ``eval/runs`` (image tree may be root-owned). Fall back to
    ``eval/outputs/benchmark_runs``, then ``OUTPUT_FOLDER``.
    """
    eval_dir = eval_dir.resolve()
    custom = (os.environ.get("BENCHMARK_RUNS_ROOT") or "").strip()
    if custom:
        base = Path(custom)
        if not base.is_absolute():
            base = (_repo_root() / custom).resolve()
        target = base / suite_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    for base in (
        eval_dir / "runs",
        eval_dir / "outputs" / "benchmark_runs",
    ):
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError:
            continue
        target = base / suite_id
        try:
            target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError:
            continue

    fallback_base: Path
    try:
        from tools.config import OUTPUT_FOLDER

        fallback_base = Path(OUTPUT_FOLDER) / "benchmark_runs"
    except ImportError:
        fallback_base = Path("/tmp/doc_redaction_benchmark_runs")
    target = fallback_base / suite_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _set_benchmark_csv_env(suite_id: str, git_commit: str) -> None:
    os.environ["BENCHMARK_SUITE_ID"] = suite_id
    os.environ["BENCHMARK_GIT_COMMIT"] = git_commit
    os.environ.setdefault("AGENT_EVAL_RUN_SUMMARY_CSV", "1")
    os.environ["APP_EVAL_ENABLED"] = "1"
    os.environ.setdefault("APP_EVAL_RUN_SUMMARY_CSV", "1")


def _read_case_instructions(case_dir: Path) -> str:
    path = case_dir / "instructions.txt"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _harness_defaults_from_gold(
    case_dir: Path,
    overrides: BenchmarkStackOverrides | None = None,
) -> Any:
    """Map gold case artifacts to ``HarnessRunDefaults`` (main-app AgentCore, not LangGraph)."""
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.agentcore_harness import HarnessRunDefaults

    settings_path = case_dir / "app_eval_settings.json"
    if settings_path.is_file():
        settings = apply_benchmark_overrides(
            settings_from_json(str(settings_path)), overrides
        )
        sig = list(settings.handwrite_signature_checkbox)
        ocr = effective_ocr_method_label(settings)
        return HarnessRunDefaults(
            ocr_method=ocr or "Local text",
            pii_method=(settings.pii_identification_method or "Local").strip(),
            redact_entities=settings.merged_entities(),
            allow_list=list(settings.allow_list),
            deny_list=list(settings.deny_list),
            handwrite_signature_checkbox=sig or None,
            page_min=int(settings.page_min or 0),
            page_max=int(settings.page_max or 0),
        )

    manifest: dict[str, Any] = {}
    manifest_path = case_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, json.JSONDecodeError):
            manifest = {}

    ocr = str(manifest.get("ocr_method") or "Local text").strip()
    pii = str(manifest.get("pii_method") or "Local").strip()
    if overrides:
        if overrides.text_extraction_method:
            ocr = overrides.text_extraction_method.strip()
        if overrides.pii_identification_method:
            pii = overrides.pii_identification_method.strip()
        if overrides.chosen_local_ocr_model:
            ocr = effective_ocr_method_label(
                AppEvalRedactionSettings(
                    text_extraction_method=ocr,
                    chosen_local_ocr_model=overrides.chosen_local_ocr_model,
                )
            )
    if "pin in agent" in ocr.casefold():
        ocr = "Local text"
    if "pin in agent" in pii.casefold():
        pii = "Local"
    return HarnessRunDefaults(ocr_method=ocr, pii_method=pii)


def _harness_configured() -> tuple[bool, str]:
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.agentcore_harness import harness_feature_ready

    return harness_feature_ready()


def _accumulate_surface_outcome(
    rec: dict[str, Any],
    max_technical: int,
    any_qa_fail: bool,
) -> tuple[int, bool]:
    if rec.get("skipped"):
        return max_technical, any_qa_fail
    tech = int(rec.get("exit_code", 0) or 0)
    if tech != 0:
        max_technical = max(max_technical, tech)
    if rec.get("qa_completed") and rec.get("qa_pass") is False:
        any_qa_fail = True
    return max_technical, any_qa_fail


def _apply_qa_outcome(
    record: dict[str, Any],
    *,
    technical_exit: int,
    qa_pass: bool | None,
    message: str = "",
) -> dict[str, Any]:
    """``exit_code`` = run health only; ``qa_pass`` = gold/policy result (may be poor)."""
    record["exit_code"] = int(technical_exit)
    record["qa_pass"] = qa_pass
    record["qa_completed"] = qa_pass is not None
    if message:
        record["message"] = message
    return record


def _settings_to_cli_args(
    settings: AppEvalRedactionSettings,
    *,
    pdf_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build ``cli_redact.main(direct_mode_args=...)`` for one main-app benchmark case."""
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    deny = list(settings.deny_list)
    allow = list(settings.allow_list)

    # Mirror the engine default (chosen_llm_entities falls back to the comprehend
    # list when unset): an empty llm_redact_entities would trip the "no standard
    # entities" guard in analyze_text even for non-LLM PII methods.
    llm_entities = list(settings.chosen_llm_entities)
    if not llm_entities:
        llm_entities = list(settings.chosen_redact_comprehend_entities)

    args: dict[str, Any] = {
        "input_file": [str(pdf_path.resolve())],
        "output_dir": str(output_dir.resolve()),
        "local_redact_entities": list(settings.chosen_redact_entities),
        "aws_redact_entities": list(settings.chosen_redact_comprehend_entities),
        "llm_redact_entities": llm_entities,
        "ocr_method": settings.text_extraction_method or "Local text",
        "pii_detector": settings.pii_identification_method or "Local",
        "page_min": int(settings.page_min or 0),
        "page_max": int(settings.page_max or 0),
        "text_extraction_only": bool(settings.text_extraction_only),
        "handwrite_signature_extraction": list(settings.handwrite_signature_checkbox),
        "custom_llm_instructions": settings.custom_llm_instructions or "",
        "app_eval": True,
        "save_to_user_folders": False,
    }
    if deny:
        args["deny_list"] = deny
    if allow:
        args["allow_list"] = allow
    local_model = (settings.chosen_local_ocr_model or "").strip()
    if local_model:
        args["chosen_local_ocr_model"] = local_model
    ocr_model = (settings.chosen_ocr_model or "").strip()
    if ocr_model:
        local_key = local_model.casefold().replace("_", "-")
        if "inference-server" in local_key:
            args["inference_server_vlm_model"] = ocr_model
        else:
            args["vlm_model_choice"] = ocr_model
    pii_model = (settings.pii_llm_model or "").strip()
    if pii_model:
        pii_key = (
            (settings.pii_identification_method or "")
            .casefold()
            .replace("_", " ")
            .replace("-", " ")
        )
        if "inference server" in pii_key:
            args["inference_server_pii_model"] = pii_model
        else:
            args["llm_model_choice"] = pii_model
    # Effective inference-server VLM endpoint (set from --llm-base-url/--model in
    # main()); passed explicitly so it applies even if tools.config constants were
    # already imported before the env was updated.
    vlm_api_url = (os.environ.get("INFERENCE_SERVER_API_URL") or "").strip()
    if vlm_api_url:
        args["inference_server_api_url"] = vlm_api_url
    vlm_model = (os.environ.get("DEFAULT_INFERENCE_SERVER_VLM_MODEL") or "").strip()
    if vlm_model:
        args["inference_server_vlm_model"] = vlm_model
    return args


def _invoke_main_redact_subprocess(
    direct: dict[str, Any],
    *,
    app_config_path: str | None,
) -> int:
    """Fresh Python process so ``APP_CONFIG_PATH`` loads before ``tools.config``."""
    eval_dir = Path(__file__).resolve().parent
    child = eval_dir / "_benchmark_main_child.py"
    env = os.environ.copy()
    env["BENCHMARK_DIRECT_MODE_JSON"] = json.dumps(direct)
    if app_config_path:
        env["BENCHMARK_APP_CONFIG_PATH"] = str(
            Path(app_config_path).expanduser().resolve()
        )
    else:
        env.pop("BENCHMARK_APP_CONFIG_PATH", None)
    proc = subprocess.run(
        [sys.executable, str(child)],
        cwd=str(_repo_root()),
        env=env,
        check=False,
    )
    return int(proc.returncode or 0)


def _find_case_eval_summary(
    runs_root: Path,
    case_dir: Path,
    *,
    eval_root: Path | None = None,
) -> Path | None:
    """
    Newest eval summary belonging to *case_dir*: first scoped to the case's own
    folder under the suite runs root, then mirrored ``app_*`` runs matched on the
    ``gold_case`` recorded inside each summary (their folder names use the document,
    not the case). Never falls back to another case's summary. Returns None when
    the case produced no eval output.
    """
    case_key = case_dir.name
    candidates = sorted(
        runs_root.rglob("eval/run_*/eval_summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    scoped = [p for p in candidates if case_key in str(p)]
    if scoped:
        return scoped[-1]
    if eval_root is None:
        _, _, eval_root = _ensure_agent_imports()
    mirrored = sorted(
        (eval_root / "runs").glob("app_*/eval_summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    wanted_gold = str(case_dir.resolve())
    for candidate in reversed(mirrored):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("gold_case") or "") == wanted_gold:
            return candidate
    return None


def _run_main_surface(
    case_dir: Path,
    suite_id: str,
    runs_root: Path,
    overrides: BenchmarkStackOverrides | None = None,
) -> dict[str, Any]:
    """Redact via CLI then score with app post-run eval."""
    readiness = case_readiness(case_dir)
    record: dict[str, Any] = {
        "surface": "main",
        "case_id": readiness.case_id,
        "case_dir": str(case_dir),
        "skipped": False,
        "exit_code": 0,
        "run_dir": "",
        "eval_summary": "",
        "message": "",
    }
    if not readiness.main_surface_ready:
        record["skipped"] = True
        record["message"] = "main surface skipped: " + (
            "; ".join(readiness.notes) if readiness.notes else "not ready"
        )
        return record

    settings_path = case_dir / "app_eval_settings.json"
    settings = apply_benchmark_overrides(
        settings_from_json(str(settings_path)), overrides
    )
    pdf_path = resolve_gold_case_pdf(case_dir)
    out_dir = runs_root / "main" / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    direct = _settings_to_cli_args(settings, pdf_path=pdf_path, output_dir=out_dir)
    app_config = (settings.app_config_path or "").strip()
    if overrides and overrides.app_config_path:
        app_config = overrides.app_config_path.strip()

    prev_app_eval = os.environ.get("APP_EVAL_ENABLED")
    prev_gold = os.environ.get("APP_EVAL_GOLD")
    prev_auto_gold = os.environ.get("APP_EVAL_AUTO_GOLD")
    os.environ["APP_EVAL_ENABLED"] = "1"
    os.environ["APP_EVAL_GOLD"] = str(case_dir.resolve())
    os.environ["APP_EVAL_AUTO_GOLD"] = "0"
    try:
        if app_config:
            rc = _invoke_main_redact_subprocess(direct, app_config_path=app_config)
            if rc != 0:
                record["exit_code"] = rc
                record["message"] = f"cli_redact subprocess exited {rc}"
                return record
        else:
            repo = _repo_root()
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            import cli_redact

            cli_redact.main(direct_mode_args=direct)
    except SystemExit as exc:
        rc = int(exc.code) if isinstance(exc.code, int) else 1
        if rc != 0:
            record["exit_code"] = rc
            record["message"] = f"cli_redact exited {rc}"
            return record
    except Exception as exc:
        record["exit_code"] = 1
        record["message"] = f"cli_redact failed: {exc}"
        return record
    finally:
        if prev_app_eval is None:
            os.environ.pop("APP_EVAL_ENABLED", None)
        else:
            os.environ["APP_EVAL_ENABLED"] = prev_app_eval
        if prev_gold is None:
            os.environ.pop("APP_EVAL_GOLD", None)
        else:
            os.environ["APP_EVAL_GOLD"] = prev_gold
        if prev_auto_gold is None:
            os.environ.pop("APP_EVAL_AUTO_GOLD", None)
        else:
            os.environ["APP_EVAL_AUTO_GOLD"] = prev_auto_gold

    summary_path = _find_case_eval_summary(runs_root, case_dir)
    if summary_path is None:
        return _apply_qa_outcome(
            record,
            technical_exit=1,
            qa_pass=None,
            message=(
                f"no eval summary written for case {case_dir.name} "
                f"under {runs_root} (eval disabled or run failed)"
            ),
        )

    if summary_path.is_file():
        record["eval_summary"] = str(summary_path)
        record["run_dir"] = str(summary_path.parent)
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            msg = (
                f"gold_case={data.get('gold_case')!s}; "
                f"overall_pass={data.get('overall_pass')!s}"
            )
            if data.get("gold_case") in (None, ""):
                return _apply_qa_outcome(
                    record,
                    technical_exit=1,
                    qa_pass=None,
                    message=msg + " (gold not pinned — use APP_EVAL_GOLD)",
                )
            qa = data.get("overall_pass")
            if isinstance(qa, bool):
                qa_pass: bool | None = qa
            else:
                qa_pass = None
            return _apply_qa_outcome(
                record, technical_exit=0, qa_pass=qa_pass, message=msg
            )
        except (OSError, json.JSONDecodeError) as exc:
            return _apply_qa_outcome(
                record,
                technical_exit=1,
                qa_pass=None,
                message=f"eval_summary unreadable: {exc}",
            )
    return _apply_qa_outcome(
        record,
        technical_exit=1,
        qa_pass=None,
        message="no eval summary written under benchmark output dir",
    )


def _run_agent_surface(
    case_dir: Path,
    *,
    orchestrator: str,
    provider: str | None,
    model: str | None,
    timeout_s: float,
    suite_id: str,
    runs_root: Path,
) -> dict[str, Any]:
    readiness = case_readiness(case_dir)
    record: dict[str, Any] = {
        "surface": "agent",
        "case_id": readiness.case_id,
        "case_dir": str(case_dir),
        "skipped": False,
        "exit_code": 0,
        "run_dir": "",
        "message": "",
    }
    if not readiness.agent_surface_ready:
        record["skipped"] = True
        record["message"] = "agent surface skipped: " + (
            "; ".join(readiness.notes) if readiness.notes else "not ready"
        )
        return record

    run_dir = runs_root / "agent" / case_dir.name
    try:
        gates, out_dir = run_regression(
            gold_case_dir=case_dir,
            orchestrator=orchestrator,  # type: ignore[arg-type]
            run_dir=run_dir,
            gates_only=False,
            provider=provider,
            model=model,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        record["exit_code"] = 1
        record["message"] = str(exc)
        return record

    record["run_dir"] = str(out_dir)
    policy = gates.policy_gate or {}
    pass_strict = policy.get("pass_strict")
    gate_exit = int(gates.exit_code)
    if gate_exit >= 2:
        _apply_qa_outcome(
            record,
            technical_exit=gate_exit,
            qa_pass=None,
            message="; ".join(gates.messages) if gates.messages else "ok",
        )
    else:
        qa_pass = bool(pass_strict) if gates.policy_gate else None
        _apply_qa_outcome(
            record,
            technical_exit=0,
            qa_pass=qa_pass,
            message="; ".join(gates.messages) if gates.messages else "ok",
        )

    try:
        from eval_run_summary_csv import append_regression_run_row

        append_regression_run_row(
            gates=gates,
            run_dir=out_dir,
            gold_case_dir=case_dir,
            orchestrator=orchestrator,
            run_source="benchmark_agent",
        )
    except ImportError:
        pass

    summary_path = out_dir / "eval_summary.json"
    if not summary_path.is_file():
        policy = gates.policy_gate or {}
        pass_strict = policy.get("pass_strict")
        summary = {
            "document_name": case_dir.name,
            "gold_case": str(case_dir.resolve()),
            "workflow_incomplete": False,
            "overall_pass": gates.exit_code == 0 and bool(pass_strict),
            "metrics": [],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return record


def _run_harness_surface(
    case_dir: Path,
    suite_id: str,
    runs_root: Path,
    overrides: BenchmarkStackOverrides | None = None,
) -> dict[str, Any]:
    """Main-app Bedrock AgentCore harness (inline tools), same eval stack as Gradio agentic redact."""
    readiness = case_readiness(case_dir)
    record: dict[str, Any] = {
        "surface": "harness",
        "case_id": readiness.case_id,
        "case_dir": str(case_dir),
        "skipped": False,
        "exit_code": 0,
        "run_dir": "",
        "eval_summary": "",
        "message": "",
    }

    harness_ok, harness_reason = _harness_configured()
    if not harness_ok:
        record["skipped"] = True
        record["message"] = f"harness not configured: {harness_reason}"
        return record

    if not readiness.harness_surface_ready:
        record["skipped"] = True
        record["message"] = "harness surface skipped: " + (
            "; ".join(readiness.notes) if readiness.notes else "not ready"
        )
        return record

    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from tools.agentcore_harness import HarnessSession, run_agentic_redaction
    from tools.harness_eval_support import (
        HarnessEvalRecorder,
        harness_post_run_eval_enabled,
    )

    pdf_path = resolve_gold_case_pdf(case_dir)
    instructions = _read_case_instructions(case_dir)
    if not instructions:
        record["skipped"] = True
        record["message"] = "harness surface skipped: empty instructions.txt"
        return record

    defaults = _harness_defaults_from_gold(case_dir, overrides)
    out_root = runs_root / "harness" / case_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    session_hash = f"bench_{suite_id}_{readiness.case_id}".replace(" ", "_")[:48]
    eval_recorder: HarnessEvalRecorder | None = None
    prev_source = os.environ.get("EVAL_APPEND_RUN_SOURCE")
    os.environ["EVAL_APPEND_RUN_SOURCE"] = "benchmark_harness"

    try:
        session = HarnessSession(
            pdf_path,
            defaults=defaults,
            output_root=out_root,
        )
        record["run_dir"] = str(session.session_dir)

        eval_on = harness_post_run_eval_enabled() or bool(
            (os.environ.get("BENCHMARK_SUITE_ID") or "").strip()
        )
        if eval_on:
            eval_recorder = HarnessEvalRecorder(session_hash=session_hash)
            eval_recorder.begin(
                workspace=session.session_dir,
                pdf_path=session.source_pdf,
                document_name=session.source_pdf.name,
                instructions=instructions,
                text_extract_method=defaults.ocr_method,
                pii_method=defaults.pii_method,
                ocr_method_short=defaults.ocr_method,
            )
            eval_recorder.gold_case_dir = case_dir.resolve()

        page_range = "all"
        if defaults.page_min or defaults.page_max:
            page_range = f"{defaults.page_min or 1}-{defaults.page_max or 'end'}"

        final = None
        for progress in run_agentic_redaction(
            pdf_path,
            instructions,
            defaults=defaults,
            page_range=page_range,
            output_root=out_root,
            session=session,
            eval_recorder=eval_recorder,
            eval_session_hash=session_hash,
        ):
            final = progress

        if final is None:
            record["exit_code"] = 1
            record["message"] = "harness run produced no progress"
            return record

        if not final.done:
            record["exit_code"] = 1
            record["message"] = final.summary or "harness run did not complete"
            return record

        record["message"] = final.summary or "ok"
        eval_root = session.session_dir / "eval"
        summary_path: Path | None = None
        if eval_root.is_dir():
            candidates = sorted(eval_root.glob("run_*/eval_summary.json"))
            if candidates:
                summary_path = candidates[-1]
        if summary_path and summary_path.is_file():
            record["eval_summary"] = str(summary_path)
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if data.get("workflow_incomplete"):
                    _apply_qa_outcome(
                        record,
                        technical_exit=1,
                        qa_pass=None,
                        message=record.get("message", "") + "; workflow_incomplete",
                    )
                else:
                    op = data.get("overall_pass")
                    qa = op if isinstance(op, bool) else None
                    _apply_qa_outcome(
                        record,
                        technical_exit=0,
                        qa_pass=qa,
                        message=record.get("message", ""),
                    )
            except (OSError, json.JSONDecodeError):
                pass
        elif eval_recorder is not None:
            _apply_qa_outcome(
                record,
                technical_exit=1,
                qa_pass=None,
                message=record.get("message", "")
                + "; no eval_summary.json (eval disabled or failed)",
            )
    except Exception as exc:
        record["exit_code"] = 1
        record["message"] = f"harness run failed: {exc}"
    finally:
        if prev_source is None:
            os.environ.pop("EVAL_APPEND_RUN_SOURCE", None)
        else:
            os.environ["EVAL_APPEND_RUN_SOURCE"] = prev_source

    return record


def run_benchmark_suite(
    *,
    gold_root: Path,
    surfaces: set[SurfaceChoice],
    orchestrator: str,
    provider: str | None,
    model: str | None,
    timeout_s: float,
    suite_id: str | None,
    allow_fail: bool,
    fail_on_qa: bool,
    case_filter: str | None,
    stack_overrides: BenchmarkStackOverrides | None = None,
) -> tuple[dict[str, Any], int]:
    git_commit = _git_commit_short()
    suite_id = suite_id or _new_suite_id()
    _set_benchmark_csv_env(suite_id, git_commit)

    _, _, eval_dir = _ensure_agent_imports()
    runs_root = resolve_benchmark_runs_root(eval_dir, suite_id)
    outputs_dir = eval_dir / "outputs"
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        outputs_dir = runs_root.parent

    cases = discover_benchmark_gold_cases(gold_root)
    if case_filter:
        cases = [
            c
            for c in cases
            if c.name == case_filter or case_readiness(c).case_id == case_filter
        ]
        if not cases:
            raise FileNotFoundError(
                f"No benchmark case named {case_filter!r} under {gold_root}"
            )

    envelope: dict[str, Any] = {
        "benchmark_suite_id": suite_id,
        "git_commit": git_commit,
        "gold_root": str(gold_root.resolve()),
        "surfaces": sorted(surfaces),
        "orchestrator": orchestrator,
        "stack_overrides": asdict(stack_overrides) if stack_overrides else {},
        "doc_redaction_gradio_url": (
            os.environ.get("DOC_REDACTION_GRADIO_URL") or ""
        ).strip()
        or None,
        "agent_llm_base_url": (os.environ.get("AGENT_LLAMA_BASE_URL") or "").strip()
        or None,
        "inference_server_api_url": (
            os.environ.get("INFERENCE_SERVER_API_URL") or ""
        ).strip()
        or None,
        "inference_server_vlm_model": (
            os.environ.get("DEFAULT_INFERENCE_SERVER_VLM_MODEL") or ""
        ).strip()
        or None,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [],
        "exit_code": 0,
    }

    max_technical = 0
    any_qa_fail = False
    for case_dir in cases:
        case_report: dict[str, Any] = {
            "case_id": case_readiness(case_dir).case_id,
            "case_dir": str(case_dir),
            "surfaces": {},
        }
        if "agent" in surfaces:
            agent_rec = _run_agent_surface(
                case_dir,
                orchestrator=orchestrator,
                provider=provider,
                model=model,
                timeout_s=timeout_s,
                suite_id=suite_id,
                runs_root=runs_root,
            )
            case_report["surfaces"]["agent"] = agent_rec
            max_technical, any_qa_fail = _accumulate_surface_outcome(
                agent_rec, max_technical, any_qa_fail
            )

        if "main" in surfaces:
            main_rec = _run_main_surface(case_dir, suite_id, runs_root, stack_overrides)
            case_report["surfaces"]["main"] = main_rec
            max_technical, any_qa_fail = _accumulate_surface_outcome(
                main_rec, max_technical, any_qa_fail
            )

        if "harness" in surfaces:
            harness_rec = _run_harness_surface(
                case_dir, suite_id, runs_root, stack_overrides
            )
            case_report["surfaces"]["harness"] = harness_rec
            max_technical, any_qa_fail = _accumulate_surface_outcome(
                harness_rec, max_technical, any_qa_fail
            )

        envelope["cases"].append(case_report)

    envelope["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    envelope["qa_any_fail"] = any_qa_fail
    if allow_fail:
        suite_exit = 0
    elif fail_on_qa and any_qa_fail and max_technical == 0:
        suite_exit = 1
    else:
        suite_exit = max_technical
    envelope["exit_code"] = suite_exit

    out_json = outputs_dir / f"{suite_id}.json"
    out_json.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    envelope["report_path"] = str(out_json)
    return envelope, envelope["exit_code"]


def _write_gold_audit(gold_root: Path, dest: Path) -> int:
    reports = audit_benchmark_gold_cases(gold_root)
    lines = [
        "# Benchmark gold audit",
        "",
        f"Gold root: `{gold_root.resolve()}`",
        "",
        "| case_id | benchmark_eligible | agent | harness | main | notes |",
        "|---------|-------------------|-------|---------|------|-------|",
    ]
    for row in reports:
        notes = "; ".join(row.notes[:3])
        if len(row.notes) > 3:
            notes += f" (+{len(row.notes) - 3} more)"
        lines.append(
            f"| {row.case_id} | {row.benchmark_eligible} | "
            f"{row.agent_surface_ready} | {row.harness_surface_ready} | "
            f"{row.main_surface_ready} | {notes} |"
        )
    lines.extend(
        [
            "",
            "Regenerate: `python agent-redact/eval/run_benchmark_suite.py --audit-only`",
            "",
        ]
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit: {dest}")
    return 0


class _RecordExplicitFlag(argparse.Action):
    """Record the option string on ``namespace._explicit_flags`` when a flag is
    given on the command line, so env-file defaults can be told apart from flags
    the user actually typed."""

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[no-untyped-def]
        flags = getattr(namespace, "_explicit_flags", None)
        if flags is None:
            flags = set()
            setattr(namespace, "_explicit_flags", flags)
        if option_string:
            flags.add(option_string)
        setattr(namespace, self.dest, values)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gold-root",
        type=Path,
        default=None,
        help="Gold cases root (default: APP_EVAL_GOLD_ROOT / AGENT_EVAL_GOLD_ROOT).",
    )
    p.add_argument(
        "--surfaces",
        default="agent,main",
        help="Comma-separated: agent (LangGraph/Pi), harness (AgentCore), main (default: agent,main).",
    )
    p.add_argument(
        "--orchestrator",
        choices=("langgraph", "pi", "both"),
        default=_env_default("BENCHMARK_ORCHESTRATOR", "AGENT_ORCHESTRATOR")
        or "langgraph",
        help="Agent orchestrator (default: $BENCHMARK_ORCHESTRATOR / $AGENT_ORCHESTRATOR / langgraph).",
    )
    p.add_argument(
        "--provider",
        default=_env_default("BENCHMARK_PROVIDER", "AGENT_DEFAULT_PROVIDER"),
        help="Agent LLM provider (default: $BENCHMARK_PROVIDER / $AGENT_DEFAULT_PROVIDER).",
    )
    p.add_argument(
        "--model",
        default=_env_default("BENCHMARK_MODEL", "AGENT_DEFAULT_MODEL"),
        help="Agent LLM model (default: $BENCHMARK_MODEL / $AGENT_DEFAULT_MODEL).",
    )
    p.add_argument(
        "--gradio-url",
        default=_env_default("BENCHMARK_GRADIO_URL", "DOC_REDACTION_GRADIO_URL"),
        help="URL of the running main doc_redaction app that the agent surface drives "
        "via gradio_client (sets DOC_REDACTION_GRADIO_URL). Default: "
        "$BENCHMARK_GRADIO_URL / $DOC_REDACTION_GRADIO_URL from the environment or "
        "config/agent.env; the app itself falls back to http://127.0.0.1:7860.",
    )
    p.add_argument(
        "--llm-base-url",
        default=_env_default(
            "BENCHMARK_LLM_BASE_URL", "AGENT_LLAMA_BASE_URL", "OPENAI_BASE_URL"
        ),
        help="Base URL of the OpenAI-compatible inference server serving the agent "
        "orchestrator LLM, e.g. the llama.cpp server for --provider llama-cpp "
        "(sets AGENT_LLAMA_BASE_URL; /v1 appended if missing, and OPENAI_BASE_URL "
        "if unset). Default: $BENCHMARK_LLM_BASE_URL / $AGENT_LLAMA_BASE_URL / "
        "$OPENAI_BASE_URL from the environment or config/agent.env.",
    )
    p.add_argument(
        "--ocr-method",
        action=_RecordExplicitFlag,
        default=_env_default("BENCHMARK_OCR_METHOD", "DEFAULT_TEXT_EXTRACTION_MODEL"),
        help="Override gold app_eval text_extraction_method / harness OCR for ALL cases "
        "(default: $BENCHMARK_OCR_METHOD / $DEFAULT_TEXT_EXTRACTION_MODEL from the env "
        "or app config file).",
    )
    p.add_argument(
        "--pii-method",
        action=_RecordExplicitFlag,
        default=_env_default("BENCHMARK_PII_METHOD", "DEFAULT_PII_DETECTION_MODEL"),
        help="Override gold pii_identification_method for ALL cases "
        "(default: $BENCHMARK_PII_METHOD / $DEFAULT_PII_DETECTION_MODEL).",
    )
    p.add_argument(
        "--pii-llm-model",
        action=_RecordExplicitFlag,
        default=_env_default("BENCHMARK_PII_LLM_MODEL", "CLOUD_LLM_PII_MODEL_CHOICE"),
        help="Override the PII-detector LLM model (gold pii_llm_model) for ALL cases, "
        "e.g. the Bedrock / inference-server / local transformers model used when the "
        "PII method is an LLM route (default: $BENCHMARK_PII_LLM_MODEL / "
        "$CLOUD_LLM_PII_MODEL_CHOICE).",
    )
    p.add_argument(
        "--chosen-local-ocr-model",
        action=_RecordExplicitFlag,
        default=_env_default("BENCHMARK_LOCAL_OCR_MODEL", "DEFAULT_LOCAL_OCR_MODEL"),
        help="Override local OCR engine (e.g. paddle, tesseract) for ALL cases; also "
        "switches text extraction to the local OCR option unless --ocr-method is given "
        "(default: $BENCHMARK_LOCAL_OCR_MODEL / $DEFAULT_LOCAL_OCR_MODEL).",
    )
    p.add_argument(
        "--chosen-ocr-model",
        action=_RecordExplicitFlag,
        default=_env_default("BENCHMARK_OCR_MODEL", "CLOUD_VLM_MODEL_CHOICE"),
        help="Override the VLM / inference-server / hybrid second-pass OCR model "
        "(gold chosen_ocr_model) for ALL cases, e.g. for hybrid-paddle-vlm or "
        "hybrid-paddle-inference-server (default: $BENCHMARK_OCR_MODEL / "
        "$CLOUD_VLM_MODEL_CHOICE).",
    )
    p.add_argument(
        "--app-config-path",
        action=_RecordExplicitFlag,
        default=_env_default("BENCHMARK_APP_CONFIG_PATH", "APP_CONFIG_PATH"),
        help="Load this env file in a fresh process for main-app legs "
        "(default: $BENCHMARK_APP_CONFIG_PATH / $APP_CONFIG_PATH).",
    )
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--suite-id", default=None, help="Override benchmark suite id.")
    p.add_argument(
        "--case", dest="case_filter", help="Run a single case id / folder name."
    )
    p.add_argument(
        "--allow-fail",
        action="store_true",
        help="Always exit 0 (ignores technical and QA failures).",
    )
    p.add_argument(
        "--fail-on-qa",
        action="store_true",
        help="Exit 1 when eval/gates score poorly (default: only technical failures fail).",
    )
    p.add_argument(
        "--audit-only",
        action="store_true",
        help="Write gold/BENCHMARK_GOLD_AUDIT.md and exit (no LLM runs).",
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="Print benchmark-eligible cases and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    gold_root = (args.gold_root or default_gold_root()).resolve()

    if args.audit_only:
        _, _, eval_dir = _ensure_agent_imports()
        return _write_gold_audit(
            gold_root, eval_dir / "gold" / "BENCHMARK_GOLD_AUDIT.md"
        )

    if args.list_cases:
        for case_dir in discover_benchmark_gold_cases(gold_root):
            r = case_readiness(case_dir)
            print(
                f"{r.case_id}\tagent={r.agent_surface_ready}"
                f"\tharness={r.harness_surface_ready}\tmain={r.main_surface_ready}"
            )
        return 0

    try:
        surfaces = _parse_surfaces(args.surfaces)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.llm_base_url:
        url_error = _validate_llm_base_url(args.llm_base_url)
        if url_error:
            print(url_error, file=sys.stderr)
            return 2

    # An explicitly typed --llm-base-url also overrides a stale OPENAI_BASE_URL
    # (shell/config), which the DeepEval judge prefers when USE_OPENAI_MODEL=1.
    raw_argv = argv if argv is not None else sys.argv[1:]
    llm_base_url_explicit = "--llm-base-url" in {
        tok.split("=", 1)[0] for tok in raw_argv if tok.startswith("--")
    }

    _apply_gradio_url_env(args.gradio_url)
    _apply_llm_base_url_env(
        args.llm_base_url, force_openai_base_url=llm_base_url_explicit
    )
    _apply_vlm_endpoint_env(args.llm_base_url, args.model)

    try:
        explicit_flags = getattr(args, "_explicit_flags", None) or set()
        stack_overrides = overrides_from_argv(
            ocr_method=args.ocr_method,
            pii_method=args.pii_method,
            chosen_local_ocr_model=args.chosen_local_ocr_model,
            app_config_path=args.app_config_path,
            chosen_ocr_model=args.chosen_ocr_model,
            pii_llm_model=args.pii_llm_model,
            ocr_method_explicit="--ocr-method" in explicit_flags,
            local_ocr_model_explicit="--chosen-local-ocr-model" in explicit_flags,
        )
        envelope, code = run_benchmark_suite(
            gold_root=gold_root,
            surfaces=surfaces,
            orchestrator=args.orchestrator,
            provider=args.provider,
            model=args.model,
            timeout_s=args.timeout_s,
            suite_id=args.suite_id,
            allow_fail=bool(args.allow_fail),
            fail_on_qa=bool(args.fail_on_qa),
            case_filter=args.case_filter,
            stack_overrides=stack_overrides,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(envelope, indent=2))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
