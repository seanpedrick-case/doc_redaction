#!/usr/bin/env python3
"""Package doc_redaction LangGraph agent code into an AgentCore app folder."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import tomllib

_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache"
)

# Runtime Python deps to merge into the AgentCore app's pyproject.toml (not full pi-agent stack).
RUNTIME_DEPENDENCIES: dict[str, str] = {
    "gradio_client": ">=1.0.0",
    "httpx": ">=0.28.0",
    "python-dotenv": ">=1.0.0",
    "langchain-openai": ">=1.0.0",
    "langchain-core": ">=1.0.0",
    "langgraph": ">=1.0.2",
    "langchain-aws": ">=1.0.0",
    "pymupdf": ">=1.24.0",
    "pandas": ">=2.0.0",
}

MAIN_PY = '''"""doc_redaction LangGraph agent — packaged by agent-redact/agentcore/package_runtime.py."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent
_PI_DIR = _APP_ROOT / "pi"
for path in (_APP_ROOT, _PI_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from invoke_agent import bootstrap_runtime_env, invoke_redaction_agent  # noqa: E402

bootstrap_runtime_env(_APP_ROOT)

from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402

app = BedrockAgentCoreApp()


@app.entrypoint
async def handler(request: dict):
  async for event in invoke_redaction_agent(request):
    yield event


if __name__ == "__main__":
    app.run()
'''


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_agentcore_app() -> Path:
    return _repo_root() / "agent-redact" / "RedactionAgent" / "app" / "RedactionAgent"


def _rmtree_robust(path: Path) -> None:
    """Remove a directory tree on Windows / OneDrive (clears read-only files first)."""

    def _on_rm_error(func, location, _exc_info) -> None:
        os.chmod(location, stat.S_IWRITE)
        func(location)

    shutil.rmtree(path, onerror=_on_rm_error)


def _copy_tree(src: Path, dest: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  copy tree {src} -> {dest}")
        return
    if dest.exists():
        _rmtree_robust(dest)
    shutil.copytree(src, dest, ignore=_COPY_IGNORE)


def _copy_file(src: Path, dest: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  copy file {src} -> {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _replace_dependencies_block(text: str, deps: list[str]) -> str:
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith("dependencies"):
            out.append("dependencies = [")
            for dep in deps:
                out.append(f'    "{dep}",')
            out.append("]")
            index += 1
            while index < len(lines) and lines[index].strip() != "]":
                index += 1
            index += 1
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _merge_pyproject(pyproject_path: Path, *, dry_run: bool) -> None:
    text = pyproject_path.read_text(encoding="utf-8")
    if dry_run:
        print(f"  merge deps into {pyproject_path}")
        return
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"Could not parse {pyproject_path}: {exc}") from exc

    existing: dict[str, str] = {}
    for item in data.get("project", {}).get("dependencies", []):
        if isinstance(item, str):
            name = re.split(r"[<>=!~\[]", item, maxsplit=1)[0].strip()
            existing[name.lower()] = item

    for name, spec in RUNTIME_DEPENDENCIES.items():
        key = name.lower()
        if key not in existing:
            existing[key] = f"{name}{spec}"

    merged = [existing[k] for k in sorted(existing, key=str.lower)]
    pyproject_path.write_text(
        _replace_dependencies_block(text, merged), encoding="utf-8"
    )


def package_runtime(
    target: Path,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Sync monorepo redaction agent sources into *target* (AgentCore app folder)."""
    repo = _repo_root()
    agent_redact = repo / "agent-redact"
    agentcore = agent_redact / "agentcore"
    actions: list[str] = []

    def log(msg: str) -> None:
        actions.append(msg)
        print(msg)

    log(f"Packaging doc_redaction runtime -> {target}")

    _copy_tree(
        agent_redact / "redaction_langgraph",
        target / "redaction_langgraph",
        dry_run=dry_run,
    )

    pi_dest = target / "pi"
    for name in ("remote_redaction.py",):
        _copy_file(agent_redact / "pi" / name, pi_dest / name, dry_run=dry_run)

    _copy_file(
        agentcore / "bundle_support" / "session_workspace.py",
        pi_dest / "session_workspace.py",
        dry_run=dry_run,
    )

    for module in ("invoke_agent.py", "session_store.py", "workspace_sync.py"):
        _copy_file(agentcore / module, target / module, dry_run=dry_run)

    if dry_run:
        log(f"  write {target / 'main.py'}")
    else:
        (target / "main.py").write_text(MAIN_PY, encoding="utf-8")
        log(f"wrote {target / 'main.py'}")

    pyproject = target / "pyproject.toml"
    if pyproject.is_file():
        _merge_pyproject(pyproject, dry_run=dry_run)
        log(f"merged runtime dependencies into {pyproject}")
    elif dry_run:
        log(f"  skip pyproject merge (no {pyproject} — run agentcore create first)")
    else:
        raise SystemExit(f"Missing {pyproject} — run agentcore create first.")

    env_example = target / "agentcore.env.example"
    env_local = target / "agentcore.env"
    example_text = """# Loaded at runtime startup when present in the CodeZip (see invoke_agent.bootstrap_runtime_env).
# Also set these on the AgentCore runtime in AWS if you prefer console/config-bundle env.
# CDK + AgentCore: use main Express HTTPS (ExpressServiceEndpoint), not Service Connect.
DOC_REDACTION_GRADIO_URL=https://your-doc-redaction-host.example
PI_DEFAULT_PROVIDER=amazon-bedrock
PI_DEFAULT_MODEL=anthropic.claude-sonnet-4-6
AWS_REGION=eu-west-2
PI_WORKSPACE_DIR=/tmp/agentcore-workspace
PI_DEFAULT_OCR_METHOD=paddle
PI_DEFAULT_PII_METHOD=Local
"""
    if dry_run:
        log(f"  write {env_example}")
        if env_local.is_file():
            log(f"  keep existing {env_local}")
    else:
        env_example.write_text(example_text, encoding="utf-8")
        log(f"wrote {env_example}")
        if not env_local.is_file():
            env_local.write_text(example_text, encoding="utf-8")
            log(f"wrote {env_local} (copy from example — edit before deploy)")

    return actions


def run_deploy(agentcore_project: Path) -> None:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env.setdefault("UV_LINK_MODE", "copy")
    subprocess.run(
        ["agentcore", "deploy"],
        cwd=str(agentcore_project),
        check=True,
        env=env,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package doc_redaction LangGraph agent into an AgentCore app folder.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=_default_agentcore_app(),
        help="AgentCore app folder (default: agent-redact/RedactionAgent/app/RedactionAgent)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Run agentcore deploy from the RedactionAgent project after packaging",
    )
    args = parser.parse_args(argv)

    target = args.target.resolve()
    package_runtime(target, dry_run=args.dry_run)

    if args.deploy:
        if args.dry_run:
            print("Skipping deploy (--dry-run).")
            return 0
        project = target.parent.parent
        if not (project / "agentcore" / "agentcore.json").is_file():
            raise SystemExit(f"Not an AgentCore project: {project}")
        print(f"Running agentcore deploy in {project} ...")
        run_deploy(project)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
