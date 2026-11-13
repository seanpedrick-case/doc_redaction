#!/usr/bin/env python3
"""
Deploy the minimal AgentCore Harness stack for a locally running ``app.py``.

Usage (from repo root or ``cdk/``)::

    python cdk/deploy_harness_local.py --profile your-aws-profile

After deploy, attach the printed managed policy to your IAM user/role, merge the
``LocalAppConfigSnippet`` into ``config/app_config.env``, then start the app with
``USE_AGENTCORE=True``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

CDK_DIR = Path(__file__).resolve().parent
REPO_ROOT = CDK_DIR.parent


def _run(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _sts_identity(profile: str, region: str) -> tuple[str, str]:
    import boto3

    session = boto3.Session(profile_name=profile or None, region_name=region or None)
    ident = session.client("sts").get_caller_identity()
    return str(ident["Account"]), str(ident.get("Arn", ""))


def _stack_outputs(stack_name: str, region: str, profile: str) -> Dict[str, str]:
    import boto3

    session = boto3.Session(profile_name=profile or None, region_name=region)
    cf = session.client("cloudformation")
    resp = cf.describe_stacks(StackName=stack_name)
    out: Dict[str, str] = {}
    for item in resp["Stacks"][0].get("Outputs") or []:
        key = item.get("OutputKey") or ""
        if key:
            out[key] = item.get("OutputValue") or ""
    return out


def _parse_snippet(snippet: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in snippet.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip()
    return values


def _patch_app_config(updates: Dict[str, str], *, dry_run: bool) -> None:
    path = REPO_ROOT / "config" / "app_config.env"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    seen = set()
    new_lines: List[str] = []
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        new_lines.append(line)
    for key, value in sorted(updates.items()):
        if key not in seen:
            new_lines.append(f"{key}={value}")
    merged = "\n".join(new_lines).rstrip() + "\n"
    if dry_run:
        print("\n--- Would write config/app_config.env ---\n")
        print(merged)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged, encoding="utf-8")
    print(f"Updated {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", ""))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-2"))
    parser.add_argument(
        "--destroy",
        action="store_true",
        help="Delete the harness-only stack",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip cdk deploy confirmation",
    )
    parser.add_argument(
        "--write-app-config",
        action="store_true",
        help="Merge stack outputs into config/app_config.env after deploy",
    )
    parser.add_argument(
        "--dry-run-app-config",
        action="store_true",
        help="With --write-app-config, print the merged file only",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(CDK_DIR))
    from harness_local_config import STACK_NAME  # noqa: E402

    account, principal_arn = _sts_identity(args.profile, args.region)
    print(f"Account: {account}")
    print(f"Principal: {principal_arn}")
    print(f"Region: {args.region}")
    print(f"Stack: {STACK_NAME}")

    deploy_env = os.environ.copy()
    deploy_env["CDK_DEFAULT_ACCOUNT"] = account
    deploy_env["CDK_DEFAULT_REGION"] = args.region
    deploy_env["AWS_ACCOUNT_ID"] = account
    deploy_env["AWS_REGION"] = args.region
    if args.profile:
        deploy_env["AWS_PROFILE"] = args.profile

    app_cmd = f"{sys.executable} app_harness_local.py"
    cdk_args = [
        "cdk",
        "destroy" if args.destroy else "deploy",
        STACK_NAME,
        "-a",
        app_cmd,
    ]
    if args.yes:
        cdk_args.extend(["--require-approval", "never"])

    try:
        _run(cdk_args, cwd=CDK_DIR, env=deploy_env)
    except subprocess.CalledProcessError as exc:
        print(
            "\nDeploy failed. Bootstrap once if needed:\n"
            f"  cdk bootstrap aws://{account}/{args.region}\n",
            file=sys.stderr,
        )
        return exc.returncode or 1

    if args.destroy:
        return 0

    outputs = _stack_outputs(STACK_NAME, args.region, args.profile)

    print("\n--- Stack outputs ---")
    for key in sorted(outputs):
        print(f"  {key}: {outputs[key]}")

    snippet = outputs.get("LocalAppConfigSnippet", "")
    if snippet:
        print("\n--- Add to config/app_config.env ---\n")
        print(snippet)
        print(
            "\nAttach IAM managed policy to your principal:\n"
            f"  {outputs.get('LocalInvokerPolicyName', 'DocRedactionHarnessLocalInvoker-...')}\n"
        )
    if args.write_app_config and snippet:
        _patch_app_config(_parse_snippet(snippet), dry_run=args.dry_run_app_config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
