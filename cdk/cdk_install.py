#!/usr/bin/env python3
"""
Interactive CDK installer for doc_redaction.

Walks through demo vs production deployment profiles, writes config/cdk_config.env
and cdk.json, optionally runs cdk deploy, post-deploy Cognito callback fixups (API,
no second deploy), and post_cdk_build_quickstart.py.

Usage examples::

    # Full interactive install + deploy + quickstart
    python cdk_install.py

    # Demo sandbox, config only
    python cdk_install.py --profile demo --config-only --yes

    # Production with flags (partial non-interactive)
    python cdk_install.py --profile production --vpc-name my-vpc \\
        --cert-arn arn:aws:acm:eu-west-2:123:certificate/abc \\
        --domain redaction.example.com --yes

    # Redeploy using existing config, skip quickstart
    python cdk_install.py --deploy-only --skip-quickstart

    # Remove existing stacks before a clean install (non-interactive)
    python cdk_install.py --profile headless --vpc-name my-vpc \\
        --force-delete-stacks --yes

    # Demo with Agent agent (Express) at /agent/
    python cdk_install.py --profile demo --enable-pi --yes --config-only

    # Headless batch (S3 → Lambda → one-shot ECS direct mode)
    python cdk_install.py --profile headless --vpc-name my-vpc --yes
    python cdk_install.py --profile production --headless --vpc-name my-vpc --yes

    # Production with Pi agent mode on dedicated hostname
    python cdk_install.py --profile production --enable-pi-legacy \\
        --pi-alb-routing host --pi-host-header agent.example.com --yes
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CDK_DIR = Path(__file__).resolve().parent
REPO_ROOT = CDK_DIR.parent
CONFIG_DIR = CDK_DIR / "config"
ENV_PATH = CONFIG_DIR / "cdk_config.env"
APP_CONFIG_ENV_PATH = CONFIG_DIR / "app_config.env"
APP_CONFIG_ENV_EXAMPLE = CONFIG_DIR / "app_config.env.example"
PI_AGENT_ENV_PATH = CONFIG_DIR / "pi_agent.env"
PI_AGENT_ENV_EXAMPLE = REPO_ROOT / "config" / "pi_agent.env.example"
PI_ALB_ROUTING_MODES = ("path", "host", "both")
CDK_JSON_PATH = CDK_DIR / "cdk.json"
CDK_JSON_EXAMPLE = CDK_DIR / "cdk.json.example"
QUICKSTART_SCRIPT = CDK_DIR / "post_cdk_build_quickstart.py"

REGIONAL_STACK = "RedactionStack"
CLOUDFRONT_STACK = "RedactionStackCloudfront"
CLOUDFRONT_STACK_REGION = "us-east-1"
APPREGISTRY_STACK_SUFFIX = "AppRegistryStack"

DEMO_PRESET: Dict[str, str] = {
    "USE_ECS_EXPRESS_MODE": "True",
    "ECS_EXPRESS_USE_PUBLIC_SUBNETS": "True",
    "ENABLE_ECS_VPC_INTERFACE_ENDPOINTS": "True",
    "USE_CLOUDFRONT": "False",
    "RUN_USEAST_STACK": "False",
    "ENABLE_RESOURCE_DELETE_PROTECTION": "False",
    "ENABLE_APPREGISTRY": "True",
    "ACM_SSL_CERTIFICATE_ARN": "",
    "SSL_CERTIFICATE_DOMAIN": "",
}

PRODUCTION_PRESET: Dict[str, str] = {
    "USE_ECS_EXPRESS_MODE": "False",
    "USE_CLOUDFRONT": "True",
    "RUN_USEAST_STACK": "True",
    "ENABLE_RESOURCE_DELETE_PROTECTION": "True",
    "ENABLE_APPREGISTRY": "True",
}

HEADLESS_PRESET: Dict[str, str] = {
    "USE_ECS_EXPRESS_MODE": "False",
    "USE_CLOUDFRONT": "False",
    "RUN_USEAST_STACK": "False",
    "ENABLE_RESOURCE_DELETE_PROTECTION": "False",
    "ENABLE_APPREGISTRY": "True",
    "ENABLE_HEADLESS_DEPLOYMENT": "True",
    "ENABLE_S3_BATCH_ECS_TRIGGER": "True",
    "COGNITO_AUTH": "False",
    "ACM_SSL_CERTIFICATE_ARN": "",
    "SSL_CERTIFICATE_DOMAIN": "",
}

DEFAULT_CDK_JSON_CONTEXT: Dict[str, Any] = {
    "@aws-cdk/aws-ec2:restrictDefaultSecurityGroup": False,
}


# ---------------------------------------------------------------------------
# Wizard helpers
# ---------------------------------------------------------------------------


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} ({default_label}): ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter y or n.")


def ask_choice(prompt: str, options: Sequence[str], default_index: int = 0) -> int:
    print(prompt)
    for i, opt in enumerate(options):
        marker = "*" if i == default_index else " "
        print(f"  {marker} {i + 1}) {opt}")
    while True:
        raw = input(
            f"Choice [1-{len(options)}] (default {default_index + 1}): "
        ).strip()
        if not raw:
            return default_index
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("Invalid choice.")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def backup_file(path: Path) -> Optional[Path]:
    if not path.is_file():
        return None
    backup = path.with_suffix(path.suffix + f".bak.{timestamp()}")
    shutil.copy2(path, backup)
    print(f"Backed up {path.name} -> {backup.name}")
    return backup


# ---------------------------------------------------------------------------
# Python / cdk.json
# ---------------------------------------------------------------------------


def _venv_python_paths() -> List[Path]:
    candidates: List[Path] = []
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        root = Path(virtual_env)
        if os.name == "nt":
            candidates.append(root / "Scripts" / "python.exe")
        else:
            candidates.append(root / "bin" / "python")

    rel_venvs = [
        CDK_DIR / ".venv",
        REPO_ROOT / ".venv",
        REPO_ROOT.parent / ".venv",
    ]
    for venv_root in rel_venvs:
        if os.name == "nt":
            candidates.append(venv_root / "Scripts" / "python.exe")
        else:
            candidates.append(venv_root / "bin" / "python")

    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    seen: set[str] = set()
    unique: List[Path] = []
    for p in [Path(sys.executable)] + candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _python_has_aws_cdk(python_exe: Path) -> Tuple[bool, str]:
    if not python_exe.is_file():
        return False, "not found"
    cmd = [
        str(python_exe),
        "-c",
        "import aws_cdk; print(getattr(aws_cdk, '__version__', 'unknown'))",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(CDK_DIR),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            version = (result.stdout or "").strip().splitlines()[-1]
            return True, version
        return False, (result.stderr or result.stdout or "import failed").strip()[:200]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def discover_python_candidates() -> List[Tuple[Path, str]]:
    found: List[Tuple[Path, str]] = []
    for candidate in _venv_python_paths():
        ok, detail = _python_has_aws_cdk(candidate)
        if ok:
            found.append((candidate.resolve(), detail))
    return found


def resolve_python_executable(
    override: Optional[str] = None,
    interactive: bool = True,
    assume_yes: bool = False,
) -> Path:
    if override:
        path = Path(override).resolve()
        ok, detail = _python_has_aws_cdk(path)
        if not ok:
            raise SystemExit(
                f"Python at {path} cannot import aws-cdk-lib: {detail}\n"
                "Install CDK deps: pip install -r requirements.txt"
            )
        print(f"Using Python: {path} (aws-cdk-lib {detail})")
        return path

    candidates = discover_python_candidates()
    if not candidates:
        raise SystemExit(
            "No Python interpreter with aws-cdk-lib found.\n"
            "Activate your venv and run: pip install -r requirements.txt\n"
            "Or pass --python /path/to/python"
        )

    if len(candidates) == 1 or assume_yes or not interactive:
        path, version = candidates[0]
        print(f"Using Python: {path} (aws-cdk-lib {version})")
        return path

    print("Python interpreters with aws-cdk-lib:")
    labels = [f"{p} (aws-cdk-lib {v})" for p, v in candidates]
    idx = ask_choice("Select Python for cdk.json", labels, default_index=0)
    path, version = candidates[idx]
    print(f"Selected: {path} (aws-cdk-lib {version})")
    return path


def format_cdk_app_command(python_exe: Path) -> str:
    """Format the cdk.json ``app`` command for the current OS."""
    py = str(python_exe)
    if os.name == "nt":
        py = py.replace("/", "\\")
    return f"{py} app.py"


def load_cdk_json_context(existing_path: Path) -> Dict[str, Any]:
    if existing_path.is_file():
        try:
            data = json.loads(existing_path.read_text(encoding="utf-8"))
            ctx = data.get("context")
            if isinstance(ctx, dict):
                return dict(ctx)
        except json.JSONDecodeError:
            pass
    if CDK_JSON_EXAMPLE.is_file():
        try:
            data = json.loads(CDK_JSON_EXAMPLE.read_text(encoding="utf-8"))
            ctx = data.get("context")
            if isinstance(ctx, dict):
                return dict(ctx)
        except json.JSONDecodeError:
            pass
    return dict(DEFAULT_CDK_JSON_CONTEXT)


def write_cdk_json(
    python_exe: Path,
    *,
    force: bool = False,
    skip: bool = False,
) -> Optional[Path]:
    if skip:
        if not CDK_JSON_PATH.is_file():
            raise SystemExit(
                f"{CDK_JSON_PATH.name} not found; cannot use --skip-cdk-json."
            )
        print(f"Keeping existing {CDK_JSON_PATH.name} (--skip-cdk-json)")
        return CDK_JSON_PATH

    if CDK_JSON_PATH.is_file() and force:
        backup_file(CDK_JSON_PATH)
    elif not CDK_JSON_PATH.is_file():
        pass
    else:
        # Update app only — backup first
        backup_file(CDK_JSON_PATH)

    context = load_cdk_json_context(CDK_JSON_PATH)
    payload = {
        "app": format_cdk_app_command(python_exe),
        "output": "cdk.out",
        "context": context,
    }
    CDK_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {CDK_JSON_PATH}")
    return CDK_JSON_PATH


def build_cdk_subprocess_env(
    values: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Environment for child Python / CDK processes.

    Merges config/cdk_config.env over the current process environment so stale
    empty defaults from an earlier cdk_config import during the wizard do not
    block load_dotenv (override=False) in app.py.
    """
    env = os.environ.copy()
    env["CDK_CONFIG_PATH"] = str(ENV_PATH)
    cdk_folder = ""
    if values:
        cdk_folder = (values.get("CDK_FOLDER") or "").strip()
    if not cdk_folder and ENV_PATH.is_file():
        cdk_folder = (read_env_file(ENV_PATH).get("CDK_FOLDER") or "").strip()
    if not cdk_folder:
        cdk_folder = str(CDK_DIR).replace("\\", "/") + "/"
    env["CDK_FOLDER"] = cdk_folder

    file_values: Dict[str, str] = {}
    if ENV_PATH.is_file():
        file_values = read_env_file(ENV_PATH)
    elif values:
        file_values = dict(values)

    for key, val in file_values.items():
        env[key] = str(val)
    return env


def apply_cdk_runtime_env(values: Dict[str, str]) -> None:
    """Expose written config to app.py imports (smoke test, cdk synth/deploy)."""
    os.environ.update(build_cdk_subprocess_env(values))


def smoke_test_python_app(python_exe: Path) -> None:
    cmd = [str(python_exe), "-c", "import aws_cdk; import app"]
    print("Smoke test: import aws_cdk and app ...")
    result = subprocess.run(
        cmd,
        cwd=str(CDK_DIR),
        env=build_cdk_subprocess_env(),
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("Python smoke test failed. Fix dependencies before deploying.")


def run_smoke_test_if_needed(
    python_exe: Optional[Path],
    args: argparse.Namespace,
) -> None:
    if args.config_only or args.skip_cdk_json or not python_exe:
        return
    smoke_test_python_app(python_exe)


# ---------------------------------------------------------------------------
# AWS discovery
# ---------------------------------------------------------------------------


def get_aws_identity(region: Optional[str] = None) -> Tuple[str, str]:
    import boto3

    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    account = ident["Account"]
    resolved_region = (
        region
        or session.region_name
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "eu-west-2"
    )
    return account, resolved_region


def resolve_cdk_executable() -> str:
    """Return the AWS CDK CLI on PATH (cdk.cmd on Windows)."""
    for name in ("cdk", "cdk.cmd", "cdk.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "AWS CDK CLI not found on PATH.\n"
        "Install: npm install -g aws-cdk  (or use npx aws-cdk)"
    )


def check_cdk_cli() -> str:
    cdk_exe = resolve_cdk_executable()
    result = subprocess.run(
        [cdk_exe, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "AWS CDK CLI not found on PATH.\n"
            "Install: npm install -g aws-cdk  (or use npx aws-cdk)"
        )
    version = (result.stdout or result.stderr or "").strip()
    if version.startswith("Python "):
        raise SystemExit(
            "CDK CLI check failed: 'cdk' resolved to the Python interpreter instead "
            "of the AWS CDK CLI. Install aws-cdk globally (npm install -g aws-cdk) "
            "and ensure it is on PATH before python.exe."
        )
    return version


def cdk_bootstrap_needed(account: str, region: str) -> bool:
    import boto3
    from botocore.exceptions import ClientError

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        cfn.describe_stacks(StackName="CDKToolkit")
        return False
    except ClientError as exc:
        if "does not exist" in str(exc):
            return True
        raise


def list_vpcs(region: str) -> List[Dict[str, str]]:
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_vpcs()
    vpcs: List[Dict[str, str]] = []
    for vpc in response.get("Vpcs", []):
        name = ""
        for tag in vpc.get("Tags", []):
            if tag.get("Key") == "Name":
                name = tag.get("Value", "")
                break
        cidrs = vpc_cidr_blocks_from_describe(vpc)
        vpcs.append(
            {
                "id": vpc["VpcId"],
                "name": name or vpc["VpcId"],
                "cidr": cidrs[0] if cidrs else "",
                "cidrs": ",".join(cidrs),
            }
        )
    return sorted(vpcs, key=lambda x: x["name"])


def vpc_cidr_blocks_from_describe(vpc: Dict[str, Any]) -> List[str]:
    """Primary and associated IPv4 CIDR blocks for one ``describe_vpcs`` entry."""
    blocks: List[str] = []
    seen: set[str] = set()
    primary = (vpc.get("CidrBlock") or "").strip()
    if primary and primary not in seen:
        seen.add(primary)
        blocks.append(primary)
    for assoc in vpc.get("CidrBlockAssociationSet") or []:
        cidr = (assoc.get("CidrBlock") or "").strip()
        if cidr and cidr not in seen:
            seen.add(cidr)
            blocks.append(cidr)
    return blocks


def list_vpc_cidr_blocks_in_region(region: str) -> List[str]:
    """All IPv4 VPC CIDR blocks in the account/region (primary + associations)."""
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    blocks: List[str] = []
    seen: set[str] = set()
    for vpc in ec2.describe_vpcs().get("Vpcs", []):
        for cidr in vpc_cidr_blocks_from_describe(vpc):
            if cidr not in seen:
                seen.add(cidr)
                blocks.append(cidr)
    return blocks


VPC_CIDR_PREFIX_LEN = 24
SUBNET_CIDR_PREFIX_LEN = 28
_VPC_CIDR_SEARCH_SUPERNETS = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")


def parse_ipv4_networks(cidrs: Sequence[str]) -> List[ipaddress.IPv4Network]:
    networks: List[ipaddress.IPv4Network] = []
    for cidr in cidrs:
        raw = (cidr or "").strip()
        if not raw:
            continue
        try:
            networks.append(ipaddress.ip_network(raw, strict=False))
        except ValueError:
            continue
    return networks


def suggest_vpc_cidr_block(
    existing_vpc_cidrs: Sequence[str],
    *,
    prefix_len: int = VPC_CIDR_PREFIX_LEN,
    search_supernets: Sequence[str] = _VPC_CIDR_SEARCH_SUPERNETS,
) -> str:
    """Return the lowest non-overlapping ``prefix_len`` block in RFC1918 space."""
    occupied = parse_ipv4_networks(existing_vpc_cidrs)
    for supernet_cidr in search_supernets:
        supernet = ipaddress.ip_network(supernet_cidr, strict=False)
        for candidate in supernet.subnets(new_prefix=prefix_len):
            if any(candidate.overlaps(block) for block in occupied):
                continue
            return str(candidate)
    raise ValueError(
        f"No unused /{prefix_len} VPC CIDR found in region "
        f"({len(occupied)} existing block(s))."
    )


def suggest_subnet_cidr_blocks(
    vpc_cidr: str,
    existing_subnet_cidrs: Sequence[str],
    count: int,
    *,
    prefix_len: int = SUBNET_CIDR_PREFIX_LEN,
    reserved_cidrs: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return ``count`` lowest non-overlapping ``prefix_len`` blocks inside ``vpc_cidr``."""
    if count < 1:
        return []
    vpc_net = ipaddress.ip_network(vpc_cidr.strip(), strict=False)
    occupied = parse_ipv4_networks(existing_subnet_cidrs)
    if reserved_cidrs:
        occupied.extend(parse_ipv4_networks(reserved_cidrs))

    suggestions: List[str] = []
    for candidate in vpc_net.subnets(new_prefix=prefix_len):
        if any(candidate.overlaps(block) for block in occupied):
            continue
        suggestions.append(str(candidate))
        occupied.append(candidate)
        if len(suggestions) >= count:
            return suggestions

    raise ValueError(
        f"Only {len(suggestions)} available /{prefix_len} subnet block(s) in "
        f"{vpc_cidr}; need {count}."
    )


def prompt_new_vpc_cidr(answers: "InstallAnswers", *, interactive: bool) -> None:
    """Fill ``answers.new_vpc_cidr`` via auto-suggest and/or manual wizard prompt."""
    if answers.new_vpc_cidr:
        return

    fallback = "10.0.0.0/24"
    auto_prompt = (
        f"Auto-select lowest available /{VPC_CIDR_PREFIX_LEN} VPC CIDR "
        f"in {answers.aws_region}?"
    )

    def _auto_select() -> bool:
        try:
            occupied = list_vpc_cidr_blocks_in_region(answers.aws_region)
            answers.new_vpc_cidr = suggest_vpc_cidr_block(occupied)
            print(f"Selected VPC CIDR: {answers.new_vpc_cidr}")
            return True
        except ValueError as exc:
            print(f"Warning: {exc}")
            return False

    if interactive:
        if ask_yes_no(auto_prompt, default=True):
            if not _auto_select():
                answers.new_vpc_cidr = ask("New VPC CIDR", fallback)
        else:
            default_cidr = fallback
            try:
                occupied = list_vpc_cidr_blocks_in_region(answers.aws_region)
                default_cidr = suggest_vpc_cidr_block(occupied)
            except ValueError:
                pass
            answers.new_vpc_cidr = ask("New VPC CIDR", default_cidr)
    elif not _auto_select():
        answers.new_vpc_cidr = fallback


def list_availability_zones(region: str) -> List[str]:
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    zones = ec2.describe_availability_zones(
        Filters=[{"Name": "state", "Values": ["available"]}]
    )
    return [z["ZoneName"] for z in zones.get("AvailabilityZones", [])]


def list_subnets_in_vpc(vpc_id: str, region: str) -> List[Dict[str, str]]:
    sys.path.insert(0, str(CDK_DIR))
    from cdk_functions import _get_existing_subnets_in_vpc

    data = _get_existing_subnets_in_vpc(vpc_id)
    subnets: List[Dict[str, str]] = []
    for name, info in data.get("by_name", {}).items():
        subnets.append(
            {
                "name": name,
                "id": info.get("id", ""),
                "cidr": info.get("cidr", ""),
                "az": info.get("az", ""),
            }
        )
    return sorted(subnets, key=lambda x: x["name"])


def list_acm_certificates(region: str) -> List[Dict[str, str]]:
    import boto3

    acm = boto3.client("acm", region_name=region)
    certs: List[Dict[str, str]] = []
    paginator = acm.get_paginator("list_certificates")
    for page in paginator.paginate(CertificateStatuses=["ISSUED"]):
        for summary in page.get("CertificateSummaryList", []):
            arn = summary["CertificateArn"]
            detail = acm.describe_certificate(CertificateArn=arn)["Certificate"]
            domain = detail.get("DomainName", "")
            sans = detail.get("SubjectAlternativeNames", [])
            label = domain
            if sans and sans[0] != domain:
                label = f"{domain} (+{len(sans) - 1} SANs)"
            certs.append({"arn": arn, "domain": domain, "label": label})
    return certs


def list_igws_for_vpc(vpc_id: str, region: str) -> List[Dict[str, str]]:
    import boto3

    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_internet_gateways(
        Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}]
    )
    return [
        {"id": igw["InternetGatewayId"]} for igw in response.get("InternetGateways", [])
    ]


def list_albs_in_vpc(vpc_id: str, region: str) -> List[Dict[str, str]]:
    import boto3

    elbv2 = boto3.client("elbv2", region_name=region)
    response = elbv2.describe_load_balancers()
    albs: List[Dict[str, str]] = []
    for lb in response.get("LoadBalancers", []):
        if lb.get("VpcId") == vpc_id:
            albs.append(
                {
                    "arn": lb["LoadBalancerArn"],
                    "dns": lb["DNSName"],
                    "name": lb["LoadBalancerName"],
                }
            )
    return albs


# ---------------------------------------------------------------------------
# Env builder / writer
# ---------------------------------------------------------------------------


def format_list_env(values: Sequence[str], *, use_single_quotes: bool = False) -> str:
    if not values:
        return "[]"
    if use_single_quotes:
        inner = ", ".join(f"'{v}'" for v in values)
    else:
        inner = ", ".join(f'"{v}"' for v in values)
    return f"[{inner}]"


SUBNET_TIER_MODES = ("auto", "existing", "create")


def parse_subnet_name_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_subnet_tier_modes(args: argparse.Namespace) -> Tuple[str, str]:
    """CLI: --subnet-mode sets both tiers; per-tier flags override individually."""
    default = getattr(args, "subnet_mode", None) or "auto"
    public = getattr(args, "public_subnet_mode", None) or default
    private = getattr(args, "private_subnet_mode", None) or default
    return public, private


def apply_subnet_cli_flags(args: argparse.Namespace, answers: "InstallAnswers") -> None:
    if getattr(args, "public_subnet_names", None):
        answers.public_subnet_names = parse_subnet_name_csv(args.public_subnet_names)
    if getattr(args, "private_subnet_names", None):
        answers.private_subnet_names = parse_subnet_name_csv(args.private_subnet_names)


def lookup_subnets_by_name(
    vpc_subnets: Sequence[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    return {subnet["name"]: subnet for subnet in vpc_subnets if subnet.get("name")}


def fill_existing_subnet_tier_metadata(
    names: Sequence[str],
    lookup: Dict[str, Dict[str, str]],
    *,
    cidrs_out: List[str],
    azs_out: List[str],
) -> List[str]:
    """Resolve CIDR/AZ for existing subnet names (order preserved)."""
    errors: List[str] = []
    cidrs_out.clear()
    azs_out.clear()
    for name in names:
        info = lookup.get(name)
        if not info:
            errors.append(f"Subnet '{name}' was not found in the VPC.")
            continue
        cidr = (info.get("cidr") or "").strip()
        az = (info.get("az") or "").strip()
        if not az:
            errors.append(f"Subnet '{name}' has no availability zone in AWS.")
            continue
        cidrs_out.append(cidr)
        azs_out.append(az)
    return errors


def enrich_existing_subnet_details_from_aws(answers: "InstallAnswers") -> List[str]:
    """Look up CIDR/AZ for tiers using existing named subnets (wizard or CLI)."""
    if answers.vpc_mode != "existing" or not answers.vpc_name:
        return []

    tiers: List[Tuple[str, str, List[str], List[str], List[str]]] = []
    if answers.public_subnet_mode == "existing" and answers.public_subnet_names:
        tiers.append(
            (
                "Public",
                answers.public_subnet_mode,
                answers.public_subnet_names,
                answers.public_subnet_cidrs,
                answers.public_subnet_azs,
            )
        )
    if answers.private_subnet_mode == "existing" and answers.private_subnet_names:
        tiers.append(
            (
                "Private",
                answers.private_subnet_mode,
                answers.private_subnet_names,
                answers.private_subnet_cidrs,
                answers.private_subnet_azs,
            )
        )
    if not tiers:
        return []

    vpc_list = list_vpcs(answers.aws_region)
    vpc_id = next(
        (vpc["id"] for vpc in vpc_list if vpc["name"] == answers.vpc_name),
        None,
    )
    if not vpc_id:
        return [f"VPC '{answers.vpc_name}' not found in {answers.aws_region}."]

    lookup = lookup_subnets_by_name(list_subnets_in_vpc(vpc_id, answers.aws_region))
    errors: List[str] = []
    for _label, _mode, names, cidrs_out, azs_out in tiers:
        if len(cidrs_out) == len(names) == len(azs_out) and names:
            continue
        errors.extend(
            fill_existing_subnet_tier_metadata(
                names,
                lookup,
                cidrs_out=cidrs_out,
                azs_out=azs_out,
            )
        )
    return errors


def apply_subnet_tier_env(
    values: Dict[str, str],
    answers: "InstallAnswers",
    *,
    tier: str,
    mode: str,
) -> None:
    """Write PUBLIC_* or PRIVATE_* subnet keys for one tier (auto | existing | create)."""
    prefix = "PUBLIC" if tier == "public" else "PRIVATE"
    names = (
        answers.public_subnet_names
        if tier == "public"
        else answers.private_subnet_names
    )
    cidrs = (
        answers.public_subnet_cidrs
        if tier == "public"
        else answers.private_subnet_cidrs
    )
    azs = answers.public_subnet_azs if tier == "public" else answers.private_subnet_azs

    if mode == "auto":
        values[f"{prefix}_SUBNETS_TO_USE"] = ""
        values[f"{prefix}_SUBNET_CIDR_BLOCKS"] = ""
        values[f"{prefix}_SUBNET_AVAILABILITY_ZONES"] = ""
    elif mode == "existing":
        values[f"{prefix}_SUBNETS_TO_USE"] = format_list_env(names)
        if names and len(cidrs) == len(names) == len(azs):
            values[f"{prefix}_SUBNET_CIDR_BLOCKS"] = format_list_env(
                cidrs, use_single_quotes=True
            )
            values[f"{prefix}_SUBNET_AVAILABILITY_ZONES"] = format_list_env(
                azs, use_single_quotes=True
            )
        else:
            values[f"{prefix}_SUBNET_CIDR_BLOCKS"] = ""
            values[f"{prefix}_SUBNET_AVAILABILITY_ZONES"] = ""
    else:
        values[f"{prefix}_SUBNETS_TO_USE"] = format_list_env(names)
        values[f"{prefix}_SUBNET_CIDR_BLOCKS"] = format_list_env(
            cidrs, use_single_quotes=True
        )
        values[f"{prefix}_SUBNET_AVAILABILITY_ZONES"] = format_list_env(
            azs, use_single_quotes=True
        )


def validate_subnet_answers(answers: "InstallAnswers") -> List[str]:
    errors: List[str] = []
    if answers.vpc_mode != "existing":
        return errors
    tiers: List[tuple] = [
        (
            "Public",
            answers.public_subnet_mode,
            answers.public_subnet_names,
            answers.public_subnet_cidrs,
        ),
    ]
    if not answers_use_express_mode(answers) and not answers_use_public_subnets_only(
        answers
    ):
        tiers.append(
            (
                "Private",
                answers.private_subnet_mode,
                answers.private_subnet_names,
                answers.private_subnet_cidrs,
            )
        )
    for label, mode, names, cidrs in tiers:
        if mode == "existing" and not names:
            errors.append(
                f"{label} subnets: provide at least one existing subnet name."
            )
        if mode == "create" and not names:
            errors.append(f"{label} subnets: create mode requires subnet names.")
        if mode == "create" and names and cidrs and len(names) != len(cidrs):
            errors.append(f"{label} subnets: CIDR count must match subnet name count.")
    return errors


def ask_subnet_tier_mode(tier_label: str) -> str:
    idx = ask_choice(
        f"{tier_label} subnets",
        [
            "Auto-discover suitable subnets",
            "Use existing named subnets",
            "Create new stack-specific subnets (name + CIDR + AZ)",
        ],
        default_index=0,
    )
    return SUBNET_TIER_MODES[idx]


def configure_subnet_tier(
    answers: "InstallAnswers",
    tier: str,
    mode: str,
    vpc_subnets: List[Dict[str, str]],
    azs: Sequence[str],
    *,
    interactive: bool,
    vpc_cidr: str = "",
    reserved_subnet_cidrs: Optional[Sequence[str]] = None,
) -> None:
    """Collect subnet names/CIDRs for one tier in the wizard."""
    is_public = tier == "public"
    label = "Public" if is_public else "Private"
    prefix_label = "Public" if is_public else "Private"

    if mode == "auto":
        return

    if mode == "existing":
        names = (
            answers.public_subnet_names if is_public else answers.private_subnet_names
        )
        if interactive and not names:
            print(
                f"Enter {label.lower()} subnet names (comma-separated), or pick from:"
            )
            for subnet in vpc_subnets:
                print(f"  - {subnet['name']} ({subnet['cidr']}, {subnet['az']})")
            names = parse_subnet_name_csv(ask(f"{label} subnet names", ""))
        if is_public:
            answers.public_subnet_names = names
            cidrs_out = answers.public_subnet_cidrs
            azs_out = answers.public_subnet_azs
        else:
            answers.private_subnet_names = names
            cidrs_out = answers.private_subnet_cidrs
            azs_out = answers.private_subnet_azs
        if names:
            lookup = lookup_subnets_by_name(vpc_subnets)
            missing = fill_existing_subnet_tier_metadata(
                names,
                lookup,
                cidrs_out=cidrs_out,
                azs_out=azs_out,
            )
            for err in missing:
                print(f"Warning: {err}")
        return

    n_az = len(azs)
    names = [f"{answers.cdk_prefix}{prefix_label}Subnet{i + 1}" for i in range(n_az)]
    if is_public:
        answers.public_subnet_names = names
        cidrs_list = answers.public_subnet_cidrs
        azs_list = answers.public_subnet_azs
        cidr_base = 0
    else:
        answers.private_subnet_names = names
        cidrs_list = answers.private_subnet_cidrs
        azs_list = answers.private_subnet_azs
        cidr_base = 10

    if interactive:
        print(f"Suggested AZs for {label.lower()} subnets: {', '.join(azs)}")
        auto_assign = ask_yes_no(
            f"Auto-assign lowest available /{SUBNET_CIDR_PREFIX_LEN} CIDR blocks "
            f"for {label.lower()} subnets?",
            default=True,
        )
    else:
        auto_assign = bool(vpc_cidr)

    existing_subnet_cidrs = [
        subnet.get("cidr", "") for subnet in vpc_subnets if subnet.get("cidr")
    ]
    suggested: Optional[List[str]] = None
    if auto_assign and vpc_cidr:
        try:
            suggested = suggest_subnet_cidr_blocks(
                vpc_cidr,
                existing_subnet_cidrs,
                len(names),
                reserved_cidrs=reserved_subnet_cidrs,
            )
        except ValueError as exc:
            print(f"Warning: {exc}")
            if not interactive:
                raise SystemExit(str(exc)) from exc

    cidrs_list.clear()
    azs_list.clear()
    for i, name in enumerate(names):
        if suggested:
            cidr = suggested[i]
            if interactive:
                print(f"  {name}: {cidr}")
        elif interactive:
            default_cidr = f"10.0.{cidr_base + i}.0/{SUBNET_CIDR_PREFIX_LEN}"
            if vpc_cidr:
                try:
                    default_cidr = suggest_subnet_cidr_blocks(
                        vpc_cidr,
                        existing_subnet_cidrs + cidrs_list,
                        1,
                        reserved_cidrs=reserved_subnet_cidrs,
                    )[0]
                except ValueError:
                    pass
            cidr = ask(f"CIDR for {label.lower()} {name}", default_cidr)
        else:
            cidr = f"10.0.{cidr_base + i}.0/{SUBNET_CIDR_PREFIX_LEN}"
        cidrs_list.append(cidr)
        azs_list.append(azs[i % len(azs)])


@dataclass
class InstallAnswers:
    profile: str = "demo"
    aws_account_id: str = ""
    aws_region: str = ""
    cdk_prefix: str = ""
    cognito_domain_prefix: str = ""
    github_branch: str = "main"
    vpc_mode: str = "existing"  # new | existing
    vpc_name: str = ""
    new_vpc_cidr: str = ""
    public_subnet_mode: str = "auto"  # auto | existing | create
    private_subnet_mode: str = "auto"  # auto | existing | create
    public_subnet_names: List[str] = field(default_factory=list)
    private_subnet_names: List[str] = field(default_factory=list)
    public_subnet_cidrs: List[str] = field(default_factory=list)
    private_subnet_cidrs: List[str] = field(default_factory=list)
    public_subnet_azs: List[str] = field(default_factory=list)
    private_subnet_azs: List[str] = field(default_factory=list)
    existing_igw_id: str = ""
    existing_alb_arn: str = ""
    existing_alb_dns: str = ""
    acm_cert_arn: str = ""
    ssl_domain: str = ""
    cloudfront_geo: str = ""
    enable_pi_express: bool = False
    enable_pi_legacy: bool = False
    enable_service_connect: bool = False
    enable_s3_batch: bool = False
    enable_headless: bool = False
    ecs_memory: str = "4096"
    pi_alb_routing: str = "path"
    pi_alb_path_prefix: str = "/agent"
    pi_alb_host_header: str = ""
    pi_alb_listener_rule_priority: str = ""
    pi_gradio_port: str = "7862"
    sc_discovery_name: str = "redaction"
    pi_default_provider: str = "amazon-bedrock"
    write_pi_agent_env: bool = True
    overwrite_pi_agent_env: bool = False
    write_app_config_env: bool = True
    overwrite_app_config_env: bool = False
    custom_overrides: Dict[str, str] = field(default_factory=dict)
    python_path: Optional[str] = None

    @property
    def pi_enabled(self) -> bool:
        return self.enable_pi_express or self.enable_pi_legacy


# Cognito hosted-UI prefix domains cannot contain these substrings (AWS docs).
COGNITO_DOMAIN_RESERVED_SUBSTRINGS = ("amazon", "cognito", "aws")


def sanitize_cognito_domain_prefix(raw: str, *, max_length: int = 63) -> str:
    """Normalize a Cognito domain prefix: allowed chars, no reserved words."""
    prefix = re.sub(r"[^a-z0-9-]", "-", (raw or "").lower())
    for reserved in COGNITO_DOMAIN_RESERVED_SUBSTRINGS:
        prefix = prefix.replace(reserved, "")
    prefix = re.sub(r"-{2,}", "-", prefix).strip("-")
    if max_length > 0:
        prefix = prefix[:max_length].strip("-")
    if not prefix or not re.match(r"^[a-z0-9]", prefix):
        return "redaction-app"
    prefix = prefix.rstrip("-")
    if not prefix:
        return "redaction-app"
    for reserved in COGNITO_DOMAIN_RESERVED_SUBSTRINGS:
        if reserved in prefix:
            return "redaction-app"
    return prefix


def default_cognito_domain_prefix_from_cdk_prefix(cdk_prefix: str) -> str:
    """Derive a short installer default from CDK_PREFIX (globally unique hint)."""
    slug = re.sub(r"[^a-z0-9-]", "-", (cdk_prefix or "").lower()).strip("-")
    return sanitize_cognito_domain_prefix(slug, max_length=20)


def validate_cognito_domain_prefix(prefix: str) -> Optional[str]:
    """Return an error message when prefix violates Cognito domain rules."""
    cleaned = (prefix or "").strip().lower()
    if not cleaned:
        return "COGNITO_USER_POOL_DOMAIN_PREFIX is required."
    reserved = [word for word in COGNITO_DOMAIN_RESERVED_SUBSTRINGS if word in cleaned]
    if reserved:
        return (
            "COGNITO_USER_POOL_DOMAIN_PREFIX cannot contain reserved words: "
            + ", ".join(reserved)
            + "."
        )
    if not re.match(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", cleaned):
        return (
            "COGNITO_USER_POOL_DOMAIN_PREFIX must use lowercase letters, "
            "numbers, and hyphens only."
        )
    return None


def normalize_pi_path_prefix(raw: str) -> str:
    segment = (raw or "agent").strip().strip("/")
    return f"/{segment}" if segment else "/agent"


def default_pi_listener_priority(use_cloudfront: bool = False) -> str:
    """Default Pi path/host rule priority (1–2 reserved for CloudFront / Express rules)."""
    del use_cloudfront  # kept for call-site compatibility
    return "3"


def merge_preset(
    profile: str, overrides: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    base: Dict[str, str] = {}
    if profile == "demo":
        base.update(DEMO_PRESET)
    elif profile == "production":
        base.update(PRODUCTION_PRESET)
    elif profile == "headless":
        base.update(HEADLESS_PRESET)
    if overrides:
        base.update(overrides)
    return base


def answers_preset_profile(answers: "InstallAnswers") -> str:
    """Preset profile for env defaults (legacy ``headless`` maps to demo)."""
    if answers.profile == "headless":
        return "demo"
    return answers.profile


def answers_use_headless(answers: "InstallAnswers") -> bool:
    return answers.profile == "headless" or answers.enable_headless


def profile_allows_headless_add_on(answers: "InstallAnswers") -> bool:
    """Headless batch is incompatible with the Demonstration (Express) route."""
    if answers.profile == "demo":
        return False
    if answers.profile in ("production", "headless"):
        return True
    if answers.profile == "custom":
        return answers.custom_overrides.get("USE_ECS_EXPRESS_MODE") != "True"
    return False


def headless_profile_error(answers: "InstallAnswers") -> Optional[str]:
    if not answers_use_headless(answers):
        return None
    if answers.profile == "headless":
        return None
    if answers.profile == "demo":
        return (
            "Headless batch mode is not available with the Demonstration (Express) "
            "profile. Use the Headless shortcut profile, Production with headless, "
            "or Custom without ECS Express."
        )
    if (
        answers.profile == "custom"
        and answers.custom_overrides.get("USE_ECS_EXPRESS_MODE") == "True"
    ):
        return (
            "Headless batch mode requires USE_ECS_EXPRESS_MODE=False. "
            "Disable ECS Express in the custom profile or omit --headless."
        )
    return None


def validate_install_answers(answers: "InstallAnswers") -> List[str]:
    errors: List[str] = []
    msg = headless_profile_error(answers)
    if msg:
        errors.append(msg)
    return errors


def answers_use_express_mode(answers: "InstallAnswers") -> bool:
    """True when the selected install profile/options enable ECS Express ingress."""
    if answers_use_headless(answers):
        return False
    preset = merge_preset(answers.profile, answers.custom_overrides)
    return preset.get("USE_ECS_EXPRESS_MODE") == "True"


def answers_use_public_subnets_only(answers: "InstallAnswers") -> bool:
    """Express ingress or demo-style headless batch (public subnets, no private install)."""
    if answers_use_express_mode(answers):
        return True
    if not answers_use_headless(answers):
        return False
    return answers_preset_profile(answers) in ("demo", "custom")


def build_env_values(answers: InstallAnswers) -> Dict[str, str]:
    cdk_folder = str(CDK_DIR).replace("\\", "/")
    if not cdk_folder.endswith("/"):
        cdk_folder += "/"

    values: Dict[str, str] = merge_preset(
        answers_preset_profile(answers), answers.custom_overrides
    )
    values.update(
        {
            "CDK_PREFIX": answers.cdk_prefix,
            "AWS_REGION": answers.aws_region,
            "AWS_ACCOUNT_ID": answers.aws_account_id,
            "CDK_FOLDER": cdk_folder,
            "CONTEXT_FILE": "precheck.context.json",
            "COGNITO_USER_POOL_DOMAIN_PREFIX": sanitize_cognito_domain_prefix(
                answers.cognito_domain_prefix
            ),
            "GITHUB_REPO_BRANCH": answers.github_branch,
            "ECS_TASK_MEMORY_SIZE": answers.ecs_memory,
            "EXISTING_IGW_ID": answers.existing_igw_id,
            "EXISTING_LOAD_BALANCER_ARN": answers.existing_alb_arn,
            "EXISTING_LOAD_BALANCER_DNS": answers.existing_alb_dns
            or "placeholder_load_balancer_dns.net",
            "ENABLE_PI_AGENT_EXPRESS_SERVICE": (
                "True" if answers.enable_pi_express else "False"
            ),
            "ENABLE_PI_AGENT_ECS_SERVICE": (
                "True" if answers.enable_pi_legacy else "False"
            ),
            "ENABLE_ECS_SERVICE_CONNECT": (
                "True" if answers.enable_service_connect else "False"
            ),
            "ENABLE_S3_BATCH_ECS_TRIGGER": (
                "True"
                if answers.enable_s3_batch or answers_use_headless(answers)
                else "False"
            ),
            "ENABLE_HEADLESS_DEPLOYMENT": (
                "True" if answers_use_headless(answers) else "False"
            ),
        }
    )

    if answers_use_headless(answers):
        values.update(
            {
                "ENABLE_HEADLESS_DEPLOYMENT": "True",
                "ENABLE_S3_BATCH_ECS_TRIGGER": "True",
                "USE_ECS_EXPRESS_MODE": "False",
                "USE_CLOUDFRONT": "False",
                "RUN_USEAST_STACK": "False",
                "COGNITO_AUTH": "False",
                "ENABLE_PI_AGENT_EXPRESS_SERVICE": "False",
                "ENABLE_PI_AGENT_ECS_SERVICE": "False",
                "ENABLE_ECS_SERVICE_CONNECT": "False",
            }
        )

    use_cloudfront = values.get("USE_CLOUDFRONT") == "True"
    if answers.pi_enabled:
        routing = answers.pi_alb_routing.strip().lower()
        path_prefix = normalize_pi_path_prefix(answers.pi_alb_path_prefix)
        priority = (
            answers.pi_alb_listener_rule_priority.strip()
            or default_pi_listener_priority(use_cloudfront)
        )
        values.update(
            {
                "PI_ALB_ROUTING": routing,
                "PI_ALB_PATH_PREFIX": path_prefix,
                "PI_ALB_LISTENER_RULE_PRIORITY": priority,
                "PI_GRADIO_PORT": answers.pi_gradio_port,
                "ECS_SERVICE_CONNECT_DISCOVERY_NAME": answers.sc_discovery_name,
            }
        )
        if routing in ("host", "both"):
            values["PI_ALB_HOST_HEADER"] = answers.pi_alb_host_header.strip()
        if answers.enable_pi_express:
            values["ECS_EXPRESS_SC_PORT_NAME"] = "port-7860"
            values["ECS_PI_EXPRESS_SC_PORT_NAME"] = f"port-{answers.pi_gradio_port}"

    if answers.profile == "production":
        values["ACM_SSL_CERTIFICATE_ARN"] = answers.acm_cert_arn
        values["SSL_CERTIFICATE_DOMAIN"] = answers.ssl_domain
        values["CLOUDFRONT_DOMAIN"] = "cloudfront_placeholder.net"
        if answers.cloudfront_geo:
            values["CLOUDFRONT_GEO_RESTRICTION"] = answers.cloudfront_geo

    if answers.vpc_mode == "new":
        values["VPC_NAME"] = ""
        values["NEW_VPC_CIDR"] = answers.new_vpc_cidr
        values["NEW_VPC_DEFAULT_NAME"] = f"{answers.cdk_prefix}vpc"
        values["PUBLIC_SUBNETS_TO_USE"] = ""
        values["PRIVATE_SUBNETS_TO_USE"] = ""
        values["PUBLIC_SUBNET_CIDR_BLOCKS"] = ""
        values["PRIVATE_SUBNET_CIDR_BLOCKS"] = ""
        values["PUBLIC_SUBNET_AVAILABILITY_ZONES"] = ""
        values["PRIVATE_SUBNET_AVAILABILITY_ZONES"] = ""
    else:
        values["VPC_NAME"] = answers.vpc_name
        values["NEW_VPC_CIDR"] = ""
        apply_subnet_tier_env(
            values, answers, tier="public", mode=answers.public_subnet_mode
        )
        if answers_use_public_subnets_only(answers):
            values["ECS_EXPRESS_USE_PUBLIC_SUBNETS"] = "True"
            values["PRIVATE_SUBNETS_TO_USE"] = ""
            values["PRIVATE_SUBNET_CIDR_BLOCKS"] = ""
            values["PRIVATE_SUBNET_AVAILABILITY_ZONES"] = ""
        else:
            apply_subnet_tier_env(
                values, answers, tier="private", mode=answers.private_subnet_mode
            )

    if values.get("ENABLE_APPREGISTRY") == "True":
        values["APPREGISTRY_STACK_NAME"] = (
            f"{answers.cdk_prefix}{APPREGISTRY_STACK_SUFFIX}"
        )

    return values


def validate_env_values(values: Dict[str, str]) -> List[str]:
    errors: List[str] = []

    express = values.get("USE_ECS_EXPRESS_MODE") == "True"
    acm = (values.get("ACM_SSL_CERTIFICATE_ARN") or "").strip()
    if express and acm:
        errors.append(
            "USE_ECS_EXPRESS_MODE=True cannot be used with ACM_SSL_CERTIFICATE_ARN."
        )

    if values.get("ENABLE_ECS_SERVICE_CONNECT") == "True" and express:
        errors.append(
            "ENABLE_ECS_SERVICE_CONNECT requires legacy Fargate (not Express)."
        )

    if values.get("ENABLE_S3_BATCH_ECS_TRIGGER") == "True" and express:
        errors.append(
            "ENABLE_S3_BATCH_ECS_TRIGGER requires legacy Fargate (not Express)."
        )

    headless = values.get("ENABLE_HEADLESS_DEPLOYMENT") == "True"
    if headless:
        if values.get("ENABLE_S3_BATCH_ECS_TRIGGER") != "True":
            errors.append(
                "ENABLE_HEADLESS_DEPLOYMENT requires ENABLE_S3_BATCH_ECS_TRIGGER=True."
            )
        if express:
            errors.append(
                "ENABLE_HEADLESS_DEPLOYMENT cannot be combined with USE_ECS_EXPRESS_MODE=True "
                "(use --profile headless, production --headless, or custom without Express)."
            )
        if values.get("USE_CLOUDFRONT") == "True":
            errors.append(
                "ENABLE_HEADLESS_DEPLOYMENT is incompatible with USE_CLOUDFRONT=True."
            )

    pi_ecs = values.get("ENABLE_PI_AGENT_ECS_SERVICE") == "True"
    pi_express = values.get("ENABLE_PI_AGENT_EXPRESS_SERVICE") == "True"
    if pi_ecs and pi_express:
        errors.append("Enable at most one agent mode (ECS or Express).")
    if pi_express and not express:
        errors.append(
            "ENABLE_PI_AGENT_EXPRESS_SERVICE requires USE_ECS_EXPRESS_MODE=True."
        )
    if pi_ecs and express:
        errors.append(
            "ENABLE_PI_AGENT_ECS_SERVICE requires USE_ECS_EXPRESS_MODE=False."
        )

    vpc_name = (values.get("VPC_NAME") or "").strip()
    new_cidr = (values.get("NEW_VPC_CIDR") or "").strip()
    if not vpc_name and not new_cidr:
        errors.append("Set VPC_NAME (existing VPC) or NEW_VPC_CIDR (new VPC).")

    if values.get("USE_CLOUDFRONT") == "True" and not express:
        if not acm:
            errors.append(
                "Production CloudFront path requires ACM_SSL_CERTIFICATE_ARN."
            )
        if not (values.get("SSL_CERTIFICATE_DOMAIN") or "").strip():
            errors.append("Production CloudFront path requires SSL_CERTIFICATE_DOMAIN.")

    if not (values.get("CDK_PREFIX") or "").strip():
        errors.append("CDK_PREFIX is required.")
    cognito_prefix_error = validate_cognito_domain_prefix(
        values.get("COGNITO_USER_POOL_DOMAIN_PREFIX", "")
    )
    if cognito_prefix_error:
        errors.append(cognito_prefix_error)

    if pi_ecs and values.get("ENABLE_ECS_SERVICE_CONNECT") != "True":
        errors.append(
            "ENABLE_PI_AGENT_ECS_SERVICE=True requires ENABLE_ECS_SERVICE_CONNECT=True."
        )

    if headless and (
        pi_ecs or pi_express or values.get("ENABLE_ECS_SERVICE_CONNECT") == "True"
    ):
        errors.append(
            "ENABLE_HEADLESS_DEPLOYMENT is incompatible with agent mode or Service Connect."
        )

    pi_routing = (values.get("PI_ALB_ROUTING") or "").strip().lower()
    if pi_ecs or pi_express:
        if pi_routing not in PI_ALB_ROUTING_MODES:
            errors.append(
                f"PI_ALB_ROUTING must be one of {list(PI_ALB_ROUTING_MODES)}; got '{pi_routing}'."
            )
        elif (
            pi_routing in ("host", "both")
            and not (values.get("PI_ALB_HOST_HEADER") or "").strip()
        ):
            errors.append(
                "PI_ALB_HOST_HEADER is required when PI_ALB_ROUTING is 'host' or 'both'."
            )

    return errors


def build_app_config_env_values(values: Dict[str, str]) -> Dict[str, str]:
    """AWS deployment keys merged into config/app_config.env for ECS tasks."""
    prefix_lower = (values.get("CDK_PREFIX") or "").lower()
    headless = values.get("ENABLE_HEADLESS_DEPLOYMENT") == "True"
    express = values.get("USE_ECS_EXPRESS_MODE") == "True"
    alb_cognito = express and not headless

    def _name(env_key: str, suffix: str) -> str:
        return (values.get(env_key) or "").strip() or f"{prefix_lower}{suffix}"

    return {
        "COGNITO_AUTH": "False" if headless or alb_cognito else "True",
        "RUN_AWS_FUNCTIONS": "True",
        "DISPLAY_FILE_NAMES_IN_LOGS": "False",
        "SESSION_OUTPUT_FOLDER": "True",
        "SAVE_LOGS_TO_DYNAMODB": "True",
        "SHOW_COSTS": "True",
        "SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS": "True",
        "LOAD_PREVIOUS_TEXTRACT_JOBS_S3": "True",
        "DOCUMENT_REDACTION_BUCKET": _name("S3_LOG_CONFIG_BUCKET_NAME", "s3-logs"),
        "TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET": _name(
            "S3_OUTPUT_BUCKET_NAME", "s3-output"
        ),
        "ACCESS_LOG_DYNAMODB_TABLE_NAME": _name(
            "ACCESS_LOG_DYNAMODB_TABLE_NAME", "dynamodb-access-logs"
        ),
        "FEEDBACK_LOG_DYNAMODB_TABLE_NAME": _name(
            "FEEDBACK_LOG_DYNAMODB_TABLE_NAME", "dynamodb-feedback-logs"
        ),
        "USAGE_LOG_DYNAMODB_TABLE_NAME": _name(
            "USAGE_LOG_DYNAMODB_TABLE_NAME", "dynamodb-usage-logs"
        ),
    }


def write_app_config_env_file(
    answers: InstallAnswers,
    values: Dict[str, str],
    *,
    overwrite: bool = False,
) -> Optional[Path]:
    if not answers.write_app_config_env:
        return None

    updates = build_app_config_env_values(values)
    APP_CONFIG_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if APP_CONFIG_ENV_PATH.is_file() and not overwrite:
        existing = read_env_file(APP_CONFIG_ENV_PATH)
        existing.update(updates)
        backup_file(APP_CONFIG_ENV_PATH)
        lines = [f"{k}={v}" for k, v in existing.items()]
        APP_CONFIG_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Updated {APP_CONFIG_ENV_PATH} (AWS deployment keys merged)")
        return APP_CONFIG_ENV_PATH

    if APP_CONFIG_ENV_EXAMPLE.is_file() and not overwrite:
        if APP_CONFIG_ENV_PATH.is_file():
            backup_file(APP_CONFIG_ENV_PATH)
        shutil.copy2(APP_CONFIG_ENV_EXAMPLE, APP_CONFIG_ENV_PATH)
        existing = read_env_file(APP_CONFIG_ENV_PATH)
        existing.update(updates)
        lines = [f"{k}={v}" for k, v in existing.items()]
        APP_CONFIG_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Created {APP_CONFIG_ENV_PATH} from example + AWS defaults")
        return APP_CONFIG_ENV_PATH

    if APP_CONFIG_ENV_PATH.is_file():
        backup_file(APP_CONFIG_ENV_PATH)
    lines = [
        "# Generated by cdk_install.py — app runtime config for AWS ECS",
        *[f"{k}={v}" for k, v in updates.items()],
    ]
    APP_CONFIG_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {APP_CONFIG_ENV_PATH}")
    return APP_CONFIG_ENV_PATH


def build_pi_agent_env_values(answers: InstallAnswers) -> Dict[str, str]:
    """Runtime settings for the Pi agent Gradio app (uploaded to S3 as pi_agent.env)."""
    path_prefix = normalize_pi_path_prefix(answers.pi_alb_path_prefix)
    values = {
        "PI_DEPLOYMENT_PROFILE": "aws-ecs",
        "PI_DEFAULT_PROVIDER": answers.pi_default_provider,
        "DOC_REDACTION_GRADIO_URL": f"http://{answers.sc_discovery_name}:7860",
        "RUN_AWS_FUNCTIONS": "True",
        "AWS_REGION": answers.aws_region,
        "PI_GRADIO_PORT": answers.pi_gradio_port,
        "PI_DEFAULT_OCR_METHOD": "AWS Textract service - all PDF types",
        "PI_DEFAULT_PII_METHOD": "AWS Comprehend",
    }
    if answers.pi_alb_routing.strip().lower() in ("path", "both"):
        values["PI_ROOT_PATH"] = path_prefix
        values["ROOT_PATH"] = path_prefix
        values["FASTAPI_ROOT_PATH"] = path_prefix
    return values


def write_pi_agent_env_file(
    answers: InstallAnswers,
    *,
    overwrite: bool = False,
) -> Optional[Path]:
    if not answers.pi_enabled or not answers.write_pi_agent_env:
        return None

    updates = build_pi_agent_env_values(answers)
    PI_AGENT_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if PI_AGENT_ENV_PATH.is_file() and not overwrite:
        existing = read_env_file(PI_AGENT_ENV_PATH)
        existing.update(updates)
        backup_file(PI_AGENT_ENV_PATH)
        lines = [f"{k}={v}" for k, v in existing.items()]
        PI_AGENT_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Updated {PI_AGENT_ENV_PATH} (AWS/Pi agent keys merged)")
        return PI_AGENT_ENV_PATH

    if PI_AGENT_ENV_EXAMPLE.is_file() and not overwrite:
        backup_file(PI_AGENT_ENV_PATH) if PI_AGENT_ENV_PATH.is_file() else None
        shutil.copy2(PI_AGENT_ENV_EXAMPLE, PI_AGENT_ENV_PATH)
        existing = read_env_file(PI_AGENT_ENV_PATH)
        existing.update(updates)
        lines = [f"{k}={v}" for k, v in existing.items()]
        PI_AGENT_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Created {PI_AGENT_ENV_PATH} from example + AWS defaults")
        return PI_AGENT_ENV_PATH

    backup_file(PI_AGENT_ENV_PATH) if PI_AGENT_ENV_PATH.is_file() else None
    lines = [
        "# Generated by cdk_install.py — Pi agent runtime config for AWS ECS",
        *[f"{k}={v}" for k, v in updates.items()],
    ]
    PI_AGENT_ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {PI_AGENT_ENV_PATH}")
    return PI_AGENT_ENV_PATH


def write_env_file(path: Path, values: Dict[str, str]) -> Path:
    backup_file(path)
    lines = [f"{key}={val}" for key, val in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return path


def read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    return values


def patch_env_file(path: Path, updates: Dict[str, str]) -> None:
    values = read_env_file(path)
    values.update(updates)
    write_env_file(path, values)


def print_summary(values: Dict[str, str], python_exe: Optional[Path] = None) -> None:
    keys = [
        "CDK_PREFIX",
        "AWS_REGION",
        "AWS_ACCOUNT_ID",
        "USE_ECS_EXPRESS_MODE",
        "USE_CLOUDFRONT",
        "ENABLE_HEADLESS_DEPLOYMENT",
        "ENABLE_S3_BATCH_ECS_TRIGGER",
        "ENABLE_RESOURCE_DELETE_PROTECTION",
        "VPC_NAME",
        "NEW_VPC_CIDR",
        "ACM_SSL_CERTIFICATE_ARN",
        "SSL_CERTIFICATE_DOMAIN",
        "ENABLE_PI_AGENT_EXPRESS_SERVICE",
        "ENABLE_PI_AGENT_ECS_SERVICE",
        "PI_ALB_ROUTING",
        "PI_ALB_PATH_PREFIX",
        "PI_ALB_HOST_HEADER",
    ]
    print("\n--- Configuration summary ---")
    for key in keys:
        if key in values:
            print(f"  {key}={values[key]}")
    if python_exe:
        print(f"  cdk.json app={format_cdk_app_command(python_exe)}")
    print("----------------------------\n")


# ---------------------------------------------------------------------------
# Deploy runner
# ---------------------------------------------------------------------------


def _deploy_env() -> Dict[str, str]:
    return build_cdk_subprocess_env()


def run_cdk_command(
    args: List[str], *, check: bool = True
) -> subprocess.CompletedProcess:
    cdk_exe = resolve_cdk_executable()
    cmd = [cdk_exe, *args]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(CDK_DIR),
        env=_deploy_env(),
        check=check,
    )


@dataclass(frozen=True)
class ExistingStack:
    name: str
    region: str
    status: str
    termination_protection: bool = False


def _should_check_cloudfront_stack(
    config_values: Optional[Dict[str, str]] = None,
) -> bool:
    """Skip us-east-1 CloudFront stack lookup when config disables that path."""
    if not config_values:
        return True
    use_cloudfront = config_values.get("USE_CLOUDFRONT")
    run_useast = config_values.get("RUN_USEAST_STACK")
    if use_cloudfront is not None and use_cloudfront != "True":
        return False
    if run_useast is not None and run_useast != "True":
        return False
    return True


def _stack_check_skippable_error(exc: Exception) -> bool:
    """Permission / SCP errors in one region must not block checks in others."""
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    code = exc.response.get("Error", {}).get("Code", "")
    return code in (
        "AccessDenied",
        "UnauthorizedOperation",
        "AuthorizationError",
        "AccessDeniedException",
    )


def derived_appregistry_stack_name(
    config_values: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Resolve AppRegistry stack name from config (explicit or CDK_PREFIX default)."""
    if not config_values:
        return None
    explicit = (config_values.get("APPREGISTRY_STACK_NAME") or "").strip()
    if explicit:
        return explicit
    prefix = (config_values.get("CDK_PREFIX") or "").strip()
    if prefix:
        return f"{prefix}{APPREGISTRY_STACK_SUFFIX}"
    return None


def list_regional_appregistry_stack_names(region: str) -> List[str]:
    """List active AppRegistry stacks in the regional account (suffix match)."""
    import boto3
    from botocore.exceptions import ClientError

    cfn = boto3.client("cloudformation", region_name=region)
    skip_statuses = {"DELETE_COMPLETE", "DELETE_IN_PROGRESS"}
    names: List[str] = []
    try:
        paginator = cfn.get_paginator("list_stacks")
        for page in paginator.paginate():
            for summary in page.get("StackSummaries", []):
                name = summary.get("StackName", "")
                status = summary.get("StackStatus", "")
                if not name.endswith(APPREGISTRY_STACK_SUFFIX):
                    continue
                if status in skip_statuses:
                    continue
                names.append(name)
    except ClientError as exc:
        if _stack_check_skippable_error(exc):
            code = exc.response.get("Error", {}).get("Code", "ClientError")
            print(
                f"Warning: could not list AppRegistry stacks in {region} "
                f"({code}). Continuing with configured stack names only."
            )
            return []
        raise
    return sorted(dict.fromkeys(names))


def appregistry_stack_names_to_check(
    config_values: Optional[Dict[str, str]] = None,
    *,
    discovered_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Merge configured and discovered AppRegistry stack names (deduped)."""
    names: List[str] = []
    derived = derived_appregistry_stack_name(config_values)
    if derived:
        names.append(derived)
    for name in discovered_names or ():
        if name not in names:
            names.append(name)
    return names


def stacks_to_check(
    regional_region: str,
    config_values: Optional[Dict[str, str]] = None,
    *,
    discovered_appregistry_names: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """Return (stack_name, region) pairs in safe deletion order."""
    checks: List[Tuple[str, str]] = []
    if _should_check_cloudfront_stack(config_values):
        checks.append((CLOUDFRONT_STACK, CLOUDFRONT_STACK_REGION))
    for appregistry_name in appregistry_stack_names_to_check(
        config_values,
        discovered_names=discovered_appregistry_names,
    ):
        checks.append((appregistry_name, regional_region))
    checks.append((REGIONAL_STACK, regional_region))
    return checks


def describe_existing_stack(stack_name: str, region: str) -> Optional[ExistingStack]:
    import boto3
    from botocore.exceptions import ClientError

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        response = cfn.describe_stacks(StackName=stack_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("ValidationError", "ResourceNotFoundException") or (
            "does not exist" in str(exc)
        ):
            return None
        raise
    stack = response["Stacks"][0]
    status = stack.get("StackStatus", "")
    if status == "DELETE_COMPLETE":
        return None
    return ExistingStack(
        name=stack_name,
        region=region,
        status=status,
        termination_protection=bool(stack.get("EnableTerminationProtection", False)),
    )


def discover_existing_doc_redaction_stacks(
    regional_region: str,
    config_values: Optional[Dict[str, str]] = None,
) -> List[ExistingStack]:
    """Find deployed doc_redaction CloudFormation stacks in the target account."""
    from botocore.exceptions import ClientError

    discovered_appregistry = list_regional_appregistry_stack_names(regional_region)
    checks = stacks_to_check(
        regional_region,
        config_values,
        discovered_appregistry_names=discovered_appregistry,
    )
    ordered: List[ExistingStack] = []
    for stack_name, region in checks:
        try:
            info = describe_existing_stack(stack_name, region)
        except ClientError as exc:
            if _stack_check_skippable_error(exc):
                code = exc.response.get("Error", {}).get("Code", "ClientError")
                print(
                    f"Warning: could not check stack {stack_name!r} in {region} "
                    f"({code}). Continuing with other regions."
                )
                continue
            raise
        if info:
            ordered.append(info)
    return ordered


def force_delete_cloudformation_stacks(
    stacks: Sequence[ExistingStack],
    *,
    wait: bool = True,
) -> None:
    """Delete stacks via CloudFormation (disables termination protection first)."""
    import boto3
    from botocore.exceptions import ClientError

    clients: Dict[str, Any] = {}

    def cfn_for(region: str):
        if region not in clients:
            clients[region] = boto3.client("cloudformation", region_name=region)
        return clients[region]

    for stack in stacks:
        cfn = cfn_for(stack.region)
        if stack.termination_protection:
            print(f"Disabling termination protection on {stack.name} ({stack.region})")
            cfn.update_termination_protection(
                EnableTerminationProtection=False,
                StackName=stack.name,
            )
        if stack.status == "DELETE_IN_PROGRESS":
            print(f"{stack.name} ({stack.region}) is already deleting.")
        else:
            print(f"Deleting stack {stack.name} ({stack.region}) ...")
            try:
                cfn.delete_stack(StackName=stack.name)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code == "ValidationError" and "DELETE_IN_PROGRESS" in str(exc):
                    print(f"{stack.name} is already deleting.")
                else:
                    raise

    if not wait:
        return

    for stack in stacks:
        cfn = cfn_for(stack.region)
        print(f"Waiting for {stack.name} ({stack.region}) to finish deleting ...")
        try:
            cfn.get_waiter("stack_delete_complete").wait(
                StackName=stack.name,
                WaiterConfig={"Delay": 15, "MaxAttempts": 120},
            )
        except ClientError:
            detail = describe_existing_stack(stack.name, stack.region)
            status = detail.status if detail else "unknown"
            raise SystemExit(
                f"Stack {stack.name} ({stack.region}) did not delete cleanly "
                f"(status={status}). Resolve in the CloudFormation console and retry."
            ) from None


def handle_existing_stacks_at_start(
    args: argparse.Namespace,
    regional_region: str,
    *,
    config_values: Optional[Dict[str, str]] = None,
) -> None:
    """At wizard start: report existing stacks and optionally force-delete them."""
    if getattr(args, "skip_stack_check", False) or args.config_only or args.synth_only:
        return

    if config_values is None and ENV_PATH.is_file():
        config_values = read_env_file(ENV_PATH)

    try:
        existing = discover_existing_doc_redaction_stacks(
            regional_region, config_values
        )
    except Exception as exc:
        print(f"Existing stack check skipped: {exc}")
        return

    if not existing:
        return

    print("\n--- Existing doc_redaction CloudFormation stacks ---")
    for stack in existing:
        line = f"  {stack.name} ({stack.region}): {stack.status}"
        if stack.termination_protection:
            line += " [termination protection ON]"
        print(line)

    if args.force_delete_stacks:
        should_delete = True
    elif args.yes:
        print(
            "\nStacks already exist in this account/region. "
            "Pass --force-delete-stacks to remove them before deploy, "
            "or omit it to update in place."
        )
        return
    else:
        should_delete = ask_yes_no(
            "Force-delete these stacks before continuing? "
            "(disables termination protection and deletes all stack resources)",
            default=False,
        )

    if not should_delete:
        print("Keeping existing stacks (deploy will update them in place).\n")
        return

    if not args.force_delete_stacks and not args.yes:
        if not ask_yes_no(
            "This permanently deletes AWS resources in these stacks. Proceed?",
            default=False,
        ):
            print("Stack deletion cancelled.\n")
            return

    force_delete_cloudformation_stacks(existing)
    print("Existing stacks deleted.\n")


def fetch_stack_output(
    stack_name: str,
    output_key: str,
    region: str,
) -> Optional[str]:
    import boto3
    from botocore.exceptions import ClientError

    cfn = boto3.client("cloudformation", region_name=region)
    try:
        response = cfn.describe_stacks(StackName=stack_name)
    except ClientError:
        return None
    for stack in response.get("Stacks", []):
        for output in stack.get("Outputs", []):
            if output.get("OutputKey") == output_key:
                return output.get("OutputValue")
    return None


def apply_post_deploy_fixup(values: Dict[str, str], assume_yes: bool) -> bool:
    """Return True if post-deploy fixup changed env and/or Cognito callbacks."""
    from cdk_post_deploy import (
        apply_cognito_alb_callback_fixup,
        cognito_alb_callbacks_need_update,
    )

    if values.get("ENABLE_HEADLESS_DEPLOYMENT") == "True":
        return False

    region = values.get("AWS_REGION", "")
    express = values.get("USE_ECS_EXPRESS_MODE") == "True"
    cloudfront = values.get("USE_CLOUDFRONT") == "True"
    fixup_applied = False

    pool_id = fetch_stack_output(REGIONAL_STACK, "CognitoPoolId", region)
    client_id = fetch_stack_output(REGIONAL_STACK, "CognitoAppClientId", region)
    if not pool_id or not client_id:
        print(
            "CognitoPoolId or CognitoAppClientId stack output missing; "
            "skipping Cognito callback fixup."
        )
        return False

    if express and not cloudfront:
        endpoint = fetch_stack_output(REGIONAL_STACK, "ExpressServiceEndpoint", region)
        if not endpoint:
            print("No ExpressServiceEndpoint output found; skipping Cognito URL fixup.")
            return False
        endpoint = endpoint.rstrip("/")
        current = (values.get("ECS_EXPRESS_COGNITO_REDIRECT_BASE") or "").strip()
        env_needs_patch = current != endpoint
        cognito_needs_update = cognito_alb_callbacks_need_update(
            pool_id, client_id, endpoint, aws_region=region
        )
        if not env_needs_patch and not cognito_needs_update:
            print("Express Cognito redirect base and callback URLs already set.")
        else:
            if env_needs_patch:
                print(f"Setting ECS_EXPRESS_COGNITO_REDIRECT_BASE={endpoint}")
                patch_env_file(
                    ENV_PATH,
                    {
                        "ECS_EXPRESS_COGNITO_REDIRECT_BASE": endpoint,
                        "COGNITO_REDIRECTION_URL": endpoint,
                    },
                )
                fixup_applied = True
            if cognito_needs_update and (
                assume_yes
                or ask_yes_no(
                    "Update Cognito app client callback URLs for the Express endpoint "
                    "(no CDK redeploy)?",
                    True,
                )
            ):
                if apply_cognito_alb_callback_fixup(
                    user_pool_id=pool_id,
                    client_id=client_id,
                    redirect_base=endpoint,
                    aws_region=region,
                ):
                    fixup_applied = True
            elif env_needs_patch and cognito_needs_update:
                print(
                    "Env updated with Express URL; Cognito callbacks unchanged. "
                    "ALB login will fail until callback URLs are updated."
                )

    elif cloudfront:
        cf_domain = fetch_stack_output(
            CLOUDFRONT_STACK, "CloudFrontDistributionURL", CLOUDFRONT_STACK_REGION
        )
        if not cf_domain:
            print(
                "No CloudFrontDistributionURL output found; skipping CloudFront fixup."
            )
            return fixup_applied
        redirect_base = f"https://{cf_domain.strip()}"
        current = (values.get("CLOUDFRONT_DOMAIN") or "").strip()
        env_needs_patch = current != cf_domain.strip()
        cognito_needs_update = cognito_alb_callbacks_need_update(
            pool_id, client_id, redirect_base, aws_region=region
        )
        if not env_needs_patch and not cognito_needs_update:
            print("CloudFront domain and Cognito callback URLs already set.")
            return fixup_applied
        if env_needs_patch:
            print(f"Setting CLOUDFRONT_DOMAIN={cf_domain}")
            patch_env_file(
                ENV_PATH,
                {
                    "CLOUDFRONT_DOMAIN": cf_domain,
                    "COGNITO_REDIRECTION_URL": redirect_base,
                },
            )
            fixup_applied = True
        if cognito_needs_update and (
            assume_yes
            or ask_yes_no(
                "Update Cognito app client callback URLs for the CloudFront domain "
                "(no CDK redeploy)?",
                True,
            )
        ):
            if apply_cognito_alb_callback_fixup(
                user_pool_id=pool_id,
                client_id=client_id,
                redirect_base=redirect_base,
                aws_region=region,
            ):
                fixup_applied = True
        elif env_needs_patch and cognito_needs_update:
            print(
                "Env updated with CloudFront domain; Cognito callbacks unchanged. "
                "ALB login will fail until callback URLs are updated."
            )

    if express:
        from cdk_functions import normalize_pi_alb_path_prefix
        from cdk_post_deploy import (
            apply_cognito_secret_fixup_from_stack,
            apply_express_alb_listener_target_group_fixup,
            apply_express_disable_in_app_cognito_auth,
            remove_express_listener_rules_without_cognito,
        )

        cluster_name = (
            values.get("CLUSTER_NAME") or f"{values.get('CDK_PREFIX', '')}Cluster"
        )
        main_service = values.get("ECS_EXPRESS_SERVICE_NAME") or values.get(
            "ECS_SERVICE_NAME", ""
        )
        pi_enabled = values.get("ENABLE_PI_AGENT_EXPRESS_SERVICE") == "True"
        pi_service = values.get("ECS_PI_EXPRESS_SERVICE_NAME") if pi_enabled else None
        pi_prefix = normalize_pi_alb_path_prefix(
            values.get("PI_ALB_PATH_PREFIX", "/agent")
        )
        secret_name = values.get("COGNITO_USER_POOL_CLIENT_SECRET_NAME") or (
            f"{values.get('CDK_PREFIX', '')}ParamCognitoSecret"
        )
        try:
            if remove_express_listener_rules_without_cognito(aws_region=region):
                fixup_applied = True
            if apply_express_alb_listener_target_group_fixup(
                cluster_name=cluster_name,
                main_service_name=main_service,
                pi_service_name=pi_service,
                pi_path_prefixes=(
                    [pi_prefix, f"{pi_prefix}/*"] if pi_service else None
                ),
                aws_region=region,
            ):
                fixup_applied = True
            if pool_id and client_id:
                if apply_cognito_secret_fixup_from_stack(
                    stack_name=REGIONAL_STACK,
                    secret_name=secret_name,
                    cluster_name=cluster_name,
                    main_service_name=main_service,
                    aws_region=region,
                    recycle_tasks=False,
                ):
                    fixup_applied = True
                if apply_express_disable_in_app_cognito_auth(
                    cluster_name, main_service, aws_region=region
                ):
                    fixup_applied = True
        except Exception as exc:
            print(f"Warning: could not sync Express ALB/Cognito settings: {exc}")

    return fixup_applied


def run_quickstart(python_exe: Path) -> None:
    if not QUICKSTART_SCRIPT.is_file():
        raise SystemExit(f"Quickstart script not found: {QUICKSTART_SCRIPT}")
    cmd = [str(python_exe), str(QUICKSTART_SCRIPT)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(CDK_DIR), env=_deploy_env(), check=True)


# ---------------------------------------------------------------------------
# Pi agent configuration
# ---------------------------------------------------------------------------


def apply_pi_cli_flags(args: argparse.Namespace, answers: InstallAnswers) -> None:
    if getattr(args, "enable_pi", False):
        preset = merge_preset(answers.profile, answers.custom_overrides)
        if preset.get("USE_ECS_EXPRESS_MODE") == "True":
            answers.enable_pi_express = True
        else:
            answers.enable_pi_legacy = True
            answers.enable_service_connect = True
    if getattr(args, "enable_pi_express", False):
        answers.enable_pi_express = True
    if getattr(args, "enable_pi_legacy", False):
        answers.enable_pi_legacy = True
        answers.enable_service_connect = True
    if args.pi_alb_routing:
        answers.pi_alb_routing = args.pi_alb_routing
    if args.pi_path_prefix:
        answers.pi_alb_path_prefix = args.pi_path_prefix
    if args.pi_host_header:
        answers.pi_alb_host_header = args.pi_host_header
    if args.pi_listener_priority:
        answers.pi_alb_listener_rule_priority = args.pi_listener_priority
    if args.pi_gradio_port:
        answers.pi_gradio_port = args.pi_gradio_port
    if args.sc_discovery_name:
        answers.sc_discovery_name = args.sc_discovery_name
    if args.pi_provider:
        answers.pi_default_provider = args.pi_provider
    if args.skip_pi_agent_env:
        answers.write_pi_agent_env = False


def configure_app_config_options(
    answers: InstallAnswers,
    args: argparse.Namespace,
    *,
    interactive: bool,
) -> None:
    if getattr(args, "skip_app_config_env", False):
        answers.write_app_config_env = False
        return

    if getattr(args, "overwrite_app_config_env", False):
        answers.write_app_config_env = True
        answers.overwrite_app_config_env = True
        return

    if not interactive:
        answers.write_app_config_env = True
        return

    example_label = (
        APP_CONFIG_ENV_EXAMPLE.name if APP_CONFIG_ENV_EXAMPLE.is_file() else "defaults"
    )
    if ask_yes_no(
        f"Write/update {APP_CONFIG_ENV_PATH.name} for the deployed app "
        f"(from {example_label} + AWS resource names)?",
        default=True,
    ):
        answers.write_app_config_env = True
        if APP_CONFIG_ENV_PATH.is_file():
            answers.overwrite_app_config_env = ask_yes_no(
                f"{APP_CONFIG_ENV_PATH.name} exists — replace file from example template?",
                default=False,
            )
    else:
        answers.write_app_config_env = False


def configure_pi_options(
    answers: InstallAnswers,
    args: argparse.Namespace,
    *,
    interactive: bool,
    assume_yes: bool,
) -> None:
    preset = merge_preset(answers.profile, answers.custom_overrides)
    use_express = preset.get("USE_ECS_EXPRESS_MODE") == "True"
    use_cloudfront = preset.get("USE_CLOUDFRONT") == "True"

    apply_pi_cli_flags(args, answers)

    if not answers.pi_enabled and interactive:
        if use_express:
            label = "Deploy agent mode (second Gradio app on Express, shared ALB)?"
            answers.enable_pi_express = ask_yes_no(label, default=False)
        else:
            label = "Deploy agent mode (second Gradio app on legacy Fargate + Service Connect)?"
            if ask_yes_no(label, default=False):
                answers.enable_pi_legacy = True
                answers.enable_service_connect = True

    if not answers.pi_enabled:
        return

    if not answers.pi_alb_listener_rule_priority:
        answers.pi_alb_listener_rule_priority = default_pi_listener_priority(
            use_cloudfront
        )

    if interactive and not (
        args.pi_alb_routing or args.pi_path_prefix or args.pi_host_header or assume_yes
    ):
        print("\n--- Agent mode ALB routing ---")
        ridx = ask_choice(
            "How should the shared ALB route traffic to the Agent UI?",
            [
                "Path prefix (default /agent/ — e.g. https://host/agent/)",
                "Dedicated hostname (PI_ALB_HOST_HEADER)",
                "Both path prefix and hostname",
            ],
            default_index=0,
        )
        answers.pi_alb_routing = PI_ALB_ROUTING_MODES[ridx]

        if answers.pi_alb_routing in ("path", "both"):
            answers.pi_alb_path_prefix = ask(
                "Agent path prefix",
                answers.pi_alb_path_prefix,
            )
        if answers.pi_alb_routing in ("host", "both"):
            default_host = ""
            if answers.ssl_domain:
                default_host = f"agent.{answers.ssl_domain}"
            answers.pi_alb_host_header = ask(
                "Agent ALB host header (DNS CNAME to CloudFront/ALB)",
                default_host,
            )

        if use_cloudfront:
            default_pri = default_pi_listener_priority(True)
            answers.pi_alb_listener_rule_priority = ask(
                "Agent ALB listener rule priority (default 3; priorities 1–2 reserved)",
                default_pri,
            )

        answers.sc_discovery_name = ask(
            "Service Connect discovery name for main app",
            answers.sc_discovery_name,
        )
        answers.pi_gradio_port = ask("Agent Gradio port", answers.pi_gradio_port)

        if ask_yes_no(
            f"Write/update {PI_AGENT_ENV_PATH.name} for AWS ECS (DOC_REDACTION_GRADIO_URL, etc.)?",
            default=True,
        ):
            answers.write_pi_agent_env = True
            if PI_AGENT_ENV_PATH.is_file():
                answers.overwrite_pi_agent_env = ask_yes_no(
                    "pi_agent.env exists — replace file from example template?",
                    default=False,
                )
        else:
            answers.write_pi_agent_env = False

    print(
        f"Agent mode: "
        f"{'Express' if answers.enable_pi_express else 'legacy Fargate'}, "
        f"routing={answers.pi_alb_routing}, "
        f"prefix={normalize_pi_path_prefix(answers.pi_alb_path_prefix)}"
    )


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------


def run_wizard(args: argparse.Namespace) -> InstallAnswers:
    answers = InstallAnswers()
    interactive = not args.yes
    assume_yes = args.yes

    if args.profile:
        answers.profile = args.profile
    elif interactive:
        idx = ask_choice(
            "Deployment profile",
            [
                "Demonstration (Express, no CloudFront, no delete protection)",
                "Production (ACM cert, CloudFront, delete protection)",
                "Headless batch only (demo-style networking, no Express web UI)",
                "Custom (configure individual toggles)",
            ],
        )
        answers.profile = ("demo", "production", "headless", "custom")[idx]
    else:
        answers.profile = "demo"

    if answers.profile == "headless":
        answers.enable_headless = True
    elif getattr(args, "headless", False):
        if answers.profile == "demo":
            raise SystemExit(
                "Headless batch mode is not available with --profile demo (Express). "
                "Use --profile headless, --profile production --headless, "
                "or --profile custom without ECS Express."
            )
        answers.enable_headless = True
    elif interactive and answers.profile == "production":
        answers.enable_headless = ask_yes_no(
            "Enable headless batch-only deployment (S3 → Lambda → one-shot ECS, "
            "no always-on web UI)?",
            default=False,
        )

    if answers.profile == "custom" and interactive:
        answers.custom_overrides["USE_ECS_EXPRESS_MODE"] = (
            "True" if ask_yes_no("Use ECS Express Mode?", False) else "False"
        )
        answers.custom_overrides["USE_CLOUDFRONT"] = (
            "True" if ask_yes_no("Use CloudFront?", True) else "False"
        )
        answers.custom_overrides["RUN_USEAST_STACK"] = (
            "True"
            if answers.custom_overrides.get("USE_CLOUDFRONT") == "True"
            and ask_yes_no(
                "Deploy us-east-1 CloudFront stack (RUN_USEAST_STACK)?", True
            )
            else "False"
        )
        answers.custom_overrides["ENABLE_RESOURCE_DELETE_PROTECTION"] = (
            "True" if ask_yes_no("Enable delete protection?", True) else "False"
        )
        answers.custom_overrides["ENABLE_APPREGISTRY"] = (
            "True" if ask_yes_no("Enable AppRegistry?", True) else "False"
        )
        if getattr(args, "headless", False):
            if answers.custom_overrides.get("USE_ECS_EXPRESS_MODE") == "True":
                raise SystemExit(
                    "Headless batch mode requires USE_ECS_EXPRESS_MODE=False in "
                    "the custom profile."
                )
            answers.enable_headless = True
        elif profile_allows_headless_add_on(answers):
            answers.enable_headless = ask_yes_no(
                "Enable headless batch-only deployment (S3 → Lambda → one-shot ECS, "
                "no always-on web UI)?",
                default=False,
            )

    headless_err = headless_profile_error(answers)
    if headless_err:
        raise SystemExit(headless_err)

    try:
        account, region = get_aws_identity(args.region)
    except Exception as exc:
        raise SystemExit(f"AWS credentials error: {exc}") from exc

    answers.aws_account_id = args.account or account
    answers.aws_region = args.region or region

    if interactive and not args.account:
        answers.aws_account_id = ask("AWS account ID", answers.aws_account_id)
    if interactive and not args.region:
        answers.aws_region = ask("AWS region", answers.aws_region)

    default_prefix = f"{answers.profile.capitalize()}-Redaction-"
    answers.cdk_prefix = args.cdk_prefix or (
        ask("CDK resource prefix (e.g. MyOrg-Redaction-)", default_prefix)
        if interactive
        else default_prefix
    )
    if not answers.cdk_prefix.endswith("-"):
        answers.cdk_prefix += "-"

    default_cognito = default_cognito_domain_prefix_from_cdk_prefix(answers.cdk_prefix)
    raw_cognito_prefix = args.cognito_prefix or (
        ask(
            "Cognito hosted UI domain prefix (must be globally unique; "
            "cannot contain aws, amazon, or cognito)",
            default_cognito,
        )
        if interactive
        else default_cognito
    )
    answers.cognito_domain_prefix = sanitize_cognito_domain_prefix(raw_cognito_prefix)

    # VPC
    if args.new_vpc_cidr:
        answers.vpc_mode = "new"
        answers.new_vpc_cidr = args.new_vpc_cidr
    elif args.vpc_name:
        answers.vpc_mode = "existing"
        answers.vpc_name = args.vpc_name
    elif interactive:
        vpc_idx = ask_choice(
            "VPC",
            ["Create new VPC", "Use existing VPC"],
        )
        answers.vpc_mode = "new" if vpc_idx == 0 else "existing"
    else:
        answers.vpc_mode = "existing"

    if answers.vpc_mode == "new":
        prompt_new_vpc_cidr(answers, interactive=interactive)
    else:
        if not answers.vpc_name:
            vpcs = list_vpcs(answers.aws_region)
            if not vpcs:
                raise SystemExit("No VPCs found in region.")
            if interactive:
                labels = [f"{v['name']} ({v['cidr']})" for v in vpcs]
                vidx = ask_choice("Select VPC", labels)
                answers.vpc_name = vpcs[vidx]["name"]
            else:
                raise SystemExit(
                    "--vpc-name required for existing VPC in non-interactive mode."
                )
        vpc_list = list_vpcs(answers.aws_region)
        vpc_id = next(
            (v["id"] for v in vpc_list if v["name"] == answers.vpc_name),
            None,
        )

        # Subnets (public only for Express or demo-style headless)
        apply_subnet_cli_flags(args, answers)
        public_subnets_only = answers_use_public_subnets_only(answers)
        if public_subnets_only:
            if interactive:
                answers.public_subnet_mode = ask_subnet_tier_mode("Public")
            else:
                answers.public_subnet_mode, _ = resolve_subnet_tier_modes(args)
            answers.private_subnet_mode = "auto"
            answers.private_subnet_names = []
            answers.private_subnet_cidrs = []
            answers.private_subnet_azs = []
        elif interactive:
            layout_idx = ask_choice(
                "Subnets in existing VPC",
                [
                    "Auto-discover public and private subnets",
                    "Use existing named subnets (both tiers)",
                    "Create new stack-specific subnets (both tiers)",
                    "Configure public and private separately",
                ],
                default_index=0,
            )
            if layout_idx < 3:
                shared_mode = SUBNET_TIER_MODES[layout_idx]
                answers.public_subnet_mode = shared_mode
                answers.private_subnet_mode = shared_mode
            else:
                answers.public_subnet_mode = ask_subnet_tier_mode("Public")
                answers.private_subnet_mode = ask_subnet_tier_mode("Private")
        else:
            answers.public_subnet_mode, answers.private_subnet_mode = (
                resolve_subnet_tier_modes(args)
            )

        azs = list_availability_zones(answers.aws_region)[:3]
        vpc_cidr = next(
            (v["cidr"] for v in vpc_list if v["name"] == answers.vpc_name),
            "",
        )
        if vpc_id:
            subnets = list_subnets_in_vpc(vpc_id, answers.aws_region)
            configure_subnet_tier(
                answers,
                "public",
                answers.public_subnet_mode,
                subnets,
                azs,
                interactive=interactive,
                vpc_cidr=vpc_cidr,
            )
            if not public_subnets_only:
                reserved_public = (
                    list(answers.public_subnet_cidrs)
                    if answers.public_subnet_mode == "create"
                    else []
                )
                configure_subnet_tier(
                    answers,
                    "private",
                    answers.private_subnet_mode,
                    subnets,
                    azs,
                    interactive=interactive,
                    vpc_cidr=vpc_cidr,
                    reserved_subnet_cidrs=reserved_public,
                )

        # Optional infra reuse
        if (
            vpc_id
            and interactive
            and ask_yes_no("Configure optional existing infra (IGW/ALB)?", False)
        ):
            igws = list_igws_for_vpc(vpc_id, answers.aws_region)
            if igws:
                # igws_ids = [g["id"] for g in igws]
                print("Internet gateways:", ", ".join(g["id"] for g in igws))
                answers.existing_igw_id = ask("EXISTING_IGW_ID (blank=new)", "")
            albs = list_albs_in_vpc(vpc_id, answers.aws_region)
            if albs:
                labels = [f"{a['name']} ({a['dns']})" for a in albs]
                if ask_yes_no("Reuse an existing ALB?", False):
                    aidx = ask_choice("Select ALB", labels)
                    answers.existing_alb_arn = albs[aidx]["arn"]
                    answers.existing_alb_dns = albs[aidx]["dns"]

    # Production TLS
    preset = merge_preset(answers_preset_profile(answers), answers.custom_overrides)
    use_express = preset.get("USE_ECS_EXPRESS_MODE") == "True"
    if (
        preset.get("USE_CLOUDFRONT") == "True"
        and preset.get("USE_ECS_EXPRESS_MODE") != "True"
        and not answers_use_headless(answers)
    ):
        if args.cert_arn:
            answers.acm_cert_arn = args.cert_arn
            answers.ssl_domain = args.domain or ""
        elif interactive:
            certs = list_acm_certificates(answers.aws_region)
            if not certs:
                raise SystemExit(f"No ISSUED ACM certificates in {answers.aws_region}.")
            labels = [f"{c['label']} — {c['arn']}" for c in certs]
            cidx = ask_choice("Select ACM certificate", labels)
            answers.acm_cert_arn = certs[cidx]["arn"]
            answers.ssl_domain = ask(
                "SSL certificate domain (Cognito/CloudFront URL)",
                certs[cidx]["domain"],
            )
        else:
            raise SystemExit(
                "--cert-arn and --domain required for production in non-interactive mode."
            )
        if interactive:
            answers.cloudfront_geo = ask(
                "CloudFront geo restriction (e.g. GB, blank=none)", ""
            )

    if not answers_use_headless(answers):
        configure_pi_options(
            answers,
            args,
            interactive=interactive,
            assume_yes=assume_yes,
        )
    else:
        answers.enable_s3_batch = True

    # Advanced add-ons (non-Pi agent)
    is_express = use_express and not answers_use_headless(answers)
    if answers_use_headless(answers) and interactive:
        mem = ask("ECS task memory (MB)", answers.ecs_memory)
        if mem:
            answers.ecs_memory = mem
    elif interactive and ask_yes_no("Configure other optional add-ons?", False):
        if not is_express:
            if not answers.enable_service_connect:
                answers.enable_service_connect = ask_yes_no(
                    "Enable ECS Service Connect (without agent mode)?", False
                )
            answers.enable_s3_batch = ask_yes_no("Enable S3 batch ECS trigger?", False)
        mem = ask("ECS task memory (MB)", answers.ecs_memory)
        if mem:
            answers.ecs_memory = mem

    configure_app_config_options(answers, args, interactive=interactive)

    answers.python_path = args.python
    return answers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive CDK installer for doc_redaction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--profile",
        choices=("demo", "production", "headless", "custom"),
        help="Base profile; use --profile headless or production --headless for batch-only",
    )
    p.add_argument(
        "--yes", action="store_true", help="Accept defaults / skip confirmations"
    )
    p.add_argument("--config-only", action="store_true", help="Write config files only")
    p.add_argument(
        "--deploy-only",
        action="store_true",
        help="Skip wizard; use existing cdk_config.env",
    )
    p.add_argument(
        "--synth-only", action="store_true", help="Run cdk synth only (no deploy)"
    )
    p.add_argument("--skip-deploy", action="store_true", help="Skip cdk deploy")
    p.add_argument(
        "--skip-quickstart",
        action="store_true",
        help="Skip post_cdk_build_quickstart.py",
    )
    p.add_argument(
        "--skip-stack-check",
        action="store_true",
        help="Do not check for existing CloudFormation stacks at startup",
    )
    p.add_argument(
        "--force-delete-stacks",
        action="store_true",
        help="If doc_redaction stacks already exist, delete them before continuing "
        "(disables termination protection; implies consent in non-interactive mode)",
    )
    p.add_argument(
        "--skip-cdk-json", action="store_true", help="Do not update cdk.json"
    )
    p.add_argument(
        "--refresh-cdk-json", action="store_true", help="Force rewrite cdk.json app key"
    )
    p.add_argument("--python", help="Python executable for cdk.json")
    p.add_argument("--region", help="AWS region")
    p.add_argument("--account", help="AWS account ID")
    p.add_argument("--cdk-prefix", help="CDK resource prefix")
    p.add_argument("--cognito-prefix", help="Cognito user pool domain prefix")
    p.add_argument("--vpc-name", help="Existing VPC Name tag")
    p.add_argument("--new-vpc-cidr", help="CIDR for new VPC")
    p.add_argument(
        "--subnet-mode",
        choices=SUBNET_TIER_MODES,
        help="Subnet mode for both public and private tiers (overridden by per-tier flags)",
    )
    p.add_argument(
        "--public-subnet-mode",
        choices=SUBNET_TIER_MODES,
        help="Public subnet tier: auto, existing, or create",
    )
    p.add_argument(
        "--private-subnet-mode",
        choices=SUBNET_TIER_MODES,
        help="Private subnet tier: auto, existing, or create",
    )
    p.add_argument(
        "--public-subnet-names",
        help="Comma-separated existing public subnet names (with --public-subnet-mode existing)",
    )
    p.add_argument(
        "--private-subnet-names",
        help="Comma-separated existing private subnet names (with --private-subnet-mode existing)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Batch-only add-on (production/custom without Express only; not valid with demo)",
    )
    p.add_argument("--cert-arn", help="ACM certificate ARN (production)")
    p.add_argument("--domain", help="SSL certificate domain (production)")
    pi = p.add_argument_group("Agent mode (second Gradio app)")
    pi.add_argument(
        "--enable-pi",
        action="store_true",
        help="Enable agent mode for current profile (Express on demo, legacy on production)",
    )
    pi.add_argument(
        "--enable-pi-express",
        action="store_true",
        help="Enable agent mode on ECS Express Mode (demo)",
    )
    pi.add_argument(
        "--enable-pi-legacy",
        action="store_true",
        help="Enable agent mode on legacy Fargate (implies Service Connect)",
    )
    pi.add_argument(
        "--pi-alb-routing",
        choices=PI_ALB_ROUTING_MODES,
        help="ALB routing mode for Agent UI (default: path)",
    )
    pi.add_argument(
        "--pi-path-prefix", default="", help="Agent path prefix (default: /agent)"
    )
    pi.add_argument(
        "--pi-host-header",
        default="",
        help="Dedicated hostname for Agent mode when routing is host or both",
    )
    pi.add_argument(
        "--pi-listener-priority",
        default="",
        help="ALB listener rule priority for Agent mode (default 3; priorities 1–2 reserved)",
    )
    pi.add_argument(
        "--pi-gradio-port",
        default="",
        help="Agent mode Gradio listen port (default 7862)",
    )
    pi.add_argument(
        "--sc-discovery-name",
        default="",
        help="Service Connect name for main app (default redaction)",
    )
    pi.add_argument(
        "--pi-provider",
        default="",
        help="PI_DEFAULT_PROVIDER for pi_agent.env (default amazon-bedrock)",
    )
    pi.add_argument(
        "--skip-pi-agent-env",
        action="store_true",
        help="Do not write config/pi_agent.env",
    )
    p.add_argument(
        "--skip-app-config-env",
        action="store_true",
        help="Do not write cdk/config/app_config.env from app_config.env.example",
    )
    p.add_argument(
        "--overwrite-app-config-env",
        action="store_true",
        help="Replace existing config/app_config.env from app_config.env.example (non-interactive)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("doc_redaction CDK installer\n")

    # Prerequisites
    try:
        cdk_version = check_cdk_cli()
        print(f"CDK CLI: {cdk_version}")
    except SystemExit:
        if not args.config_only:
            raise
        print("Warning: CDK CLI not found; config-only mode continuing.")

    if not args.config_only and not args.synth_only:
        try:
            _stack_account, stack_region = get_aws_identity(args.region)
            handle_existing_stacks_at_start(args, stack_region)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"Existing stack check skipped: {exc}")

    python_exe: Optional[Path] = None
    if not args.skip_cdk_json:
        python_exe = resolve_python_executable(
            override=args.python,
            interactive=not args.yes and not args.deploy_only,
            assume_yes=args.yes or args.deploy_only,
        )
        write_cdk_json(
            python_exe,
            force=args.refresh_cdk_json or not CDK_JSON_PATH.is_file(),
            skip=args.skip_cdk_json,
        )
    elif CDK_JSON_PATH.is_file():
        # Best-effort parse python from existing cdk.json
        try:
            data = json.loads(CDK_JSON_PATH.read_text(encoding="utf-8"))
            app_cmd = data.get("app", "")
            if app_cmd:
                python_exe = Path(app_cmd.split()[0])
        except (json.JSONDecodeError, IndexError):
            pass

    values: Dict[str, str]
    if args.deploy_only:
        if not ENV_PATH.is_file():
            raise SystemExit(
                f"No config at {ENV_PATH}. Run without --deploy-only first."
            )
        values = read_env_file(ENV_PATH)
        if not python_exe or not python_exe.is_file():
            python_exe = resolve_python_executable(
                override=args.python, interactive=False, assume_yes=True
            )
    else:
        answers = run_wizard(args)
        errors = validate_install_answers(answers)
        errors.extend(validate_subnet_answers(answers))
        errors.extend(enrich_existing_subnet_details_from_aws(answers))
        values = build_env_values(answers)
        errors.extend(validate_env_values(values))
        if errors:
            print("Configuration errors:")
            for err in errors:
                print(f"  - {err}")
            return 1

        if not python_exe:
            python_exe = resolve_python_executable(
                override=answers.python_path or args.python,
                interactive=not args.yes,
                assume_yes=args.yes,
            )
            if not args.skip_cdk_json:
                write_cdk_json(python_exe, force=args.refresh_cdk_json)

        print_summary(values, python_exe)
        if not args.yes and not ask_yes_no("Write config/cdk_config.env?", True):
            print("Aborted.")
            return 0

        write_env_file(ENV_PATH, values)
        if answers.write_app_config_env:
            write_app_config_env_file(
                answers,
                values,
                overwrite=answers.overwrite_app_config_env,
            )
        if answers.pi_enabled:
            write_pi_agent_env_file(
                answers,
                overwrite=answers.overwrite_pi_agent_env,
            )

    apply_cdk_runtime_env(values)

    if args.config_only:
        print("Config written. Exiting (--config-only).")
        return 0

    run_smoke_test_if_needed(python_exe, args)

    # Bootstrap prompt
    account = values.get("AWS_ACCOUNT_ID", "")
    region = values.get("AWS_REGION", "")
    if account and region:
        try:
            if cdk_bootstrap_needed(account, region):
                print(f"CDK bootstrap not found in {region}.")
                if args.yes or ask_yes_no(
                    f"Run cdk bootstrap aws://{account}/{region}?", True
                ):
                    run_cdk_command(
                        ["bootstrap", f"aws://{account}/{region}"],
                        check=True,
                    )
        except Exception as exc:
            print(f"Bootstrap check skipped: {exc}")

    run_cdk_command(["synth"], check=True)

    if args.synth_only or args.skip_deploy:
        print("Synth complete. Skipping deploy.")
        return 0

    if not args.yes and not ask_yes_no("Run cdk deploy --all?", True):
        print("Deploy skipped.")
        return 0

    run_cdk_command(["deploy", "--all", "--require-approval", "broadening"], check=True)

    values = read_env_file(ENV_PATH)
    apply_post_deploy_fixup(values, assume_yes=args.yes)

    if args.skip_quickstart:
        print("Skipping post-deploy quickstart.")
    else:
        is_headless = values.get("ENABLE_HEADLESS_DEPLOYMENT") == "True"
        is_demo = values.get("USE_ECS_EXPRESS_MODE") == "True"
        if args.yes:
            run_qs = True
        elif is_headless:
            run_qs = ask_yes_no(
                "Run post_cdk_build_quickstart.py (CodeBuild image only; no ECS service)?",
                default=True,
            )
        else:
            run_qs = ask_yes_no(
                "Run post_cdk_build_quickstart.py (CodeBuild + scale ECS)?",
                default=is_demo,
            )
        if run_qs:
            if not python_exe:
                python_exe = resolve_python_executable(
                    assume_yes=True, interactive=False
                )
            run_quickstart(python_exe)

    print("\nDone. Next steps:")
    if (
        values.get("ENABLE_PI_AGENT_EXPRESS_SERVICE") == "True"
        or values.get("ENABLE_PI_AGENT_ECS_SERVICE") == "True"
    ):
        routing = values.get("PI_ALB_ROUTING", "path")
        if routing in ("path", "both"):
            prefix = values.get("PI_ALB_PATH_PREFIX", "/agent")
            print(f"  - Agent UI path: {prefix}/ on your app URL")
        if routing in ("host", "both") and values.get("PI_ALB_HOST_HEADER"):
            print(
                f"  - Agent DNS: CNAME {values['PI_ALB_HOST_HEADER']} -> "
                f"{values.get('CLOUDFRONT_DOMAIN') or 'ALB/Express endpoint'}"
            )
        if PI_AGENT_ENV_PATH.is_file():
            print(
                f"  - Agent runtime config: {PI_AGENT_ENV_PATH} (uploaded by quickstart)"
            )
    if values.get("USE_CLOUDFRONT") == "True":
        domain = values.get("SSL_CERTIFICATE_DOMAIN", "")
        cf = values.get("CLOUDFRONT_DOMAIN", "")
        if domain and cf and cf != "cloudfront_placeholder.net":
            print(f"  - Point DNS CNAME {domain} -> {cf}")
    elif values.get("ENABLE_HEADLESS_DEPLOYMENT") == "True":
        prefix = values.get("S3_BATCH_ENV_PREFIX", "input/config/")
        suffix = values.get("S3_BATCH_ENV_SUFFIX", ".env")
        print(
            f"  - Upload job files to s3://<output-bucket>/{prefix}*{suffix} "
            "(see BatchJobEnvPrefix stack output after deploy)"
        )
        print(
            "  - Lambda starts one-shot ECS tasks with RUN_DIRECT_MODE=True "
            "(see cdk/config/lambda/lambda_function.py)"
        )
    elif values.get("USE_ECS_EXPRESS_MODE") == "True":
        ep = values.get("ECS_EXPRESS_COGNITO_REDIRECT_BASE", "")
        if ep:
            print(f"  - Express endpoint: {ep}")
    if APP_CONFIG_ENV_PATH.is_file():
        print(f"  - App runtime config: {APP_CONFIG_ENV_PATH} (uploaded by quickstart)")
    print(f"  - Config: {ENV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
