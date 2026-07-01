import ipaddress
import json
import os
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple, Union

S3BucketAvailability = Literal["owned", "available", "globally_taken"]
CognitoDomainAvailability = Literal["available", "taken"]

import boto3
import pandas as pd
from aws_cdk import (
    App,
    CfnOutput,
    CfnTag,
    CustomResource,
    Duration,
    Fn,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elb
from aws_cdk import aws_elasticloadbalancingv2_actions as elb_act
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_wafv2 as wafv2
from aws_cdk import custom_resources as cr
from botocore.exceptions import ClientError, NoCredentialsError
from cdk_config import (
    ACCESS_LOG_DYNAMODB_TABLE_NAME,
    APP_CONFIG_ENV_BASENAME,
    AWS_REGION,
    ECS_AVAILABILITY_ZONE_REBALANCING,
    ENABLE_RESOURCE_DELETE_PROTECTION,
    FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    NAT_GATEWAY_EIP_NAME,
    PRIVATE_SUBNET_AVAILABILITY_ZONES,
    PRIVATE_SUBNET_CIDR_BLOCKS,
    PRIVATE_SUBNETS_TO_USE,
    PUBLIC_SUBNET_AVAILABILITY_ZONES,
    PUBLIC_SUBNET_CIDR_BLOCKS,
    PUBLIC_SUBNETS_TO_USE,
    S3_LOG_CONFIG_BUCKET_NAME,
    S3_OUTPUT_BUCKET_NAME,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
)
from constructs import Construct
from dotenv import dotenv_values, set_key

# CDK CLI stores lookup-provider results under these key prefixes in cdk.context.json.
_CDK_LOOKUP_CONTEXT_PREFIXES = (
    "vpc-provider:",
    "load-balancer:",
    "availability-zones:",
    "hosted-zone:",
    "security-group:",
    "key-provider:",
    "ami:",
)


def _ensure_folder_exists(output_folder: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


def is_resource_delete_protection_enabled() -> bool:
    """Whether stack and resource delete protection is enabled (see ENABLE_RESOURCE_DELETE_PROTECTION)."""
    return str(ENABLE_RESOURCE_DELETE_PROTECTION).strip().lower() in (
        "true",
        "1",
        "yes",
    )


def resource_deletion_protection_flag() -> bool:
    """AWS deletion_protection attribute (ALB, DynamoDB tables, Cognito user pools)."""
    return is_resource_delete_protection_enabled()


def ecs_availability_zone_rebalancing(
    setting: str,
) -> ecs.AvailabilityZoneRebalancing:
    """Map ``ECS_AVAILABILITY_ZONE_REBALANCING`` env value to the CDK enum."""
    if setting == "ENABLED":
        return ecs.AvailabilityZoneRebalancing.ENABLED
    return ecs.AvailabilityZoneRebalancing.DISABLED


def managed_resource_removal_policy() -> RemovalPolicy:
    """Removal policy for CDK-managed resources without a native deletion_protection flag."""
    return (
        RemovalPolicy.RETAIN
        if is_resource_delete_protection_enabled()
        else RemovalPolicy.DESTROY
    )


def s3_auto_delete_objects_on_stack_destroy() -> bool:
    """Empty S3 buckets automatically when the stack is destroyed (dev/teardown only)."""
    return not is_resource_delete_protection_enabled()


def ecr_empty_on_delete() -> bool:
    """Force-delete ECR images when the repository is removed on stack destroy."""
    return not is_resource_delete_protection_enabled()


def purge_cdk_lookup_context(file_path: str) -> int:
    """Remove stale CDK lookup cache entries that require the bootstrap lookup role."""
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        context_data = json.load(f)
    cleaned = {
        key: value
        for key, value in context_data.items()
        if not key.startswith(_CDK_LOOKUP_CONTEXT_PREFIXES)
    }
    removed = len(context_data) - len(cleaned)
    if removed:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
        print(f"Removed {removed} stale CDK lookup context key(s) from {file_path}.")
    return removed


def log_aws_credential_context(
    expected_account_id: Optional[str] = None,
    expected_region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Print the active AWS identity and non-secret credential hints for CDK debugging.

    Helps distinguish SSO/assumed-role sessions from long-lived access keys in
    ~/.aws/credentials or environment variables.
    """
    profile = os.environ.get("AWS_PROFILE") or "(not set — using default profile chain)"
    default_region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "(not set in environment)"
    )
    env_access_key_set = bool(os.environ.get("AWS_ACCESS_KEY_ID"))
    env_secret_key_set = bool(os.environ.get("AWS_SECRET_ACCESS_KEY"))
    env_session_token_set = bool(os.environ.get("AWS_SESSION_TOKEN"))

    print("\n--- AWS credential context (CDK / boto3) ---")
    print(f"AWS_PROFILE: {profile}")
    print(f"AWS_REGION / AWS_DEFAULT_REGION (env): {default_region}")
    print(
        "Environment credential variables: "
        f"AWS_ACCESS_KEY_ID={'set' if env_access_key_set else 'not set'}, "
        f"AWS_SECRET_ACCESS_KEY={'set' if env_secret_key_set else 'not set'}, "
        f"AWS_SESSION_TOKEN={'set' if env_session_token_set else 'not set'}"
    )
    if expected_account_id:
        print(f"Configured CDK target account (AWS_ACCOUNT_ID): {expected_account_id}")
    if expected_region:
        print(f"Configured CDK target region (AWS_REGION): {expected_region}")

    session = boto3.Session()
    active_profile = session.profile_name or "(default)"
    print(f"boto3 session profile: {active_profile}")
    print(f"boto3 session region: {session.region_name or '(not set)'}")

    credentials = session.get_credentials()
    credential_summary: Dict[str, Any] = {
        "profile": profile,
        "session_profile": active_profile,
    }

    if credentials is None:
        print("WARNING: No AWS credentials found in the default provider chain.")
        print("--- End AWS credential context ---\n")
        credential_summary["error"] = "no_credentials"
        return credential_summary

    frozen = credentials.get_frozen_credentials()
    access_key = frozen.access_key or ""
    access_key_prefix = (access_key[:4] + "...") if len(access_key) >= 4 else "(none)"
    credential_summary["access_key_prefix"] = access_key_prefix

    if env_access_key_set:
        credential_source = "environment variables (highest precedence)"
    elif access_key.startswith("AKIA"):
        credential_source = "long-lived access key (likely ~/.aws/credentials [default] or named profile)"
    elif access_key.startswith("ASIA"):
        credential_source = "temporary credentials (SSO, assumed role, or STS session)"
    else:
        credential_source = (
            "resolved credentials (source could not be classified from key prefix)"
        )

    print(f"Inferred credential type: {credential_source}")
    credential_summary["inferred_credential_type"] = credential_source

    if env_access_key_set and profile != "(not set — using default profile chain)":
        print(
            "NOTE: AWS_ACCESS_KEY_ID is set in the environment, so it overrides "
            f"profile '{profile}' and SSO."
        )

    try:
        sts = session.client("sts", region_name=session.region_name or expected_region)
        identity = sts.get_caller_identity()
    except (ClientError, NoCredentialsError) as exc:
        print(f"WARNING: sts:GetCallerIdentity failed: {exc}")
        print("--- End AWS credential context ---\n")
        credential_summary["error"] = str(exc)
        return credential_summary

    account = identity.get("Account", "")
    arn = identity.get("Arn", "")
    user_id = identity.get("UserId", "")

    print(f"Caller account: {account}")
    print(f"Caller ARN: {arn}")
    print(f"Caller UserId: {user_id}")

    if ":assumed-role/" in arn:
        principal_kind = "assumed IAM role (typical for SSO or role chaining)"
    elif ":user/" in arn:
        principal_kind = "IAM user (typical for static access keys in credentials file)"
    elif ":federated-user/" in arn:
        principal_kind = "federated user"
    else:
        principal_kind = "other IAM principal"

    print(f"Principal kind: {principal_kind}")
    credential_summary.update(
        {
            "account": account,
            "arn": arn,
            "user_id": user_id,
            "principal_kind": principal_kind,
        }
    )

    if expected_account_id and account and account != str(expected_account_id):
        print(
            "WARNING: Caller account does not match configured AWS_ACCOUNT_ID. "
            "CDK will target the configured account but act as this identity — "
            "deployments and lookups may fail. Set AWS_PROFILE to your SSO profile "
            "and unset AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY if needed."
        )
        credential_summary["account_mismatch"] = True
    elif expected_account_id and account == str(expected_account_id):
        print("Caller account matches configured AWS_ACCOUNT_ID.")

    if profile == "(not set — using default profile chain)":
        print(
            "TIP: Set AWS_PROFILE to your SSO profile name so Python and the CDK CLI "
            "(Node) use the same session. Example: "
            '$env:AWS_PROFILE = "YourSsoProfileName"'
        )

    print("--- End AWS credential context ---\n")
    return credential_summary


# --- Function to load context from file ---


def _context_value_for_cdk(value):
    """CDK/JSII context cannot use JSON null; normalize for Windows synth."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return {k: _context_value_for_cdk(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_context_value_for_cdk(v) for v in value]
    return value


def load_context_from_file(app: App, file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)
            for key, value in context_data.items():
                app.node.set_context(key, _context_value_for_cdk(value))
            print(f"Loaded context from {file_path}")
    else:
        print(f"Context file not found: {file_path}")


# --- Helper to parse environment variables into lists ---
def _get_env_list(env_var_name: str) -> List[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    value = env_var_name[1:-1].strip().replace('"', "").replace("'", "")
    if not value:
        return []
    # Split by comma and filter out any empty strings that might result from extra commas
    return [s.strip() for s in value.split(",") if s.strip()]


# 1. Try to load CIDR/AZs from environment variables
if PUBLIC_SUBNETS_TO_USE:
    PUBLIC_SUBNETS_TO_USE = _get_env_list(PUBLIC_SUBNETS_TO_USE)
if PRIVATE_SUBNETS_TO_USE:
    PRIVATE_SUBNETS_TO_USE = _get_env_list(PRIVATE_SUBNETS_TO_USE)

if PUBLIC_SUBNET_CIDR_BLOCKS:
    PUBLIC_SUBNET_CIDR_BLOCKS = _get_env_list("PUBLIC_SUBNET_CIDR_BLOCKS")
if PUBLIC_SUBNET_AVAILABILITY_ZONES:
    PUBLIC_SUBNET_AVAILABILITY_ZONES = _get_env_list("PUBLIC_SUBNET_AVAILABILITY_ZONES")
if PRIVATE_SUBNET_CIDR_BLOCKS:
    PRIVATE_SUBNET_CIDR_BLOCKS = _get_env_list("PRIVATE_SUBNET_CIDR_BLOCKS")
if PRIVATE_SUBNET_AVAILABILITY_ZONES:
    PRIVATE_SUBNET_AVAILABILITY_ZONES = _get_env_list(
        "PRIVATE_SUBNET_AVAILABILITY_ZONES"
    )


def resolve_policy_file_paths(
    paths: List[str],
    *,
    cdk_folder: str = "",
) -> List[str]:
    """Resolve JSON policy paths relative to ``CDK_FOLDER`` when not absolute."""
    resolved: List[str] = []
    base = (cdk_folder or "").strip().rstrip("/\\")
    for raw in paths:
        path = (raw or "").strip()
        if not path:
            continue
        if os.path.isabs(path):
            resolved.append(os.path.normpath(path))
        elif base:
            resolved.append(os.path.normpath(os.path.join(base, path)))
        else:
            resolved.append(os.path.normpath(path))
    return resolved


def attach_managed_policy_arns(role: iam.IRole, policy_arns: List[str]) -> None:
    """Attach existing customer-managed policies by full ARN."""
    for arn in policy_arns:
        arn = (arn or "").strip()
        if not arn:
            continue
        role.add_managed_policy(
            iam.ManagedPolicy.from_managed_policy_arn(role, arn, arn)
        )
        print(f"Attached managed policy ARN to {role.node.id}: {arn}")


def check_for_existing_role(role_name: str):
    try:
        iam = boto3.client("iam")
        # iam.get_role(RoleName=role_name)

        response = iam.get_role(RoleName=role_name)
        role = response["Role"]["Arn"]

        print("Response Role:", role)

        return True, role, ""
    except iam.exceptions.NoSuchEntityException:
        return False, "", ""
    except Exception as e:
        raise Exception("Getting information on IAM role failed due to:", e)


from typing import List


def add_statement_to_policy(role: iam.IRole, policy_document: Dict[str, Any]):
    """
    Adds individual policy statements from a parsed policy document to a CDK Role.

    Args:
        role: The CDK Role construct to attach policies to.
        policy_document: A Python dictionary representing an IAM policy document.
    """
    # Ensure the loaded JSON is a valid policy document structure
    if "Statement" not in policy_document or not isinstance(
        policy_document["Statement"], list
    ):
        print("Warning: Policy document does not contain a 'Statement' list. Skipping.")
        return  # Do not return role, just log and exit

    for statement_dict in policy_document["Statement"]:
        try:
            # Create a CDK PolicyStatement from the dictionary
            cdk_policy_statement = iam.PolicyStatement.from_json(statement_dict)

            # Add the policy statement to the role
            role.add_to_policy(cdk_policy_statement)
            print(f"  - Added statement: {statement_dict.get('Sid', 'No Sid')}")
        except Exception as e:
            print(
                f"Warning: Could not process policy statement: {statement_dict}. Error: {e}"
            )


def vpc_endpoint_aws_service_name(service_suffix: str, region: str) -> str:
    """Full EC2 ``ServiceName`` for a VPC endpoint (matches describe_vpc_endpoints)."""
    return f"com.amazonaws.{region}.{service_suffix}"


def list_existing_vpc_endpoint_service_names(
    vpc_id: str,
    *,
    region_name: Optional[str] = None,
) -> FrozenSet[str]:
    """Return AWS service names for non-deleted VPC endpoints in the given VPC."""
    ec2_client = boto3.client("ec2", region_name=region_name)
    service_names: set[str] = set()
    paginator = ec2_client.get_paginator("describe_vpc_endpoints")
    for page in paginator.paginate(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}],
    ):
        for endpoint in page.get("VpcEndpoints", []):
            state = endpoint.get("State")
            if state in ("available", "pending", "pendingAcceptance"):
                service_name = endpoint.get("ServiceName")
                if service_name:
                    service_names.add(service_name)
    return frozenset(service_names)


def resolve_ecs_vpc_endpoint_subnet_selection(
    *,
    use_express_ingress: bool,
    express_use_public_subnets: bool,
    public_subnets: List[ec2.ISubnet],
    private_subnets: List[ec2.ISubnet],
) -> Optional[ec2.SubnetSelection]:
    """
    Choose subnets for ECS-related **interface** VPC endpoints.

    Interface ENIs must sit in the same subnets ECS tasks use. Express Mode with
    ``ECS_EXPRESS_USE_PUBLIC_SUBNETS=True`` runs tasks in public subnets; legacy
    Fargate and Express-on-private use private. S3 gateway routes use
    ``resolve_ecs_s3_gateway_subnet_selection`` instead.
    """
    if use_express_ingress:
        if express_use_public_subnets:
            if not public_subnets:
                return None
            return ec2.SubnetSelection(subnets=public_subnets)
        if not private_subnets:
            return None
        return ec2.SubnetSelection(subnets=private_subnets)
    if private_subnets:
        return ec2.SubnetSelection(subnets=private_subnets)
    if public_subnets:
        return ec2.SubnetSelection(subnets=public_subnets)
    return None


def resolve_ecs_s3_gateway_subnet_selection(
    *,
    public_subnets: List[ec2.ISubnet],
    private_subnets: List[ec2.ISubnet],
) -> Optional[ec2.SubnetSelection]:
    """
    All stack-managed subnets for the S3 **gateway** endpoint.

    Gateway endpoints attach to route tables; every public and private subnet the
    stack imports or creates should get the S3 prefix-list route, not only the ECS
    task tier.
    """
    all_subnets: List[ec2.ISubnet] = []
    seen_subnet_ids: set[str] = set()
    for subnet in public_subnets + private_subnets:
        subnet_id = subnet.subnet_id
        if subnet_id in seen_subnet_ids:
            continue
        seen_subnet_ids.add(subnet_id)
        all_subnets.append(subnet)
    if not all_subnets:
        return None
    return ec2.SubnetSelection(subnets=all_subnets)


def list_vpc_associated_cidr_blocks(vpc: dict) -> List[str]:
    """All associated IPv4 CIDR blocks for a VPC (primary and secondary)."""
    cidrs: List[str] = []
    seen: set[str] = set()
    for assoc in vpc.get("CidrBlockAssociationSet") or []:
        state = (assoc.get("CidrBlockState") or {}).get("State")
        if state and state != "associated":
            continue
        cidr = assoc.get("CidrBlock")
        if cidr and cidr not in seen:
            seen.add(cidr)
            cidrs.append(cidr)
    primary = vpc.get("CidrBlock")
    if primary and primary not in seen:
        cidrs.insert(0, primary)
    return cidrs


def resolve_vpc_endpoint_ingress_cidr_blocks(
    *,
    vpc_cidr_block: Optional[str] = None,
    vpc_cidr_blocks: Optional[List[str]] = None,
    fallback_vpc_cidr: Optional[str] = None,
) -> List[str]:
    """CIDR list for VPC endpoint SG ingress (deduplicated, stable order)."""
    resolved: List[str] = []
    seen: set[str] = set()
    for cidr in vpc_cidr_blocks or []:
        if cidr and cidr not in seen:
            seen.add(cidr)
            resolved.append(cidr)
    if not resolved:
        single = vpc_cidr_block or fallback_vpc_cidr
        if single:
            resolved = [single]
    return resolved


def add_vpc_endpoint_https_ingress_from_vpc_cidrs(
    endpoint_sg: ec2.SecurityGroup,
    *,
    vpc_cidr_block: Optional[str] = None,
    vpc_cidr_blocks: Optional[List[str]] = None,
    fallback_vpc_cidr: Optional[str] = None,
) -> None:
    """Allow HTTPS from every CIDR associated with the VPC."""
    cidrs = resolve_vpc_endpoint_ingress_cidr_blocks(
        vpc_cidr_block=vpc_cidr_block,
        vpc_cidr_blocks=vpc_cidr_blocks,
        fallback_vpc_cidr=fallback_vpc_cidr,
    )
    if not cidrs:
        raise ValueError(
            "VPC CIDR block(s) are required for the VPC endpoint security group. "
            "Re-run check_resources.py so vpc_cidr_blocks is stored in precheck context."
        )
    for cidr in cidrs:
        endpoint_sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(cidr),
            connection=ec2.Port.tcp(443),
            description=(
                "HTTPS from VPC workloads"
                if len(cidrs) == 1
                else f"HTTPS from VPC workloads ({cidr})"
            ),
        )


def create_ecs_vpc_endpoints_for_private_subnets(
    scope: Construct,
    *,
    vpc: ec2.IVpc,
    subnets: Optional[ec2.SubnetSelection],
    logical_id_prefix: str = "Ecs",
    include_secrets_and_kms: bool = True,
    vpc_cidr_block: Optional[str] = None,
    vpc_cidr_blocks: Optional[List[str]] = None,
    skip_service_names: Optional[FrozenSet[str]] = None,
    aws_region: str,
    s3_gateway_subnets: Optional[ec2.SubnetSelection] = None,
) -> None:
    """
    Interface and S3 gateway VPC endpoints for ECS workloads.

    Interface endpoints use ``subnets`` (ECS task tier). The S3 gateway uses
    ``s3_gateway_subnets`` when provided, otherwise ``subnets``. Without
    ``ecr.api`` / ``ecr.dkr`` endpoints (or a working NAT path), tasks fail with
    ``GetAuthorizationToken`` timeouts.

    ``skip_service_names`` should list full AWS endpoint service names already present
    in the VPC (from pre-check) so shared VPCs do not fail on duplicate private DNS.
    """
    s3_subnets = s3_gateway_subnets or subnets
    if not subnets and not s3_subnets:
        return
    skip = skip_service_names or frozenset()

    interface_services: List[Tuple[str, str, ec2.InterfaceVpcEndpointAwsService]] = [
        ("EcrApi", "ecr.api", ec2.InterfaceVpcEndpointAwsService.ECR),
        ("EcrDkr", "ecr.dkr", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
        ("CloudWatchLogs", "logs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
    ]
    if include_secrets_and_kms:
        interface_services.extend(
            [
                (
                    "SecretsManager",
                    "secretsmanager",
                    ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
                ),
                ("Kms", "kms", ec2.InterfaceVpcEndpointAwsService.KMS),
            ]
        )

    interface_services_to_create = []
    for suffix, service_suffix, service in interface_services:
        service_name = vpc_endpoint_aws_service_name(service_suffix, aws_region)
        if service_name in skip:
            print(
                f"Skipping {logical_id_prefix}{suffix}Endpoint "
                f"({service_name} already exists in this VPC)."
            )
            continue
        interface_services_to_create.append((suffix, service))

    s3_service_name = vpc_endpoint_aws_service_name("s3", aws_region)
    s3_service = ec2.GatewayVpcEndpointAwsService.S3
    create_s3_gateway = s3_service_name not in skip
    if not create_s3_gateway:
        print(
            f"Skipping {logical_id_prefix}S3GatewayEndpoint "
            f"({s3_service_name} already exists in this VPC)."
        )

    if not interface_services_to_create and not create_s3_gateway:
        print(
            "All ECS-related VPC endpoints already exist in this VPC; "
            "nothing to create."
        )
        return

    endpoint_sg = None
    if interface_services_to_create and subnets:
        endpoint_sg = ec2.SecurityGroup(
            scope,
            f"{logical_id_prefix}VpcEndpointSecurityGroup",
            vpc=vpc,
            description="HTTPS ingress for ECS-related VPC interface endpoints",
            allow_all_outbound=True,
        )
        add_vpc_endpoint_https_ingress_from_vpc_cidrs(
            endpoint_sg,
            vpc_cidr_block=vpc_cidr_block,
            vpc_cidr_blocks=vpc_cidr_blocks,
            fallback_vpc_cidr=vpc.vpc_cidr_block,
        )

    if subnets:
        for suffix, service in interface_services_to_create:
            vpc.add_interface_endpoint(
                f"{logical_id_prefix}{suffix}Endpoint",
                service=service,
                subnets=subnets,
                security_groups=[endpoint_sg],
                private_dns_enabled=True,
            )
    elif interface_services_to_create:
        print(
            "Skipping ECS interface VPC endpoints: no task-tier subnet selection "
            "was provided."
        )

    if create_s3_gateway and s3_subnets:
        try:
            vpc.add_gateway_endpoint(
                f"{logical_id_prefix}S3GatewayEndpoint",
                service=s3_service,
                subnets=[s3_subnets],
            )
        except Exception as exc:
            print(
                "Note: could not add S3 gateway VPC endpoint (one may already exist on "
                f"this VPC): {exc}"
            )


def add_s3_enforce_ssl_policy(bucket: s3.IBucket) -> None:
    """Deny non-TLS S3 requests (Security Hub S3.5). Compatible with all CDK versions."""
    bucket.add_to_resource_policy(
        iam.PolicyStatement(
            effect=iam.Effect.DENY,
            principals=[iam.AnyPrincipal()],
            actions=["s3:*"],
            resources=[bucket.bucket_arn, f"{bucket.bucket_arn}/*"],
            conditions={"Bool": {"aws:SecureTransport": "false"}},
        )
    )


def add_custom_policies(
    scope: Construct,  # Not strictly used here, but good practice if you expand to ManagedPolicies
    role: iam.IRole,
    policy_file_locations: Optional[List[str]] = None,
    custom_policy_text: Optional[str] = None,
) -> iam.IRole:
    """
    Loads custom policies from JSON files or a string and attaches them to a CDK Role.

    Args:
        scope: The scope in which to define constructs (if needed, e.g., for iam.ManagedPolicy).
        role: The CDK Role construct to attach policies to.
        policy_file_locations: List of file paths to JSON policy documents.
        custom_policy_text: A JSON string representing a policy document.

    Returns:
        The modified CDK Role construct.
    """
    if policy_file_locations is None:
        policy_file_locations = []

    current_source = "unknown source"  # For error messages

    try:
        if policy_file_locations:
            print(f"Attempting to add policies from files to role {role.node.id}...")
            for path in policy_file_locations:
                current_source = f"file: {path}"
                try:
                    with open(path, "r") as f:
                        policy_document = json.load(f)
                    print(f"Processing policy from {current_source}...")
                    add_statement_to_policy(role, policy_document)
                except FileNotFoundError:
                    print(f"Warning: Policy file not found at {path}. Skipping.")
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Invalid JSON in policy file {path}: {e}. Skipping."
                    )
                except Exception as e:
                    print(
                        f"An unexpected error occurred processing policy from {path}: {e}. Skipping."
                    )

        if custom_policy_text:
            current_source = "custom policy text string"
            print(
                f"Attempting to add policy from custom text to role {role.node.id}..."
            )
            try:
                # *** FIX: Parse the JSON string into a Python dictionary ***
                policy_document = json.loads(custom_policy_text)
                print(f"Processing policy from {current_source}...")
                add_statement_to_policy(role, policy_document)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in custom_policy_text: {e}. Skipping.")
            except Exception as e:
                print(
                    f"An unexpected error occurred processing policy from custom_policy_text: {e}. Skipping."
                )

        # You might want a final success message, but individual processing messages are also good.
        print(f"Finished processing custom policies for role {role.node.id}.")

    except Exception as e:
        print(
            f"An unhandled error occurred during policy addition for {current_source}: {e}"
        )

    return role


# Import the S3 Bucket class if you intend to return a CDK object later
# from aws_cdk import aws_s3 as s3


def _s3_bucket_listed_in_account(s3_client: Any, bucket_name: str) -> bool:
    token = None
    while True:
        kwargs: Dict[str, Any] = {}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3_client.list_buckets(**kwargs)
        for entry in response.get("Buckets", []):
            if entry.get("Name") == bucket_name:
                return True
        token = response.get("NextContinuationToken")
        if not token:
            return False


def resolve_s3_bucket_availability(
    bucket_name: str,
    *,
    s3_client: Any = None,
) -> Tuple[S3BucketAvailability, str]:
    """
    Resolve whether an S3 bucket name is owned in this account, globally free,
    or taken by another AWS account.

    S3 names are globally unique. ``head_bucket`` returns 404 when the name is
    free, 200 when this principal can access it, and 403 when the name is taken
    elsewhere (or access is denied in-account).
    """
    client = s3_client or boto3.client("s3")
    try:
        client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' exists and is accessible in this account.")
        return "owned", bucket_name
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket", "NotFound"):
            print(f"Bucket '{bucket_name}' is available (name not taken globally).")
            return "available", bucket_name
        if error_code in ("403", "AccessDenied"):
            if _s3_bucket_listed_in_account(client, bucket_name):
                print(
                    f"Bucket '{bucket_name}' exists in this account "
                    "(head_bucket denied; confirmed via list_buckets)."
                )
                return "owned", bucket_name
            print(
                f"Bucket '{bucket_name}' is taken globally by another AWS account "
                f"(head_bucket returned {error_code}; not listed in this account)."
            )
            return "globally_taken", bucket_name
        print(
            f"An unexpected AWS ClientError occurred checking bucket '{bucket_name}': {e}"
        )
        raise
    except Exception as e:
        print(
            f"An unexpected non-ClientError occurred checking bucket '{bucket_name}': {e}"
        )
        raise


def check_s3_bucket_exists(
    bucket_name: str,
) -> Tuple[bool, Optional[str]]:
    """
    Return whether the bucket exists in **this** AWS account (for CDK import).

    For global availability (including cross-account collisions), use
    ``resolve_s3_bucket_availability``.
    """
    status, name = resolve_s3_bucket_availability(bucket_name)
    if status == "owned":
        return True, name
    return False, None


def resolve_cognito_domain_prefix_availability(
    domain_prefix: str,
    *,
    region_name: Optional[str] = None,
    cognito_client: Any = None,
) -> CognitoDomainAvailability:
    """Return whether a Cognito hosted UI domain prefix is free in the region.

    Cognito prefixes are unique per region across all AWS accounts. When a prefix
    is owned elsewhere, ``describe_user_pool_domain`` raises
    ``ResourceNotFoundException`` ("does not exist in this account") — the same
    symptom CloudFormation reports as ``AlreadyExists`` on create.
    """
    prefix = (domain_prefix or "").strip().lower()
    if not prefix:
        return "available"
    region = region_name or os.environ.get("AWS_REGION")
    client = cognito_client or boto3.client("cognito-idp", region_name=region)
    try:
        response = client.describe_user_pool_domain(Domain=prefix)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            print(
                f"Cognito domain prefix {prefix!r} is not available in {region} "
                f"(likely taken by another AWS account; "
                f"{e.response['Error'].get('Message', 'ResourceNotFoundException')})."
            )
            return "taken"
        raise
    description = response.get("DomainDescription") or {}
    if description.get("UserPoolId"):
        print(
            f"Cognito domain prefix {prefix!r} is in use by user pool "
            f"{description['UserPoolId']} in this account."
        )
        return "taken"
    print(f"Cognito domain prefix {prefix!r} is available in {region}.")
    return "available"


def default_secrets_manager_kms_key_arn(region: str, account_id: str) -> str:
    """AWS managed CMK alias used by Secrets Manager when no customer key is set."""
    return f"arn:aws:kms:{region}:{account_id}:key/aws/secretsmanager"


def get_secret_kms_key_arn(
    secret_id: str,
    *,
    region_name: Optional[str] = None,
) -> Optional[str]:
    """Return the KMS key ARN that encrypts a Secrets Manager secret."""
    client = boto3.client("secretsmanager", region_name=region_name)
    try:
        description = client.describe_secret(SecretId=secret_id)
    except Exception as exc:
        print(f"Warning: could not describe secret for KMS key: {exc}")
        return None
    kms_key_id = description.get("KmsKeyId")
    if not kms_key_id:
        return None
    if str(kms_key_id).startswith("arn:"):
        return str(kms_key_id)
    region = region_name or AWS_REGION
    account_id = (
        description.get("OwningAccount")
        or boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    )
    return f"arn:aws:kms:{region}:{account_id}:key/{kms_key_id}"


def build_ecs_task_role_kms_policy(
    *,
    shared_kms_key_arn: Optional[str] = None,
) -> Dict[str, Any]:
    """Task role: shared CMK for S3 bucket encryption (when USE_CUSTOM_KMS_KEY=1)."""
    statements: List[Dict[str, Any]] = [
        {
            "Sid": "STSCallerIdentity",
            "Effect": "Allow",
            "Action": ["sts:GetCallerIdentity"],
            "Resource": "*",
        },
    ]
    if shared_kms_key_arn:
        statements.append(
            {
                "Sid": "KMSS3Access",
                "Effect": "Allow",
                "Action": ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"],
                "Resource": shared_kms_key_arn,
            }
        )
    return {"Version": "2012-10-17", "Statement": statements}


def build_ecs_execution_role_kms_policy(
    *,
    secret_kms_key_arn: str,
) -> Dict[str, Any]:
    """Execution role: decrypt the CMK that encrypts the Cognito secret only."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "STSCallerIdentity",
                "Effect": "Allow",
                "Action": ["sts:GetCallerIdentity"],
                "Resource": "*",
            },
            {
                "Sid": "KMSSecretDecrypt",
                "Effect": "Allow",
                "Action": ["kms:Decrypt"],
                "Resource": secret_kms_key_arn,
            },
        ],
    }


# Example usage in your check_resources.py:
# exists, bucket_name_if_exists = check_s3_bucket_exists(log_bucket_name)
# context_data[f"exists:{log_bucket_name}"] = exists
# # You don't necessarily need to store the name in context if using from_bucket_name


# Delete an S3 bucket
def delete_s3_bucket(bucket_name: str):
    s3 = boto3.client("s3")

    try:
        # List and delete all objects
        response = s3.list_object_versions(Bucket=bucket_name)
        versions = response.get("Versions", []) + response.get("DeleteMarkers", [])
        for version in versions:
            s3.delete_object(
                Bucket=bucket_name, Key=version["Key"], VersionId=version["VersionId"]
            )

        # Delete the bucket
        s3.delete_bucket(Bucket=bucket_name)
        return {"Status": "SUCCESS"}
    except Exception as e:
        return {"Status": "FAILED", "Reason": str(e)}


# Function to get subnet ID from subnet name
def get_subnet_id(vpc: str, ec2_client: str, subnet_name: str):
    response = ec2_client.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc.vpc_id]}]
    )

    for subnet in response["Subnets"]:
        if subnet["Tags"] and any(
            tag["Key"] == "Name" and tag["Value"] == subnet_name
            for tag in subnet["Tags"]
        ):
            return subnet["SubnetId"]

    return None


def check_ecr_repo_exists(repo_name: str) -> tuple[bool, dict]:
    """
    Checks if an ECR repository with the given name exists.

    Args:
        repo_name: The name of the ECR repository to check.

    Returns:
        True if the repository exists, False otherwise.
    """
    ecr_client = boto3.client("ecr")
    try:
        print("ecr repo_name to check:", repo_name)
        response = ecr_client.describe_repositories(repositoryNames=[repo_name])
        # If describe_repositories succeeds and returns a list of repositories,
        # and the list is not empty, the repository exists.
        return len(response["repositories"]) > 0, response["repositories"][0]
    except ClientError as e:
        # Check for the specific error code indicating the repository doesn't exist
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            return False, {}
        else:
            # Re-raise other exceptions to handle unexpected errors
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}


def check_codebuild_project_exists(
    project_name: str,
):  # Adjust return type hint as needed
    """
    Checks if a CodeBuild project with the given name exists.

    Args:
        project_name: The name of the CodeBuild project to check.

    Returns:
        A tuple:
        - The first element is True if the project exists, False otherwise.
        - The second element is the project object (dictionary) if found,
          None otherwise.
    """
    codebuild_client = boto3.client("codebuild")
    try:
        # Use batch_get_projects with a list containing the single project name
        response = codebuild_client.batch_get_projects(names=[project_name])

        # The response for batch_get_projects includes 'projects' (found)
        # and 'projectsNotFound' (not found).
        if response["projects"]:
            # If the project is found in the 'projects' list
            print(f"CodeBuild project '{project_name}' found.")
            project = response["projects"][0]
            return (
                True,
                project["arn"],
                project.get("serviceRole"),
            )
        elif (
            response["projectsNotFound"]
            and project_name in response["projectsNotFound"]
        ):
            # If the project name is explicitly in the 'projectsNotFound' list
            print(f"CodeBuild project '{project_name}' not found.")
            return False, None, None
        else:
            # This case is less expected for a single name lookup,
            # but could happen if there's an internal issue or the response
            # structure is slightly different than expected for an error.
            # It's safer to assume it wasn't found if not in 'projects'.
            print(
                f"CodeBuild project '{project_name}' not found (not in 'projects' list)."
            )
            return False, None, None

    except ClientError as e:
        # Catch specific ClientErrors. batch_get_projects might not throw
        # 'InvalidInputException' for a non-existent project name if the
        # name format is valid. It typically just lists it in projectsNotFound.
        # However, other ClientErrors are possible (e.g., permissions).
        print(
            f"An AWS ClientError occurred checking CodeBuild project '{project_name}': {e}"
        )
        # Decide how to handle other ClientErrors - raising might be safer
        raise  # Re-raise the original exception
    except Exception as e:
        print(
            f"An unexpected non-ClientError occurred checking CodeBuild project '{project_name}': {e}"
        )
        # Decide how to handle other errors
        raise  # Re-raise the original exception


def public_github_repository_url(owner: str, repo: str) -> str:
    """HTTPS clone URL for a public GitHub repository."""
    owner_clean = owner.strip().strip("/")
    repo_clean = repo.strip().strip("/")
    return f"https://github.com/{owner_clean}/{repo_clean}.git"


def public_github_codebuild_source(
    owner: str,
    repo: str,
    branch_or_ref: str,
) -> codebuild.ISource:
    """CodeBuild source for a public GitHub repo (no CodeConnections/OAuth)."""
    return codebuild.Source.git_hub(
        owner=owner,
        repo=repo,
        branch_or_ref=branch_or_ref,
        webhook=False,
        report_build_status=False,
    )


def configure_public_github_codebuild_source(
    project: codebuild.Project,
    owner: str,
    repo: str,
) -> None:
    """Ensure CodeBuild clones a public repo without account-level GitHub credentials."""
    cfn = project.node.default_child
    if not isinstance(cfn, codebuild.CfnProject):
        return
    cfn.add_property_deletion_override("Source.Auth")
    cfn.add_property_override(
        "Source.Location", public_github_repository_url(owner, repo)
    )
    cfn.add_property_override("Source.ReportBuildStatus", False)
    cfn.add_property_override("Triggers.Webhook", False)


def get_vpc_id_by_name(vpc_name: str):
    """
    Finds a VPC ID by its 'Name' tag.

    Returns ``(vpc_id, nat_gateways, vpc_cidr_block, vpc_cidr_blocks)`` or ``None``
    if not found. ``vpc_cidr_block`` is the primary block; ``vpc_cidr_blocks`` lists
    every associated IPv4 CIDR (primary + secondary).
    """
    ec2_client = boto3.client("ec2")
    try:
        response = ec2_client.describe_vpcs(
            Filters=[{"Name": "tag:Name", "Values": [vpc_name]}]
        )
        if response and response["Vpcs"]:
            vpc = response["Vpcs"][0]
            vpc_id = vpc["VpcId"]
            vpc_cidr_block = vpc.get("CidrBlock")
            vpc_cidr_blocks = list_vpc_associated_cidr_blocks(vpc)
            cidr_summary = (
                ", ".join(vpc_cidr_blocks) if vpc_cidr_blocks else vpc_cidr_block
            )
            print(
                f"VPC '{vpc_name}' found with ID: {vpc_id}"
                + (f", CIDR(s): {cidr_summary}" if cidr_summary else "")
            )

            # In get_vpc_id_by_name, after finding VPC ID:

            # Look for NAT Gateways in this VPC
            ec2_client = boto3.client("ec2")
            nat_gateways = []
            try:
                response = ec2_client.describe_nat_gateways(
                    Filters=[
                        {"Name": "vpc-id", "Values": [vpc_id]},
                        # Optional: Add a tag filter if you consistently tag your NATs
                        # {'Name': 'tag:Name', 'Values': [f"{prefix}-nat-gateway"]}
                    ]
                )
                nat_gateways = response.get("NatGateways", [])
            except Exception as e:
                print(
                    f"Warning: Could not describe NAT Gateways in VPC '{vpc_id}': {e}"
                )
                # Decide how to handle this error - proceed or raise?

            # Decide how to identify the specific NAT Gateway you want to check for.

            return vpc_id, nat_gateways, vpc_cidr_block, vpc_cidr_blocks
        else:
            print(f"VPC '{vpc_name}' not found.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred finding VPC '{vpc_name}': {e}")
        raise


# --- Helper to fetch all existing subnets in a VPC once ---
def _get_existing_subnets_in_vpc(vpc_id: str) -> Dict[str, Any]:
    """
    Fetches all subnets in a given VPC.
    Returns a dictionary with 'by_name' (map of name to subnet data),
    'by_id' (map of id to subnet data), and 'cidr_networks' (list of ipaddress.IPv4Network).
    """
    ec2_client = boto3.client("ec2")
    existing_subnets_data = {
        "by_name": {},  # {subnet_name: {'id': 'subnet-id', 'cidr': 'x.x.x.x/x'}}
        "by_id": {},  # {subnet_id: {'name': 'subnet-name', 'cidr': 'x.x.x.x/x/x'}}
        "cidr_networks": [],  # List of ipaddress.IPv4Network objects
    }
    try:
        subnet_to_route_table: Dict[str, str] = {}
        rt_response = ec2_client.describe_route_tables(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        for route_table in rt_response.get("RouteTables", []):
            route_table_id = route_table["RouteTableId"]
            for association in route_table.get("Associations", []):
                associated_subnet_id = association.get("SubnetId")
                if associated_subnet_id:
                    subnet_to_route_table[associated_subnet_id] = route_table_id

        response = ec2_client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        for s in response.get("Subnets", []):
            subnet_id = s["SubnetId"]
            cidr_block = s.get("CidrBlock")
            # Extract 'Name' tag, which is crucial for lookup by name
            name_tag = next(
                (tag["Value"] for tag in s.get("Tags", []) if tag["Key"] == "Name"),
                None,
            )

            subnet_info = {
                "id": subnet_id,
                "cidr": cidr_block,
                "name": name_tag,
                "az": s.get("AvailabilityZone"),
                "route_table_id": subnet_to_route_table.get(subnet_id),
            }

            if name_tag:
                existing_subnets_data["by_name"][name_tag] = subnet_info
            existing_subnets_data["by_id"][subnet_id] = subnet_info

            if cidr_block:
                try:
                    existing_subnets_data["cidr_networks"].append(
                        ipaddress.ip_network(cidr_block, strict=False)
                    )
                except ValueError:
                    print(
                        f"Warning: Existing subnet {subnet_id} has an invalid CIDR: {cidr_block}. Skipping for overlap check."
                    )

        print(
            f"Fetched {len(response.get('Subnets', []))} existing subnets from VPC '{vpc_id}'."
        )
    except Exception as e:
        print(
            f"Error describing existing subnets in VPC '{vpc_id}': {e}. Cannot perform full validation."
        )
        raise  # Re-raise if this essential step fails

    return existing_subnets_data


def get_internet_gateways_attached_to_vpc(vpc_id: str) -> List[str]:
    """Return Internet Gateway IDs currently attached to the VPC."""
    ec2_client = boto3.client("ec2")
    response = ec2_client.describe_internet_gateways(
        Filters=[{"Name": "attachment.vpc-id", "Values": [vpc_id]}]
    )
    return [
        igw["InternetGatewayId"]
        for igw in response.get("InternetGateways", [])
        if igw.get("InternetGatewayId")
    ]


def internet_gateway_exists(igw_id: str) -> bool:
    ec2_client = boto3.client("ec2")
    response = ec2_client.describe_internet_gateways(InternetGatewayIds=[igw_id])
    return bool(response.get("InternetGateways"))


def route_table_default_internet_gateway(route_table_id: str) -> Optional[str]:
    """
    Return the Internet Gateway ID for 0.0.0.0/0 on this route table, if any.
    """
    ec2_client = boto3.client("ec2")
    response = ec2_client.describe_route_tables(RouteTableIds=[route_table_id])
    tables = response.get("RouteTables", [])
    if not tables:
        return None
    for route in tables[0].get("Routes", []):
        if route.get("DestinationCidrBlock") != "0.0.0.0/0":
            continue
        gateway_id = route.get("GatewayId") or ""
        if gateway_id.startswith("igw-"):
            return gateway_id
    return None


def route_table_has_non_igw_default_route(route_table_id: str) -> bool:
    """True if 0.0.0.0/0 exists but does not target an Internet Gateway."""
    ec2_client = boto3.client("ec2")
    response = ec2_client.describe_route_tables(RouteTableIds=[route_table_id])
    tables = response.get("RouteTables", [])
    if not tables:
        return False
    for route in tables[0].get("Routes", []):
        if route.get("DestinationCidrBlock") != "0.0.0.0/0":
            continue
        gateway_id = route.get("GatewayId") or ""
        if gateway_id.startswith("igw-"):
            return False
        # Active default route via NAT instance, TGW, etc.
        if (
            route.get("NatGatewayId")
            or route.get("TransitGatewayId")
            or route.get("GatewayId")
        ):
            return True
    return False


def audit_public_subnet_internet_connectivity(
    vpc_id: str,
    configured_igw_id: str,
    public_subnet_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate / discover Internet Gateway usage for legacy public subnets.

    Returns context fields for CDK:
      internet_gateway_id, internet_gateway_needs_vpc_attachment,
      public_subnets_needing_igw_route (list of {name, subnet_id, route_table_id}).
    """
    configured_igw_id = (configured_igw_id or "").strip()
    attached_igws = get_internet_gateways_attached_to_vpc(vpc_id)

    resolved_igw_id = configured_igw_id
    needs_attachment = False

    if configured_igw_id:
        if not internet_gateway_exists(configured_igw_id):
            raise ValueError(
                f"EXISTING_IGW_ID '{configured_igw_id}' was not found in this account/region."
            )
        if configured_igw_id not in attached_igws:
            # Ensure it is not attached to another VPC
            ec2_client = boto3.client("ec2")
            detail = ec2_client.describe_internet_gateways(
                InternetGatewayIds=[configured_igw_id]
            )
            for attachment in detail.get("InternetGateways", [{}])[0].get(
                "Attachments", []
            ):
                other_vpc = attachment.get("VpcId")
                if other_vpc and other_vpc != vpc_id:
                    raise ValueError(
                        f"EXISTING_IGW_ID '{configured_igw_id}' is attached to VPC "
                        f"'{other_vpc}', not target VPC '{vpc_id}'. Detach it or choose "
                        "the IGW attached to this VPC."
                    )
            needs_attachment = True
    elif attached_igws:
        if len(attached_igws) > 1:
            raise ValueError(
                f"VPC '{vpc_id}' has multiple attached Internet Gateways "
                f"({', '.join(attached_igws)}). Set EXISTING_IGW_ID to the one to use."
            )
        resolved_igw_id = attached_igws[0]
        print(
            f"EXISTING_IGW_ID not set; using Internet Gateway attached to VPC: "
            f"{resolved_igw_id}"
        )
    elif public_subnet_entries:
        raise ValueError(
            f"VPC '{vpc_id}' has no Internet Gateway attached and EXISTING_IGW_ID is "
            "empty. Set EXISTING_IGW_ID to an existing IGW for this VPC (CDK will "
            "attach it if detached)."
        )

    subnets_needing_route: List[Dict[str, str]] = []
    for entry in public_subnet_entries:
        name = entry.get("name") or "unknown"
        route_table_id = entry.get("route_table_id")
        subnet_id = entry.get("subnet_id") or entry.get("id") or ""
        if not route_table_id:
            print(
                f"Warning: public subnet '{name}' has no route table association in "
                "pre-check; skipping IGW route audit (CDK may still add routes after create)."
            )
            continue
        existing_igw = route_table_default_internet_gateway(route_table_id)
        if existing_igw:
            if resolved_igw_id and existing_igw != resolved_igw_id:
                raise ValueError(
                    f"Public subnet '{name}' route table '{route_table_id}' has "
                    f"0.0.0.0/0 -> {existing_igw}, but EXISTING_IGW_ID / resolved IGW "
                    f"is '{resolved_igw_id}'. Fix the route table manually or align "
                    "EXISTING_IGW_ID."
                )
            continue
        if route_table_has_non_igw_default_route(route_table_id):
            raise ValueError(
                f"Public subnet '{name}' route table '{route_table_id}' has a default "
                "route that does not use an Internet Gateway (e.g. NAT/TGW). Remove "
                "or change it before adding 0.0.0.0/0 -> IGW for an internet-facing ALB."
            )
        subnets_needing_route.append(
            {
                "name": name,
                "subnet_id": subnet_id,
                "route_table_id": route_table_id,
            }
        )

    return {
        "internet_gateway_id": resolved_igw_id,
        "internet_gateway_needs_vpc_attachment": needs_attachment,
        "public_subnets_needing_igw_route": subnets_needing_route,
    }


def wire_public_subnet_internet_access(
    scope: Construct,
    logical_id_prefix: str,
    *,
    vpc_id: str,
    internet_gateway_id: str,
    needs_igw_vpc_attachment: bool,
    subnets_needing_route: List[Dict[str, str]],
) -> Optional[ec2.CfnVPCGatewayAttachment]:
    """
    Attach the Internet Gateway to the VPC (if needed) and add 0.0.0.0/0 routes on
    imported public subnet route tables that lack an IGW default route.
    """
    if not internet_gateway_id:
        return None

    attachment = None
    if needs_igw_vpc_attachment:
        attachment = ec2.CfnVPCGatewayAttachment(
            scope,
            f"{logical_id_prefix}IgwVpcAttachment",
            vpc_id=vpc_id,
            internet_gateway_id=internet_gateway_id,
        )
        print(
            f"CDK: will attach Internet Gateway '{internet_gateway_id}' to VPC '{vpc_id}'."
        )

    seen_route_tables: set[str] = set()
    for i, entry in enumerate(subnets_needing_route):
        route_table_id = entry.get("route_table_id")
        if not route_table_id or route_table_id in seen_route_tables:
            continue
        seen_route_tables.add(route_table_id)
        safe_name = (entry.get("name") or f"rt{i}").replace("-", "")[:40]
        route = ec2.CfnRoute(
            scope,
            f"{logical_id_prefix}IgwRoute{safe_name}{i}",
            route_table_id=route_table_id,
            destination_cidr_block="0.0.0.0/0",
            gateway_id=internet_gateway_id,
        )
        if attachment is not None:
            route.add_dependency(attachment)
        print(
            f"CDK: will add 0.0.0.0/0 -> {internet_gateway_id} on route table "
            f"'{route_table_id}' (subnet '{entry.get('name', '')}')."
        )

    return attachment


# --- Modified validate_subnet_creation_parameters to take pre-fetched data ---
def validate_subnet_creation_parameters(
    vpc_id: str,
    proposed_subnets_data: List[
        Dict[str, str]
    ],  # e.g., [{'name': 'my-public-subnet', 'cidr': '10.0.0.0/24', 'az': 'us-east-1a'}]
    existing_aws_subnets_data: Dict[
        str, Any
    ],  # Pre-fetched data from _get_existing_subnets_in_vpc
) -> None:
    """
    Validates proposed subnet names and CIDR blocks against existing AWS subnets
    in the specified VPC and against each other.
    This function uses pre-fetched AWS subnet data.

    Args:
        vpc_id: The ID of the VPC (for logging/error messages).
        proposed_subnets_data: A list of dictionaries, where each dict represents
                               a proposed subnet with 'name', 'cidr', and 'az'.
        existing_aws_subnets_data: Dictionary containing existing AWS subnet data
                                   (e.g., from _get_existing_subnets_in_vpc).

    Raises:
        ValueError: If any proposed subnet name or CIDR block
                    conflicts with existing AWS resources or other proposed resources.
    """
    if not proposed_subnets_data:
        print("No proposed subnet data provided for validation. Skipping.")
        return

    print(
        f"--- Starting pre-synth validation for VPC '{vpc_id}' with proposed subnets ---"
    )

    print("Existing subnet data:", pd.DataFrame(existing_aws_subnets_data["by_name"]))

    existing_aws_subnet_names = set(existing_aws_subnets_data["by_name"].keys())
    existing_aws_cidr_networks = existing_aws_subnets_data["cidr_networks"]

    # Sets to track names and list to track networks for internal batch consistency
    proposed_names_seen: set[str] = set()
    proposed_cidr_networks_seen: List[ipaddress.IPv4Network] = []

    for i, proposed_subnet in enumerate(proposed_subnets_data):
        subnet_name = proposed_subnet.get("name")
        cidr_block_str = proposed_subnet.get("cidr")
        availability_zone = proposed_subnet.get("az")

        if not all([subnet_name, cidr_block_str, availability_zone]):
            raise ValueError(
                f"Proposed subnet at index {i} is incomplete. Requires 'name', 'cidr', and 'az'."
            )

        # 1. Check for duplicate names within the proposed batch
        if subnet_name in proposed_names_seen:
            raise ValueError(
                f"Proposed subnet name '{subnet_name}' is duplicated within the input list."
            )
        proposed_names_seen.add(subnet_name)

        # 2. Check for duplicate names against existing AWS subnets
        if subnet_name in existing_aws_subnet_names:
            print(
                f"Proposed subnet name '{subnet_name}' already exists in VPC '{vpc_id}'."
            )

        # Parse proposed CIDR
        try:
            proposed_net = ipaddress.ip_network(cidr_block_str, strict=False)
        except ValueError as e:
            raise ValueError(
                f"Invalid CIDR format '{cidr_block_str}' for proposed subnet '{subnet_name}': {e}"
            )

        # 3. Check for overlapping CIDRs within the proposed batch
        for existing_proposed_net in proposed_cidr_networks_seen:
            if proposed_net.overlaps(existing_proposed_net):
                raise ValueError(
                    f"Proposed CIDR '{cidr_block_str}' for subnet '{subnet_name}' "
                    f"overlaps with another proposed CIDR '{str(existing_proposed_net)}' "
                    f"within the same batch."
                )

        # 4. Check for overlapping CIDRs against existing AWS subnets
        for existing_aws_net in existing_aws_cidr_networks:
            if proposed_net.overlaps(existing_aws_net):
                raise ValueError(
                    f"Proposed CIDR '{cidr_block_str}' for subnet '{subnet_name}' "
                    f"overlaps with an existing AWS subnet CIDR '{str(existing_aws_net)}' "
                    f"in VPC '{vpc_id}'."
                )

        # If all checks pass for this subnet, add its network to the list for subsequent checks
        proposed_cidr_networks_seen.append(proposed_net)
        print(
            f"Validation successful for proposed subnet '{subnet_name}' with CIDR '{cidr_block_str}'."
        )

    print(
        f"--- All proposed subnets passed pre-synth validation checks for VPC '{vpc_id}'. ---"
    )


# --- Modified check_subnet_exists_by_name (Uses pre-fetched data) ---
def check_subnet_exists_by_name(
    subnet_name: str, existing_aws_subnets_data: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a subnet with the given name exists within the pre-fetched data.

    Args:
        subnet_name: The 'Name' tag value of the subnet to check.
        existing_aws_subnets_data: Dictionary containing existing AWS subnet data
                                   (e.g., from _get_existing_subnets_in_vpc).

    Returns:
        A tuple:
        - The first element is True if the subnet exists, False otherwise.
        - The second element is the Subnet ID if found, None otherwise.
    """
    subnet_info = existing_aws_subnets_data["by_name"].get(subnet_name)
    if subnet_info:
        print(f"Subnet '{subnet_name}' found with ID: {subnet_info['id']}")
        return True, subnet_info["id"]
    else:
        print(f"Subnet '{subnet_name}' not found.")
        return False, None


def create_nat_gateway(
    scope: Construct,
    public_subnet_for_nat: ec2.ISubnet,  # Expects a proper ISubnet
    nat_gateway_name: str,
    nat_gateway_id_context_key: str,
) -> str:
    """
    Creates a single NAT Gateway in the specified public subnet.
    It does not handle lookup from context; the calling stack should do that.
    Returns the CloudFormation Ref of the NAT Gateway ID.
    """
    print(
        f"Defining a new NAT Gateway '{nat_gateway_name}' in subnet '{public_subnet_for_nat.subnet_id}'."
    )

    # Create an Elastic IP for the NAT Gateway
    eip = ec2.CfnEIP(
        scope,
        NAT_GATEWAY_EIP_NAME,
        tags=[CfnTag(key="Name", value=NAT_GATEWAY_EIP_NAME)],
    )

    # Create the NAT Gateway
    nat_gateway_logical_id = nat_gateway_name.replace("-", "") + "NatGateway"
    nat_gateway = ec2.CfnNatGateway(
        scope,
        nat_gateway_logical_id,
        subnet_id=public_subnet_for_nat.subnet_id,  # Associate with the public subnet
        allocation_id=eip.attr_allocation_id,  # Associate with the EIP
        tags=[CfnTag(key="Name", value=nat_gateway_name)],
    )
    # The NAT GW depends on the EIP. The dependency on the subnet is implicit via subnet_id.
    nat_gateway.add_dependency(eip)

    # *** CRUCIAL: Use CfnOutput to export the ID after deployment ***
    # This is how you will get the ID to put into cdk.context.json
    CfnOutput(
        scope,
        "SingleNatGatewayIdOutput",
        value=nat_gateway.ref,
        description=f"Physical ID of the Single NAT Gateway. Add this to cdk.context.json under the key '{nat_gateway_id_context_key}'.",
        export_name=f"{scope.stack_name}-NatGatewayId",  # Make export name unique
    )

    print(
        f"CDK: Defined new NAT Gateway '{nat_gateway.ref}'. Its physical ID will be available in the stack outputs after deployment."
    )
    # Return the tokenised reference for use within this synthesis
    return nat_gateway.ref


def create_subnets(
    scope: Construct,
    vpc: ec2.IVpc,
    prefix: str,
    subnet_names: List[str],
    cidr_blocks: List[str],
    availability_zones: List[str],
    is_public: bool,
    internet_gateway_id: Optional[str] = None,
    single_nat_gateway_id: Optional[str] = None,
    internet_gateway_attachment: Optional[ec2.CfnVPCGatewayAttachment] = None,
) -> Tuple[List[ec2.CfnSubnet], List[ec2.CfnRouteTable]]:
    """
    Creates subnets using L2 constructs but returns the underlying L1 Cfn objects
    for backward compatibility.
    """
    # --- Validations remain the same ---
    if not (len(subnet_names) == len(cidr_blocks) == len(availability_zones) > 0):
        raise ValueError(
            "Subnet names, CIDR blocks, and Availability Zones lists must be non-empty and match in length."
        )
    if is_public and not internet_gateway_id:
        raise ValueError("internet_gateway_id must be provided for public subnets.")
    if not is_public and not single_nat_gateway_id:
        raise ValueError(
            "single_nat_gateway_id must be provided for private subnets when using a single NAT Gateway."
        )

    # --- We will populate these lists with the L1 objects to return ---
    created_subnets: List[ec2.CfnSubnet] = []
    created_route_tables: List[ec2.CfnRouteTable] = []

    subnet_type_tag = "public" if is_public else "private"

    for i, subnet_name in enumerate(subnet_names):
        logical_id = f"{prefix}{subnet_type_tag.capitalize()}Subnet{i+1}"

        # 1. Create the L2 Subnet (this is the easy part)
        subnet = ec2.Subnet(
            scope,
            logical_id,
            vpc_id=vpc.vpc_id,
            cidr_block=cidr_blocks[i],
            availability_zone=availability_zones[i],
            map_public_ip_on_launch=is_public,
        )
        Tags.of(subnet).add("Name", subnet_name)
        Tags.of(subnet).add("Type", subnet_type_tag)

        if is_public and internet_gateway_attachment is not None:
            subnet.node.add_dependency(internet_gateway_attachment)

        if is_public:
            try:
                subnet.add_route(
                    "DefaultInternetRoute",
                    router_id=internet_gateway_id,
                    router_type=ec2.RouterType.GATEWAY,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not create 0.0.0.0/0 -> Internet Gateway route for public "
                    f"subnet '{subnet_name}'. Ensure EXISTING_IGW_ID is attached to this "
                    f"VPC ({internet_gateway_id}): {e}"
                ) from e
            print(f"CDK: Defined public L2 subnet '{subnet_name}' and added IGW route.")
        else:
            try:
                subnet.add_route(
                    "DefaultNatRoute",
                    router_id=single_nat_gateway_id,
                    router_type=ec2.RouterType.NAT_GATEWAY,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not create 0.0.0.0/0 -> NAT Gateway route for private "
                    f"subnet '{subnet_name}': {e}"
                ) from e
            print(
                f"CDK: Defined private L2 subnet '{subnet_name}' and added NAT GW route."
            )

        route_table = subnet.route_table

        created_subnets.append(subnet)
        created_route_tables.append(route_table)

    return created_subnets, created_route_tables


def ingress_rule_exists(security_group: str, peer: str, port: str):
    for rule in security_group.connections.security_groups:
        if port:
            if rule.peer == peer and rule.connection == port:
                return True
        else:
            if rule.peer == peer:
                return True
    return False


def check_for_existing_user_pool(user_pool_name: str):
    cognito_client = boto3.client("cognito-idp")
    list_pools_response = cognito_client.list_user_pools(
        MaxResults=60
    )  # MaxResults up to 60

    # ListUserPools might require pagination if you have more than 60 pools
    # This simple example doesn't handle pagination, which could miss your pool

    existing_user_pool_id = ""

    for pool in list_pools_response.get("UserPools", []):
        if pool.get("Name") == user_pool_name:
            existing_user_pool_id = pool["Id"]
            print(
                f"Found existing user pool by name '{user_pool_name}' with ID: {existing_user_pool_id}"
            )
            break  # Found the one we're looking for

    if existing_user_pool_id:
        return True, existing_user_pool_id, pool
    else:
        return False, "", ""


def check_for_existing_user_pool_client(user_pool_id: str, user_pool_client_name: str):
    """
    Checks if a Cognito User Pool Client with the given name exists in the specified User Pool.

    Args:
        user_pool_id: The ID of the Cognito User Pool.
        user_pool_client_name: The name of the User Pool Client to check for.

    Returns:
        A tuple:
        - True, client_id, client_details if the client exists.
        - False, "", {} otherwise.
    """
    cognito_client = boto3.client("cognito-idp")
    next_token = None

    while True:
        try:
            kwargs = {"UserPoolId": user_pool_id, "MaxResults": 60}
            if next_token:
                kwargs["NextToken"] = next_token
            response = cognito_client.list_user_pool_clients(**kwargs)
        except cognito_client.exceptions.ResourceNotFoundException:
            print(f"Error: User pool with ID '{user_pool_id}' not found.")
            return False, "", {}

        except Exception as e:
            print(
                f"Could not list app clients for pool '{user_pool_id}' "
                f"(client name '{user_pool_client_name}'): {e}"
            )
            return False, "", {}

        for client in response.get("UserPoolClients", []):
            if client.get("ClientName") == user_pool_client_name:
                print(
                    f"Found existing user pool client '{user_pool_client_name}' with ID: {client['ClientId']}"
                )
                return True, client["ClientId"], client

        next_token = response.get("NextToken")
        if not next_token:
            break

    print(
        f"No app client named '{user_pool_client_name}' in user pool '{user_pool_id}'."
    )
    return False, "", {}


def check_for_secret(secret_name: str, secret_value: dict = ""):
    """
    Checks if a Secrets Manager secret with the given name exists.
    If it doesn't exist, it creates the secret.

    Args:
        secret_name: The name of the Secrets Manager secret.
        secret_value: A dictionary containing the key-value pairs for the secret.

    Returns:
        Tuple of (exists, response). When exists is True, response is the
        ``get_secret_value`` API dict (includes ``ARN`` for IAM grants).
    """
    secretsmanager_client = boto3.client("secretsmanager")

    try:
        # Try to get the secret. If it doesn't exist, a ResourceNotFoundException will be raised.
        secret_response = secretsmanager_client.get_secret_value(SecretId=secret_name)
        print("Secret already exists.")
        return True, secret_response
    except secretsmanager_client.exceptions.ResourceNotFoundException:
        print("Secret not found")
        return False, {}
    except Exception as e:
        # Handle other potential exceptions during the get operation
        print(f"Error checking for secret: {e}")
        return False, {}


def get_security_group_id_by_name(
    group_name: str,
    vpc_id: str,
    region_name: str = AWS_REGION,
) -> Tuple[bool, str]:
    """Look up a security group ID by name within a VPC."""
    if not group_name or not vpc_id:
        return False, ""
    try:
        ec2_client = boto3.client("ec2", region_name=region_name)
        response = ec2_client.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [group_name]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )
        groups = response.get("SecurityGroups") or []
        if groups:
            return True, groups[0]["GroupId"]
        return False, ""
    except ClientError as e:
        print(f"Error looking up security group '{group_name}': {e}")
        return False, ""


def resolve_service_connect_client_security_group_ids(
    explicit_ids: List[str],
    security_group_names: List[str],
    get_context_str,
) -> List[str]:
    """
    Merge explicit sg- IDs with IDs resolved from pre-check context (security_group_id:{name}).
    """
    resolved: List[str] = []
    for sg_id in explicit_ids:
        if not sg_id.startswith("sg-"):
            raise ValueError(
                f"ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_IDS entry '{sg_id}' "
                "must be a security group ID (sg-...)."
            )
        if sg_id not in resolved:
            resolved.append(sg_id)

    missing_names: List[str] = []
    for sg_name in security_group_names:
        sg_id = get_context_str(f"security_group_id:{sg_name}")
        if sg_id:
            if sg_id not in resolved:
                resolved.append(sg_id)
        else:
            missing_names.append(sg_name)

    if missing_names:
        raise ValueError(
            "Could not resolve Service Connect client security group(s) in VPC "
            f"{get_context_str('vpc_id') or '(unknown)'}: "
            + ", ".join(missing_names)
            + ". Set ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_IDS, fix "
            "ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES / "
            "ECS_SERVICE_CONNECT_CLIENT_CDK_PREFIXES, and re-run check_resources.py."
        )

    return resolved


def check_alb_exists(
    load_balancer_name: str, region_name: str = None
) -> tuple[bool, dict]:
    """
    Checks if an Application Load Balancer (ALB) with the given name exists.

    Args:
        load_balancer_name: The name of the ALB to check.
        region_name: The AWS region to check in.  If None, uses the default
                     session region.

    Returns:
        A tuple:
        - The first element is True if the ALB exists, False otherwise.
        - The second element is the ALB object (dictionary) if found,
          None otherwise.  Specifically, it returns the first element of
          the LoadBalancers list from the describe_load_balancers response.
    """
    if region_name:
        elbv2_client = boto3.client("elbv2", region_name=region_name)
    else:
        elbv2_client = boto3.client("elbv2")
    try:
        response = elbv2_client.describe_load_balancers(Names=[load_balancer_name])
        if response["LoadBalancers"]:
            return (
                True,
                response["LoadBalancers"][0],
            )  # Return True and the first ALB object
        else:
            return False, {}
    except ClientError as e:
        #  If the error indicates the ALB doesn't exist, return False
        if e.response["Error"]["Code"] == "LoadBalancerNotFound":
            return False, {}
        else:
            # Re-raise other exceptions
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}


def check_fargate_task_definition_exists(
    task_definition_name: str, region_name: str = None
) -> tuple[bool, dict]:
    """
    Checks if a Fargate task definition with the given name exists.

    Args:
        task_definition_name: The name or ARN of the task definition to check.
        region_name: The AWS region to check in. If None, uses the default
                     session region.

    Returns:
        A tuple:
        - The first element is True if the task definition exists, False otherwise.
        - The second element is the task definition object (dictionary) if found,
          None otherwise.  Specifically, it returns the first element of the
          taskDefinitions list from the describe_task_definition response.
    """
    if region_name:
        ecs_client = boto3.client("ecs", region_name=region_name)
    else:
        ecs_client = boto3.client("ecs")
    try:
        response = ecs_client.describe_task_definition(
            taskDefinition=task_definition_name
        )
        # If describe_task_definition succeeds, it returns the task definition.
        # We can directly return True and the task definition.
        return True, response["taskDefinition"]
    except ClientError as e:
        # Check for the error code indicating the task definition doesn't exist.
        if (
            e.response["Error"]["Code"] == "ClientException"
            and "Task definition" in e.response["Message"]
            and "does not exist" in e.response["Message"]
        ):
            return False, {}
        else:
            # Re-raise other exceptions.
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}


def check_ecs_service_exists(
    cluster_name: str, service_name: str, region_name: str = None
) -> tuple[bool, dict]:
    """
    Checks if an ECS service with the given name exists in the specified cluster.

    Args:
        cluster_name: The name or ARN of the ECS cluster.
        service_name: The name of the ECS service to check.
        region_name: The AWS region to check in. If None, uses the default
                     session region.

    Returns:
        A tuple:
        - The first element is True if the service exists, False otherwise.
        - The second element is the service object (dictionary) if found,
          None otherwise.
    """
    if region_name:
        ecs_client = boto3.client("ecs", region_name=region_name)
    else:
        ecs_client = boto3.client("ecs")
    try:
        response = ecs_client.describe_services(
            cluster=cluster_name, services=[service_name]
        )
        if response["services"]:
            return (
                True,
                response["services"][0],
            )  # Return True and the first service object
        else:
            return False, {}
    except ClientError as e:
        # Check for the error code indicating the service doesn't exist.
        if e.response["Error"]["Code"] == "ClusterNotFoundException":
            return False, {}
        elif e.response["Error"]["Code"] == "ServiceNotFoundException":
            return False, {}
        else:
            # Re-raise other exceptions.
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}


def check_cloudfront_distribution_exists(
    distribution_name: str, region_name: str = None
) -> tuple[bool, dict | None]:
    """
    Checks if a CloudFront distribution with the given name exists.

    Args:
        distribution_name: The name of the CloudFront distribution to check.
        region_name: The AWS region to check in. If None, uses the default
                     session region.  Note: CloudFront is a global service,
                     so the region is usually 'us-east-1', but this parameter
                     is included for completeness.

    Returns:
        A tuple:
        - The first element is True if the distribution exists, False otherwise.
        - The second element is the distribution object (dictionary) if found,
          None otherwise.  Specifically, it returns the first element of the
          DistributionList from the ListDistributions response.
    """
    if region_name:
        cf_client = boto3.client("cloudfront", region_name=region_name)
    else:
        cf_client = boto3.client("cloudfront")
    try:
        response = cf_client.list_distributions()
        if "Items" in response["DistributionList"]:
            for distribution in response["DistributionList"]["Items"]:
                # CloudFront doesn't directly filter by name, so we have to iterate.
                if (
                    distribution["AliasSet"]["Items"]
                    and distribution["AliasSet"]["Items"][0] == distribution_name
                ):
                    return True, distribution
            return False, None
        else:
            return False, None
    except ClientError as e:
        #  If the error indicates the Distribution doesn't exist, return False
        if e.response["Error"]["Code"] == "NoSuchDistribution":
            return False, None
        else:
            # Re-raise other exceptions
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, None


def create_web_acl_with_common_rules(
    scope: Construct, web_acl_name: str, waf_scope: str = "CLOUDFRONT"
):
    """
    Use CDK to create a web ACL based on an AWS common rule set with overrides.
    This function now expects a 'scope' argument, typically 'self' from your stack,
    as CfnWebACL requires a construct scope.
    """

    # Create full list of rules
    rules = []
    aws_ruleset_names = [
        "AWSManagedRulesCommonRuleSet",
        "AWSManagedRulesKnownBadInputsRuleSet",
        "AWSManagedRulesAmazonIpReputationList",
    ]

    # Use a separate counter to assign unique priorities sequentially
    priority_counter = 1

    for aws_rule_name in aws_ruleset_names:
        current_rule_action_overrides = None

        # All managed rule groups need an override_action.
        # 'none' means use the managed rule group's default action.
        current_override_action = wafv2.CfnWebACL.OverrideActionProperty(none={})

        current_priority = priority_counter
        priority_counter += 1

        if aws_rule_name == "AWSManagedRulesCommonRuleSet":
            current_rule_action_overrides = [
                wafv2.CfnWebACL.RuleActionOverrideProperty(
                    name="SizeRestrictions_BODY",
                    action_to_use=wafv2.CfnWebACL.RuleActionProperty(allow={}),
                )
            ]
            # No need to set current_override_action here, it's already set above.
            # If you wanted this specific rule to have a *fixed* priority, you'd handle it differently
            # For now, it will get priority 1 from the counter.

        rule_property = wafv2.CfnWebACL.RuleProperty(
            name=aws_rule_name,
            priority=current_priority,
            statement=wafv2.CfnWebACL.StatementProperty(
                managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                    vendor_name="AWS",
                    name=aws_rule_name,
                    rule_action_overrides=current_rule_action_overrides,
                )
            ),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=aws_rule_name,
                sampled_requests_enabled=True,
            ),
            override_action=current_override_action,  # THIS IS THE CRUCIAL PART FOR ALL MANAGED RULES
        )

        rules.append(rule_property)

    # Add the rate limit rule
    rate_limit_priority = priority_counter  # Use the next available priority
    rules.append(
        wafv2.CfnWebACL.RuleProperty(
            name="RateLimitRule",
            priority=rate_limit_priority,
            statement=wafv2.CfnWebACL.StatementProperty(
                rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                    limit=1000, aggregate_key_type="IP"
                )
            ),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name="RateLimitRule",
                sampled_requests_enabled=True,
            ),
            action=wafv2.CfnWebACL.RuleActionProperty(block={}),
        )
    )

    web_acl = wafv2.CfnWebACL(
        scope,
        "WebACL",
        name=web_acl_name,
        default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
        scope=waf_scope,
        visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
            cloud_watch_metrics_enabled=True,
            metric_name="webACL",
            sampled_requests_enabled=True,
        ),
        rules=rules,
    )

    CfnOutput(scope, "WebACLArn", value=web_acl.attr_arn)
    web_acl.apply_removal_policy(managed_resource_removal_policy())

    return web_acl


def check_web_acl_exists(
    web_acl_name: str, scope: str, region_name: str = None
) -> tuple[bool, dict]:
    """
    Checks if a Web ACL with the given name and scope exists.

    Args:
        web_acl_name: The name of the Web ACL to check.
        scope: The scope of the Web ACL ('CLOUDFRONT' or 'REGIONAL').
        region_name: The AWS region to check in. Required for REGIONAL scope.
                     If None, uses the default session region.  For CLOUDFRONT,
                     the region should be 'us-east-1'.

    Returns:
        A tuple:
        - The first element is True if the Web ACL exists, False otherwise.
        - The second element is the Web ACL object (dictionary) if found,
          None otherwise.
    """
    if scope not in ["CLOUDFRONT", "REGIONAL"]:
        raise ValueError("Scope must be either 'CLOUDFRONT' or 'REGIONAL'")

    if scope == "REGIONAL" and not region_name:
        raise ValueError("Region name is required for REGIONAL scope")

    if scope == "CLOUDFRONT":
        region_name = "us-east-1"  # CloudFront scope requires us-east-1

    if region_name:
        waf_client = boto3.client("wafv2", region_name=region_name)
    else:
        waf_client = boto3.client("wafv2")
    try:
        response = waf_client.list_web_acls(Scope=scope)
        if "WebACLs" in response:
            for web_acl in response["WebACLs"]:
                if web_acl["Name"] == web_acl_name:
                    # Describe the Web ACL to get the full object.
                    describe_response = waf_client.describe_web_acl(
                        Name=web_acl_name, Scope=scope
                    )
                    return True, describe_response["WebACL"]
            return False, {}
        else:
            return False, {}
    except ClientError as e:
        # Check for the error code indicating the web ACL doesn't exist.
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return False, {}
        else:
            # Re-raise other exceptions.
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}


def add_alb_https_listener_with_cert(
    scope: Construct,
    logical_id: str,  # A unique ID for this listener construct
    alb: elb.ApplicationLoadBalancer,
    acm_certificate_arn: Optional[
        str
    ],  # Optional: If None, no HTTPS listener will be created
    default_target_group: elb.ITargetGroup,  # Mandatory: The target group to forward traffic to
    listener_port_https: int = 443,
    listener_open_to_internet: bool = False,  # Be cautious with True, ensure ALB security group restricts access
    # --- Cognito Authentication Parameters ---
    enable_cognito_auth: bool = False,
    cognito_user_pool: Optional[cognito.IUserPool] = None,
    cognito_user_pool_client: Optional[cognito.IUserPoolClient] = None,
    cognito_user_pool_domain: Optional[
        str
    ] = None,  # E.g., "my-app-domain" for "my-app-domain.auth.region.amazoncognito.com"
    cognito_auth_scope: Optional[
        str
    ] = "openid profile email",  # Default recommended scope
    cognito_auth_on_unauthenticated_request: elb.UnauthenticatedAction = elb.UnauthenticatedAction.AUTHENTICATE,
    stickiness_cookie_duration=None,
    # --- End Cognito Parameters ---
) -> Optional[elb.ApplicationListener]:
    """
    Conditionally adds an HTTPS listener to an ALB with an ACM certificate,
    and optionally enables Cognito User Pool authentication.

    Args:
        scope (Construct): The scope in which to define this construct (e.g., your CDK Stack).
        logical_id (str): A unique logical ID for the listener construct within the stack.
        alb (elb.ApplicationLoadBalancer): The Application Load Balancer to add the listener to.
        acm_certificate_arn (Optional[str]): The ARN of the ACM certificate to attach.
                                             If None, the HTTPS listener will NOT be created.
        default_target_group (elb.ITargetGroup): The default target group for the listener to forward traffic to.
                                                 This is mandatory for a functional listener.
        listener_port_https (int): The HTTPS port to listen on (default: 443).
        listener_open_to_internet (bool): Whether the listener should allow connections from all sources.
                                          If False (recommended), ensure your ALB's security group allows
                                          inbound traffic on this port from desired sources.
        enable_cognito_auth (bool): Set to True to enable Cognito User Pool authentication.
        cognito_user_pool (Optional[cognito.IUserPool]): The Cognito User Pool object. Required if enable_cognito_auth is True.
        cognito_user_pool_client (Optional[cognito.IUserPoolClient]): The Cognito User Pool App Client object. Required if enable_cognito_auth is True.
        cognito_user_pool_domain (Optional[str]): The domain prefix for your Cognito User Pool. Required if enable_cognito_auth is True.
        cognito_auth_scope (Optional[str]): The scope for the Cognito authentication.
        cognito_auth_on_unauthenticated_request (elb.UnauthenticatedAction): Action for unauthenticated requests.
                                                                           Defaults to AUTHENTICATE (redirect to login).

    Returns:
        Optional[elb.ApplicationListener]: The created ApplicationListener if successful,
                                           None if no ACM certificate ARN was provided.
    """
    https_listener = None
    if acm_certificate_arn:
        certificates_list = [elb.ListenerCertificate.from_arn(acm_certificate_arn)]
        print(
            f"Attempting to add ALB HTTPS listener on port {listener_port_https} with ACM certificate: {acm_certificate_arn}"
        )

        # Determine the default action based on whether Cognito auth is enabled
        default_action = None
        if enable_cognito_auth is True:
            if not all(
                [cognito_user_pool, cognito_user_pool_client, cognito_user_pool_domain]
            ):
                raise ValueError(
                    "Cognito User Pool, Client, and Domain must be provided if enable_cognito_auth is True."
                )
            print(
                f"Enabling Cognito authentication with User Pool: {cognito_user_pool.user_pool_id}"
            )

            default_action = elb_act.AuthenticateCognitoAction(
                next=elb.ListenerAction.forward(
                    [default_target_group]
                ),  # After successful auth, forward to TG
                user_pool=cognito_user_pool,
                user_pool_client=cognito_user_pool_client,
                user_pool_domain=cognito_user_pool_domain,
                scope=cognito_auth_scope,
                on_unauthenticated_request=cognito_auth_on_unauthenticated_request,
                session_timeout=stickiness_cookie_duration,
                # Additional options you might want to configure:
                # session_cookie_name="AWSELBCookies"
            )
        else:
            default_action = elb.ListenerAction.forward([default_target_group])
            print("Cognito authentication is NOT enabled for this listener.")

        # Add the HTTPS listener
        https_listener = alb.add_listener(
            logical_id,
            port=listener_port_https,
            open=listener_open_to_internet,
            certificates=certificates_list,
            default_action=default_action,  # Use the determined default action
        )
        print(f"ALB HTTPS listener on port {listener_port_https} defined.")
    else:
        print("ACM_CERTIFICATE_ARN is not provided. Skipping HTTPS listener creation.")

    return https_listener


def create_ecs_express_infrastructure_role(
    scope: Construct,
    logical_id: str,
    role_name: str,
) -> iam.Role:
    """IAM role for ECS Express Mode to provision ALB, ACM cert, and autoscaling."""
    role = iam.Role(
        scope,
        logical_id,
        role_name=role_name,
        assumed_by=iam.ServicePrincipal("ecs.amazonaws.com"),
    )
    role.add_managed_policy(
        iam.ManagedPolicy.from_aws_managed_policy_name(
            "service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"
        )
    )
    return role


def _secret_value_from_arn(secret_arn: str, json_key: str) -> str:
    return f"{secret_arn}:{json_key}::"


def express_ingress_listener_arn(
    express_service: ecs.CfnExpressGatewayService,
) -> str:
    return express_service.attr_ecs_managed_resource_arns_ingress_path_listener_arn


def express_ingress_load_balancer_arn(
    express_service: ecs.CfnExpressGatewayService,
) -> str:
    return express_service.attr_ecs_managed_resource_arns_ingress_path_load_balancer_arn


def express_ingress_first_target_group_arn(
    express_service: ecs.CfnExpressGatewayService,
) -> str:
    """First target group ARN; use typed list attr (get_att returns a scalar Reference)."""
    return Fn.select(
        0,
        express_service.attr_ecs_managed_resource_arns_ingress_path_target_group_arns,
    )


def express_ingress_first_load_balancer_security_group_arn(
    express_service: ecs.CfnExpressGatewayService,
) -> str:
    """First Express-managed ALB security group ARN."""
    return Fn.select(
        0,
        express_service.attr_ecs_managed_resource_arns_ingress_path_load_balancer_security_groups,
    )


def _security_group_id_from_arn(security_group_arn: str) -> str:
    """EC2 APIs expect sg- IDs; Express managed-resource attrs return full ARNs."""
    return Fn.select(1, Fn.split("security-group/", security_group_arn))


def express_ingress_first_load_balancer_security_group(
    express_service: ecs.CfnExpressGatewayService,
) -> str:
    """First ALB security group ID (sg-...) for EC2/ELB imports."""
    return _security_group_id_from_arn(
        express_ingress_first_load_balancer_security_group_arn(express_service)
    )


# Injected via Express `secrets`, not plain environment (avoid duplication/leakage).
_EXPRESS_SECRET_ENV_NAMES = frozenset(
    {"AWS_USER_POOL_ID", "AWS_CLIENT_ID", "AWS_CLIENT_SECRET"}
)


def create_basic_config_env(
    out_dir: str = "config",
    s3_log_config_bucket_name: str = S3_LOG_CONFIG_BUCKET_NAME,
    s3_output_bucket_name: str = S3_OUTPUT_BUCKET_NAME,
    access_log_dynamodb_table_name: str = ACCESS_LOG_DYNAMODB_TABLE_NAME,
    feedback_log_dynamodb_table_name: str = FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    usage_log_dynamodb_table_name: str = USAGE_LOG_DYNAMODB_TABLE_NAME,
    *,
    headless: bool = False,
    alb_cognito: bool = False,
    pi_express_backend: bool = False,
):
    """
    Create a basic app_config.env file for the deployed redaction app.

    ``alb_cognito=True`` disables in-app Gradio Cognito login when the ALB
    ``authenticate-cognito`` action already protects the service (Express Mode).

    ``pi_express_backend=True`` disables in-app login on the main redaction app when
    Pi Express calls it over Service Connect (users authenticate on the Pi UI only).
    """
    variables = {
        "COGNITO_AUTH": (
            "False" if headless or alb_cognito or pi_express_backend else "True"
        ),
        "RUN_AWS_FUNCTIONS": "True",
        "DISPLAY_FILE_NAMES_IN_LOGS": "False",
        "SESSION_OUTPUT_FOLDER": "True",
        "SAVE_LOGS_TO_DYNAMODB": "True",
        "SHOW_COSTS": "True",
        "SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS": "True",
        "LOAD_PREVIOUS_TEXTRACT_JOBS_S3": "True",
        "DOCUMENT_REDACTION_BUCKET": s3_log_config_bucket_name,
        "TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET": s3_output_bucket_name,
        "ACCESS_LOG_DYNAMODB_TABLE_NAME": access_log_dynamodb_table_name,
        "FEEDBACK_LOG_DYNAMODB_TABLE_NAME": feedback_log_dynamodb_table_name,
        "USAGE_LOG_DYNAMODB_TABLE_NAME": usage_log_dynamodb_table_name,
    }

    _ensure_folder_exists(out_dir + "/")
    env_file_path = os.path.abspath(os.path.join(out_dir, APP_CONFIG_ENV_BASENAME))

    if not os.path.exists(env_file_path):
        with open(env_file_path, "w", encoding="utf-8"):
            pass

    for key, value in variables.items():
        set_key(env_file_path, key, str(value), quote_mode="never")

    return variables


def load_app_config_env_for_express(
    config_env_path: str,
    *,
    exclude_names: Optional[FrozenSet[str]] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> List[ecs.CfnExpressGatewayService.KeyValuePairProperty]:
    """
    Load KEY=VALUE pairs from config/app_config.env for Express PrimaryContainer.environment.

    Uses the same file written by create_basic_config_env() and uploaded to S3 on the
    legacy Fargate path (environmentFiles). ``overrides`` replace keys after loading
    (e.g. ``COGNITO_AUTH=False`` when ALB handles Cognito).
    """
    exclude = exclude_names or _EXPRESS_SECRET_ENV_NAMES
    path = os.path.abspath(config_env_path)
    if not os.path.isfile(path):
        print(
            f"Warning: app config env file not found at {path}; "
            "Express container will not receive app config environment variables."
        )
        return []

    raw = dict(dotenv_values(path))
    if overrides:
        raw.update(overrides)
    environment: List[ecs.CfnExpressGatewayService.KeyValuePairProperty] = []
    for name, value in sorted(raw.items()):
        if not name or value is None or name in exclude:
            continue
        environment.append(
            ecs.CfnExpressGatewayService.KeyValuePairProperty(
                name=name,
                value=str(value),
            )
        )
    print(
        f"Loaded {len(environment)} environment variables from {path} for ECS Express Mode."
    )
    return environment


def build_express_gateway_primary_container(
    *,
    image_uri: str,
    container_port: int,
    log_group_name: str,
    aws_region: str,
    secret: secretsmanager.ISecret,
    environment: Optional[
        List[ecs.CfnExpressGatewayService.KeyValuePairProperty]
    ] = None,
) -> ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty:
    secret_arn = secret.secret_arn
    return ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
        image=image_uri,
        container_port=container_port,
        aws_logs_configuration=ecs.CfnExpressGatewayService.ExpressGatewayServiceAwsLogsConfigurationProperty(
            log_group=log_group_name,
            log_stream_prefix="ecs",
        ),
        environment=environment or None,
        secrets=[
            ecs.CfnExpressGatewayService.SecretProperty(
                name="AWS_USER_POOL_ID",
                value_from=_secret_value_from_arn(secret_arn, "REDACTION_USER_POOL_ID"),
            ),
            ecs.CfnExpressGatewayService.SecretProperty(
                name="AWS_CLIENT_ID",
                value_from=_secret_value_from_arn(secret_arn, "REDACTION_CLIENT_ID"),
            ),
            ecs.CfnExpressGatewayService.SecretProperty(
                name="AWS_CLIENT_SECRET",
                value_from=_secret_value_from_arn(
                    secret_arn, "REDACTION_CLIENT_SECRET"
                ),
            ),
        ],
    )


def express_gateway_idle_scaling_target(
    *,
    max_task_count: int = 1,
) -> ecs.CfnExpressGatewayService.ExpressGatewayScalingTargetProperty:
    """Defer running tasks until post-deploy image build (legacy Fargate uses desired_count=0)."""
    return ecs.CfnExpressGatewayService.ExpressGatewayScalingTargetProperty(
        min_task_count=0,
        max_task_count=max_task_count,
        auto_scaling_metric="AVERAGE_CPU",
        auto_scaling_target_value=60,
    )


def create_express_gateway_service(
    scope: Construct,
    logical_id: str,
    *,
    service_name: str,
    cluster_name: str,
    execution_role_arn: str,
    infrastructure_role_arn: str,
    task_role_arn: str,
    cpu: str,
    memory: str,
    health_check_path: str,
    primary_container: ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty,
    subnet_ids: List[str],
    security_group_ids: List[str],
    scaling_target: Optional[
        ecs.CfnExpressGatewayService.ExpressGatewayScalingTargetProperty
    ] = None,
) -> ecs.CfnExpressGatewayService:
    network = None
    if subnet_ids or security_group_ids:
        network = ecs.CfnExpressGatewayService.ExpressGatewayServiceNetworkConfigurationProperty(
            subnets=subnet_ids or None,
            security_groups=security_group_ids or None,
        )
    express_service = ecs.CfnExpressGatewayService(
        scope,
        logical_id,
        service_name=service_name,
        cluster=cluster_name,
        execution_role_arn=execution_role_arn,
        infrastructure_role_arn=infrastructure_role_arn,
        task_role_arn=task_role_arn,
        cpu=cpu,
        memory=memory,
        health_check_path=health_check_path,
        primary_container=primary_container,
        network_configuration=network,
        scaling_target=scaling_target or express_gateway_idle_scaling_target(),
    )
    return express_service


def _forward_target_group_action(
    target_group_arn: str,
    stickiness_seconds: int,
) -> Dict[str, Any]:
    action: Dict[str, Any] = {
        "Type": "forward",
        "Order": 2,
        "ForwardConfig": {
            "TargetGroups": [{"TargetGroupArn": target_group_arn}],
        },
    }
    if stickiness_seconds > 0:
        action["ForwardConfig"]["TargetGroupStickinessConfig"] = {
            "Enabled": True,
            "DurationSeconds": stickiness_seconds,
        }
    return action


def elbv2_cognito_auth_custom_resource_policy() -> cr.AwsCustomResourcePolicy:
    """
    IAM policy for AwsCustomResource calls that configure authenticate-cognito on ALB.

    ELB validates the user pool client during modifyListener/createRule; the caller
    (the custom-resource Lambda) needs cognito-idp:DescribeUserPoolClient.
    """
    return cr.AwsCustomResourcePolicy.from_statements(
        [
            iam.PolicyStatement(
                actions=[
                    "elasticloadbalancing:DescribeRules",
                    "elasticloadbalancing:ModifyListener",
                    "elasticloadbalancing:CreateRule",
                    "elasticloadbalancing:ModifyRule",
                    "elasticloadbalancing:DeleteRule",
                ],
                resources=["*"],
            ),
            iam.PolicyStatement(
                actions=["cognito-idp:DescribeUserPoolClient"],
                resources=["*"],
            ),
        ]
    )


def build_cognito_default_listener_actions(
    *,
    user_pool_arn: str,
    user_pool_client_id: str,
    user_pool_domain_prefix: str,
    target_group_arn: str,
    stickiness_seconds: int = 28800,
    scope: str = "openid email profile",
) -> List[Dict[str, Any]]:
    """Default actions for ELBv2 ModifyListener (authenticate-cognito + forward)."""
    return [
        {
            "Type": "authenticate-cognito",
            "Order": 1,
            "AuthenticateCognitoConfig": {
                "UserPoolArn": user_pool_arn,
                "UserPoolClientId": user_pool_client_id,
                "UserPoolDomain": user_pool_domain_prefix,
                "Scope": scope,
                "OnUnauthenticatedRequest": "authenticate",
                "SessionTimeout": stickiness_seconds,
            },
        },
        _forward_target_group_action(target_group_arn, stickiness_seconds),
    ]


def configure_express_listener_cognito_and_cloudfront(
    scope: Construct,
    logical_id_prefix: str,
    *,
    express_service: ecs.CfnExpressGatewayService,
    user_pool_arn: str,
    user_pool_client_id: str,
    user_pool_domain_prefix: str,
    use_cloudfront: bool,
    cloudfront_host_header: str,
    stickiness_seconds: int = 28800,
    allow_cloudfront_origin_without_cognito: bool = False,
) -> None:
    """
    Attach Cognito auth to the Express-managed HTTPS listener.

    By default, **no** forward-only host-header rules are added. CloudFront (and all
    other) traffic uses the listener default action: ``authenticate-cognito`` then
    forward. Set ``allow_cloudfront_origin_without_cognito=True`` only when the ALB
    must accept CloudFront origin requests without a Cognito session (legacy pattern).
    """
    listener_arn = express_ingress_listener_arn(express_service)
    target_group_arn = express_ingress_first_target_group_arn(express_service)
    default_actions = build_cognito_default_listener_actions(
        user_pool_arn=user_pool_arn,
        user_pool_client_id=user_pool_client_id,
        user_pool_domain_prefix=user_pool_domain_prefix,
        target_group_arn=target_group_arn,
        stickiness_seconds=stickiness_seconds,
    )
    modify_listener = cr.AwsCustomResource(
        scope,
        f"{logical_id_prefix}ModifyExpressListener",
        on_create=cr.AwsSdkCall(
            service="ELBv2",
            action="modifyListener",
            parameters={
                "ListenerArn": listener_arn,
                "DefaultActions": default_actions,
            },
            physical_resource_id=cr.PhysicalResourceId.of(
                f"express-listener-cognito-{logical_id_prefix}"
            ),
        ),
        on_update=cr.AwsSdkCall(
            service="ELBv2",
            action="modifyListener",
            parameters={
                "ListenerArn": listener_arn,
                "DefaultActions": default_actions,
            },
            physical_resource_id=cr.PhysicalResourceId.of(
                f"express-listener-cognito-{logical_id_prefix}"
            ),
        ),
        policy=elbv2_cognito_auth_custom_resource_policy(),
    )
    modify_listener.node.add_dependency(express_service)

    if (
        use_cloudfront
        and cloudfront_host_header
        and allow_cloudfront_origin_without_cognito
    ):
        forward_only = [
            {
                "Type": "forward",
                "Order": 1,
                "ForwardConfig": {
                    "TargetGroups": [{"TargetGroupArn": target_group_arn}],
                    "TargetGroupStickinessConfig": {
                        "Enabled": True,
                        "DurationSeconds": stickiness_seconds,
                    },
                },
            }
        ]
        _elbv2_listener_rule_custom_resource(
            scope,
            f"{logical_id_prefix}ExpressCloudFrontHostRule",
            listener_arn=listener_arn,
            priority=1,
            conditions=[
                {
                    "Field": "host-header",
                    "HostHeaderConfig": {"Values": [cloudfront_host_header]},
                }
            ],
            rule_actions=forward_only,
            dependencies=[modify_listener],
        )


def allow_express_load_balancer_to_ecs_security_group(
    scope: Construct,
    logical_id: str,
    *,
    express_service: ecs.CfnExpressGatewayService,
    ecs_security_group: ec2.ISecurityGroup,
    container_port: int,
) -> None:
    """Allow traffic from the Express-managed ALB security group to the task SG."""
    lb_sg_id = express_ingress_first_load_balancer_security_group(express_service)
    ec2.CfnSecurityGroupIngress(
        scope,
        logical_id,
        group_id=ecs_security_group.security_group_id,
        ip_protocol="tcp",
        from_port=container_port,
        to_port=container_port,
        source_security_group_id=lb_sg_id,
        description="Express Mode ALB to ECS tasks",
    )


def _dict_env_to_express_key_value_pairs(
    environment: Dict[str, str],
) -> List[ecs.CfnExpressGatewayService.KeyValuePairProperty]:
    return [
        ecs.CfnExpressGatewayService.KeyValuePairProperty(name=k, value=str(v))
        for k, v in environment.items()
        if v is not None and str(v) != ""
    ]


def normalize_pi_alb_path_prefix(raw: str, *, default: str = "pi") -> str:
    """Return a leading-slash path prefix (no trailing slash), e.g. '/pi'."""
    segment = (raw or default).strip().strip("/")
    return f"/{segment}" if segment else f"/{default}"


def normalize_pi_alb_routing_mode(raw: str) -> str:
    mode = (raw or "path").strip().lower()
    allowed = frozenset({"path", "host", "both"})
    if mode not in allowed:
        raise ValueError(
            f"PI_ALB_ROUTING must be one of {sorted(allowed)}; got '{raw}'."
        )
    return mode


def pi_alb_path_patterns(path_prefix: str) -> List[str]:
    """ALB path-pattern values for a Pi path prefix (exact + subtree)."""
    prefix = normalize_pi_alb_path_prefix(path_prefix)
    return [prefix, f"{prefix}/*"]


def pi_alb_health_check_path(path_prefix: str, routing_mode: str) -> str:
    if normalize_pi_alb_routing_mode(routing_mode) in ("path", "both"):
        return f"{normalize_pi_alb_path_prefix(path_prefix)}/"
    return "/"


def pi_alb_root_path_for_container(path_prefix: str, routing_mode: str) -> str:
    """Gradio/FastAPI ROOT_PATH to set on Pi tasks when path routing is enabled."""
    if normalize_pi_alb_routing_mode(routing_mode) in ("path", "both"):
        return normalize_pi_alb_path_prefix(path_prefix)
    return ""


def pi_listener_rule_count(routing_mode: str) -> int:
    mode = normalize_pi_alb_routing_mode(routing_mode)
    count = 0
    if mode in ("path", "both"):
        count += 1
    if mode in ("host", "both"):
        count += 1
    return count


def format_express_pi_public_url(express_endpoint: str) -> str:
    """Public Pi UI URL for a dedicated ECS Express managed HTTPS endpoint."""
    base = (express_endpoint or "").strip().rstrip("/")
    return f"{base}/" if base else ""


def format_main_express_gradio_url(express_endpoint: str) -> str:
    """HTTPS base URL for the main doc_redaction Gradio app (ExpressServiceEndpoint)."""
    from cdk_config import normalize_https_redirect_url

    return normalize_https_redirect_url(express_endpoint)


def format_pi_public_urls(
    *,
    routing_mode: str,
    path_prefix: str,
    host_header: str,
    cloudfront_domain: str = "",
    use_https: bool = True,
) -> List[str]:
    """Human-facing Pi UI URLs for stack outputs."""
    scheme = "https" if use_https else "http"
    urls: List[str] = []
    mode = normalize_pi_alb_routing_mode(routing_mode)
    prefix = normalize_pi_alb_path_prefix(path_prefix)
    if mode in ("path", "both"):
        if cloudfront_domain.strip():
            urls.append(f"{scheme}://{cloudfront_domain.strip()}{prefix}/")
        else:
            urls.append(f"{scheme}://<cloudfront-or-alb-host>{prefix}/")
    if mode in ("host", "both") and host_header.strip():
        urls.append(f"{scheme}://{host_header.strip()}/")
    return urls


def _apply_pi_root_path_env(env: Dict[str, str], pi_root_path: str) -> None:
    if pi_root_path:
        env["PI_ROOT_PATH"] = pi_root_path
        env["ROOT_PATH"] = pi_root_path
        env["FASTAPI_ROOT_PATH"] = pi_root_path


def build_pi_express_container_environment(
    *,
    service_connect_discovery_name: str,
    main_app_port: Union[str, int],
    pi_gradio_port: Union[str, int],
    cognito_auth: bool = True,
    doc_redaction_gradio_url: Optional[str] = None,
) -> Dict[str, str]:
    """Inline env for Pi on Express (no volume mounts; workspace under /tmp)."""
    port = int(main_app_port)
    pi_port = int(pi_gradio_port)
    backend_url = (doc_redaction_gradio_url or "").strip().rstrip("/")
    if not backend_url:
        backend_url = f"http://{service_connect_discovery_name}:{port}"
    env = {
        "APP_TYPE": "pi",
        "APP_CONFIG_PATH": "/workspace/doc_redaction/config/pi_agent.env.example",
        "PI_DEPLOYMENT_PROFILE": "aws-ecs",
        "PI_DEFAULT_PROVIDER": "amazon-bedrock",
        "DOC_REDACTION_GRADIO_URL": backend_url,
        "PI_GRADIO_PORT": str(pi_port),
        "GRADIO_SERVER_PORT": str(pi_port),
        "GRADIO_SERVER_NAME": "0.0.0.0",
        "PI_WORKSPACE_DIR": "/tmp/pi-workspace",
        "PI_WORKDIR": "/workspace/doc_redaction",
        "PI_UPLOAD_ROOT": "/tmp/gradio",
        "PI_SESSION_DIR": "/tmp/pi-sessions",
        "PI_CODING_AGENT_DIR": "/tmp/pi-agent",
        "ACCESS_LOGS_FOLDER": "/tmp/pi-logs/",
        "USAGE_LOGS_FOLDER": "/tmp/pi-usage/",
        "FEEDBACK_LOGS_FOLDER": "/tmp/pi-feedback/",
        "RUN_FASTAPI": "True",
        "RUN_AWS_FUNCTIONS": "True",
        "COGNITO_AUTH": "True" if cognito_auth else "False",
    }
    try:
        from cdk_config import AGENTCORE_RUNTIME_URL, ENABLE_AGENTCORE_RUNTIME

        if ENABLE_AGENTCORE_RUNTIME == "True" and AGENTCORE_RUNTIME_URL:
            env["AGENT_ORCHESTRATOR"] = "agentcore"
            env["AGENTCORE_RUNTIME_URL"] = AGENTCORE_RUNTIME_URL
    except ImportError:
        pass
    return env


def build_express_pi_primary_container(
    *,
    image_uri: str,
    container_port: int,
    log_group_name: str,
    aws_region: str,
    environment: Optional[Dict[str, str]] = None,
    secret: Optional[secretsmanager.ISecret] = None,
    cognito_auth: bool = True,
) -> ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty:
    """Express PrimaryContainer for Pi (inline env; Cognito creds from Secrets Manager)."""
    env_pairs = (
        _dict_env_to_express_key_value_pairs(environment) if environment else None
    )
    secrets: Optional[List[ecs.CfnExpressGatewayService.SecretProperty]] = None
    if secret is not None and cognito_auth:
        secret_arn = secret.secret_arn
        secrets = [
            ecs.CfnExpressGatewayService.SecretProperty(
                name="AWS_USER_POOL_ID",
                value_from=_secret_value_from_arn(secret_arn, "REDACTION_USER_POOL_ID"),
            ),
            ecs.CfnExpressGatewayService.SecretProperty(
                name="AWS_CLIENT_ID",
                value_from=_secret_value_from_arn(secret_arn, "REDACTION_CLIENT_ID"),
            ),
            ecs.CfnExpressGatewayService.SecretProperty(
                name="AWS_CLIENT_SECRET",
                value_from=_secret_value_from_arn(
                    secret_arn, "REDACTION_CLIENT_SECRET"
                ),
            ),
        ]
    return ecs.CfnExpressGatewayService.ExpressGatewayContainerProperty(
        image=image_uri,
        container_port=container_port,
        aws_logs_configuration=ecs.CfnExpressGatewayService.ExpressGatewayServiceAwsLogsConfigurationProperty(
            log_group=log_group_name,
            log_stream_prefix="ecs-pi",
        ),
        environment=env_pairs,
        secrets=secrets,
    )


_ELBV2_RULE_DELETE_IGNORE = (
    ".*(not a valid listener rule ARN|RuleNotFound|ResourceNotFound|ValidationError).*"
)
_ELBV2_LISTENER_RULE_UPSERT_PROVIDER_ID = "Elbv2ListenerRuleUpsertProvider"


def _elbv2_listener_rule_upsert_provider(scope: Construct) -> cr.Provider:
    """Shared Provider for upserting ALB listener rules (one Lambda per stack)."""
    stack = Stack.of(scope)
    existing = stack.node.try_find_child(_ELBV2_LISTENER_RULE_UPSERT_PROVIDER_ID)
    if existing is not None:
        return existing

    asset_dir = os.path.join(
        os.path.dirname(__file__), "lambda_elbv2_listener_rule_upsert"
    )
    fn = lambda_.Function(
        stack,
        "Elbv2ListenerRuleUpsertFn",
        runtime=lambda_.Runtime.PYTHON_3_12,
        handler="lambda_function.handler",
        code=lambda_.Code.from_asset(asset_dir),
        timeout=Duration.seconds(120),
        description="Upsert ALB listener rules for Express Cognito routing",
    )
    fn.add_to_role_policy(
        iam.PolicyStatement(
            actions=[
                "elasticloadbalancing:DescribeRules",
                "elasticloadbalancing:CreateRule",
                "elasticloadbalancing:ModifyRule",
                "elasticloadbalancing:DeleteRule",
            ],
            resources=["*"],
        )
    )
    fn.add_to_role_policy(
        iam.PolicyStatement(
            actions=["cognito-idp:DescribeUserPoolClient"],
            resources=["*"],
        )
    )
    return cr.Provider(
        stack,
        _ELBV2_LISTENER_RULE_UPSERT_PROVIDER_ID,
        on_event_handler=fn,
    )


def _elbv2_listener_rule_custom_resource(
    scope: Construct,
    logical_id: str,
    *,
    listener_arn: str,
    priority: int,
    conditions: List[Dict[str, Any]],
    rule_actions: List[Dict[str, Any]],
    dependencies: Optional[List[Any]] = None,
) -> CustomResource:
    """
    Create or update a numbered listener rule (upsert by priority + conditions).

    Reuses an existing rule at the same priority when conditions match, which
    avoids PriorityInUse failures after partial deploy rollbacks.
    """
    provider = _elbv2_listener_rule_upsert_provider(scope)
    resource = CustomResource(
        scope,
        logical_id,
        service_token=provider.service_token,
        resource_type="Custom::Elbv2ListenerRuleUpsert",
        properties={
            "ListenerArn": listener_arn,
            "Priority": priority,
            "Conditions": conditions,
            "Actions": rule_actions,
        },
    )
    for dep in dependencies or []:
        resource.node.add_dependency(dep)
    return resource


def _express_pi_listener_rule_custom_resource(
    scope: Construct,
    logical_id: str,
    *,
    listener_arn: str,
    priority: int,
    conditions: List[Dict[str, Any]],
    rule_actions: List[Dict[str, Any]],
    express_main_service: ecs.CfnExpressGatewayService,
    express_pi_service: ecs.CfnExpressGatewayService,
) -> CustomResource:
    return _elbv2_listener_rule_custom_resource(
        scope,
        logical_id,
        listener_arn=listener_arn,
        priority=priority,
        conditions=conditions,
        rule_actions=rule_actions,
        dependencies=[express_pi_service, express_main_service],
    )


def configure_express_pi_listener_rules(
    scope: Construct,
    logical_id_prefix: str,
    *,
    express_main_service: ecs.CfnExpressGatewayService,
    express_pi_service: ecs.CfnExpressGatewayService,
    routing_mode: str,
    path_prefix: str,
    pi_host_header: str,
    rule_priority: int,
    user_pool_arn: str,
    user_pool_client_id: str,
    user_pool_domain_prefix: str,
    stickiness_seconds: int = 28800,
) -> int:
    """
    Path and/or host-header rules on the shared Express HTTPS listener → Pi TG.
    Returns the next free listener rule priority after Pi rules.
    """
    mode = normalize_pi_alb_routing_mode(routing_mode)
    listener_arn = express_ingress_listener_arn(express_main_service)
    pi_target_group_arn = express_ingress_first_target_group_arn(express_pi_service)
    rule_actions = build_cognito_default_listener_actions(
        user_pool_arn=user_pool_arn,
        user_pool_client_id=user_pool_client_id,
        user_pool_domain_prefix=user_pool_domain_prefix,
        target_group_arn=pi_target_group_arn,
        stickiness_seconds=stickiness_seconds,
    )
    priority = rule_priority

    if mode in ("path", "both"):
        path_patterns = pi_alb_path_patterns(path_prefix)
        _express_pi_listener_rule_custom_resource(
            scope,
            f"{logical_id_prefix}ExpressAgenticPathRule",
            listener_arn=listener_arn,
            priority=priority,
            conditions=[
                {
                    "Field": "path-pattern",
                    "PathPatternConfig": {"Values": path_patterns},
                }
            ],
            rule_actions=rule_actions,
            express_main_service=express_main_service,
            express_pi_service=express_pi_service,
        )
        priority += 1

    if mode in ("host", "both") and pi_host_header.strip():
        _express_pi_listener_rule_custom_resource(
            scope,
            f"{logical_id_prefix}ExpressAgenticHostRule",
            listener_arn=listener_arn,
            priority=priority,
            conditions=[
                {
                    "Field": "host-header",
                    "HostHeaderConfig": {"Values": [pi_host_header.strip()]},
                }
            ],
            rule_actions=rule_actions,
            express_main_service=express_main_service,
            express_pi_service=express_pi_service,
        )
        priority += 1

    return priority


def _express_service_connect_configuration(
    *,
    namespace: str,
    port_name: Optional[str] = None,
    discovery_name: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, Any]:
    """ECS API serviceConnectConfiguration payload for updateService."""
    cfg: Dict[str, Any] = {"enabled": True, "namespace": namespace}
    if port_name and discovery_name and port is not None:
        cfg["services"] = [
            {
                "portName": port_name,
                "discoveryName": discovery_name,
                "clientAliases": [
                    {"port": int(port), "dnsName": discovery_name},
                ],
            }
        ]
    return cfg


def apply_service_connect_to_express_service(
    scope: Construct,
    logical_id: str,
    *,
    cluster_name: str,
    service_name: str,
    namespace: str,
    express_service: ecs.CfnExpressGatewayService,
    port_name: Optional[str] = None,
    discovery_name: Optional[str] = None,
    port: Optional[int] = None,
) -> cr.AwsCustomResource:
    """
    Enable Service Connect on an Express gateway service after create (AWS does not
    support SC at Express create time). Server config when port_name/discovery_name/port
    are set; client-only when they are omitted.
    """
    sc_cfg = _express_service_connect_configuration(
        namespace=namespace,
        port_name=port_name,
        discovery_name=discovery_name,
        port=port,
    )
    physical_id = f"{cluster_name}/{service_name}/service-connect"
    custom = cr.AwsCustomResource(
        scope,
        logical_id,
        on_create=cr.AwsSdkCall(
            service="ECS",
            action="updateService",
            parameters={
                "cluster": cluster_name,
                "service": service_name,
                "serviceConnectConfiguration": sc_cfg,
                "forceNewDeployment": True,
            },
            physical_resource_id=cr.PhysicalResourceId.of(physical_id),
        ),
        on_update=cr.AwsSdkCall(
            service="ECS",
            action="updateService",
            parameters={
                "cluster": cluster_name,
                "service": service_name,
                "serviceConnectConfiguration": sc_cfg,
                "forceNewDeployment": True,
            },
            physical_resource_id=cr.PhysicalResourceId.of(physical_id),
        ),
        policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
            resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
        ),
    )
    custom.node.add_dependency(express_service)
    return custom


def create_s3_batch_ecs_trigger_lambda(
    scope: Construct,
    logical_id: str,
    *,
    function_name: Optional[str],
    lambda_asset_path: str,
    output_bucket: s3.IBucket,
    config_bucket: s3.IBucket,
    cluster_name: str,
    task_definition_arn: str,
    container_name: str,
    subnet_ids: List[str],
    security_group_id: str,
    execution_role: iam.IRole,
    task_role: iam.IRole,
    env_prefix: str,
    env_suffix: str,
    input_prefix: str,
    config_prefix: str,
    default_params_key: str,
    default_direct_mode_task: str = "redact",
    assign_public_ip: bool = False,
) -> lambda_.Function:
    """
    Lambda triggered by job .env uploads on the output bucket; runs one-shot Fargate tasks.
    """
    lambda_role = iam.Role(
        scope,
        f"{logical_id}Role",
        assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        managed_policies=[
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        ],
    )

    lambda_role.add_to_policy(
        iam.PolicyStatement(
            actions=["ecs:RunTask"],
            resources=[task_definition_arn],
        )
    )
    lambda_role.add_to_policy(
        iam.PolicyStatement(
            actions=["ecs:RunTask"],
            resources=[
                f"arn:aws:ecs:*:*:cluster/{cluster_name}",
            ],
        )
    )
    lambda_role.add_to_policy(
        iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[execution_role.role_arn, task_role.role_arn],
            conditions={
                "StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}
            },
        )
    )
    output_bucket.grant_read(lambda_role, f"{env_prefix}*")
    config_bucket.grant_read(lambda_role)
    if default_params_key:
        output_bucket.grant_read(lambda_role, default_params_key)

    fn_kwargs: Dict[str, Any] = {
        "runtime": lambda_.Runtime.PYTHON_3_12,
        "handler": "lambda_function.lambda_handler",
        "code": lambda_.Code.from_asset(lambda_asset_path),
        "role": lambda_role,
        "timeout": Duration.seconds(60),
        "memory_size": 256,
        "environment": {
            "OUTPUT_BUCKET": output_bucket.bucket_name,
            "CONFIG_BUCKET": config_bucket.bucket_name,
            "INPUT_PREFIX": input_prefix,
            "CONFIG_PREFIX": config_prefix,
            "ENV_PREFIX": env_prefix,
            "ENV_SUFFIX": env_suffix,
            "DEFAULT_PARAMS_KEY": default_params_key,
            "ECS_CLUSTER": cluster_name,
            "ECS_TASK_DEF": task_definition_arn,
            "SUBNETS": ",".join(subnet_ids),
            "SECURITY_GROUPS": security_group_id,
            "CONTAINER_NAME": container_name,
            "DEFAULT_DIRECT_MODE_TASK": default_direct_mode_task,
            "ECS_ASSIGN_PUBLIC_IP": "true" if assign_public_ip else "false",
            "APP_CONFIG_ENV_KEY": APP_CONFIG_ENV_BASENAME,
        },
    }
    if function_name:
        fn_kwargs["function_name"] = function_name

    batch_fn = lambda_.Function(scope, logical_id, **fn_kwargs)

    output_bucket.add_event_notification(
        s3.EventType.OBJECT_CREATED,
        s3n.LambdaDestination(batch_fn),
        s3.NotificationKeyFilter(prefix=env_prefix, suffix=env_suffix),
    )

    return batch_fn


def create_headless_s3_batch_seed(
    scope: Construct,
    logical_id: str,
    *,
    destination_bucket: s3.IBucket,
    seed_asset_directory: str,
) -> None:
    """Upload input/ and input/config/ markers plus example job .env to the output bucket."""
    from aws_cdk import aws_s3_deployment as s3deploy

    s3deploy.BucketDeployment(
        scope,
        logical_id,
        sources=[s3deploy.Source.asset(seed_asset_directory)],
        destination_bucket=destination_bucket,
        prune=False,
    )


def build_pi_agent_container_environment(
    *,
    service_connect_discovery_name: str,
    main_app_port: Union[str, int],
    pi_gradio_port: Union[str, int],
    pi_root_path: str = "",
) -> Dict[str, str]:
    """Inline env for Pi agent tasks (overrides image defaults; SC URL for main app)."""
    port = int(main_app_port)
    pi_port = int(pi_gradio_port)
    env = {
        "APP_TYPE": "pi",
        "APP_CONFIG_PATH": "/workspace/doc_redaction/config/pi_agent.env",
        "PI_DEPLOYMENT_PROFILE": "aws-ecs",
        "PI_DEFAULT_PROVIDER": "amazon-bedrock",
        "DOC_REDACTION_GRADIO_URL": f"http://{service_connect_discovery_name}:{port}",
        "PI_GRADIO_PORT": str(pi_port),
        "GRADIO_SERVER_PORT": str(pi_port),
        "GRADIO_SERVER_NAME": "0.0.0.0",
        "PI_WORKSPACE_DIR": "/home/user/app/workspace",
        "PI_WORKDIR": "/workspace/doc_redaction",
        "PI_UPLOAD_ROOT": "/tmp/gradio",
        "PI_SESSION_DIR": "/tmp/pi-sessions",
        "PI_CODING_AGENT_DIR": "/tmp/pi-agent",
        "ACCESS_LOGS_FOLDER": "/tmp/pi-logs/",
        "USAGE_LOGS_FOLDER": "/tmp/pi-usage/",
        "FEEDBACK_LOGS_FOLDER": "/tmp/pi-feedback/",
        "RUN_FASTAPI": "True",
        "RUN_AWS_FUNCTIONS": "True",
        "SAVE_OUTPUTS_TO_S3": "True",
        "S3_OUTPUTS_BUCKET": S3_OUTPUT_BUCKET_NAME,
        "COGNITO_AUTH": "False",
    }
    _apply_pi_root_path_env(env, pi_root_path)
    return env


# Gradio mounted on FastAPI (tools.gradio_platform.mount_or_launch); matches agent-redact/pi/start.sh.
PI_ECS_APP_START_CMD = (
    "python3 agent-redact/pi/pi_agent_config.py && "
    "exec uvicorn gradio_app:app --app-dir agent-redact/pi "
    "--host 0.0.0.0 --port ${PI_GRADIO_PORT:-7862} "
    '--proxy-headers --forwarded-allow-ips "*"'
)

# Fargate volume mounts are root-owned; chown as root, then run the app as user (see entrypoint-ecs.sh).
PI_ECS_CONTAINER_USER = "root"
PI_ECS_CONTAINER_COMMAND = [
    "/usr/local/bin/entrypoint-ecs.sh",
    PI_ECS_APP_START_CMD,
]
# Inline fallback when the image predates entrypoint-ecs.sh (same behaviour via bash).
PI_ECS_CONTAINER_COMMAND_FALLBACK = [
    "bash",
    "-c",
    "mkdir -p /tmp/pi-agent /tmp/pi-logs /tmp/pi-usage /tmp/pi-feedback "
    "/home/user/app/workspace /tmp/gradio /tmp/pi-sessions && "
    "chown -R user:user /tmp/pi-agent /tmp/pi-logs /tmp/pi-usage /tmp/pi-feedback "
    "/home/user/app/workspace /tmp/gradio /tmp/pi-sessions && "
    "cd /workspace/doc_redaction && "
    f"exec su -s /bin/bash user -c '{PI_ECS_APP_START_CMD}'",
]


def create_pi_agent_ecs_resources(
    scope: Construct,
    logical_id_prefix: str,
    *,
    vpc: ec2.IVpc,
    cluster: ecs.ICluster,
    private_subnets: List[ec2.ISubnet],
    pi_ecr_image_uri: str,
    container_name: str,
    task_role: iam.IRole,
    execution_role: iam.IRole,
    config_bucket: s3.IBucket,
    pi_agent_env_s3_key: str,
    service_name: str,
    task_family: str,
    security_group_name: str,
    log_group_name: str,
    cpu: int,
    memory_mib: int,
    pi_gradio_port: int,
    service_connect_namespace: str,
    service_connect_discovery_name: str,
    main_app_port: int,
    use_fargate_spot: str,
    pi_root_path: str = "",
) -> Tuple[ecs.FargateService, ec2.SecurityGroup, ecs.FargateTaskDefinition]:
    """Second Fargate service for the Pi agent (joins Service Connect namespace as a client)."""
    pi_security_group = ec2.SecurityGroup(
        scope,
        f"{logical_id_prefix}SecurityGroup",
        vpc=vpc,
        security_group_name=security_group_name,
        description="Pi agent ECS tasks",
    )

    pi_log_group = logs.LogGroup(
        scope,
        f"{logical_id_prefix}LogGroup",
        log_group_name=log_group_name,
        retention=logs.RetentionDays.ONE_MONTH,
        removal_policy=managed_resource_removal_policy(),
    )

    pi_volume = ecs.Volume(name="piEphemeralVolume")
    pi_task_definition = ecs.FargateTaskDefinition(
        scope,
        f"{logical_id_prefix}TaskDefinition",
        family=task_family,
        cpu=cpu,
        memory_limit_mib=memory_mib,
        task_role=task_role,
        execution_role=execution_role,
        runtime_platform=ecs.RuntimePlatform(
            cpu_architecture=ecs.CpuArchitecture.X86_64,
            operating_system_family=ecs.OperatingSystemFamily.LINUX,
        ),
        ephemeral_storage_gib=21,
        volumes=[pi_volume],
    )

    env_files: List[ecs.EnvironmentFile] = []
    if pi_agent_env_s3_key:
        env_files.append(
            ecs.EnvironmentFile.from_bucket(config_bucket, pi_agent_env_s3_key)
        )

    pi_container = pi_task_definition.add_container(
        container_name,
        image=ecs.ContainerImage.from_registry(f"{pi_ecr_image_uri}:latest"),
        logging=ecs.LogDriver.aws_logs(
            stream_prefix="ecs-pi",
            log_group=pi_log_group,
        ),
        environment_files=env_files if env_files else None,
        environment=build_pi_agent_container_environment(
            service_connect_discovery_name=service_connect_discovery_name,
            main_app_port=main_app_port,
            pi_gradio_port=pi_gradio_port,
            pi_root_path=pi_root_path,
        ),
        command=PI_ECS_CONTAINER_COMMAND_FALLBACK,
        user=PI_ECS_CONTAINER_USER,
        essential=True,
    )

    pi_container.add_mount_points(
        ecs.MountPoint(
            source_volume=pi_volume.name,
            container_path="/home/user/app/workspace",
            read_only=False,
        ),
        ecs.MountPoint(
            source_volume=pi_volume.name,
            container_path="/tmp/gradio",
            read_only=False,
        ),
        ecs.MountPoint(
            source_volume=pi_volume.name,
            container_path="/tmp/pi-sessions",
            read_only=False,
        ),
    )

    pi_container.add_port_mappings(
        ecs.PortMapping(
            container_port=pi_gradio_port,
            host_port=pi_gradio_port,
            name=f"port-{pi_gradio_port}",
            protocol=ecs.Protocol.TCP,
            app_protocol=ecs.AppProtocol.http,
        )
    )

    pi_service = ecs.FargateService(
        scope,
        f"{logical_id_prefix}Service",
        service_name=service_name,
        cluster=cluster,
        task_definition=pi_task_definition,
        security_groups=[pi_security_group],
        vpc_subnets=ec2.SubnetSelection(subnets=private_subnets),
        platform_version=ecs.FargatePlatformVersion.LATEST,
        capacity_provider_strategies=[
            ecs.CapacityProviderStrategy(
                capacity_provider=use_fargate_spot,
                base=0,
                weight=1,
            )
        ],
        min_healthy_percent=0,
        max_healthy_percent=100,
        desired_count=0,
        availability_zone_rebalancing=ecs_availability_zone_rebalancing(
            ECS_AVAILABILITY_ZONE_REBALANCING
        ),
        service_connect_configuration=ecs.ServiceConnectProps(
            namespace=service_connect_namespace,
        ),
    )

    return pi_service, pi_security_group, pi_task_definition


def attach_pi_agent_to_shared_alb(
    scope: Construct,
    logical_id_prefix: str,
    *,
    vpc: ec2.IVpc,
    alb_security_group: ec2.ISecurityGroup,
    pi_security_group: ec2.SecurityGroup,
    pi_service: ecs.FargateService,
    pi_port: int,
    routing_mode: str,
    path_prefix: str,
    pi_host_header: str,
    listener_rule_priority: int,
    target_group_name: str,
    stickiness_cookie_duration: Duration,
    https_listener: Optional[elb.IApplicationListener],
    http_listener: Optional[elb.IApplicationListener],
    acm_certificate_arn: str,
    enable_cognito_auth: bool,
    cognito_user_pool: Optional[cognito.IUserPool],
    cognito_user_pool_client: Optional[cognito.IUserPoolClient],
    cognito_user_pool_domain: Optional[cognito.IUserPoolDomain],
) -> Tuple[elb.ApplicationTargetGroup, int]:
    """Register Pi on the shared legacy ALB (path and/or host-header listener rules)."""
    pi_security_group.add_ingress_rule(
        peer=alb_security_group,
        connection=ec2.Port.tcp(pi_port),
        description="Shared ALB to Pi agent",
    )

    pi_target_group = elb.ApplicationTargetGroup(
        scope,
        f"{logical_id_prefix}TargetGroup",
        target_group_name=target_group_name,
        port=pi_port,
        protocol=elb.ApplicationProtocol.HTTP,
        targets=[pi_service],
        stickiness_cookie_duration=stickiness_cookie_duration,
        vpc=vpc,
        health_check=elb.HealthCheck(
            path=pi_alb_health_check_path(path_prefix, routing_mode),
            healthy_http_codes="200-399",
        ),
    )

    if (
        enable_cognito_auth
        and acm_certificate_arn
        and cognito_user_pool
        and cognito_user_pool_client
        and cognito_user_pool_domain
        and https_listener
    ):
        forward_action = elb_act.AuthenticateCognitoAction(
            next=elb.ListenerAction.forward(
                [pi_target_group],
                stickiness_duration=stickiness_cookie_duration,
            ),
            user_pool=cognito_user_pool,
            user_pool_client=cognito_user_pool_client,
            user_pool_domain=cognito_user_pool_domain,
            scope="openid profile email",
            on_unauthenticated_request=elb.UnauthenticatedAction.AUTHENTICATE,
            session_timeout=stickiness_cookie_duration,
        )
    else:
        forward_action = elb.ListenerAction.forward(
            [pi_target_group],
            stickiness_duration=stickiness_cookie_duration,
        )

    mode = normalize_pi_alb_routing_mode(routing_mode)
    priority = listener_rule_priority

    def _add_rules(listener: elb.IApplicationListener, id_prefix: str) -> None:
        nonlocal priority
        if mode in ("path", "both"):
            listener.add_action(
                f"{id_prefix}PathRule",
                priority=priority,
                conditions=[
                    elb.ListenerCondition.path_patterns(
                        pi_alb_path_patterns(path_prefix)
                    )
                ],
                action=forward_action,
            )
            priority += 1
        if mode in ("host", "both") and pi_host_header.strip():
            listener.add_action(
                f"{id_prefix}HostRule",
                priority=priority,
                conditions=[
                    elb.ListenerCondition.host_headers([pi_host_header.strip()])
                ],
                action=forward_action,
            )
            priority += 1

    if https_listener:
        _add_rules(https_listener, f"{logical_id_prefix}Https")
    elif http_listener:
        _add_rules(http_listener, f"{logical_id_prefix}Http")

    if (
        http_listener
        and acm_certificate_arn
        and pi_host_header.strip()
        and mode in ("host", "both")
    ):
        redirect_priority = listener_rule_priority
        if mode in ("path", "both"):
            redirect_priority += 1
        http_listener.add_action(
            f"{logical_id_prefix}HttpRedirectRule",
            priority=redirect_priority,
            conditions=[elb.ListenerCondition.host_headers([pi_host_header.strip()])],
            action=elb.ListenerAction.redirect(
                protocol="HTTPS",
                port="443",
                host="#{host}",
                path="/#{path}",
                query="#{query}",
            ),
        )

    return pi_target_group, priority


def ensure_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not."""

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


# Re-export for app.py and other CDK entrypoints (implementation is boto3-only).
