import json
import os
from typing import Dict, List

from cdk_config import (  # Import necessary config
    ALB_NAME,
    AWS_REGION,
    CDK_CONFIG_PATH,
    CDK_FOLDER,
    CODEBUILD_PROJECT_NAME,
    CODEBUILD_ROLE_NAME,
    COGNITO_USER_POOL_CLIENT_NAME,
    COGNITO_USER_POOL_CLIENT_SECRET_NAME,
    COGNITO_USER_POOL_DOMAIN_PREFIX,
    COGNITO_USER_POOL_NAME,
    CONTEXT_FILE,
    ECR_CDK_REPO_NAME,
    ECR_PI_REPO_NAME,
    ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES_TO_LOOKUP,
    ECS_TASK_EXECUTION_ROLE_NAME,
    ECS_TASK_ROLE_NAME,
    ENABLE_ECS_SERVICE_CONNECT,
    ENABLE_HEADLESS_DEPLOYMENT,
    ENABLE_PI_AGENT_ECS_SERVICE,
    ENABLE_S3_BATCH_ECS_TRIGGER,
    EXISTING_IGW_ID,
    PRIVATE_SUBNET_AVAILABILITY_ZONES,
    PRIVATE_SUBNET_CIDR_BLOCKS,
    PRIVATE_SUBNETS_TO_USE,
    PUBLIC_SUBNET_AVAILABILITY_ZONES,
    PUBLIC_SUBNET_CIDR_BLOCKS,
    PUBLIC_SUBNETS_TO_USE,
    S3_LOG_CONFIG_BUCKET_NAME,
    S3_OUTPUT_BUCKET_NAME,
    USE_ECS_EXPRESS_MODE,
    VPC_NAME,
    WEB_ACL_NAME,
)
from cdk_functions import (  # Import your check functions (assuming they use Boto3)
    _get_existing_subnets_in_vpc,
    audit_public_subnet_internet_connectivity,
    check_alb_exists,
    check_codebuild_project_exists,
    check_ecr_repo_exists,
    check_for_existing_role,
    check_for_existing_user_pool,
    check_for_existing_user_pool_client,
    check_for_secret,
    check_subnet_exists_by_name,
    check_web_acl_exists,
    get_secret_kms_key_arn,
    get_security_group_id_by_name,
    get_vpc_id_by_name,
    list_existing_vpc_endpoint_service_names,
    resolve_cognito_domain_prefix_availability,
    resolve_s3_bucket_availability,
    validate_subnet_creation_parameters,
    # Add other check functions as needed
)

cdk_folder = CDK_FOLDER  # <FULL_PATH_TO_CDK_FOLDER_HERE>

# Full path needed to find config file
os.environ["CDK_CONFIG_PATH"] = cdk_folder + CDK_CONFIG_PATH


# --- Helper to parse environment variables into lists ---
def _get_env_list(env_var_name: str) -> List[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    value = env_var_name[1:-1].strip().replace('"', "").replace("'", "")
    if not value:
        return []
    # Split by comma and filter out any empty strings that might result from extra commas
    return [s.strip() for s in value.split(",") if s.strip()]


if PUBLIC_SUBNETS_TO_USE and not isinstance(PUBLIC_SUBNETS_TO_USE, list):
    PUBLIC_SUBNETS_TO_USE = _get_env_list(PUBLIC_SUBNETS_TO_USE)
if PRIVATE_SUBNETS_TO_USE and not isinstance(PRIVATE_SUBNETS_TO_USE, list):
    PRIVATE_SUBNETS_TO_USE = _get_env_list(PRIVATE_SUBNETS_TO_USE)
if PUBLIC_SUBNET_CIDR_BLOCKS and not isinstance(PUBLIC_SUBNET_CIDR_BLOCKS, list):
    PUBLIC_SUBNET_CIDR_BLOCKS = _get_env_list(PUBLIC_SUBNET_CIDR_BLOCKS)
if PUBLIC_SUBNET_AVAILABILITY_ZONES and not isinstance(
    PUBLIC_SUBNET_AVAILABILITY_ZONES, list
):
    PUBLIC_SUBNET_AVAILABILITY_ZONES = _get_env_list(PUBLIC_SUBNET_AVAILABILITY_ZONES)
if PRIVATE_SUBNET_CIDR_BLOCKS and not isinstance(PRIVATE_SUBNET_CIDR_BLOCKS, list):
    PRIVATE_SUBNET_CIDR_BLOCKS = _get_env_list(PRIVATE_SUBNET_CIDR_BLOCKS)
if PRIVATE_SUBNET_AVAILABILITY_ZONES and not isinstance(
    PRIVATE_SUBNET_AVAILABILITY_ZONES, list
):
    PRIVATE_SUBNET_AVAILABILITY_ZONES = _get_env_list(PRIVATE_SUBNET_AVAILABILITY_ZONES)

# Check for the existence of elements in your AWS environment to see if it's necessary to create new versions of the same


def check_and_set_context():
    context_data = {}

    # --- Find the VPC ID first ---
    if VPC_NAME:
        print("VPC_NAME:", VPC_NAME)
        vpc_lookup = get_vpc_id_by_name(VPC_NAME)
        if not vpc_lookup:
            raise RuntimeError(
                f"Required VPC '{VPC_NAME}' not found. Cannot proceed with subnet checks."
            )
        vpc_id, nat_gateways, vpc_cidr_block, vpc_cidr_blocks = vpc_lookup

        # If you expect only one, or one per AZ and you're creating one per AZ in CDK:
        if nat_gateways:
            # For simplicity, let's just check if *any* NAT exists in the VPC
            # A more robust check would match by subnet, AZ, or a specific tag.
            context_data["exists:NatGateway"] = True
            context_data["id:NatGateway"] = nat_gateways[0][
                "NatGatewayId"
            ]  # Store the ID of the first one found
        else:
            context_data["exists:NatGateway"] = False
            context_data["id:NatGateway"] = None

        context_data["vpc_id"] = vpc_id  # Store VPC ID in context
        if vpc_cidr_block:
            context_data["vpc_cidr_block"] = vpc_cidr_block
        if vpc_cidr_blocks:
            context_data["vpc_cidr_blocks"] = vpc_cidr_blocks

        existing_endpoint_services = sorted(
            list_existing_vpc_endpoint_service_names(vpc_id, region_name=AWS_REGION)
        )
        if existing_endpoint_services:
            context_data["existing_vpc_endpoint_service_names"] = (
                existing_endpoint_services
            )
            print(
                "Existing VPC endpoints in target VPC (will be skipped on deploy): "
                + ", ".join(existing_endpoint_services)
            )

        # SUBNET CHECKS
        all_proposed_subnets_data: List[Dict[str, str]] = []

        # Flag to indicate if full validation mode (with CIDR/AZs) is active
        full_validation_mode = False

        # Determine if full validation mode is possible/desired
        # It's 'desired' if CIDR/AZs are provided, and their lengths match the name lists.
        public_ready_for_full_validation = (
            len(PUBLIC_SUBNETS_TO_USE) > 0
            and len(PUBLIC_SUBNET_CIDR_BLOCKS) == len(PUBLIC_SUBNETS_TO_USE)
            and len(PUBLIC_SUBNET_AVAILABILITY_ZONES) == len(PUBLIC_SUBNETS_TO_USE)
        )
        private_ready_for_full_validation = (
            len(PRIVATE_SUBNETS_TO_USE) > 0
            and len(PRIVATE_SUBNET_CIDR_BLOCKS) == len(PRIVATE_SUBNETS_TO_USE)
            and len(PRIVATE_SUBNET_AVAILABILITY_ZONES) == len(PRIVATE_SUBNETS_TO_USE)
        )

        # Activate full validation if *any* type of subnet (public or private) has its full details provided.
        # You might adjust this logic if you require ALL subnet types to have CIDRs, or NONE.
        if public_ready_for_full_validation or private_ready_for_full_validation:
            full_validation_mode = True

            # If some are ready but others aren't, print a warning or raise an error based on your strictness
            if (
                public_ready_for_full_validation
                and not private_ready_for_full_validation
                and PRIVATE_SUBNETS_TO_USE
            ):
                print(
                    "Warning: Public subnets have CIDRs/AZs, but private subnets do not. Only public will be fully validated/created with CIDRs."
                )
            if (
                private_ready_for_full_validation
                and not public_ready_for_full_validation
                and PUBLIC_SUBNETS_TO_USE
            ):
                print(
                    "Warning: Private subnets have CIDRs/AZs, but public subnets do not. Only private will be fully validated/created with CIDRs."
                )

            # Prepare data for validate_subnet_creation_parameters for all subnets that have full details
            if public_ready_for_full_validation:
                for i, name in enumerate(PUBLIC_SUBNETS_TO_USE):
                    all_proposed_subnets_data.append(
                        {
                            "name": name,
                            "cidr": PUBLIC_SUBNET_CIDR_BLOCKS[i],
                            "az": PUBLIC_SUBNET_AVAILABILITY_ZONES[i],
                        }
                    )
            if private_ready_for_full_validation:
                for i, name in enumerate(PRIVATE_SUBNETS_TO_USE):
                    all_proposed_subnets_data.append(
                        {
                            "name": name,
                            "cidr": PRIVATE_SUBNET_CIDR_BLOCKS[i],
                            "az": PRIVATE_SUBNET_AVAILABILITY_ZONES[i],
                        }
                    )

        print(f"Target VPC ID for Boto3 lookup: {vpc_id}")

        # Fetch all existing subnets in the target VPC once to avoid repeated API calls
        try:
            existing_aws_subnets = _get_existing_subnets_in_vpc(vpc_id)
        except Exception as e:
            print(f"Failed to fetch existing VPC subnets. Aborting. Error: {e}")
            raise SystemExit(1)  # Exit immediately if we can't get baseline data

        print("\n--- Running Name-Only Subnet Existence Check Mode ---")
        # Fallback: check only by name using the existing data
        checked_public_subnets = {}
        if PUBLIC_SUBNETS_TO_USE:
            for subnet_name in PUBLIC_SUBNETS_TO_USE:
                print("subnet_name:", subnet_name)
                exists, subnet_id = check_subnet_exists_by_name(
                    subnet_name, existing_aws_subnets
                )
                checked_public_subnets[subnet_name] = {
                    "exists": exists,
                    "id": subnet_id,
                    "az": (
                        existing_aws_subnets["by_name"].get(subnet_name, {}).get("az")
                        if exists
                        else None
                    ),
                    "route_table_id": (
                        existing_aws_subnets["by_name"]
                        .get(subnet_name, {})
                        .get("route_table_id")
                        if exists
                        else None
                    ),
                }

                # If the subnet exists, remove it from the proposed subnets list
                if checked_public_subnets[subnet_name]["exists"] is True:
                    all_proposed_subnets_data = [
                        subnet
                        for subnet in all_proposed_subnets_data
                        if subnet["name"] != subnet_name
                    ]

        context_data["checked_public_subnets"] = checked_public_subnets

        checked_private_subnets = {}
        if PRIVATE_SUBNETS_TO_USE:
            for subnet_name in PRIVATE_SUBNETS_TO_USE:
                print("subnet_name:", subnet_name)
                exists, subnet_id = check_subnet_exists_by_name(
                    subnet_name, existing_aws_subnets
                )
                checked_private_subnets[subnet_name] = {
                    "exists": exists,
                    "id": subnet_id,
                    "az": (
                        existing_aws_subnets["by_name"].get(subnet_name, {}).get("az")
                        if exists
                        else None
                    ),
                    "route_table_id": (
                        existing_aws_subnets["by_name"]
                        .get(subnet_name, {})
                        .get("route_table_id")
                        if exists
                        else None
                    ),
                }

                # If the subnet exists, remove it from the proposed subnets list
                if checked_private_subnets[subnet_name]["exists"] is True:
                    all_proposed_subnets_data = [
                        subnet
                        for subnet in all_proposed_subnets_data
                        if subnet["name"] != subnet_name
                    ]

        context_data["checked_private_subnets"] = checked_private_subnets

        # Internet Gateway + public subnet default routes (legacy ALB / NAT public subnets)
        if PUBLIC_SUBNETS_TO_USE and vpc_id:
            public_entries_for_igw_audit = []
            for subnet_name in PUBLIC_SUBNETS_TO_USE:
                info = checked_public_subnets.get(subnet_name, {})
                if not info.get("exists"):
                    continue
                public_entries_for_igw_audit.append(
                    {
                        "name": subnet_name,
                        "subnet_id": info.get("id"),
                        "route_table_id": info.get("route_table_id"),
                    }
                )
            if public_entries_for_igw_audit or EXISTING_IGW_ID:
                print("\n--- Auditing Internet Gateway and public subnet routes ---")
                try:
                    igw_audit = audit_public_subnet_internet_connectivity(
                        vpc_id,
                        EXISTING_IGW_ID,
                        public_entries_for_igw_audit,
                    )
                    context_data["internet_gateway_id"] = igw_audit[
                        "internet_gateway_id"
                    ]
                    context_data["internet_gateway_needs_vpc_attachment"] = igw_audit[
                        "internet_gateway_needs_vpc_attachment"
                    ]
                    context_data["public_subnets_needing_igw_route"] = igw_audit[
                        "public_subnets_needing_igw_route"
                    ]
                    needing = igw_audit["public_subnets_needing_igw_route"]
                    if igw_audit["internet_gateway_needs_vpc_attachment"]:
                        print(
                            f"CDK will attach IGW '{igw_audit['internet_gateway_id']}' "
                            f"to VPC '{vpc_id}' on deploy."
                        )
                    if needing:
                        print(
                            f"CDK will add default internet routes on deploy for: "
                            f"{', '.join(n['name'] for n in needing)}"
                        )
                    else:
                        print(
                            "All audited public subnets already have 0.0.0.0/0 -> IGW routes."
                        )
                except ValueError as e:
                    print(
                        f"\nFATAL ERROR: Internet Gateway / public route audit failed: {e}\n"
                    )
                    raise SystemExit(1) from e

        print("\nName-only existence subnet check complete.\n")

        if full_validation_mode:
            print(
                "\n--- Running in Full Subnet Validation Mode (CIDR/AZs provided) ---"
            )
            try:
                validate_subnet_creation_parameters(
                    vpc_id, all_proposed_subnets_data, existing_aws_subnets
                )
                print("\nPre-synth validation successful. Proceeding with CDK synth.\n")

                # Populate context_data for downstream CDK construct creation.
                # Skip subnets that already exist in AWS (imported in the stack).
                context_data["public_subnets_to_create"] = []
                if public_ready_for_full_validation:
                    for i, name in enumerate(PUBLIC_SUBNETS_TO_USE):
                        if checked_public_subnets.get(name, {}).get("exists"):
                            continue
                        context_data["public_subnets_to_create"].append(
                            {
                                "name": name,
                                "cidr": PUBLIC_SUBNET_CIDR_BLOCKS[i],
                                "az": PUBLIC_SUBNET_AVAILABILITY_ZONES[i],
                                "is_public": True,
                            }
                        )
                context_data["private_subnets_to_create"] = []
                if private_ready_for_full_validation:
                    for i, name in enumerate(PRIVATE_SUBNETS_TO_USE):
                        if checked_private_subnets.get(name, {}).get("exists"):
                            continue
                        context_data["private_subnets_to_create"].append(
                            {
                                "name": name,
                                "cidr": PRIVATE_SUBNET_CIDR_BLOCKS[i],
                                "az": PRIVATE_SUBNET_AVAILABILITY_ZONES[i],
                                "is_public": False,
                            }
                        )

            except (ValueError, Exception) as e:
                print(f"\nFATAL ERROR: Subnet parameter validation failed: {e}\n")
                raise SystemExit(1)  # Exit if validation fails

    # Example checks and setting context values
    # IAM Roles
    role_name = CODEBUILD_ROLE_NAME
    exists, role_arn, _ = check_for_existing_role(role_name)
    context_data[f"exists:{role_name}"] = exists
    if exists:
        context_data[f"arn:{role_name}"] = role_arn

    role_name = ECS_TASK_ROLE_NAME
    exists, role_arn, _ = check_for_existing_role(role_name)
    context_data[f"exists:{role_name}"] = exists
    if exists:
        context_data[f"arn:{role_name}"] = role_arn

    role_name = ECS_TASK_EXECUTION_ROLE_NAME
    exists, role_arn, _ = check_for_existing_role(role_name)
    context_data[f"exists:{role_name}"] = exists
    if exists:
        context_data[f"arn:{role_name}"] = role_arn

    # S3 Buckets
    def _record_s3_bucket_context(name: str) -> None:
        status, _ = resolve_s3_bucket_availability(name)
        context_data[f"exists:{name}"] = status == "owned"
        context_data[f"globally_taken:{name}"] = status == "globally_taken"

    bucket_name = S3_LOG_CONFIG_BUCKET_NAME
    _record_s3_bucket_context(bucket_name)

    output_bucket_name = S3_OUTPUT_BUCKET_NAME
    _record_s3_bucket_context(output_bucket_name)

    # ECR Repositories
    for repo_name in (ECR_CDK_REPO_NAME, ECR_PI_REPO_NAME):
        exists, _ = check_ecr_repo_exists(repo_name)
        context_data[f"exists:{repo_name}"] = exists

    # CodeBuild Project
    project_name = CODEBUILD_PROJECT_NAME
    exists, project_arn, service_role_arn = check_codebuild_project_exists(project_name)
    context_data[f"exists:{project_name}"] = exists
    if exists:
        context_data[f"arn:{project_name}"] = project_arn
        if service_role_arn:
            context_data[f"service_role_arn:{project_name}"] = service_role_arn

    # ALB (by name lookup) — skipped when Express Mode will provision its own ALB
    alb_name = ALB_NAME[-32:] if len(ALB_NAME) > 32 else ALB_NAME
    if USE_ECS_EXPRESS_MODE == "True":
        context_data[f"exists:{alb_name}"] = False
        print(
            "USE_ECS_EXPRESS_MODE=True: skipping ALB pre-check (Express provisions ALB)."
        )
    elif ENABLE_HEADLESS_DEPLOYMENT == "True":
        context_data[f"exists:{alb_name}"] = False
        print(
            "ENABLE_HEADLESS_DEPLOYMENT=True: skipping ALB pre-check (no web ingress)."
        )
    if ENABLE_HEADLESS_DEPLOYMENT == "True":
        print(
            "ENABLE_HEADLESS_DEPLOYMENT=True: requires ENABLE_S3_BATCH_ECS_TRIGGER=True "
            "and USE_ECS_EXPRESS_MODE=False."
        )
    elif ENABLE_S3_BATCH_ECS_TRIGGER == "True":
        print(
            "ENABLE_S3_BATCH_ECS_TRIGGER=True: requires legacy Fargate (USE_ECS_EXPRESS_MODE=False)."
        )
    if ENABLE_PI_AGENT_ECS_SERVICE == "True":
        print(
            "ENABLE_PI_AGENT_ECS_SERVICE=True: requires legacy Fargate, Service Connect, "
            "and PI_ALB_ROUTING (default path=/pi on shared ALB; host mode needs PI_ALB_HOST_HEADER)."
        )
    elif ENABLE_HEADLESS_DEPLOYMENT == "True":
        print(
            "ENABLE_HEADLESS_DEPLOYMENT=True: legacy Fargate task definition + "
            "S3 batch Lambda only (no ALB pre-check)."
        )
    else:
        exists, alb_object = check_alb_exists(alb_name, region_name=AWS_REGION)
        context_data[f"exists:{alb_name}"] = exists
        if exists:
            print("alb_object:", alb_object)
            context_data[f"arn:{alb_name}"] = alb_object["LoadBalancerArn"]
            context_data[f"dns:{alb_name}"] = alb_object["DNSName"]
            context_data[f"canonical_hosted_zone_id:{alb_name}"] = alb_object[
                "CanonicalHostedZoneId"
            ]
            if alb_object.get("SecurityGroups"):
                context_data[f"security_group_id:{alb_name}"] = alb_object[
                    "SecurityGroups"
                ][0]

    # Cognito (web login only; headless batch mode has no Gradio/ALB ingress)
    if ENABLE_HEADLESS_DEPLOYMENT != "True":
        domain_prefix = (COGNITO_USER_POOL_DOMAIN_PREFIX or "").strip().lower()
        if domain_prefix:
            availability = resolve_cognito_domain_prefix_availability(
                domain_prefix, region_name=AWS_REGION
            )
            context_data[f"cognito_domain_taken:{domain_prefix}"] = (
                availability == "taken"
            )

        user_pool_name = COGNITO_USER_POOL_NAME
        exists, user_pool_id, _ = check_for_existing_user_pool(user_pool_name)
        context_data[f"exists:{user_pool_name}"] = exists
        if exists:
            context_data[f"id:{user_pool_name}"] = user_pool_id

        # Cognito User Pool Client (by name and pool ID) - requires User Pool ID from check
        if user_pool_id:
            user_pool_id_for_client_check = user_pool_id  # context_data.get(f"id:{user_pool_name}") # Use ID from context
            user_pool_client_name = COGNITO_USER_POOL_CLIENT_NAME
            if user_pool_id_for_client_check:
                exists, client_id, _ = check_for_existing_user_pool_client(
                    user_pool_client_name, user_pool_id_for_client_check
                )
                context_data[f"exists:{user_pool_client_name}"] = exists
                if exists:
                    context_data[f"id:{user_pool_client_name}"] = client_id
                else:
                    print(
                        f"User pool '{user_pool_name}' exists but app client "
                        f"'{user_pool_client_name}' does not; CDK will create a new client."
                    )

        # Secrets Manager Secret (by name)
        secret_name = COGNITO_USER_POOL_CLIENT_SECRET_NAME
        exists, secret_response = check_for_secret(secret_name)
        context_data[f"exists:{secret_name}"] = exists
        if exists:
            secret_arn = (
                secret_response.get("ARN")
                if isinstance(secret_response, dict)
                else None
            )
            if secret_arn:
                context_data[f"arn:{secret_name}"] = secret_arn
                print("Cognito secret ARN recorded for IAM grants.")
                secret_kms_key_arn = get_secret_kms_key_arn(
                    secret_name, region_name=AWS_REGION
                )
                if secret_kms_key_arn:
                    context_data[f"kms_key_arn:{secret_name}"] = secret_kms_key_arn
                    print("Cognito secret KMS key recorded for execution role decrypt.")
            else:
                print(
                    "Warning: Cognito secret exists but ARN was not returned; "
                    "CDK will use a name-based ARN wildcard in IAM policies."
                )
    else:
        print(
            "ENABLE_HEADLESS_DEPLOYMENT=True: skipping Cognito user pool and secret pre-checks."
        )

    # Service Connect client security groups (by name in VPC)
    if ENABLE_ECS_SERVICE_CONNECT == "True":
        vpc_id_for_sg = context_data.get("vpc_id")
        if vpc_id_for_sg:
            for sg_name in ECS_SERVICE_CONNECT_CLIENT_SECURITY_GROUP_NAMES_TO_LOOKUP:
                exists, sg_id = get_security_group_id_by_name(
                    sg_name, vpc_id_for_sg, region_name=AWS_REGION
                )
                context_data[f"exists:sg:{sg_name}"] = exists
                if exists:
                    context_data[f"security_group_id:{sg_name}"] = sg_id
                    print(f"Service Connect client SG '{sg_name}' -> {sg_id}")
                else:
                    print(
                        f"Warning: Service Connect client SG '{sg_name}' "
                        f"not found in VPC {vpc_id_for_sg}"
                    )
        else:
            print(
                "Warning: vpc_id missing from context; cannot resolve Service "
                "Connect client security group names."
            )

    # WAF Web ACL (by name and scope)
    web_acl_name = WEB_ACL_NAME
    exists, existing_web_acl = check_web_acl_exists(web_acl_name, scope="CLOUDFRONT")
    context_data[f"exists:{web_acl_name}"] = exists
    if exists:
        context_data[f"arn:{web_acl_name}"] = existing_web_acl["ARN"]

    # Write the context data to the file
    with open(CONTEXT_FILE, "w") as f:
        json.dump(context_data, f, indent=2)

    print(f"Context data written to {CONTEXT_FILE}")


if __name__ == "__main__":
    print(f"Pre-check context file: {CONTEXT_FILE}")
    print(
        "Running AWS pre-check (requires credentials for the target account/region)..."
    )
    try:
        check_and_set_context()
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"Pre-check failed: {exc}") from exc
    if not os.path.exists(CONTEXT_FILE):
        raise SystemExit(f"Pre-check finished but {CONTEXT_FILE} was not created.")
    print(f"Pre-check complete. Context written to {os.path.abspath(CONTEXT_FILE)}")
