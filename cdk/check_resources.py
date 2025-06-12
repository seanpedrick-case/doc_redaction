import json
import os
from cdk_config import CDK_PREFIX, VPC_NAME, AWS_REGION, PUBLIC_SUBNETS_TO_USE, PRIVATE_SUBNETS_TO_USE, CODEBUILD_ROLE_NAME, ECS_TASK_ROLE_NAME, ECS_TASK_EXECUTION_ROLE_NAME, S3_LOG_CONFIG_BUCKET_NAME, S3_OUTPUT_BUCKET_NAME, ECR_CDK_REPO_NAME, CODEBUILD_PROJECT_NAME, ALB_NAME, COGNITO_USER_POOL_NAME, COGNITO_USER_POOL_CLIENT_NAME, COGNITO_USER_POOL_CLIENT_SECRET_NAME, WEB_ACL_NAME, CONTEXT_FILE, PUBLIC_SUBNET_CIDR_BLOCKS, PRIVATE_SUBNET_CIDR_BLOCKS, PUBLIC_SUBNET_AVAILABILITY_ZONES, PRIVATE_SUBNET_AVAILABILITY_ZONES, CDK_FOLDER, CDK_CONFIG_PATH  # Import necessary config
from cdk_functions import ( # Import your check functions (assuming they use Boto3)
    get_vpc_id_by_name,
    check_subnet_exists_by_name,
    check_for_existing_role,
    check_s3_bucket_exists,
    check_ecr_repo_exists,
    check_codebuild_project_exists,
    check_alb_exists,
    check_for_existing_user_pool,
    check_for_existing_user_pool_client,
    check_for_secret,
    check_cloudfront_distribution_exists,
    check_web_acl_exists,
    _get_existing_subnets_in_vpc,
    validate_subnet_creation_parameters
    # Add other check functions as needed
)

from typing import List, Dict, Any

cdk_folder = CDK_FOLDER #<FULL_PATH_TO_CDK_FOLDER_HERE>

# Full path needed to find config file
os.environ["CDK_CONFIG_PATH"] = cdk_folder + CDK_CONFIG_PATH

# --- Helper to parse environment variables into lists ---
def _get_env_list(env_var_name: str) -> List[str]:
    """Parses a comma-separated environment variable into a list of strings."""
    value = env_var_name[1:-1].strip().replace('\"', '').replace("\'","")
    if not value:
        return []
    # Split by comma and filter out any empty strings that might result from extra commas
    return [s.strip() for s in value.split(',') if s.strip()]


if PUBLIC_SUBNETS_TO_USE and not isinstance(PUBLIC_SUBNETS_TO_USE, list): PUBLIC_SUBNETS_TO_USE = _get_env_list(PUBLIC_SUBNETS_TO_USE)
if PRIVATE_SUBNETS_TO_USE and not isinstance(PRIVATE_SUBNETS_TO_USE, list): PRIVATE_SUBNETS_TO_USE = _get_env_list(PRIVATE_SUBNETS_TO_USE)
if PUBLIC_SUBNET_CIDR_BLOCKS and not isinstance(PUBLIC_SUBNET_CIDR_BLOCKS, list): PUBLIC_SUBNET_CIDR_BLOCKS = _get_env_list(PUBLIC_SUBNET_CIDR_BLOCKS)
if PUBLIC_SUBNET_AVAILABILITY_ZONES and not isinstance(PUBLIC_SUBNET_AVAILABILITY_ZONES, list): PUBLIC_SUBNET_AVAILABILITY_ZONES = _get_env_list(PUBLIC_SUBNET_AVAILABILITY_ZONES)
if PRIVATE_SUBNET_CIDR_BLOCKS and not isinstance(PRIVATE_SUBNET_CIDR_BLOCKS, list): PRIVATE_SUBNET_CIDR_BLOCKS = _get_env_list(PRIVATE_SUBNET_CIDR_BLOCKS)
if PRIVATE_SUBNET_AVAILABILITY_ZONES and not isinstance(PRIVATE_SUBNET_AVAILABILITY_ZONES, list): PRIVATE_SUBNET_AVAILABILITY_ZONES = _get_env_list(PRIVATE_SUBNET_AVAILABILITY_ZONES)

# Check for the existence of elements in your AWS environment to see if it's necessary to create new versions of the same

def check_and_set_context():
    context_data = {}

    # --- Find the VPC ID first ---
    print("VPC_NAME:", VPC_NAME)
    vpc_id, nat_gateways = get_vpc_id_by_name(VPC_NAME)

    # If you expect only one, or one per AZ and you're creating one per AZ in CDK:
    if nat_gateways:
        # For simplicity, let's just check if *any* NAT exists in the VPC
        # A more robust check would match by subnet, AZ, or a specific tag.
        context_data["exists:NatGateway"] = True
        context_data["id:NatGateway"] = nat_gateways[0]['NatGatewayId'] # Store the ID of the first one found
    else:
        context_data["exists:NatGateway"] = False
        context_data["id:NatGateway"] = None

    if not vpc_id:
        # If the VPC doesn't exist, you might not be able to check/create subnets.
        # Decide how to handle this: raise an error, set a flag, etc.
        raise RuntimeError(f"Required VPC '{VPC_NAME}' not found. Cannot proceed with subnet checks.")

    context_data["vpc_id"] = vpc_id # Store VPC ID in context

    # SUBNET CHECKS
    context_data: Dict[str, Any] = {}
    all_proposed_subnets_data: List[Dict[str, str]] = []

    # Flag to indicate if full validation mode (with CIDR/AZs) is active
    full_validation_mode = False

    # Determine if full validation mode is possible/desired
    # It's 'desired' if CIDR/AZs are provided, and their lengths match the name lists.
    public_ready_for_full_validation = (
        len(PUBLIC_SUBNETS_TO_USE) > 0 and
        len(PUBLIC_SUBNET_CIDR_BLOCKS) == len(PUBLIC_SUBNETS_TO_USE) and
        len(PUBLIC_SUBNET_AVAILABILITY_ZONES) == len(PUBLIC_SUBNETS_TO_USE)
    )
    private_ready_for_full_validation = (
        len(PRIVATE_SUBNETS_TO_USE) > 0 and
        len(PRIVATE_SUBNET_CIDR_BLOCKS) == len(PRIVATE_SUBNETS_TO_USE) and
        len(PRIVATE_SUBNET_AVAILABILITY_ZONES) == len(PRIVATE_SUBNETS_TO_USE)
    )

    # Activate full validation if *any* type of subnet (public or private) has its full details provided.
    # You might adjust this logic if you require ALL subnet types to have CIDRs, or NONE.
    if public_ready_for_full_validation or private_ready_for_full_validation:
        full_validation_mode = True

        # If some are ready but others aren't, print a warning or raise an error based on your strictness
        if public_ready_for_full_validation and not private_ready_for_full_validation and PRIVATE_SUBNETS_TO_USE:
            print("Warning: Public subnets have CIDRs/AZs, but private subnets do not. Only public will be fully validated/created with CIDRs.")
        if private_ready_for_full_validation and not public_ready_for_full_validation and PUBLIC_SUBNETS_TO_USE:
            print("Warning: Private subnets have CIDRs/AZs, but public subnets do not. Only private will be fully validated/created with CIDRs.")

        # Prepare data for validate_subnet_creation_parameters for all subnets that have full details
        if public_ready_for_full_validation:
            for i, name in enumerate(PUBLIC_SUBNETS_TO_USE):
                all_proposed_subnets_data.append({
                    'name': name,
                    'cidr': PUBLIC_SUBNET_CIDR_BLOCKS[i],
                    'az': PUBLIC_SUBNET_AVAILABILITY_ZONES[i]
                })
        if private_ready_for_full_validation:
            for i, name in enumerate(PRIVATE_SUBNETS_TO_USE):
                all_proposed_subnets_data.append({
                    'name': name,
                    'cidr': PRIVATE_SUBNET_CIDR_BLOCKS[i],
                    'az': PRIVATE_SUBNET_AVAILABILITY_ZONES[i]
                })


    print(f"Target VPC ID for Boto3 lookup: {vpc_id}")

    # Fetch all existing subnets in the target VPC once to avoid repeated API calls
    try:
        existing_aws_subnets = _get_existing_subnets_in_vpc(vpc_id)
    except Exception as e:
        print(f"Failed to fetch existing VPC subnets. Aborting. Error: {e}")
        raise SystemExit(1) # Exit immediately if we can't get baseline data
    
    print("\n--- Running Name-Only Subnet Existence Check Mode ---")
    # Fallback: check only by name using the existing data
    checked_public_subnets = {}
    if PUBLIC_SUBNETS_TO_USE:
        for subnet_name in PUBLIC_SUBNETS_TO_USE:
            print("subnet_name:", subnet_name)
            exists, subnet_id = check_subnet_exists_by_name(subnet_name, existing_aws_subnets)
            checked_public_subnets[subnet_name] = {"exists": exists, "id": subnet_id}

            # If the subnet exists, remove it from the proposed subnets list
            if checked_public_subnets[subnet_name]["exists"] == True:
                all_proposed_subnets_data = [
                    subnet for subnet in all_proposed_subnets_data 
                    if subnet['name'] != subnet_name
                ]

    context_data["checked_public_subnets"] = checked_public_subnets

    checked_private_subnets = {}
    if PRIVATE_SUBNETS_TO_USE:
        for subnet_name in PRIVATE_SUBNETS_TO_USE:
            print("subnet_name:", subnet_name)
            exists, subnet_id = check_subnet_exists_by_name(subnet_name, existing_aws_subnets)
            checked_private_subnets[subnet_name] = {"exists": exists, "id": subnet_id}

            # If the subnet exists, remove it from the proposed subnets list
            if checked_private_subnets[subnet_name]["exists"] == True:
                all_proposed_subnets_data = [
                    subnet for subnet in all_proposed_subnets_data 
                    if subnet['name'] != subnet_name
                ]

    context_data["checked_private_subnets"] = checked_private_subnets



    print("\nName-only existence subnet check complete.\n")

    if full_validation_mode:
        print("\n--- Running in Full Subnet Validation Mode (CIDR/AZs provided) ---")
        try:
            validate_subnet_creation_parameters(vpc_id, all_proposed_subnets_data, existing_aws_subnets)
            print("\nPre-synth validation successful. Proceeding with CDK synth.\n")

            # Populate context_data for downstream CDK construct creation
            context_data["public_subnets_to_create"] = []
            if public_ready_for_full_validation:
                for i, name in enumerate(PUBLIC_SUBNETS_TO_USE):
                    context_data["public_subnets_to_create"].append({
                        'name': name,
                        'cidr': PUBLIC_SUBNET_CIDR_BLOCKS[i],
                        'az': PUBLIC_SUBNET_AVAILABILITY_ZONES[i],
                        'is_public': True
                    })
            context_data["private_subnets_to_create"] = []
            if private_ready_for_full_validation:
                for i, name in enumerate(PRIVATE_SUBNETS_TO_USE):
                    context_data["private_subnets_to_create"].append({
                        'name': name,
                        'cidr': PRIVATE_SUBNET_CIDR_BLOCKS[i],
                        'az': PRIVATE_SUBNET_AVAILABILITY_ZONES[i],
                        'is_public': False
                    })

        except (ValueError, Exception) as e:
            print(f"\nFATAL ERROR: Subnet parameter validation failed: {e}\n")
            raise SystemExit(1) # Exit if validation fails

    # Example checks and setting context values
    # IAM Roles
    role_name = CODEBUILD_ROLE_NAME
    exists, _, _ = check_for_existing_role(role_name)
    context_data[f"exists:{role_name}"] = exists # Use boolean
    if exists:
         _, role_arn, _ = check_for_existing_role(role_name) # Get ARN if needed
         context_data[f"arn:{role_name}"] = role_arn

    role_name = ECS_TASK_ROLE_NAME
    exists, _, _ = check_for_existing_role(role_name)
    context_data[f"exists:{role_name}"] = exists
    if exists:
         _, role_arn, _ = check_for_existing_role(role_name)
         context_data[f"arn:{role_name}"] = role_arn

    role_name = ECS_TASK_EXECUTION_ROLE_NAME
    exists, _, _ = check_for_existing_role(role_name)
    context_data[f"exists:{role_name}"] = exists
    if exists:
         _, role_arn, _ = check_for_existing_role(role_name)
         context_data[f"arn:{role_name}"] = role_arn

    # S3 Buckets
    bucket_name = S3_LOG_CONFIG_BUCKET_NAME
    exists, _ = check_s3_bucket_exists(bucket_name)
    context_data[f"exists:{bucket_name}"] = exists
    if exists:
        # You might not need the ARN if using from_bucket_name
        pass

    output_bucket_name = S3_OUTPUT_BUCKET_NAME
    exists, _ = check_s3_bucket_exists(output_bucket_name)
    context_data[f"exists:{output_bucket_name}"] = exists
    if exists:
         pass

    # ECR Repository
    repo_name = ECR_CDK_REPO_NAME
    exists, _ = check_ecr_repo_exists(repo_name)
    context_data[f"exists:{repo_name}"] = exists
    if exists:
         pass # from_repository_name is sufficient

    # CodeBuild Project
    project_name = CODEBUILD_PROJECT_NAME
    exists, _ = check_codebuild_project_exists(project_name)
    context_data[f"exists:{project_name}"] = exists
    if exists:
         # Need a way to get the ARN from the check function
         _, project_arn = check_codebuild_project_exists(project_name) # Assuming it returns ARN
         context_data[f"arn:{project_name}"] = project_arn

    # ALB (by name lookup)
    alb_name = ALB_NAME
    exists, _ = check_alb_exists(alb_name, region_name=AWS_REGION)
    context_data[f"exists:{alb_name}"] = exists
    if exists:
        _, alb_object = check_alb_exists(alb_name, region_name=AWS_REGION) # Assuming check returns object
        print("alb_object:", alb_object)
        context_data[f"arn:{alb_name}"] = alb_object['LoadBalancerArn']


    # Cognito User Pool (by name)
    user_pool_name = COGNITO_USER_POOL_NAME
    exists, user_pool_id, _ = check_for_existing_user_pool(user_pool_name)
    context_data[f"exists:{user_pool_name}"] = exists
    if exists:
        context_data[f"id:{user_pool_name}"] = user_pool_id

    # Cognito User Pool Client (by name and pool ID) - requires User Pool ID from check
    if user_pool_id:
        user_pool_id_for_client_check = user_pool_id #context_data.get(f"id:{user_pool_name}") # Use ID from context
        user_pool_client_name = COGNITO_USER_POOL_CLIENT_NAME
        if user_pool_id_for_client_check:
            exists, client_id, _ = check_for_existing_user_pool_client(user_pool_client_name, user_pool_id_for_client_check)
            context_data[f"exists:{user_pool_client_name}"] = exists
            if exists:
                context_data[f"id:{user_pool_client_name}"] = client_id

    # Secrets Manager Secret (by name)
    secret_name = COGNITO_USER_POOL_CLIENT_SECRET_NAME
    exists, _ = check_for_secret(secret_name)
    context_data[f"exists:{secret_name}"] = exists
    # You might not need the ARN if using from_secret_name_v2


    # WAF Web ACL (by name and scope)
    web_acl_name = WEB_ACL_NAME
    exists, _ = check_web_acl_exists(web_acl_name, scope="CLOUDFRONT") # Assuming check returns object
    context_data[f"exists:{web_acl_name}"] = exists
    if exists:
        _, existing_web_acl = check_web_acl_exists(web_acl_name, scope="CLOUDFRONT")
        context_data[f"arn:{web_acl_name}"] = existing_web_acl.attr_arn

    # Write the context data to the file
    with open(CONTEXT_FILE, "w") as f:
        json.dump(context_data, f, indent=2)

    print(f"Context data written to {CONTEXT_FILE}")

