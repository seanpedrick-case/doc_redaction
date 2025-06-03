import json
from cdk_config import CDK_PREFIX, VPC_NAME, ECR_REPO_NAME, AWS_REGION, SECRETS_MANAGER_ID, PUBLIC_SUBNETS_TO_USE, PRIVATE_SUBNETS_TO_USE, CODEBUILD_ROLE_NAME, ECS_TASK_ROLE_NAME, ECS_TASK_EXECUTION_ROLE_NAME, S3_LOG_CONFIG_BUCKET_NAME, S3_OUTPUT_BUCKET_NAME, ECR_CDK_REPO_NAME, CODEBUILD_PROJECT_NAME, ALB_NAME, COGNITO_USER_POOL_NAME, COGNITO_USER_POOL_CLIENT_NAME, COGNITO_USER_POOL_CLIENT_SECRET_NAME, WEB_ACL_NAME, CONTEXT_FILE  # Import necessary config
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
    # Add other check functions as needed
)

if PUBLIC_SUBNETS_TO_USE: PUBLIC_SUBNETS_TO_USE = eval(PUBLIC_SUBNETS_TO_USE)
if PRIVATE_SUBNETS_TO_USE: PRIVATE_SUBNETS_TO_USE = eval(PRIVATE_SUBNETS_TO_USE)

# Check for the existence of elements in your AWS environment to see if it's necessary to create new versions of the same

def check_and_set_context():
    prefix = CDK_PREFIX
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

    # --- Check for Subnets ---
    checked_public_subnets = {}
    if PUBLIC_SUBNETS_TO_USE:
        for subnet_name in PUBLIC_SUBNETS_TO_USE:
            exists, subnet_id = check_subnet_exists_by_name(vpc_id, subnet_name)
            checked_public_subnets[subnet_name] = {"exists": exists, "id": subnet_id}
    context_data["checked_public_subnets"] = checked_public_subnets

    checked_private_subnets = {}
    if PRIVATE_SUBNETS_TO_USE:
        for subnet_name in PRIVATE_SUBNETS_TO_USE:
            exists, subnet_id = check_subnet_exists_by_name(vpc_id, subnet_name)
            checked_private_subnets[subnet_name] = {"exists": exists, "id": subnet_id}
    context_data["checked_private_subnets"] = checked_private_subnets

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
        context_data[f"arn:{alb_name}"] = alb_object.load_balancer_arn


    # Cognito User Pool (by name)
    user_pool_name = COGNITO_USER_POOL_NAME
    exists, user_pool_id, _ = check_for_existing_user_pool(user_pool_name)
    context_data[f"exists:{user_pool_name}"] = exists
    if exists:
        context_data[f"id:{user_pool_name}"] = user_pool_id

    # Cognito User Pool Client (by name and pool ID) - requires User Pool ID from check
    user_pool_id_for_client_check = context_data.get(f"id:{user_pool_name}") # Use ID from context
    user_pool_client_name = COGNITO_USER_POOL_CLIENT_NAME
    if user_pool_id_for_client_check:
         exists, client_id, _ = check_for_existing_user_pool_client(user_pool_client_name, user_pool_id_for_client_check)
         context_data[f"exists:{user_pool_client_name}"] = exists
         if exists:
             context_data[f"id:{user_pool_client_name}"] = client_id
             # Getting client secret from Boto3 is complex; rely on Secrets Manager being source of truth

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


if __name__ == "__main__":
    check_and_set_context()