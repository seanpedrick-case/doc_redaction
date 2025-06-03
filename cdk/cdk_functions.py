import boto3
from botocore.exceptions import ClientError
import json
from constructs import Construct
from typing import List, Tuple, Optional
#from aws_cdk import core
from aws_cdk import (
    CfnTag,
    aws_ec2 as ec2,
    aws_wafv2 as wafv2,
    aws_elasticloadbalancingv2 as elb,
    aws_certificatemanager as acm, # You might need this if you were looking up a cert, but not strictly for ARN
    aws_cognito as cognito,
    Stack,
    CfnOutput
)


from cdk_config import AWS_REGION, PUBLIC_SUBNETS_TO_USE, PRIVATE_SUBNETS_TO_USE, PUBLIC_SUBNET_CIDR_BLOCKS, PRIVATE_SUBNET_CIDR_BLOCKS, PUBLIC_SUBNET_AVAILABILITY_ZONES, PRIVATE_SUBNET_AVAILABILITY_ZONES, POLICY_FILE_LOCATIONS, CDK_PREFIX, VPC_NAME, ACM_CERTIFICATE_ARN

if PUBLIC_SUBNETS_TO_USE: PUBLIC_SUBNETS_TO_USE = eval(PUBLIC_SUBNETS_TO_USE)
if PUBLIC_SUBNET_CIDR_BLOCKS: PUBLIC_SUBNET_CIDR_BLOCKS = eval(PUBLIC_SUBNET_CIDR_BLOCKS)
if PUBLIC_SUBNET_AVAILABILITY_ZONES: PUBLIC_SUBNET_AVAILABILITY_ZONES = eval(PUBLIC_SUBNET_AVAILABILITY_ZONES)

if PRIVATE_SUBNETS_TO_USE: PRIVATE_SUBNETS_TO_USE = eval(PRIVATE_SUBNETS_TO_USE)
if PRIVATE_SUBNET_CIDR_BLOCKS: PRIVATE_SUBNET_CIDR_BLOCKS = eval(PRIVATE_SUBNET_CIDR_BLOCKS)
if PRIVATE_SUBNET_AVAILABILITY_ZONES: PRIVATE_SUBNET_AVAILABILITY_ZONES = eval(PRIVATE_SUBNET_AVAILABILITY_ZONES)

if POLICY_FILE_LOCATIONS: POLICY_FILE_LOCATIONS = eval(POLICY_FILE_LOCATIONS)

def check_for_existing_role(role_name:str):    
    try:
        iam = boto3.client('iam')
        iam.get_role(RoleName=role_name)
        
        response = iam.get_role(RoleName=role_name)
        role = response['Role']
        

        return True, role
    except iam.exceptions.NoSuchEntityException:
        return False, "", ""
    except Exception as e:
        raise Exception("Getting information on IAM role failed due to:", e)

import json
from aws_cdk import aws_iam as iam # Import the CDK IAM module
from constructs import Construct # Import if your policies need scope/context

# Assuming POLICY_FILE_LOCATIONS is defined elsewhere

def add_custom_policies(
    # You need the scope (usually the stack) to create CDK constructs
    scope: Construct,
    role: iam.IRole, # Type hint for the CDK Role object
    POLICY_FILE_LOCATIONS=POLICY_FILE_LOCATIONS
    # POLICY_FILE_ARNS=POLICY_FILE_ARNS # Add if needed later
) -> iam.IRole: # Return the modified CDK Role object
    """
    Loads custom policies from JSON files and attaches them to a CDK Role.

    Args:
        scope: The scope in which to define nested constructs (if any, though
               not strictly needed for PolicyStatement itself, required if
               you needed to create Policy resources).
        role: The CDK Role construct to attach policies to.
        POLICY_FILE_LOCATIONS: List of file paths to JSON policy documents.

    Returns:
        The modified CDK Role construct.
    """

    if POLICY_FILE_LOCATIONS:
        for i, path in enumerate(POLICY_FILE_LOCATIONS):
            try:
                # Load custom policies from JSON files
                with open(path) as f:
                    policy_document = json.load(f)

                # Ensure the loaded JSON is a valid policy document structure
                if 'Statement' not in policy_document or not isinstance(policy_document['Statement'], list):
                     print(f"Warning: Policy file {path} does not appear to be a valid policy document. Skipping.")
                     continue # Skip this file

                # PolicyStatement.from_json takes a *single statement* dictionary,
                # not the whole policy document dictionary.
                # You need to loop through the 'Statement' list if the JSON file
                # contains a full policy document.
                for statement_dict in policy_document['Statement']:
                    # Create a CDK PolicyStatement from the dictionary
                    cdk_policy_statement = iam.PolicyStatement.from_json(statement_dict)

                    # Add the policy statement to the role
                    # Use add_to_policy method of the CDK Role object
                    role.add_to_policy(cdk_policy_statement)

                print(f"Successfully added policies from {path} to role {role.node.id}")

            except FileNotFoundError:
                print(f"Warning: Policy file not found at {path}. Skipping.")
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in policy file {path}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred processing policy file {path}: {e}")
                # Decide if you want to re-raise for critical errors
                # raise e

    # If you had a list of managed policy ARNs to attach:
    # if POLICY_FILE_ARNS:
    #     for arn in POLICY_FILE_ARNS:
    #         try:
    #             managed_policy = iam.ManagedPolicy.from_managed_policy_arn(scope, f"CustomManagedPolicy{i}", arn) # Unique logical ID
    #             role.add_managed_policy(managed_policy)
    #             print(f"Attached managed policy {arn} to role {role.node.id}")
    #         except Exception as e:
    #             print(f"Warning: Could not attach managed policy {arn}: {e}")


    return role # Return the modified role object


# Import the S3 Bucket class if you intend to return a CDK object later
# from aws_cdk import aws_s3 as s3

def check_s3_bucket_exists(bucket_name: str): # Return type hint depends on what you return
    """
    Checks if an S3 bucket with the given name exists and is accessible.

    Args:
        bucket_name: The name of the S3 bucket to check.

    Returns:
        A tuple: (bool indicating existence, optional S3 Bucket object or None)
        Note: Returning a Boto3 S3 Bucket object from here is NOT ideal
              for direct use in CDK. You'll likely only need the boolean result
              or the bucket name for CDK lookups/creations.
              For this example, let's return the boolean and the name.
    """
    s3_client = boto3.client('s3')
    try:
        # Use head_bucket to check for existence and access
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' exists and is accessible.")
        return True, bucket_name # Return True and the bucket name

    except ClientError as e:
        # If a ClientError occurs, check the error code.
        # '404' means the bucket does not exist.
        # '403' means the bucket exists but you don't have permission.
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"Bucket '{bucket_name}' does not exist.")
            return False, None
        elif error_code == '403':
             # The bucket exists, but you can't access it.
             # Depending on your requirements, this might be treated as "exists"
             # or "not accessible for our purpose". For checking existence,
             # we'll say it exists here, but note the permission issue.
             # NOTE - when I tested this, it was returning 403 even for buckets that don't exist. So I will return False instead
            print(f"Bucket '{bucket_name}' returned 403, which indicates it may exist but is not accessible due to permissions, or that it doesn't exist. Returning False for existence just in case.")
            return False, bucket_name # It exists, even if not accessible
        else:
            # For other errors, it's better to raise the exception
            # to indicate something unexpected happened.
            print(f"An unexpected AWS ClientError occurred checking bucket '{bucket_name}': {e}")
            # Decide how to handle other errors - raising might be safer
            raise # Re-raise the original exception
    except Exception as e:
        print(f"An unexpected non-ClientError occurred checking bucket '{bucket_name}': {e}")
        # Decide how to handle other errors
        raise # Re-raise the original exception

# Example usage in your check_resources.py:
# exists, bucket_name_if_exists = check_s3_bucket_exists(log_bucket_name)
# context_data[f"exists:{log_bucket_name}"] = exists
# # You don't necessarily need to store the name in context if using from_bucket_name

# Delete an S3 bucket
def delete_s3_bucket(bucket_name):
    s3 = boto3.client('s3')
    
    try:
        # List and delete all objects
        response = s3.list_object_versions(Bucket=bucket_name)
        versions = response.get('Versions', []) + response.get('DeleteMarkers', [])
        for version in versions:
            s3.delete_object(Bucket=bucket_name, Key=version['Key'], VersionId=version['VersionId'])
        
        # Delete the bucket
        s3.delete_bucket(Bucket=bucket_name)
        return {'Status': 'SUCCESS'}
    except Exception as e:
        return {'Status': 'FAILED', 'Reason': str(e)}

# Function to get subnet ID from subnet name
def get_subnet_id(vpc, ec2_client, subnet_name):
    response = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc.vpc_id]}])

    for subnet in response['Subnets']:
        if subnet['Tags'] and any(tag['Key'] == 'Name' and tag['Value'] == subnet_name for tag in subnet['Tags']):
            return subnet['SubnetId']
    
    return None

def check_ecr_repo_exists(repo_name: str) -> tuple[bool, dict]:
    """
    Checks if an ECR repository with the given name exists.

    Args:
        repo_name: The name of the ECR repository to check.

    Returns:
        True if the repository exists, False otherwise.
    """
    ecr_client = boto3.client('ecr')
    try:
        print("ecr repo_name to check:", repo_name)
        response = ecr_client.describe_repositories(repositoryNames=[repo_name])
        # If describe_repositories succeeds and returns a list of repositories,
        # and the list is not empty, the repository exists.
        return len(response['repositories']) > 0, response['repositories'][0]
    except ClientError as e:
        # Check for the specific error code indicating the repository doesn't exist
        if e.response['Error']['Code'] == 'RepositoryNotFoundException':
            return False, {}
        else:
            # Re-raise other exceptions to handle unexpected errors
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}
    


# You might eventually want to return a CDK object for use in your stack
# from aws_cdk import aws_codebuild as codebuild

def check_codebuild_project_exists(project_name: str): # Adjust return type hint as needed
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
    codebuild_client = boto3.client('codebuild')
    try:
        # Use batch_get_projects with a list containing the single project name
        response = codebuild_client.batch_get_projects(names=[project_name])

        # The response for batch_get_projects includes 'projects' (found)
        # and 'projectsNotFound' (not found).
        if response['projects']:
            # If the project is found in the 'projects' list
            print(f"CodeBuild project '{project_name}' found.")
            return True, response['projects'][0]  # Return True and the project details dict
        elif response['projectsNotFound'] and project_name in response['projectsNotFound']:
             # If the project name is explicitly in the 'projectsNotFound' list
             print(f"CodeBuild project '{project_name}' not found.")
             return False, None
        else:
            # This case is less expected for a single name lookup,
            # but could happen if there's an internal issue or the response
            # structure is slightly different than expected for an error.
            # It's safer to assume it wasn't found if not in 'projects'.
            print(f"CodeBuild project '{project_name}' not found (not in 'projects' list).")
            return False, None

    except ClientError as e:
        # Catch specific ClientErrors. batch_get_projects might not throw
        # 'InvalidInputException' for a non-existent project name if the
        # name format is valid. It typically just lists it in projectsNotFound.
        # However, other ClientErrors are possible (e.g., permissions).
        print(f"An AWS ClientError occurred checking CodeBuild project '{project_name}': {e}")
        # Decide how to handle other ClientErrors - raising might be safer
        raise # Re-raise the original exception
    except Exception as e:
        print(f"An unexpected non-ClientError occurred checking CodeBuild project '{project_name}': {e}")
        # Decide how to handle other errors
        raise # Re-raise the original exception

# Example usage in your check_resources.py:
# exists, project_details = check_codebuild_project_exists(codebuild_project_name)
# context_data[f"exists:{codebuild_project_name}"] = exists
# if exists:
#     context_data[f"arn:{codebuild_project_name}"] = project_details['arn'] # Get ARN from the details dict

def get_vpc_id_by_name(vpc_name: str) -> Optional[str]:
    """
    Finds a VPC ID by its 'Name' tag.
    """
    ec2_client = boto3.client('ec2')
    try:
        response = ec2_client.describe_vpcs(
            Filters=[
                {'Name': 'tag:Name', 'Values': [vpc_name]}
            ]
        )
        if response and response['Vpcs']:
            vpc_id = response['Vpcs'][0]['VpcId']
            print(f"VPC '{vpc_name}' found with ID: {vpc_id}")

            # In get_vpc_id_by_name, after finding VPC ID:

            # Look for NAT Gateways in this VPC
            ec2_client = boto3.client('ec2')
            nat_gateways = []
            try:
                response = ec2_client.describe_nat_gateways(
                    Filters=[
                        {'Name': 'vpc-id', 'Values': [vpc_id]},
                        # Optional: Add a tag filter if you consistently tag your NATs
                        # {'Name': 'tag:Name', 'Values': [f"{prefix}-nat-gateway"]}
                    ]
                )
                nat_gateways = response.get('NatGateways', [])
            except Exception as e:
                print(f"Warning: Could not describe NAT Gateways in VPC '{vpc_id}': {e}")
                # Decide how to handle this error - proceed or raise?

            # Decide how to identify the specific NAT Gateway you want to check for.
            


            return vpc_id, nat_gateways
        else:
            print(f"VPC '{vpc_name}' not found.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred finding VPC '{vpc_name}': {e}")
        raise

def check_and_set_context():
    prefix = CDK_PREFIX
    context_data = {}

    # --- Find the VPC ID first ---
    vpc_id = get_vpc_id_by_name(VPC_NAME)
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

def check_subnet_exists_by_name(vpc_id: str, subnet_name: str) -> Tuple[bool, Optional[str]]:
    """
    Checks if a subnet with the given name exists within a specific VPC.

    Args:
        vpc_id: The ID of the VPC to check within.
        subnet_name: The 'Name' tag value of the subnet to check.

    Returns:
        A tuple:
        - The first element is True if the subnet exists, False otherwise.
        - The second element is the Subnet ID if found, None otherwise.
    """
    ec2_client = boto3.client('ec2')
    try:
        response = ec2_client.describe_subnets(
            Filters=[
                {'Name': 'vpc-id', 'Values': [vpc_id]},
                {'Name': 'tag:Name', 'Values': [subnet_name]} # Filter by the 'Name' tag
            ]
        )
        if response and response['Subnets']:
            # Found a subnet with the given name in the VPC
            print(f"Subnet '{subnet_name}' found in VPC '{vpc_id}' with ID: {response['Subnets'][0]['SubnetId']}")
            return True, response['Subnets'][0]['SubnetId']
        else:
            # No subnet found with that name and VPC ID
            print(f"Subnet '{subnet_name}' not found in VPC '{vpc_id}'.")
            return False, None
    except Exception as e:
        print(f"An unexpected error occurred checking for subnet '{subnet_name}' in VPC '{vpc_id}': {e}")
        # Decide how to handle other errors - raising might be safer
        raise # Re-raise the original exception


# You might also need these types if managing route tables explicitly
# from aws_cdk.aws_ec2 import IRouteTable, CfnRouteTable, CfnSubnetRouteTableAssociation

# Assuming these are not directly used in this function, but defined elsewhere if needed
# PUBLIC_SUBNET_CIDR_BLOCKS, PRIVATE_SUBNET_CIDR_BLOCKS, etc.

def create_subnets(
    scope: Construct,
    vpc: ec2.IVpc,
    prefix: str,
    subnet_names: List[str],
    cidr_blocks: List[str],
    availability_zones: List[str],
    is_public: bool
) -> List[ec2.ISubnet]: # Return only the list of created subnets (ISubnet objects)
    """
    Creates a list of subnets within a VPC.

    Args:
        scope: The scope in which to define the constructs (typically the stack).
        vpc: The VPC construct (created or looked up).
        prefix: A prefix for resource names.
        subnet_names: A list of names for the subnets to create.
        cidr_blocks: A list of CIDR blocks for the subnets. Must match subnet_names length.
        availability_zones: A list of Availability Zones for the subnets. Must match subnet_names length.
        is_public: Boolean indicating if the subnets should be public (map public IP).

    Returns:
        A list of the created ISubnet objects.
    """
    if not (len(subnet_names) == len(cidr_blocks) == len(availability_zones) > 0):
        raise ValueError("Subnet names, CIDR blocks, and Availability Zones lists must be non-empty and match in length.")

    created_subnets: List[ec2.ISubnet] = []
    subnet_type_tag = "public" if is_public else "private"

    for i, subnet_name in enumerate(subnet_names):
        # Generate a unique logical ID for the subnet within the stack.
        # Using the subnet name and index helps ensure uniqueness even if names repeat across calls.
        logical_id = f"{prefix}{subnet_type_tag.capitalize()}Subnet{i+1}Created"

        subnet = ec2.Subnet(
            scope,
            logical_id,
            vpc_id=vpc.vpc_id, # Link to the VPC ID
            cidr_block=cidr_blocks[i],
            availability_zone=availability_zones[i],
            map_public_ip_on_launch=is_public#,
             # Add tags, including the 'Name' tag used for lookups
            #tags=[
            #    CfnTag(key="Name", value=subnet_name),
            #    CfnTag(key="Type", value=subnet_type_tag) # Optional: helpful for organization/selection
            #]
        )
        created_subnets.append(subnet)
        print(f"Defined {subnet_type_tag} subnet '{subnet_name}' ({cidr_blocks[i]}) in {availability_zones[i]}.")


    # --- Route Table Management for Private Subnets (If using Scenario 1) ---
    # If is_public is False and you are managing private subnet route tables explicitly,
    # you would create them and their associations here.
    # You would also need to return the list of created route table objects.

    created_private_route_tables = []

    # Example (Conceptual, assuming is_public is False):
    if not is_public:
        created_private_route_tables: List[ec2.IRouteTable] = []
        for i, created_subnet in enumerate(created_subnets):
             logical_id = f"{prefix}PrivateRouteTable{i+1}Created"
             route_table = ec2.CfnRouteTable(
                 scope,
                 logical_id,
                 vpc_id=vpc.vpc_id,
                 tags=[CfnTag(key="Name", value=f"{prefix}-private-rt-{i+1}-created")]
             )
             created_private_route_tables.append(route_table)

             logical_id_assoc = f"{prefix}PrivateRTAssoc{i+1}Created"
             ec2.CfnSubnetRouteTableAssociation(
                 scope,
                 logical_id_assoc,
                 subnet_id=created_subnet.subnet_id,
                 route_table_id=route_table.ref
             )
             print(f"Created route table and association for private subnet {created_subnet.subnet_id}")
        # If managing route tables here, you would return:
        return created_subnets, created_private_route_tables
    #     # Or adjust your stack to receive the route tables from this function.
    return created_subnets, []
                
def ingress_rule_exists(security_group:str, peer:str, port:str):
    for rule in security_group.connections.security_groups:
        if port:
            if rule.peer == peer and rule.connection == port:
                return True
        else:
            if rule.peer == peer:
                return True
    return False

def check_for_existing_user_pool(user_pool_name:str):
    cognito_client = boto3.client("cognito-idp")
    list_pools_response = cognito_client.list_user_pools(MaxResults=60) # MaxResults up to 60
            
    # ListUserPools might require pagination if you have more than 60 pools
    # This simple example doesn't handle pagination, which could miss your pool

    existing_user_pool_id = ""

    for pool in list_pools_response.get('UserPools', []):
        if pool.get('Name') == user_pool_name:
            existing_user_pool_id = pool['Id']
            print(f"Found existing user pool by name '{user_pool_name}' with ID: {existing_user_pool_id}")
            break # Found the one we're looking for

    if existing_user_pool_id:
        return True, existing_user_pool_id, pool
    else:
        return False, "", ""
    
import boto3

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
            response = cognito_client.list_user_pool_clients(
                UserPoolId=user_pool_id,
                MaxResults=60,
                NextToken=next_token
            )
        except cognito_client.exceptions.ResourceNotFoundException:
            print(f"Error: User pool with ID '{user_pool_id}' not found.")
            return False, "", {}

        for client in response.get('UserPoolClients', []):
            if client.get('ClientName') == user_pool_client_name:
                print(f"Found existing user pool client '{user_pool_client_name}' with ID: {client['ClientId']}")
                return True, client['ClientId'], client

        next_token = response.get('NextToken')
        if not next_token:
            break

    return False, "", {}

def check_for_secret(secret_name: str, secret_value: dict=""):
    """
    Checks if a Secrets Manager secret with the given name exists.
    If it doesn't exist, it creates the secret.

    Args:
        secret_name: The name of the Secrets Manager secret.
        secret_value: A dictionary containing the key-value pairs for the secret.

    Returns:
        True if the secret existed or was created, False otherwise (due to other errors).
    """
    secretsmanager_client = boto3.client("secretsmanager")

    try:
        # Try to get the secret. If it doesn't exist, a ResourceNotFoundException will be raised.
        secret_value = secretsmanager_client.get_secret_value(SecretId=secret_name)
        print(f"Secret '{secret_name}' already exists.")
        return True, secret_value
    except secretsmanager_client.exceptions.ResourceNotFoundException:
        print("Secret not found")
        return False, {}
    except Exception as e:
        # Handle other potential exceptions during the get operation
        print(f"Error checking for secret '{secret_name}': {e}")
        return False, {}
    
def check_alb_exists(load_balancer_name: str, region_name: str = None) -> tuple[bool, dict]:
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
        elbv2_client = boto3.client('elbv2', region_name=region_name)
    else:
        elbv2_client = boto3.client('elbv2')
    try:
        response = elbv2_client.describe_load_balancers(Names=[load_balancer_name])
        if response['LoadBalancers']:
            return True, response['LoadBalancers'][0]  # Return True and the first ALB object
        else:
            return False, {}
    except ClientError as e:
        #  If the error indicates the ALB doesn't exist, return False
        if e.response['Error']['Code'] == 'LoadBalancerNotFound':
            return False, {}
        else:
            # Re-raise other exceptions
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}
    
def check_fargate_task_definition_exists(task_definition_name: str, region_name: str = None) -> tuple[bool, dict]:
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
        ecs_client = boto3.client('ecs', region_name=region_name)
    else:
        ecs_client = boto3.client('ecs')
    try:
        response = ecs_client.describe_task_definition(taskDefinition=task_definition_name)
        # If describe_task_definition succeeds, it returns the task definition.
        # We can directly return True and the task definition.
        return True, response['taskDefinition']
    except ClientError as e:
        # Check for the error code indicating the task definition doesn't exist.
        if e.response['Error']['Code'] == 'ClientException' and 'Task definition' in e.response['Message'] and 'does not exist' in e.response['Message']:
            return False, {}
        else:
            # Re-raise other exceptions.
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}
    
def check_ecs_service_exists(cluster_name: str, service_name: str, region_name: str = None) -> tuple[bool, dict]:
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
        ecs_client = boto3.client('ecs', region_name=region_name)
    else:
        ecs_client = boto3.client('ecs')
    try:
        response = ecs_client.describe_services(cluster=cluster_name, services=[service_name])
        if response['services']:
            return True, response['services'][0]  # Return True and the first service object
        else:
            return False, {}
    except ClientError as e:
        # Check for the error code indicating the service doesn't exist.
        if e.response['Error']['Code'] == 'ClusterNotFoundException':
            return False, {}
        elif e.response['Error']['Code'] == 'ServiceNotFoundException':
            return False, {}
        else:
            # Re-raise other exceptions.
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}
    
def check_cloudfront_distribution_exists(distribution_name: str, region_name: str = None) -> tuple[bool, dict | None]:
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
        cf_client = boto3.client('cloudfront', region_name=region_name)
    else:
        cf_client = boto3.client('cloudfront')
    try:
        response = cf_client.list_distributions()
        if 'Items' in response['DistributionList']:
            for distribution in response['DistributionList']['Items']:
                # CloudFront doesn't directly filter by name, so we have to iterate.
                if distribution['AliasSet']['Items'] and distribution['AliasSet']['Items'][0] == distribution_name:
                    return True, distribution
            return False, None
        else:
            return False, None
    except ClientError as e:
        #  If the error indicates the Distribution doesn't exist, return False
        if e.response['Error']['Code'] == 'NoSuchDistribution':
            return False, None
        else:
            # Re-raise other exceptions
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, None

import aws_cdk.aws_wafv2 as wafv2
from aws_cdk import CfnOutput # Assuming you might want to export the WebACL ARN

def create_web_acl_with_common_rules(scope, web_acl_name: str):
    '''
    Use CDK to create a web ACL based on an AWS common rule set with overrides.
    This function now expects a 'scope' argument, typically 'self' from your stack,
    as CfnWebACL requires a construct scope.
    '''

    # Create full list of rules
    rules = []
    aws_ruleset_names = [
        "AWSManagedRulesCommonRuleSet",
        "AWSManagedRulesKnownBadInputsRuleSet",
        "AWSManagedRulesAmazonIpReputationList"
    ]

    for i, aws_rule_name in enumerate(aws_ruleset_names):
        # Initialize variables for conditional properties
        current_rule_action_overrides = None
        current_override_action = None
        current_priority = i + 1 # Default priority

        # Determine conditional values BEFORE constructing the RuleProperty
        if aws_rule_name == "AWSManagedRulesCommonRuleSet":
            current_rule_action_overrides = [
                wafv2.CfnWebACL.RuleActionOverrideProperty(
                    name="SizeRestrictions_BODY",
                    # Allow means this specific rule within the managed group won't block
                    action_to_use=wafv2.CfnWebACL.RuleActionProperty(
                        allow={}
                    )
                )
            ]
            # For the entire managed rule group, 'none' means apply the AWS recommended action
            # unless a specific rule within it is overridden by rule_action_overrides.
            current_override_action = wafv2.CfnWebACL.OverrideActionProperty(none={})
            current_priority = 2 # Set specific priority for this rule set

        # Construct the RuleProperty using the determined values
        rule_property = wafv2.CfnWebACL.RuleProperty(
            name=aws_rule_name,
            priority=current_priority, # Use the dynamic priority
            statement=wafv2.CfnWebACL.StatementProperty(
                managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                    vendor_name="AWS",
                    name=aws_rule_name,
                    # Pass the dynamically determined overrides here
                    rule_action_overrides=current_rule_action_overrides
                )
            ),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=aws_rule_name,
                sampled_requests_enabled=True
            ),
            # Pass the dynamically determined override_action for the *entire rule group*
            override_action=current_override_action
        )

        rules.append(rule_property)

    # Add the rate limit rule
    # Ensure its priority is unique and typically after managed rules
    rate_limit_priority = max([rule.priority for rule in rules]) + 1 if rules else 1
    rules.append(wafv2.CfnWebACL.RuleProperty(
        name="RateLimitRule",
        priority=rate_limit_priority,
        statement=wafv2.CfnWebACL.StatementProperty(
            rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                limit=1000, # Max requests per 5-minute period
                aggregate_key_type="IP" # Aggregate by client IP address
            )
        ),
        visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
            cloud_watch_metrics_enabled=True,
            metric_name="RateLimitRule",
            sampled_requests_enabled=True
        ),
        action=wafv2.CfnWebACL.RuleActionProperty(
            block={} # Block requests exceeding the limit
        )
    ))

    # Create the WAF web ACL
    web_acl = wafv2.CfnWebACL(
        scope, # Pass the scope (e.g., 'self' from your stack)
        "WebACL",
        name=web_acl_name,
        # Default action applied if no rule matches
        default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
        scope="CLOUDFRONT", # Essential for CloudFront distributions
        visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
            cloud_watch_metrics_enabled=True,
            metric_name="webACL", # Metric name for the entire WebACL
            sampled_requests_enabled=True
        ),
        rules=rules # The list of rules we just constructed
    )

    # Optional: Output the WebACL ARN if you need to reference it elsewhere
    # CfnOutput(scope, "WebACLArn", value=web_acl.attr_arn)

    return web_acl
    

def check_web_acl_exists(web_acl_name: str, scope: str, region_name: str = None) -> tuple[bool, dict]:
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
    if scope not in ['CLOUDFRONT', 'REGIONAL']:
        raise ValueError("Scope must be either 'CLOUDFRONT' or 'REGIONAL'")

    if scope == 'REGIONAL' and not region_name:
        raise ValueError("Region name is required for REGIONAL scope")

    if scope == 'CLOUDFRONT':
        region_name = 'us-east-1'  # CloudFront scope requires us-east-1
    
    if region_name:
        waf_client = boto3.client('wafv2', region_name=region_name)
    else:
        waf_client = boto3.client('wafv2')
    try:
        response = waf_client.list_web_acls(Scope=scope)
        if 'WebACLs' in response:
            for web_acl in response['WebACLs']:
                if web_acl['Name'] == web_acl_name:
                    # Describe the Web ACL to get the full object.
                    describe_response = waf_client.describe_web_acl(Name=web_acl_name, Scope=scope)
                    return True, describe_response['WebACL']
            return False, {}
        else:
            return False, {}
    except ClientError as e:
        # Check for the error code indicating the web ACL doesn't exist.
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            return False, {}
        else:
            # Re-raise other exceptions.
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False, {}
    


def add_alb_https_listener_with_cert(
    scope: Construct,
    logical_id: str, # A unique ID for this listener construct
    alb: elb.ApplicationLoadBalancer,
    acm_certificate_arn: Optional[str], # Optional: If None, no HTTPS listener will be created
    default_target_group: elb.ITargetGroup, # Mandatory: The target group to forward traffic to
    listener_port_https: int = 443,
    listener_open_to_internet: bool = False, # Be cautious with True, ensure ALB security group restricts access
    # --- Cognito Authentication Parameters ---
    enable_cognito_auth: bool = False,
    cognito_user_pool: Optional[cognito.IUserPool] = None,
    cognito_user_pool_client: Optional[cognito.IUserPoolClient] = None,
    cognito_user_pool_domain: Optional[str] = None, # E.g., "my-app-domain" for "my-app-domain.auth.region.amazoncognito.com"
    cognito_auth_scope: Optional[str] = "openid profile email", # Default recommended scope
    cognito_auth_on_unauthenticated_request: elb.UnauthenticatedAction = elb.UnauthenticatedAction.AUTHENTICATE,
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
        print(f"Attempting to add ALB HTTPS listener on port {listener_port_https} with ACM certificate: {acm_certificate_arn}")

        # Determine the default action based on whether Cognito auth is enabled
        default_action = None
        if enable_cognito_auth:
            if not all([cognito_user_pool, cognito_user_pool_client, cognito_user_pool_domain]):
                raise ValueError(
                    "Cognito User Pool, Client, and Domain must be provided if enable_cognito_auth is True."
                )
            print(f"Enabling Cognito authentication with User Pool: {cognito_user_pool.user_pool_id}")

            default_action = elb.ListenerAction.authenticate_cognito(
                user_pool=cognito_user_pool,
                user_pool_client=cognito_user_pool_client,
                user_pool_domain=cognito_user_pool_domain,
                next=elb.ListenerAction.forward([default_target_group]), # After successful auth, forward to TG
                scope=cognito_auth_scope,
                on_unauthenticated_request=cognito_auth_on_unauthenticated_request,
                # Additional options you might want to configure:
                # session_cookie_name="AWSELBCookies",
                # session_timeout=Duration.hours(1)
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
            default_action=default_action # Use the determined default action
        )
        print(f"ALB HTTPS listener on port {listener_port_https} defined.")
    else:
        print("ACM_CERTIFICATE_ARN is not provided. Skipping HTTPS listener creation.")

    return https_listener