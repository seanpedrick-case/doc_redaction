import ipaddress
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
from aws_cdk import App, CfnOutput, CfnTag, Tags
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_elasticloadbalancingv2 as elb
from aws_cdk import aws_elasticloadbalancingv2_actions as elb_act
from aws_cdk import aws_iam as iam
from aws_cdk import aws_wafv2 as wafv2
from botocore.exceptions import ClientError
from cdk_config import (
    ACCESS_LOG_DYNAMODB_TABLE_NAME,
    AWS_REGION,
    FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    NAT_GATEWAY_EIP_NAME,
    POLICY_FILE_LOCATIONS,
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
from dotenv import set_key


# --- Function to load context from file ---
def load_context_from_file(app: App, file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            context_data = json.load(f)
            for key, value in context_data.items():
                app.node.set_context(key, value)
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

if POLICY_FILE_LOCATIONS:
    POLICY_FILE_LOCATIONS = _get_env_list(POLICY_FILE_LOCATIONS)


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

# Assume POLICY_FILE_LOCATIONS is defined globally or passed as a default
# For example:
# POLICY_FILE_LOCATIONS = ["./policies/my_read_policy.json", "./policies/my_write_policy.json"]


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


def check_s3_bucket_exists(
    bucket_name: str,
):  # Return type hint depends on what you return
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
    s3_client = boto3.client("s3")
    try:
        # Use head_bucket to check for existence and access
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' exists and is accessible.")
        return True, bucket_name  # Return True and the bucket name

    except ClientError as e:
        # If a ClientError occurs, check the error code.
        # '404' means the bucket does not exist.
        # '403' means the bucket exists but you don't have permission.
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            print(f"Bucket '{bucket_name}' does not exist.")
            return False, None
        elif error_code == "403":
            # The bucket exists, but you can't access it.
            # Depending on your requirements, this might be treated as "exists"
            # or "not accessible for our purpose". For checking existence,
            # we'll say it exists here, but note the permission issue.
            # NOTE - when I tested this, it was returning 403 even for buckets that don't exist. So I will return False instead
            print(
                f"Bucket '{bucket_name}' returned 403, which indicates it may exist but is not accessible due to permissions, or that it doesn't exist. Returning False for existence just in case."
            )
            return False, bucket_name  # It exists, even if not accessible
        else:
            # For other errors, it's better to raise the exception
            # to indicate something unexpected happened.
            print(
                f"An unexpected AWS ClientError occurred checking bucket '{bucket_name}': {e}"
            )
            # Decide how to handle other errors - raising might be safer
            raise  # Re-raise the original exception
    except Exception as e:
        print(
            f"An unexpected non-ClientError occurred checking bucket '{bucket_name}': {e}"
        )
        # Decide how to handle other errors
        raise  # Re-raise the original exception


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
            return (
                True,
                response["projects"][0]["arn"],
            )  # Return True and the project details dict
        elif (
            response["projectsNotFound"]
            and project_name in response["projectsNotFound"]
        ):
            # If the project name is explicitly in the 'projectsNotFound' list
            print(f"CodeBuild project '{project_name}' not found.")
            return False, None
        else:
            # This case is less expected for a single name lookup,
            # but could happen if there's an internal issue or the response
            # structure is slightly different than expected for an error.
            # It's safer to assume it wasn't found if not in 'projects'.
            print(
                f"CodeBuild project '{project_name}' not found (not in 'projects' list)."
            )
            return False, None

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


def get_vpc_id_by_name(vpc_name: str) -> Optional[str]:
    """
    Finds a VPC ID by its 'Name' tag.
    """
    ec2_client = boto3.client("ec2")
    try:
        response = ec2_client.describe_vpcs(
            Filters=[{"Name": "tag:Name", "Values": [vpc_name]}]
        )
        if response and response["Vpcs"]:
            vpc_id = response["Vpcs"][0]["VpcId"]
            print(f"VPC '{vpc_name}' found with ID: {vpc_id}")

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

            return vpc_id, nat_gateways
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
        "by_id": {},  # {subnet_id: {'name': 'subnet-name', 'cidr': 'x.x.x.x/x'}}
        "cidr_networks": [],  # List of ipaddress.IPv4Network objects
    }
    try:
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

            subnet_info = {"id": subnet_id, "cidr": cidr_block, "name": name_tag}

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

        if is_public:
            # The subnet's route_table is automatically created by the L2 Subnet construct
            try:
                subnet.add_route(
                    "DefaultInternetRoute",  # A logical ID for the CfnRoute resource
                    router_id=internet_gateway_id,
                    router_type=ec2.RouterType.GATEWAY,
                    # destination_cidr_block="0.0.0.0/0" is the default for this method
                )
            except Exception as e:
                print("Could not create IGW route for public subnet due to:", e)
            print(f"CDK: Defined public L2 subnet '{subnet_name}' and added IGW route.")
        else:
            try:
                # Using .add_route() for private subnets as well for consistency
                subnet.add_route(
                    "DefaultNatRoute",  # A logical ID for the CfnRoute resource
                    router_id=single_nat_gateway_id,
                    router_type=ec2.RouterType.NAT_GATEWAY,
                )
            except Exception as e:
                print("Could not create NAT gateway route for public subnet due to:", e)
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
    next_token = "string"

    while True:
        try:
            response = cognito_client.list_user_pool_clients(
                UserPoolId=user_pool_id, MaxResults=60, NextToken=next_token
            )
        except cognito_client.exceptions.ResourceNotFoundException:
            print(f"Error: User pool with ID '{user_pool_id}' not found.")
            return False, "", {}

        except cognito_client.exceptions.InvalidParameterException:
            print(f"Error: No app clients for '{user_pool_id}' found.")
            return False, "", {}

        except Exception as e:
            print("Could not check User Pool clients due to:", e)

        for client in response.get("UserPoolClients", []):
            if client.get("ClientName") == user_pool_client_name:
                print(
                    f"Found existing user pool client '{user_pool_client_name}' with ID: {client['ClientId']}"
                )
                return True, client["ClientId"], client

        next_token = response.get("NextToken")
        if not next_token:
            break

    return False, "", {}


def check_for_secret(secret_name: str, secret_value: dict = ""):
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
        print("Secret already exists.")
        return True, secret_value
    except secretsmanager_client.exceptions.ResourceNotFoundException:
        print("Secret not found")
        return False, {}
    except Exception as e:
        # Handle other potential exceptions during the get operation
        print(f"Error checking for secret: {e}")
        return False, {}


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


def ensure_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not."""

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


def create_basic_config_env(
    out_dir: str = "config",
    S3_LOG_CONFIG_BUCKET_NAME=S3_LOG_CONFIG_BUCKET_NAME,
    S3_OUTPUT_BUCKET_NAME=S3_OUTPUT_BUCKET_NAME,
    ACCESS_LOG_DYNAMODB_TABLE_NAME=ACCESS_LOG_DYNAMODB_TABLE_NAME,
    FEEDBACK_LOG_DYNAMODB_TABLE_NAME=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    USAGE_LOG_DYNAMODB_TABLE_NAME=USAGE_LOG_DYNAMODB_TABLE_NAME,
):
    """
    Create a basic config.env file for the user to use with their newly deployed redaction app.
    """
    variables = {
        "COGNITO_AUTH": "1",
        "RUN_AWS_FUNCTIONS": "1",
        "DISPLAY_FILE_NAMES_IN_LOGS": "False",
        "SESSION_OUTPUT_FOLDER": "True",
        "SAVE_LOGS_TO_DYNAMODB": "True",
        "SHOW_COSTS": "True",
        "SHOW_WHOLE_DOCUMENT_TEXTRACT_CALL_OPTIONS": "True",
        "LOAD_PREVIOUS_TEXTRACT_JOBS_S3": "True",
        "DOCUMENT_REDACTION_BUCKET": S3_LOG_CONFIG_BUCKET_NAME,
        "TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET": S3_OUTPUT_BUCKET_NAME,
        "ACCESS_LOG_DYNAMODB_TABLE_NAME": ACCESS_LOG_DYNAMODB_TABLE_NAME,
        "FEEDBACK_LOG_DYNAMODB_TABLE_NAME": FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
        "USAGE_LOG_DYNAMODB_TABLE_NAME": USAGE_LOG_DYNAMODB_TABLE_NAME,
    }

    # Write variables to .env file
    ensure_folder_exists(out_dir + "/")
    env_file_path = os.path.abspath(os.path.join(out_dir, "config.env"))

    # It's good practice to ensure the file exists before calling set_key repeatedly.
    # set_key will create it, but for a loop, it might be cleaner to ensure it's empty/exists once.
    if not os.path.exists(env_file_path):
        with open(env_file_path, "w"):
            pass  # Create empty file

    for key, value in variables.items():
        set_key(env_file_path, key, str(value), quote_mode="never")

    return variables


def start_codebuild_build(PROJECT_NAME: str, AWS_REGION: str = AWS_REGION):
    """
    Start an existing Codebuild project build
    """

    # --- Initialize CodeBuild client ---
    client = boto3.client("codebuild", region_name=AWS_REGION)

    try:
        print(f"Attempting to start build for project: {PROJECT_NAME}")

        response = client.start_build(projectName=PROJECT_NAME)

        build_id = response["build"]["id"]
        print(f"Successfully started build with ID: {build_id}")
        print(f"Build ARN: {response['build']['arn']}")
        print("Build URL (approximate - construct based on region and ID):")
        print(
            f"https://{AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/{PROJECT_NAME}/build/{build_id.split(':')[-1]}/detail"
        )

        # You can inspect the full response if needed
        # print("\nFull response:")
        # import json
        # print(json.dumps(response, indent=2))

    except client.exceptions.ResourceNotFoundException:
        print(f"Error: Project '{PROJECT_NAME}' not found in region '{AWS_REGION}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def upload_file_to_s3(
    local_file_paths: List[str],
    s3_key: str,
    s3_bucket: str,
    RUN_AWS_FUNCTIONS: str = "1",
):
    """
    Uploads a file from local machine to Amazon S3.

    Args:
    - local_file_path: Local file path(s) of the file(s) to upload.
    - s3_key: Key (path) to the file in the S3 bucket.
    - s3_bucket: Name of the S3 bucket.

    Returns:
    - Message as variable/printed to console
    """
    final_out_message = []
    final_out_message_str = ""

    if RUN_AWS_FUNCTIONS == "1":
        try:
            if s3_bucket and local_file_paths:

                s3_client = boto3.client("s3", region_name=AWS_REGION)

                if isinstance(local_file_paths, str):
                    local_file_paths = [local_file_paths]

                for file in local_file_paths:
                    if s3_client:
                        # print(s3_client)
                        try:
                            # Get file name off file path
                            file_name = os.path.basename(file)

                            s3_key_full = s3_key + file_name
                            print("S3 key: ", s3_key_full)

                            s3_client.upload_file(file, s3_bucket, s3_key_full)
                            out_message = (
                                "File " + file_name + " uploaded successfully!"
                            )
                            print(out_message)

                        except Exception as e:
                            out_message = f"Error uploading file(s): {e}"
                            print(out_message)

                        final_out_message.append(out_message)
                        final_out_message_str = "\n".join(final_out_message)

                    else:
                        final_out_message_str = "Could not connect to AWS."
            else:
                final_out_message_str = (
                    "At least one essential variable is empty, could not upload to S3"
                )
        except Exception as e:
            final_out_message_str = "Could not upload files to S3 due to: " + str(e)
            print(final_out_message_str)
    else:
        final_out_message_str = "App not set to run AWS functions"

    return final_out_message_str


# Initialize ECS client
def start_ecs_task(cluster_name, service_name):
    ecs_client = boto3.client("ecs")

    try:
        # Update the service to set the desired count to 1
        ecs_client.update_service(
            cluster=cluster_name, service=service_name, desiredCount=1
        )
        return {
            "statusCode": 200,
            "body": f"Service {service_name} in cluster {cluster_name} has been updated to 1 task.",
        }
    except Exception as e:
        return {"statusCode": 500, "body": f"Error updating service: {str(e)}"}
