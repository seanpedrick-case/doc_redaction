import json  # You might still need json if loading task_definition.json
import os
from typing import Any, Dict, List

from aws_cdk import (
    CfnOutput,  # <-- Import CfnOutput directly
    Duration,
    RemovalPolicy,
    SecretValue,
    Stack,
)
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_codebuild as codebuild
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb  # Import the DynamoDB module
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_wafv2 as wafv2
from cdk_config import (
    ACCESS_LOG_DYNAMODB_TABLE_NAME,
    ACM_SSL_CERTIFICATE_ARN,
    ALB_NAME,
    ALB_NAME_SECURITY_GROUP_NAME,
    ALB_TARGET_GROUP_NAME,
    AWS_ACCOUNT_ID,
    AWS_MANAGED_TASK_ROLES_LIST,
    AWS_REGION,
    CDK_PREFIX,
    CLOUDFRONT_DISTRIBUTION_NAME,
    CLOUDFRONT_GEO_RESTRICTION,
    CLUSTER_NAME,
    CODEBUILD_PROJECT_NAME,
    CODEBUILD_ROLE_NAME,
    COGNITO_REDIRECTION_URL,
    COGNITO_USER_POOL_CLIENT_NAME,
    COGNITO_USER_POOL_CLIENT_SECRET_NAME,
    COGNITO_USER_POOL_DOMAIN_PREFIX,
    COGNITO_USER_POOL_NAME,
    CUSTOM_HEADER,
    CUSTOM_HEADER_VALUE,
    CUSTOM_KMS_KEY_NAME,
    DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS,
    ECR_CDK_REPO_NAME,
    ECS_LOG_GROUP_NAME,
    ECS_READ_ONLY_FILE_SYSTEM,
    ECS_SECURITY_GROUP_NAME,
    ECS_SERVICE_NAME,
    ECS_TASK_CPU_SIZE,
    ECS_TASK_EXECUTION_ROLE_NAME,
    ECS_TASK_MEMORY_SIZE,
    ECS_TASK_ROLE_NAME,
    ECS_USE_FARGATE_SPOT,
    EXISTING_IGW_ID,
    FARGATE_TASK_DEFINITION_NAME,
    FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
    GITHUB_REPO_BRANCH,
    GITHUB_REPO_NAME,
    GITHUB_REPO_USERNAME,
    GRADIO_SERVER_PORT,
    LOAD_BALANCER_WEB_ACL_NAME,
    NAT_GATEWAY_NAME,
    NEW_VPC_CIDR,
    NEW_VPC_DEFAULT_NAME,
    PRIVATE_SUBNET_AVAILABILITY_ZONES,
    PRIVATE_SUBNET_CIDR_BLOCKS,
    PRIVATE_SUBNETS_TO_USE,
    PUBLIC_SUBNET_AVAILABILITY_ZONES,
    PUBLIC_SUBNET_CIDR_BLOCKS,
    PUBLIC_SUBNETS_TO_USE,
    S3_LOG_CONFIG_BUCKET_NAME,
    S3_OUTPUT_BUCKET_NAME,
    SAVE_LOGS_TO_DYNAMODB,
    SINGLE_NAT_GATEWAY_ID,
    TASK_DEFINITION_FILE_LOCATION,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
    USE_CLOUDFRONT,
    USE_CUSTOM_KMS_KEY,
    VPC_NAME,
    WEB_ACL_NAME,
)
from cdk_functions import (  # Only keep CDK-native functions
    add_alb_https_listener_with_cert,
    add_custom_policies,
    create_nat_gateway,
    create_subnets,
    create_web_acl_with_common_rules,
)
from constructs import Construct


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

if AWS_MANAGED_TASK_ROLES_LIST:
    AWS_MANAGED_TASK_ROLES_LIST = _get_env_list(AWS_MANAGED_TASK_ROLES_LIST)


class CdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Helper to get context values ---
        def get_context_bool(key: str, default: bool = False) -> bool:
            return self.node.try_get_context(key) or default

        def get_context_str(key: str, default: str = None) -> str:
            return self.node.try_get_context(key) or default

        def get_context_dict(key: str, default: dict = None) -> dict:
            return self.node.try_get_context(key) or default

        def get_context_list_of_dicts(key: str) -> List[Dict[str, Any]]:
            ctx_value = self.node.try_get_context(key)
            if not isinstance(ctx_value, list):
                print(
                    f"Warning: Context key '{key}' not found or not a list. Returning empty list."
                )
                return []
            # Optional: Add validation that all items in the list are dicts
            return ctx_value

        self.template_options.description = "Deployment of the 'doc_redaction' PDF, image, and XLSX/CSV redaction app. Git repo available at: https://github.com/seanpedrick-case/doc_redaction."

        # --- VPC and Subnets (Assuming VPC is always lookup, Subnets are created/returned by create_subnets) ---
        new_vpc_created = False
        if VPC_NAME:
            print("Looking for current VPC:", VPC_NAME)
            try:
                vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_name=VPC_NAME)
                print("Successfully looked up VPC:", vpc.vpc_id)
            except Exception as e:
                raise Exception(
                    f"Could not look up VPC with name '{VPC_NAME}' due to: {e}"
                )

        elif NEW_VPC_DEFAULT_NAME:
            new_vpc_created = True
            print(
                f"NEW_VPC_DEFAULT_NAME ('{NEW_VPC_DEFAULT_NAME}') is set. Creating a new VPC."
            )

            # Configuration for the new VPC
            # You can make these configurable via context as well, e.g.,
            # new_vpc_cidr = self.node.try_get_context("new_vpc_cidr") or "10.0.0.0/24"
            # new_vpc_max_azs = self.node.try_get_context("new_vpc_max_azs") or 2 # Use 2 AZs by default for HA
            # new_vpc_nat_gateways = self.node.try_get_context("new_vpc_nat_gateways") or new_vpc_max_azs # One NAT GW per AZ for HA
            # or 1 for cost savings if acceptable
            if not NEW_VPC_CIDR:
                raise Exception(
                    "App has been instructed to create a new VPC but not VPC CDR range provided to variable NEW_VPC_CIDR"
                )

            print("Provided NEW_VPC_CIDR range:", NEW_VPC_CIDR)

            new_vpc_cidr = NEW_VPC_CIDR
            new_vpc_max_azs = 2  # Creates resources in 2 AZs. Adjust as needed.

            # For "a NAT gateway", you can set nat_gateways=1.
            # For resilience (NAT GW per AZ), set nat_gateways=new_vpc_max_azs.
            # The Vpc construct will create NAT Gateway(s) if subnet_type PRIVATE_WITH_EGRESS is used
            # and nat_gateways > 0.
            new_vpc_nat_gateways = (
                1  # Creates a single NAT Gateway for cost-effectiveness.
            )
            # If you need one per AZ for higher availability, set this to new_vpc_max_azs.

            vpc = ec2.Vpc(
                self,
                "MyNewLogicalVpc",  # This is the CDK construct ID
                vpc_name=NEW_VPC_DEFAULT_NAME,
                ip_addresses=ec2.IpAddresses.cidr(new_vpc_cidr),
                max_azs=new_vpc_max_azs,
                nat_gateways=new_vpc_nat_gateways,  # Number of NAT gateways to create
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name="Public",  # Name prefix for public subnets
                        subnet_type=ec2.SubnetType.PUBLIC,
                        cidr_mask=28,  # Adjust CIDR mask as needed (e.g., /24 provides ~250 IPs per subnet)
                    ),
                    ec2.SubnetConfiguration(
                        name="Private",  # Name prefix for private subnets
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,  # Ensures these subnets have NAT Gateway access
                        cidr_mask=28,  # Adjust CIDR mask as needed
                    ),
                    # You could also add ec2.SubnetType.PRIVATE_ISOLATED if needed
                ],
                # Internet Gateway is created and configured automatically for PUBLIC subnets.
                # Route tables for public subnets will point to the IGW.
                # Route tables for PRIVATE_WITH_EGRESS subnets will point to the NAT Gateway(s).
            )
            print(
                f"Successfully created new VPC: {vpc.vpc_id} with name '{NEW_VPC_DEFAULT_NAME}'"
            )
            # If nat_gateways > 0, vpc.nat_gateway_ips will contain EIPs if Vpc created them.
            # vpc.public_subnets, vpc.private_subnets, vpc.isolated_subnets are populated.

        else:
            raise Exception(
                "VPC_NAME for current VPC not found, and NEW_VPC_DEFAULT_NAME not found to create a new VPC"
            )

        # --- Subnet Handling (Check Context and Create/Import) ---
        # Initialize lists to hold ISubnet objects (L2) and CfnSubnet/CfnRouteTable (L1)
        # We will store ISubnet for consistency, as CfnSubnet has a .subnet_id property
        self.public_subnets: List[ec2.ISubnet] = []
        self.private_subnets: List[ec2.ISubnet] = []
        # Store L1 CfnRouteTables explicitly if you need to reference them later
        self.private_route_tables_cfn: List[ec2.CfnRouteTable] = []
        self.public_route_tables_cfn: List[ec2.CfnRouteTable] = (
            []
        )  # New: to store public RTs

        names_to_create_private = []
        names_to_create_public = []

        if not PUBLIC_SUBNETS_TO_USE and not PRIVATE_SUBNETS_TO_USE:
            print(
                "Warning: No public or private subnets specified in *_SUBNETS_TO_USE. Attempting to select from existing VPC subnets."
            )

            print("vpc.public_subnets:", vpc.public_subnets)
            print("vpc.private_subnets:", vpc.private_subnets)

            if (
                vpc.public_subnets
            ):  # These are already one_per_az if max_azs was used and Vpc created them
                self.public_subnets.extend(vpc.public_subnets)
            else:
                self.node.add_warning("No public subnets found in the VPC.")

            # Get private subnets with egress specifically
            # selected_private_subnets_with_egress = vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)

            print(
                f"Selected from VPC: {len(self.public_subnets)} public, {len(self.private_subnets)} private_with_egress subnets."
            )

            if (
                len(self.public_subnets) < 1 or len(self.private_subnets) < 1
            ):  # Simplified check for new VPC
                # If new_vpc_max_azs was 1, you'd have 1 of each. If 2, then 2 of each.
                # The original check ' < 2' might be too strict if new_vpc_max_azs=1
                pass  # For new VPC, allow single AZ setups if configured that way. The VPC construct ensures one per AZ up to max_azs.

            if not self.public_subnets and not self.private_subnets:
                print(
                    "Error: No public or private subnets could be found in the VPC for automatic selection. "
                    "You must either specify subnets in *_SUBNETS_TO_USE or ensure the VPC has discoverable subnets."
                )
                raise RuntimeError("No suitable subnets found for automatic selection.")
            else:
                print(
                    f"Automatically selected {len(self.public_subnets)} public and {len(self.private_subnets)} private subnets based on VPC properties."
                )

            selected_public_subnets = vpc.select_subnets(
                subnet_type=ec2.SubnetType.PUBLIC, one_per_az=True
            )
            private_subnets_egress = vpc.select_subnets(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, one_per_az=True
            )

            if private_subnets_egress.subnets:
                self.private_subnets.extend(private_subnets_egress.subnets)
            else:
                self.node.add_warning(
                    "No PRIVATE_WITH_EGRESS subnets found in the VPC."
                )

            try:
                private_subnets_isolated = vpc.select_subnets(
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, one_per_az=True
                )
            except Exception as e:
                private_subnets_isolated = []
                print("Could not find any isolated subnets due to:", e)

            ###
            combined_subnet_objects = []

            if private_subnets_isolated:
                if private_subnets_egress.subnets:
                    # Add the first PRIVATE_WITH_EGRESS subnet
                    combined_subnet_objects.append(private_subnets_egress.subnets[0])
            elif not private_subnets_isolated:
                if private_subnets_egress.subnets:
                    # Add the first PRIVATE_WITH_EGRESS subnet
                    combined_subnet_objects.extend(private_subnets_egress.subnets)
            else:
                self.node.add_warning(
                    "No PRIVATE_WITH_EGRESS subnets found to select the first one."
                )

            # Add all PRIVATE_ISOLATED subnets *except* the first one (if they exist)
            try:
                if len(private_subnets_isolated.subnets) > 1:
                    combined_subnet_objects.extend(private_subnets_isolated.subnets[1:])
                elif (
                    private_subnets_isolated.subnets
                ):  # Only 1 isolated subnet, add a warning if [1:] was desired
                    self.node.add_warning(
                        "Only one PRIVATE_ISOLATED subnet found, private_subnets_isolated.subnets[1:] will be empty."
                    )
                else:
                    self.node.add_warning("No PRIVATE_ISOLATED subnets found.")
            except Exception as e:
                print("Could not identify private isolated subnets due to:", e)

            # Create an ec2.SelectedSubnets object from the combined private subnet list.
            selected_private_subnets = vpc.select_subnets(
                subnets=combined_subnet_objects
            )

            print("selected_public_subnets:", selected_public_subnets)
            print("selected_private_subnets:", selected_private_subnets)

            if (
                len(selected_public_subnets.subnet_ids) < 2
                or len(selected_private_subnets.subnet_ids) < 2
            ):
                raise Exception(
                    "Need at least two public or private subnets in different availability zones"
                )

            if not selected_public_subnets and not selected_private_subnets:
                # If no subnets could be found even with automatic selection, raise an error.
                # This ensures the stack doesn't proceed if it absolutely needs subnets.
                print(
                    "Error: No existing public or private subnets could be found in the VPC for automatic selection. "
                    "You must either specify subnets in *_SUBNETS_TO_USE or ensure the VPC has discoverable subnets."
                )
                raise RuntimeError("No suitable subnets found for automatic selection.")
            else:
                self.public_subnets = selected_public_subnets.subnets
                self.private_subnets = selected_private_subnets.subnets
                print(
                    f"Automatically selected {len(self.public_subnets)} public and {len(self.private_subnets)} private subnets based on VPC discovery."
                )

                print("self.public_subnets:", self.public_subnets)
                print("self.private_subnets:", self.private_subnets)
                # Since subnets are now assigned, we can exit this processing block.
                # The rest of the original code (which iterates *_SUBNETS_TO_USE) will be skipped.

        checked_public_subnets_ctx = get_context_dict("checked_public_subnets")
        get_context_dict("checked_private_subnets")

        public_subnets_data_for_creation_ctx = get_context_list_of_dicts(
            "public_subnets_to_create"
        )
        private_subnets_data_for_creation_ctx = get_context_list_of_dicts(
            "private_subnets_to_create"
        )

        # --- 3. Process Public Subnets ---
        print("\n--- Processing Public Subnets ---")
        # Import existing public subnets
        if checked_public_subnets_ctx:
            for i, subnet_name in enumerate(PUBLIC_SUBNETS_TO_USE):
                subnet_info = checked_public_subnets_ctx.get(subnet_name)
                if subnet_info and subnet_info.get("exists"):
                    subnet_id = subnet_info.get("id")
                    if not subnet_id:
                        raise RuntimeError(
                            f"Context for existing public subnet '{subnet_name}' is missing 'id'."
                        )
                    try:
                        ec2.Subnet.from_subnet_id(
                            self,
                            f"ImportedPublicSubnet{subnet_name.replace('-', '')}{i}",
                            subnet_id,
                        )
                        # self.public_subnets.append(imported_subnet)
                        print(
                            f"Imported existing public subnet: {subnet_name} (ID: {subnet_id})"
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to import public subnet '{subnet_name}' with ID '{subnet_id}'. Error: {e}"
                        )

        # Create new public subnets based on public_subnets_data_for_creation_ctx
        if public_subnets_data_for_creation_ctx:
            names_to_create_public = [
                s["name"] for s in public_subnets_data_for_creation_ctx
            ]
            cidrs_to_create_public = [
                s["cidr"] for s in public_subnets_data_for_creation_ctx
            ]
            azs_to_create_public = [
                s["az"] for s in public_subnets_data_for_creation_ctx
            ]

            if names_to_create_public:
                print(
                    f"Attempting to create {len(names_to_create_public)} new public subnets: {names_to_create_public}"
                )
                newly_created_public_subnets, newly_created_public_rts_cfn = (
                    create_subnets(
                        self,
                        vpc,
                        CDK_PREFIX,
                        names_to_create_public,
                        cidrs_to_create_public,
                        azs_to_create_public,
                        is_public=True,
                        internet_gateway_id=EXISTING_IGW_ID,
                    )
                )
                self.public_subnets.extend(newly_created_public_subnets)
                self.public_route_tables_cfn.extend(newly_created_public_rts_cfn)

        if (
            not self.public_subnets
            and not names_to_create_public
            and not PUBLIC_SUBNETS_TO_USE
        ):
            raise Exception("No public subnets found or created, exiting.")

        # --- NAT Gateway Creation/Lookup ---
        print("Creating NAT gateway/located existing")
        self.single_nat_gateway_id = None

        nat_gw_id_from_context = SINGLE_NAT_GATEWAY_ID

        if nat_gw_id_from_context:
            print(
                f"Using existing NAT Gateway ID from context: {nat_gw_id_from_context}"
            )
            self.single_nat_gateway_id = nat_gw_id_from_context

        elif (
            new_vpc_created
            and new_vpc_nat_gateways > 0
            and hasattr(vpc, "nat_gateways")
            and vpc.nat_gateways
        ):
            self.single_nat_gateway_id = vpc.nat_gateways[0].gateway_id
            print(
                f"Using NAT Gateway {self.single_nat_gateway_id} created by the new VPC construct."
            )

        if not self.single_nat_gateway_id:
            print("Creating a new NAT gateway")

            if hasattr(vpc, "nat_gateways") and vpc.nat_gateways:
                print("Existing NAT gateway found in vpc")
                pass

                # If not in context, create a new one, but only if we have a public subnet.
            elif self.public_subnets:
                print("NAT Gateway ID not found in context. Creating a new one.")
                # Place the NAT GW in the first available public subnet
                first_public_subnet = self.public_subnets[0]

                self.single_nat_gateway_id = create_nat_gateway(
                    self,
                    first_public_subnet,
                    nat_gateway_name=NAT_GATEWAY_NAME,
                    nat_gateway_id_context_key=SINGLE_NAT_GATEWAY_ID,
                )
            else:
                print(
                    "WARNING: No public subnets available and NAT gateway not found in existing VPC. Cannot create a NAT Gateway."
                )

        # --- 4. Process Private Subnets ---
        print("\n--- Processing Private Subnets ---")
        # ... (rest of your existing subnet processing logic for checked_private_subnets_ctx) ...
        # (This part for importing existing subnets remains the same)

        # Create new private subnets
        if private_subnets_data_for_creation_ctx:
            names_to_create_private = [
                s["name"] for s in private_subnets_data_for_creation_ctx
            ]
            cidrs_to_create_private = [
                s["cidr"] for s in private_subnets_data_for_creation_ctx
            ]
            azs_to_create_private = [
                s["az"] for s in private_subnets_data_for_creation_ctx
            ]

            if names_to_create_private:
                print(
                    f"Attempting to create {len(names_to_create_private)} new private subnets: {names_to_create_private}"
                )
                # --- CALL THE NEW CREATE_SUBNETS FUNCTION FOR PRIVATE ---
                # Ensure self.single_nat_gateway_id is available before this call
                if not self.single_nat_gateway_id:
                    raise ValueError(
                        "A single NAT Gateway ID is required for private subnets but was not resolved."
                    )

                newly_created_private_subnets_cfn, newly_created_private_rts_cfn = (
                    create_subnets(
                        self,
                        vpc,
                        CDK_PREFIX,
                        names_to_create_private,
                        cidrs_to_create_private,
                        azs_to_create_private,
                        is_public=False,
                        single_nat_gateway_id=self.single_nat_gateway_id,  # Pass the single NAT Gateway ID
                    )
                )
                self.private_subnets.extend(newly_created_private_subnets_cfn)
                self.private_route_tables_cfn.extend(newly_created_private_rts_cfn)
                print(
                    f"Successfully defined {len(newly_created_private_subnets_cfn)} new private subnets and their route tables for creation."
                )
        else:
            print(
                "No private subnets specified for creation in context ('private_subnets_to_create')."
            )

        # if not self.private_subnets:
        #     raise Exception("No private subnets found or created, exiting.")

        if (
            not self.private_subnets
            and not names_to_create_private
            and not PRIVATE_SUBNETS_TO_USE
        ):
            # This condition might need adjustment for new VPCs.
            raise Exception("No private subnets found or created, exiting.")

        # --- 5. Sanity Check and Output ---
        # Output the single NAT Gateway ID for verification
        if self.single_nat_gateway_id:
            CfnOutput(
                self,
                "SingleNatGatewayId",
                value=self.single_nat_gateway_id,
                description="ID of the single NAT Gateway resolved or created.",
            )
        elif (
            NEW_VPC_DEFAULT_NAME
            and (self.node.try_get_context("new_vpc_nat_gateways") or 1) > 0
        ):
            print(
                "INFO: A new VPC was created with NAT Gateway(s). Their routing is handled by the VPC construct. No single_nat_gateway_id was explicitly set for separate output."
            )
        else:
            out_message = "WARNING: No single NAT Gateway was resolved or created explicitly by the script's logic after VPC setup."
            print(out_message)
            raise Exception(out_message)

        # --- Outputs for other stacks/regions ---
        # These are crucial for cross-stack, cross-region referencing

        self.params = dict()
        self.params["vpc_id"] = vpc.vpc_id
        self.params["private_subnets"] = self.private_subnets
        self.params["private_route_tables"] = self.private_route_tables_cfn
        self.params["public_subnets"] = self.public_subnets
        self.params["public_route_tables"] = self.public_route_tables_cfn

        private_subnet_selection = ec2.SubnetSelection(subnets=self.private_subnets)
        public_subnet_selection = ec2.SubnetSelection(subnets=self.public_subnets)

        for sub in private_subnet_selection.subnets:
            print(
                "private subnet:",
                sub.subnet_id,
                "is in availability zone:",
                sub.availability_zone,
            )

        for sub in public_subnet_selection.subnets:
            print(
                "public subnet:",
                sub.subnet_id,
                "is in availability zone:",
                sub.availability_zone,
            )

        print("Private subnet route tables:", self.private_route_tables_cfn)

        # Add the S3 Gateway Endpoint to the VPC
        if names_to_create_private:
            try:
                s3_gateway_endpoint = vpc.add_gateway_endpoint(
                    "S3GatewayEndpoint",
                    service=ec2.GatewayVpcEndpointAwsService.S3,
                    subnets=[private_subnet_selection],
                )
            except Exception as e:
                print("Could not add S3 gateway endpoint to subnets due to:", e)

            # Output some useful information
            CfnOutput(
                self,
                "VpcIdOutput",
                value=vpc.vpc_id,
                description="The ID of the VPC where the S3 Gateway Endpoint is deployed.",
            )
            CfnOutput(
                self,
                "S3GatewayEndpointService",
                value=s3_gateway_endpoint.vpc_endpoint_id,
                description="The id for the S3 Gateway Endpoint.",
            )  # Specify the S3 service

        # --- IAM Roles ---
        if USE_CUSTOM_KMS_KEY == "1":
            kms_key = kms.Key(
                self,
                "RedactionSharedKmsKey",
                alias=CUSTOM_KMS_KEY_NAME,
                removal_policy=RemovalPolicy.DESTROY,
            )

            custom_sts_kms_policy_dict = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "STSCallerIdentity",
                        "Effect": "Allow",
                        "Action": ["sts:GetCallerIdentity"],
                        "Resource": "*",
                    },
                    {
                        "Sid": "KMSAccess",
                        "Effect": "Allow",
                        "Action": ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"],
                        "Resource": kms_key.key_arn,  # Use key_arn, as it's the full ARN, safer than key_id
                    },
                ],
            }
        else:
            kms_key = None

            custom_sts_kms_policy_dict = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "STSCallerIdentity",
                        "Effect": "Allow",
                        "Action": ["sts:GetCallerIdentity"],
                        "Resource": "*",
                    },
                    {
                        "Sid": "KMSSecretsManagerDecrypt",  # Explicitly add decrypt for default key
                        "Effect": "Allow",
                        "Action": ["kms:Decrypt"],
                        "Resource": f"arn:aws:kms:{AWS_REGION}:{AWS_ACCOUNT_ID}:key/aws/secretsmanager",
                    },
                ],
            }
        custom_sts_kms_policy = json.dumps(custom_sts_kms_policy_dict, indent=4)

        try:
            codebuild_role_name = CODEBUILD_ROLE_NAME

            if get_context_bool(f"exists:{codebuild_role_name}"):
                # If exists, lookup/import the role using ARN from context
                role_arn = get_context_str(f"arn:{codebuild_role_name}")
                if not role_arn:
                    raise ValueError(
                        f"Context value 'arn:{codebuild_role_name}' is required if role exists."
                    )
                codebuild_role = iam.Role.from_role_arn(
                    self, "CodeBuildRole", role_arn=role_arn
                )
                print("Using existing CodeBuild role")
            else:
                # If not exists, create the role
                codebuild_role = iam.Role(
                    self,
                    "CodeBuildRole",  # Logical ID
                    role_name=codebuild_role_name,  # Explicit resource name
                    assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
                )
                codebuild_role.add_managed_policy(
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "EC2InstanceProfileForImageBuilderECRContainerBuilds"
                    )
                )
                print("Successfully created new CodeBuild role")

            task_role_name = ECS_TASK_ROLE_NAME
            if get_context_bool(f"exists:{task_role_name}"):
                role_arn = get_context_str(f"arn:{task_role_name}")
                if not role_arn:
                    raise ValueError(
                        f"Context value 'arn:{task_role_name}' is required if role exists."
                    )
                task_role = iam.Role.from_role_arn(self, "TaskRole", role_arn=role_arn)
                print("Using existing ECS task role")
            else:
                task_role = iam.Role(
                    self,
                    "TaskRole",  # Logical ID
                    role_name=task_role_name,  # Explicit resource name
                    assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
                )
                for role in AWS_MANAGED_TASK_ROLES_LIST:
                    print(f"Adding {role} to policy")
                    task_role.add_managed_policy(
                        iam.ManagedPolicy.from_aws_managed_policy_name(f"{role}")
                    )
                task_role = add_custom_policies(
                    self, task_role, custom_policy_text=custom_sts_kms_policy
                )
                print("Successfully created new ECS task role")

            execution_role_name = ECS_TASK_EXECUTION_ROLE_NAME
            if get_context_bool(f"exists:{execution_role_name}"):
                role_arn = get_context_str(f"arn:{execution_role_name}")
                if not role_arn:
                    raise ValueError(
                        f"Context value 'arn:{execution_role_name}' is required if role exists."
                    )
                execution_role = iam.Role.from_role_arn(
                    self, "ExecutionRole", role_arn=role_arn
                )
                print("Using existing ECS execution role")
            else:
                execution_role = iam.Role(
                    self,
                    "ExecutionRole",  # Logical ID
                    role_name=execution_role_name,  # Explicit resource name
                    assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
                )
                for role in AWS_MANAGED_TASK_ROLES_LIST:
                    execution_role.add_managed_policy(
                        iam.ManagedPolicy.from_aws_managed_policy_name(f"{role}")
                    )
                execution_role = add_custom_policies(
                    self, execution_role, custom_policy_text=custom_sts_kms_policy
                )
                print("Successfully created new ECS execution role")

        except Exception as e:
            raise Exception("Failed at IAM role step due to:", e)

        # --- S3 Buckets ---
        try:
            log_bucket_name = S3_LOG_CONFIG_BUCKET_NAME
            if get_context_bool(f"exists:{log_bucket_name}"):
                bucket = s3.Bucket.from_bucket_name(
                    self, "LogConfigBucket", bucket_name=log_bucket_name
                )
                print("Using existing S3 bucket", log_bucket_name)
            else:
                if USE_CUSTOM_KMS_KEY == "1" and isinstance(kms_key, kms.Key):
                    bucket = s3.Bucket(
                        self,
                        "LogConfigBucket",
                        bucket_name=log_bucket_name,
                        versioned=False,
                        removal_policy=RemovalPolicy.DESTROY,
                        auto_delete_objects=True,
                        encryption=s3.BucketEncryption.KMS,
                        encryption_key=kms_key,
                    )
                else:
                    bucket = s3.Bucket(
                        self,
                        "LogConfigBucket",
                        bucket_name=log_bucket_name,
                        versioned=False,
                        removal_policy=RemovalPolicy.DESTROY,
                        auto_delete_objects=True,
                    )

                print("Created S3 bucket", log_bucket_name)

            # Add policies - this will apply to both created and imported buckets
            # CDK handles idempotent policy additions
            bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role],  # Pass the role object directly
                    actions=["s3:GetObject", "s3:PutObject"],
                    resources=[f"{bucket.bucket_arn}/*"],
                )
            )
            bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role],
                    actions=["s3:ListBucket"],
                    resources=[bucket.bucket_arn],
                )
            )

            output_bucket_name = S3_OUTPUT_BUCKET_NAME
            if get_context_bool(f"exists:{output_bucket_name}"):
                output_bucket = s3.Bucket.from_bucket_name(
                    self, "OutputBucket", bucket_name=output_bucket_name
                )
                print("Using existing Output bucket", output_bucket_name)
            else:
                if USE_CUSTOM_KMS_KEY == "1" and isinstance(kms_key, kms.Key):
                    output_bucket = s3.Bucket(
                        self,
                        "OutputBucket",
                        bucket_name=output_bucket_name,
                        lifecycle_rules=[
                            s3.LifecycleRule(
                                expiration=Duration.days(
                                    int(DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS)
                                )
                            )
                        ],
                        versioned=False,
                        removal_policy=RemovalPolicy.DESTROY,
                        auto_delete_objects=True,
                        encryption=s3.BucketEncryption.KMS,
                        encryption_key=kms_key,
                    )
                else:
                    output_bucket = s3.Bucket(
                        self,
                        "OutputBucket",
                        bucket_name=output_bucket_name,
                        lifecycle_rules=[
                            s3.LifecycleRule(
                                expiration=Duration.days(
                                    int(DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS)
                                )
                            )
                        ],
                        versioned=False,
                        removal_policy=RemovalPolicy.DESTROY,
                        auto_delete_objects=True,
                    )

                print("Created Output bucket:", output_bucket_name)

            # Add policies to output bucket
            output_bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role],
                    actions=["s3:GetObject", "s3:PutObject"],
                    resources=[f"{output_bucket.bucket_arn}/*"],
                )
            )
            output_bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role],
                    actions=["s3:ListBucket"],
                    resources=[output_bucket.bucket_arn],
                )
            )

        except Exception as e:
            raise Exception("Could not handle S3 buckets due to:", e)

        # --- Elastic Container Registry ---
        try:
            full_ecr_repo_name = ECR_CDK_REPO_NAME
            if get_context_bool(f"exists:{full_ecr_repo_name}"):
                ecr_repo = ecr.Repository.from_repository_name(
                    self, "ECRRepo", repository_name=full_ecr_repo_name
                )
                print("Using existing ECR repository")
            else:
                ecr_repo = ecr.Repository(
                    self, "ECRRepo", repository_name=full_ecr_repo_name
                )  # Explicitly set repository_name
                print("Created ECR repository", full_ecr_repo_name)

            ecr_image_loc = ecr_repo.repository_uri
        except Exception as e:
            raise Exception("Could not handle ECR repo due to:", e)

        # --- CODEBUILD ---
        try:
            codebuild_project_name = CODEBUILD_PROJECT_NAME
            if get_context_bool(f"exists:{codebuild_project_name}"):
                # Lookup CodeBuild project by ARN from context
                project_arn = get_context_str(f"arn:{codebuild_project_name}")
                if not project_arn:
                    raise ValueError(
                        f"Context value 'arn:{codebuild_project_name}' is required if project exists."
                    )
                codebuild_project = codebuild.Project.from_project_arn(
                    self, "CodeBuildProject", project_arn=project_arn
                )
                print("Using existing CodeBuild project")
            else:
                codebuild_project = codebuild.Project(
                    self,
                    "CodeBuildProject",  # Logical ID
                    project_name=codebuild_project_name,  # Explicit resource name
                    source=codebuild.Source.git_hub(
                        owner=GITHUB_REPO_USERNAME,
                        repo=GITHUB_REPO_NAME,
                        branch_or_ref=GITHUB_REPO_BRANCH,
                    ),
                    environment=codebuild.BuildEnvironment(
                        build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                        privileged=True,
                        environment_variables={
                            "ECR_REPO_NAME": codebuild.BuildEnvironmentVariable(
                                value=full_ecr_repo_name
                            ),
                            "AWS_DEFAULT_REGION": codebuild.BuildEnvironmentVariable(
                                value=AWS_REGION
                            ),
                            "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(
                                value=AWS_ACCOUNT_ID
                            ),
                        },
                    ),
                    build_spec=codebuild.BuildSpec.from_object(
                        {
                            "version": "0.2",
                            "phases": {
                                "pre_build": {
                                    "commands": [
                                        "echo Logging in to Amazon ECR",
                                        "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com",
                                    ]
                                },
                                "build": {
                                    "commands": [
                                        "echo Building the Docker image",
                                        "docker build -t $ECR_REPO_NAME:latest .",
                                        "docker tag $ECR_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_NAME:latest",
                                    ]
                                },
                                "post_build": {
                                    "commands": [
                                        "echo Pushing the Docker image",
                                        "docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_NAME:latest",
                                    ]
                                },
                            },
                        }
                    ),
                )
                print("Successfully created CodeBuild project", codebuild_project_name)

            # Grant permissions - applies to both created and imported project role
            ecr_repo.grant_pull_push(codebuild_project.role)

        except Exception as e:
            raise Exception("Could not handle Codebuild project due to:", e)

        # --- Security Groups ---
        try:
            ecs_security_group_name = ECS_SECURITY_GROUP_NAME

            try:
                ecs_security_group = ec2.SecurityGroup(
                    self,
                    "ECSSecurityGroup",  # Logical ID
                    security_group_name=ecs_security_group_name,  # Explicit resource name
                    vpc=vpc,
                )
                print(f"Created Security Group: {ecs_security_group_name}")
            except Exception as e:  # If lookup fails, create
                print("Failed to create ECS security group due to:", e)

            alb_security_group_name = ALB_NAME_SECURITY_GROUP_NAME

            try:
                alb_security_group = ec2.SecurityGroup(
                    self,
                    "ALBSecurityGroup",  # Logical ID
                    security_group_name=alb_security_group_name,  # Explicit resource name
                    vpc=vpc,
                )
                print(f"Created Security Group: {alb_security_group_name}")
            except Exception as e:  # If lookup fails, create
                print("Failed to create ALB security group due to:", e)

            # Define Ingress Rules - CDK will manage adding/removing these as needed
            ec2_port_gradio_server_port = ec2.Port.tcp(
                int(GRADIO_SERVER_PORT)
            )  # Ensure port is int
            ecs_security_group.add_ingress_rule(
                peer=alb_security_group,
                connection=ec2_port_gradio_server_port,
                description="ALB traffic",
            )

            alb_security_group.add_ingress_rule(
                peer=ec2.Peer.prefix_list("pl-93a247fa"),
                connection=ec2.Port.all_traffic(),
                description="CloudFront traffic",
            )

        except Exception as e:
            raise Exception("Could not handle security groups due to:", e)

        # --- DynamoDB tables for logs (optional) ---

        if SAVE_LOGS_TO_DYNAMODB == "True":
            try:
                print("Creating DynamoDB tables for logs")

                dynamodb.Table(
                    self,
                    "RedactionAccessDataTable",
                    table_name=ACCESS_LOG_DYNAMODB_TABLE_NAME,
                    partition_key=dynamodb.Attribute(
                        name="id", type=dynamodb.AttributeType.STRING
                    ),
                    billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                    removal_policy=RemovalPolicy.DESTROY,
                )

                dynamodb.Table(
                    self,
                    "RedactionFeedbackDataTable",
                    table_name=FEEDBACK_LOG_DYNAMODB_TABLE_NAME,
                    partition_key=dynamodb.Attribute(
                        name="id", type=dynamodb.AttributeType.STRING
                    ),
                    billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                    removal_policy=RemovalPolicy.DESTROY,
                )

                dynamodb.Table(
                    self,
                    "RedactionUsageDataTable",
                    table_name=USAGE_LOG_DYNAMODB_TABLE_NAME,
                    partition_key=dynamodb.Attribute(
                        name="id", type=dynamodb.AttributeType.STRING
                    ),
                    billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                    removal_policy=RemovalPolicy.DESTROY,
                )

            except Exception as e:
                raise Exception("Could not create DynamoDB tables due to:", e)

        # --- ALB ---
        try:
            load_balancer_name = ALB_NAME
            if len(load_balancer_name) > 32:
                load_balancer_name = load_balancer_name[-32:]
            if get_context_bool(f"exists:{load_balancer_name}"):
                # Lookup ALB by ARN from context
                alb_arn = get_context_str(f"arn:{load_balancer_name}")
                if not alb_arn:
                    raise ValueError(
                        f"Context value 'arn:{load_balancer_name}' is required if ALB exists."
                    )
                alb = elbv2.ApplicationLoadBalancer.from_lookup(
                    self, "ALB", load_balancer_arn=alb_arn  # Logical ID
                )
                print(f"Using existing Application Load Balancer {load_balancer_name}.")
            else:
                alb = elbv2.ApplicationLoadBalancer(
                    self,
                    "ALB",  # Logical ID
                    load_balancer_name=load_balancer_name,  # Explicit resource name
                    vpc=vpc,
                    internet_facing=True,
                    security_group=alb_security_group,  # Link to SG
                    vpc_subnets=public_subnet_selection,  # Link to subnets
                )
                print("Successfully created new Application Load Balancer")
        except Exception as e:
            raise Exception("Could not handle application load balancer due to:", e)

        # --- Cognito User Pool ---
        try:
            if get_context_bool(f"exists:{COGNITO_USER_POOL_NAME}"):
                # Lookup by ID from context
                user_pool_id = get_context_str(f"id:{COGNITO_USER_POOL_NAME}")
                if not user_pool_id:
                    raise ValueError(
                        f"Context value 'id:{COGNITO_USER_POOL_NAME}' is required if User Pool exists."
                    )
                user_pool = cognito.UserPool.from_user_pool_id(
                    self, "UserPool", user_pool_id=user_pool_id
                )
                print(f"Using existing user pool {user_pool_id}.")
            else:
                user_pool = cognito.UserPool(
                    self,
                    "UserPool",
                    user_pool_name=COGNITO_USER_POOL_NAME,
                    mfa=cognito.Mfa.OFF,  # Adjust as needed
                    sign_in_aliases=cognito.SignInAliases(email=True),
                    removal_policy=RemovalPolicy.DESTROY,
                )  # Adjust as needed
                print(f"Created new user pool {user_pool.user_pool_id}.")

            # If you're using a certificate, assume that you will be using the ALB Cognito login features. You need different redirect URLs to accept the token that comes from Cognito authentication.
            if ACM_SSL_CERTIFICATE_ARN:
                redirect_uris = [
                    COGNITO_REDIRECTION_URL,
                    COGNITO_REDIRECTION_URL + "/oauth2/idpresponse",
                ]
            else:
                redirect_uris = [COGNITO_REDIRECTION_URL]

            user_pool_client_name = COGNITO_USER_POOL_CLIENT_NAME
            if get_context_bool(f"exists:{user_pool_client_name}"):
                # Lookup by ID from context (requires User Pool object)
                user_pool_client_id = get_context_str(f"id:{user_pool_client_name}")
                if not user_pool_client_id:
                    raise ValueError(
                        f"Context value 'id:{user_pool_client_name}' is required if User Pool Client exists."
                    )
                user_pool_client = cognito.UserPoolClient.from_user_pool_client_id(
                    self, "UserPoolClient", user_pool_client_id=user_pool_client_id
                )
                print(f"Using existing user pool client {user_pool_client_id}.")
            else:
                user_pool_client = cognito.UserPoolClient(
                    self,
                    "UserPoolClient",
                    auth_flows=cognito.AuthFlow(
                        user_srp=True, user_password=True
                    ),  # Example: enable SRP for secure sign-in
                    user_pool=user_pool,
                    generate_secret=True,
                    user_pool_client_name=user_pool_client_name,
                    supported_identity_providers=[
                        cognito.UserPoolClientIdentityProvider.COGNITO
                    ],
                    o_auth=cognito.OAuthSettings(
                        flows=cognito.OAuthFlows(authorization_code_grant=True),
                        scopes=[
                            cognito.OAuthScope.OPENID,
                            cognito.OAuthScope.EMAIL,
                            cognito.OAuthScope.PROFILE,
                        ],
                        callback_urls=redirect_uris,
                    ),
                )

            CfnOutput(
                self, "CognitoAppClientId", value=user_pool_client.user_pool_client_id
            )

            print(
                f"Created new user pool client {user_pool_client.user_pool_client_id}."
            )

            # Add a domain to the User Pool (crucial for ALB integration)
            user_pool_domain = user_pool.add_domain(
                "UserPoolDomain",
                cognito_domain=cognito.CognitoDomainOptions(
                    domain_prefix=COGNITO_USER_POOL_DOMAIN_PREFIX
                ),
            )

            # Apply removal_policy to the created UserPoolDomain construct
            user_pool_domain.apply_removal_policy(policy=RemovalPolicy.DESTROY)

            CfnOutput(
                self, "CognitoUserPoolLoginUrl", value=user_pool_domain.base_url()
            )

        except Exception as e:
            raise Exception("Could not handle Cognito resources due to:", e)

        # --- Secrets Manager Secret ---
        try:
            secret_name = COGNITO_USER_POOL_CLIENT_SECRET_NAME
            if get_context_bool(f"exists:{secret_name}"):
                # Lookup by name
                secret = secretsmanager.Secret.from_secret_name_v2(
                    self, "CognitoSecret", secret_name=secret_name
                )
                print("Using existing Secret.")
            else:
                if USE_CUSTOM_KMS_KEY == "1" and isinstance(kms_key, kms.Key):
                    secret = secretsmanager.Secret(
                        self,
                        "CognitoSecret",  # Logical ID
                        secret_name=secret_name,  # Explicit resource name
                        secret_object_value={
                            "REDACTION_USER_POOL_ID": SecretValue.unsafe_plain_text(
                                user_pool.user_pool_id
                            ),  # Use the CDK attribute
                            "REDACTION_CLIENT_ID": SecretValue.unsafe_plain_text(
                                user_pool_client.user_pool_client_id
                            ),  # Use the CDK attribute
                            "REDACTION_CLIENT_SECRET": user_pool_client.user_pool_client_secret,  # Use the CDK attribute
                        },
                        encryption_key=kms_key,
                    )
                else:
                    secret = secretsmanager.Secret(
                        self,
                        "CognitoSecret",  # Logical ID
                        secret_name=secret_name,  # Explicit resource name
                        secret_object_value={
                            "REDACTION_USER_POOL_ID": SecretValue.unsafe_plain_text(
                                user_pool.user_pool_id
                            ),  # Use the CDK attribute
                            "REDACTION_CLIENT_ID": SecretValue.unsafe_plain_text(
                                user_pool_client.user_pool_client_id
                            ),  # Use the CDK attribute
                            "REDACTION_CLIENT_SECRET": user_pool_client.user_pool_client_secret,  # Use the CDK attribute
                        },
                    )

                print(
                    "Created new secret in Secrets Manager for Cognito user pool and related details."
                )

        except Exception as e:
            raise Exception("Could not handle Secrets Manager secret due to:", e)

        # --- Fargate Task Definition ---
        try:
            fargate_task_definition_name = FARGATE_TASK_DEFINITION_NAME

            read_only_file_system = ECS_READ_ONLY_FILE_SYSTEM == "True"

            if os.path.exists(TASK_DEFINITION_FILE_LOCATION):
                with open(TASK_DEFINITION_FILE_LOCATION) as f:  # Use correct path
                    task_def_params = json.load(f)
                # Need to ensure taskRoleArn and executionRoleArn in JSON are correct ARN strings
            else:
                epheremal_storage_volume_name = "appEphemeralVolume"

                task_def_params = {}
                task_def_params["taskRoleArn"] = (
                    task_role.role_arn
                )  # Use CDK role object ARN
                task_def_params["executionRoleArn"] = (
                    execution_role.role_arn
                )  # Use CDK role object ARN
                task_def_params["memory"] = ECS_TASK_MEMORY_SIZE
                task_def_params["cpu"] = ECS_TASK_CPU_SIZE
                container_def = {
                    "name": full_ecr_repo_name,
                    "image": ecr_image_loc + ":latest",
                    "essential": True,
                    "portMappings": [
                        {
                            "containerPort": int(GRADIO_SERVER_PORT),
                            "hostPort": int(GRADIO_SERVER_PORT),
                            "protocol": "tcp",
                            "appProtocol": "http",
                        }
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": ECS_LOG_GROUP_NAME,
                            "awslogs-region": AWS_REGION,
                            "awslogs-stream-prefix": "ecs",
                        },
                    },
                    "environmentFiles": [
                        {"value": bucket.bucket_arn + "/config.env", "type": "s3"}
                    ],
                    "memoryReservation": int(task_def_params["memory"])
                    - 512,  # Reserve some memory for the container
                    "mountPoints": [
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/app/logs",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/app/feedback",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/app/usage",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/app/input",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/app/output",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/app/tmp",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/app/config",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/tmp/matplotlib_cache",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/tmp",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/var/tmp",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/tmp/tld",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/tmp/gradio_tmp",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/.paddlex",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/home/user/.local/share/spacy/data",
                            "readOnly": False,
                        },
                        {
                            "sourceVolume": epheremal_storage_volume_name,
                            "containerPath": "/usr/share/tessdata",
                            "readOnly": False,
                        },
                    ],
                    "readonlyRootFilesystem": read_only_file_system,
                }
                task_def_params["containerDefinitions"] = [container_def]

            log_group_name_from_config = task_def_params["containerDefinitions"][0][
                "logConfiguration"
            ]["options"]["awslogs-group"]

            cdk_managed_log_group = logs.LogGroup(
                self,
                "MyTaskLogGroup",  # CDK Logical ID
                log_group_name=log_group_name_from_config,
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.DESTROY,
            )

            epheremal_storage_volume_cdk_obj = ecs.Volume(
                name=epheremal_storage_volume_name
            )

            fargate_task_definition = ecs.FargateTaskDefinition(
                self,
                "FargateTaskDefinition",  # Logical ID
                family=fargate_task_definition_name,
                cpu=int(task_def_params["cpu"]),
                memory_limit_mib=int(task_def_params["memory"]),
                task_role=task_role,
                execution_role=execution_role,
                runtime_platform=ecs.RuntimePlatform(
                    cpu_architecture=ecs.CpuArchitecture.X86_64,
                    operating_system_family=ecs.OperatingSystemFamily.LINUX,
                ),
                ephemeral_storage_gib=21,  # Minimum is 21 GiB
                volumes=[epheremal_storage_volume_cdk_obj],
            )
            print("Fargate task definition defined.")

            # Add container definitions to the task definition object
            if task_def_params["containerDefinitions"]:
                container_def_params = task_def_params["containerDefinitions"][0]

                if container_def_params.get("environmentFiles"):
                    env_files = []
                    for env_file_param in container_def_params["environmentFiles"]:
                        # Need to parse the ARN to get the bucket object and key
                        env_file_arn_parts = env_file_param["value"].split(":::")
                        bucket_name_and_key = env_file_arn_parts[-1]
                        env_bucket_name, env_key = bucket_name_and_key.split("/", 1)

                        env_file = ecs.EnvironmentFile.from_bucket(bucket, env_key)

                        env_files.append(env_file)

                container = fargate_task_definition.add_container(
                    container_def_params["name"],
                    image=ecs.ContainerImage.from_registry(
                        container_def_params["image"]
                    ),
                    logging=ecs.LogDriver.aws_logs(
                        stream_prefix=container_def_params["logConfiguration"][
                            "options"
                        ]["awslogs-stream-prefix"],
                        log_group=cdk_managed_log_group,
                    ),
                    secrets={
                        "AWS_USER_POOL_ID": ecs.Secret.from_secrets_manager(
                            secret, "REDACTION_USER_POOL_ID"
                        ),
                        "AWS_CLIENT_ID": ecs.Secret.from_secrets_manager(
                            secret, "REDACTION_CLIENT_ID"
                        ),
                        "AWS_CLIENT_SECRET": ecs.Secret.from_secrets_manager(
                            secret, "REDACTION_CLIENT_SECRET"
                        ),
                    },
                    environment_files=env_files,
                    readonly_root_filesystem=read_only_file_system,
                )

                for port_mapping in container_def_params["portMappings"]:
                    container.add_port_mappings(
                        ecs.PortMapping(
                            container_port=int(port_mapping["containerPort"]),
                            host_port=int(port_mapping["hostPort"]),
                            name="port-" + str(port_mapping["containerPort"]),
                            app_protocol=ecs.AppProtocol.http,
                            protocol=ecs.Protocol.TCP,
                        )
                    )

                container.add_port_mappings(
                    ecs.PortMapping(
                        container_port=80,
                        host_port=80,
                        name="port-80",
                        app_protocol=ecs.AppProtocol.http,
                        protocol=ecs.Protocol.TCP,
                    )
                )

                if container_def_params.get("mountPoints"):
                    mount_points = []
                    for mount_point in container_def_params["mountPoints"]:
                        mount_points.append(
                            ecs.MountPoint(
                                container_path=mount_point["containerPath"],
                                read_only=mount_point["readOnly"],
                                source_volume=epheremal_storage_volume_name,
                            )
                        )
                    container.add_mount_points(*mount_points)

        except Exception as e:
            raise Exception("Could not handle Fargate task definition due to:", e)

        # --- ECS Cluster ---
        try:
            cluster = ecs.Cluster(
                self,
                "ECSCluster",  # Logical ID
                cluster_name=CLUSTER_NAME,  # Explicit resource name
                enable_fargate_capacity_providers=True,
                vpc=vpc,
            )
            print("Successfully created new ECS cluster")
        except Exception as e:
            raise Exception("Could not handle ECS cluster due to:", e)

        # --- ECS Service ---
        try:
            ecs_service_name = ECS_SERVICE_NAME

            if ECS_USE_FARGATE_SPOT == "True":
                use_fargate_spot = "FARGATE_SPOT"
            if ECS_USE_FARGATE_SPOT == "False":
                use_fargate_spot = "FARGATE"

            # Check if service exists - from_service_arn or from_service_name (needs cluster)
            try:
                # from_service_name is useful if you have the cluster object
                ecs_service = ecs.FargateService.from_service_attributes(
                    self,
                    "ECSService",  # Logical ID
                    cluster=cluster,  # Requires the cluster object
                    service_name=ecs_service_name,
                )
                print(f"Using existing ECS service {ecs_service_name}.")
            except Exception:
                # Service will be created with a count of 0, because you haven't yet actually built the initial Docker container with CodeBuild
                ecs_service = ecs.FargateService(
                    self,
                    "ECSService",  # Logical ID
                    service_name=ecs_service_name,  # Explicit resource name
                    platform_version=ecs.FargatePlatformVersion.LATEST,
                    capacity_provider_strategies=[
                        ecs.CapacityProviderStrategy(
                            capacity_provider=use_fargate_spot, base=0, weight=1
                        )
                    ],
                    cluster=cluster,
                    task_definition=fargate_task_definition,  # Link to TD
                    security_groups=[ecs_security_group],  # Link to SG
                    vpc_subnets=ec2.SubnetSelection(
                        subnets=self.private_subnets
                    ),  # Link to subnets
                    min_healthy_percent=0,
                    max_healthy_percent=100,
                    desired_count=0,
                )
                print("Successfully created new ECS service")

            # Note: Auto-scaling setup would typically go here if needed for the service

        except Exception as e:
            raise Exception("Could not handle ECS service due to:", e)

        # --- Grant Secret Read Access (Applies to both created and imported roles) ---
        try:
            secret.grant_read(task_role)
            secret.grant_read(execution_role)
        except Exception as e:
            raise Exception("Could not grant access to Secrets Manager due to:", e)

        # --- ALB TARGET GROUPS AND LISTENERS ---
        # This section should primarily define the resources if they are managed by this stack.
        # CDK handles adding/removing targets and actions on updates.
        # If they might pre-exist outside the stack, you need lookups.
        cookie_duration = Duration.hours(12)
        target_group_name = ALB_TARGET_GROUP_NAME  # Explicit resource name
        cloudfront_distribution_url = "cloudfront_placeholder.net"  # Need to replace this afterwards with the actual cloudfront_distribution.domain_name

        try:
            # --- CREATING TARGET GROUPS AND ADDING THE CLOUDFRONT LISTENER RULE ---

            target_group = elbv2.ApplicationTargetGroup(
                self,
                "AppTargetGroup",  # Logical ID
                target_group_name=target_group_name,  # Explicit resource name
                port=int(GRADIO_SERVER_PORT),  # Ensure port is int
                protocol=elbv2.ApplicationProtocol.HTTP,
                targets=[ecs_service],  # Link to ECS Service
                stickiness_cookie_duration=cookie_duration,
                vpc=vpc,  # Target Groups need VPC
            )
            print(f"ALB target group {target_group_name} defined.")

            # First HTTP
            listener_port = 80
            # Check if Listener exists - from_listener_arn or lookup by port/ALB

            http_listener = alb.add_listener(
                "HttpListener",  # Logical ID
                port=listener_port,
                open=False,  # Be cautious with open=True, usually restrict source SG
            )
            print(f"ALB listener on port {listener_port} defined.")

            if ACM_SSL_CERTIFICATE_ARN:
                http_listener.add_action(
                    "DefaultAction",  # Logical ID for the default action
                    action=elbv2.ListenerAction.redirect(
                        protocol="HTTPS",
                        host="#{host}",
                        port="443",
                        path="/#{path}",
                        query="#{query}",
                    ),
                )
            else:
                if USE_CLOUDFRONT == "True":

                    # The following default action can be added for the listener after a host header rule is added to the listener manually in the Console as suggested in the above comments.
                    http_listener.add_action(
                        "DefaultAction",  # Logical ID for the default action
                        action=elbv2.ListenerAction.fixed_response(
                            status_code=403,
                            content_type="text/plain",
                            message_body="Access denied",
                        ),
                    )

                    # Add the Listener Rule for the specific CloudFront Host Header
                    http_listener.add_action(
                        "CloudFrontHostHeaderRule",
                        action=elbv2.ListenerAction.forward(
                            target_groups=[target_group],
                            stickiness_duration=cookie_duration,
                        ),
                        priority=1,  # Example priority. Adjust as needed. Lower is evaluated first.
                        conditions=[
                            elbv2.ListenerCondition.host_headers(
                                [cloudfront_distribution_url]
                            )  # May have to redefine url in console afterwards if not specified in config file
                        ],
                    )

                else:
                    # Add the Listener Rule for the specific CloudFront Host Header
                    http_listener.add_action(
                        "CloudFrontHostHeaderRule",
                        action=elbv2.ListenerAction.forward(
                            target_groups=[target_group],
                            stickiness_duration=cookie_duration,
                        ),
                    )

                print("Added targets and actions to ALB HTTP listener.")

            # Now the same for HTTPS if you have an ACM certificate
            if ACM_SSL_CERTIFICATE_ARN:
                listener_port_https = 443
                # Check if Listener exists - from_listener_arn or lookup by port/ALB

                https_listener = add_alb_https_listener_with_cert(
                    self,
                    "MyHttpsListener",  # Logical ID for the HTTPS listener
                    alb,
                    acm_certificate_arn=ACM_SSL_CERTIFICATE_ARN,
                    default_target_group=target_group,
                    enable_cognito_auth=True,
                    cognito_user_pool=user_pool,
                    cognito_user_pool_client=user_pool_client,
                    cognito_user_pool_domain=user_pool_domain,
                    listener_open_to_internet=True,
                    stickiness_cookie_duration=cookie_duration,
                )

                if https_listener:
                    CfnOutput(
                        self, "HttpsListenerArn", value=https_listener.listener_arn
                    )

                print(f"ALB listener on port {listener_port_https} defined.")

                # if USE_CLOUDFRONT == 'True':
                #     # Add default action to the listener
                #     https_listener.add_action(
                #         "DefaultAction", # Logical ID for the default action
                #         action=elbv2.ListenerAction.fixed_response(
                #             status_code=403,
                #             content_type="text/plain",
                #             message_body="Access denied",
                #         ),
                #     )

                #     # Add the Listener Rule for the specific CloudFront Host Header
                #     https_listener.add_action(
                #         "CloudFrontHostHeaderRuleHTTPS",
                #         action=elbv2.ListenerAction.forward(target_groups=[target_group],stickiness_duration=cookie_duration),
                #         priority=1, # Example priority. Adjust as needed. Lower is evaluated first.
                #         conditions=[
                #             elbv2.ListenerCondition.host_headers([cloudfront_distribution_url])
                #         ]
                #     )
                # else:
                #     https_listener.add_action(
                #         "CloudFrontHostHeaderRuleHTTPS",
                #         action=elbv2.ListenerAction.forward(target_groups=[target_group],stickiness_duration=cookie_duration))

                print("Added targets and actions to ALB HTTPS listener.")

        except Exception as e:
            raise Exception(
                "Could not handle ALB target groups and listeners due to:", e
            )

        # Create WAF to attach to load balancer
        try:
            web_acl_name = LOAD_BALANCER_WEB_ACL_NAME
            if get_context_bool(f"exists:{web_acl_name}"):
                # Lookup WAF ACL by ARN from context
                web_acl_arn = get_context_str(f"arn:{web_acl_name}")
                if not web_acl_arn:
                    raise ValueError(
                        f"Context value 'arn:{web_acl_name}' is required if Web ACL exists."
                    )

                web_acl = create_web_acl_with_common_rules(
                    self, web_acl_name, waf_scope="REGIONAL"
                )  # Assuming it takes scope and name
                print(f"Handled ALB WAF web ACL {web_acl_name}.")
            else:
                web_acl = create_web_acl_with_common_rules(
                    self, web_acl_name, waf_scope="REGIONAL"
                )  # Assuming it takes scope and name
                print(f"Created ALB WAF web ACL {web_acl_name}.")

            wafv2.CfnWebACLAssociation(
                self,
                id="alb_waf_association",
                resource_arn=alb.load_balancer_arn,
                web_acl_arn=web_acl.attr_arn,
            )

        except Exception as e:
            raise Exception("Could not handle create ALB WAF web ACL due to:", e)

        # --- Outputs for other stacks/regions ---

        self.params = dict()
        self.params["alb_arn_output"] = alb.load_balancer_arn
        self.params["alb_security_group_id"] = alb_security_group.security_group_id
        self.params["alb_dns_name"] = alb.load_balancer_dns_name

        CfnOutput(
            self,
            "AlbArnOutput",
            value=alb.load_balancer_arn,
            description="ARN of the Application Load Balancer",
            export_name=f"{self.stack_name}-AlbArn",
        )  # Export name must be unique within the account/region

        CfnOutput(
            self,
            "AlbSecurityGroupIdOutput",
            value=alb_security_group.security_group_id,
            description="ID of the ALB's Security Group",
            export_name=f"{self.stack_name}-AlbSgId",
        )
        CfnOutput(self, "ALBName", value=alb.load_balancer_name)

        CfnOutput(self, "RegionalAlbDnsName", value=alb.load_balancer_dns_name)

        CfnOutput(self, "CognitoPoolId", value=user_pool.user_pool_id)
        # Add other outputs if needed

        CfnOutput(self, "ECRRepoUri", value=ecr_repo.repository_uri)


# --- CLOUDFRONT DISTRIBUTION in separate stack (us-east-1 required) ---
class CdkStackCloudfront(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        alb_arn: str,
        alb_sec_group_id: str,
        alb_dns_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- Helper to get context values ---
        def get_context_bool(key: str, default: bool = False) -> bool:
            return self.node.try_get_context(key) or default

        def get_context_str(key: str, default: str = None) -> str:
            return self.node.try_get_context(key) or default

        def get_context_dict(scope: Construct, key: str, default: dict = None) -> dict:
            return scope.node.try_get_context(key) or default

        print(f"CloudFront Stack: Received ALB ARN: {alb_arn}")
        print(f"CloudFront Stack: Received ALB Security Group ID: {alb_sec_group_id}")

        if not alb_arn:
            raise ValueError("ALB ARN must be provided to CloudFront stack")
        if not alb_sec_group_id:
            raise ValueError(
                "ALB Security Group ID must be provided to CloudFront stack"
            )

        # 2. Import the ALB using its ARN
        # This imports an existing ALB as a construct in the CloudFront stack's context.
        # CloudFormation will understand this reference at deploy time.
        alb = elbv2.ApplicationLoadBalancer.from_application_load_balancer_attributes(
            self,
            "ImportedAlb",
            load_balancer_arn=alb_arn,
            security_group_id=alb_sec_group_id,
            load_balancer_dns_name=alb_dns_name,
        )

        try:
            web_acl_name = WEB_ACL_NAME
            if get_context_bool(f"exists:{web_acl_name}"):
                # Lookup WAF ACL by ARN from context
                web_acl_arn = get_context_str(f"arn:{web_acl_name}")
                if not web_acl_arn:
                    raise ValueError(
                        f"Context value 'arn:{web_acl_name}' is required if Web ACL exists."
                    )

                web_acl = create_web_acl_with_common_rules(
                    self, web_acl_name
                )  # Assuming it takes scope and name
                print(f"Handled Cloudfront WAF web ACL {web_acl_name}.")
            else:
                web_acl = create_web_acl_with_common_rules(
                    self, web_acl_name
                )  # Assuming it takes scope and name
                print(f"Created Cloudfront WAF web ACL {web_acl_name}.")

            # Add ALB as CloudFront Origin
            origin = origins.LoadBalancerV2Origin(
                alb,  # Use the created or looked-up ALB object
                custom_headers={CUSTOM_HEADER: CUSTOM_HEADER_VALUE},
                origin_shield_enabled=False,
                protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
            )

            if CLOUDFRONT_GEO_RESTRICTION:
                geo_restrict = cloudfront.GeoRestriction.allowlist(
                    CLOUDFRONT_GEO_RESTRICTION
                )
            else:
                geo_restrict = None

            cloudfront_distribution = cloudfront.Distribution(
                self,
                "CloudFrontDistribution",  # Logical ID
                comment=CLOUDFRONT_DISTRIBUTION_NAME,  # Use name as comment for easier identification
                geo_restriction=geo_restrict,
                default_behavior=cloudfront.BehaviorOptions(
                    origin=origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER,
                ),
                web_acl_id=web_acl.attr_arn,
            )
            print(f"Cloudfront distribution {CLOUDFRONT_DISTRIBUTION_NAME} defined.")

        except Exception as e:
            raise Exception("Could not handle Cloudfront distribution due to:", e)

        # --- Outputs ---
        CfnOutput(
            self, "CloudFrontDistributionURL", value=cloudfront_distribution.domain_name
        )
