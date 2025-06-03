import os
import json # You might still need json if loading task_definition.json
from typing import List
from datetime import timedelta
from aws_cdk import (
    Stack,
    CfnTag,    # <-- Import CfnTag directly
    CfnOutput, # <-- Import CfnOutput directly
    App, Environment, Duration, RemovalPolicy,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_s3 as s3,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_codebuild as codebuild,
    aws_cognito as cognito,
    aws_secretsmanager as secretsmanager,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_elasticloadbalancingv2 as elbv2,
    aws_logs as logs,
    SecretValue,
    CfnOutput
)

from constructs import Construct
from cdk_config import CDK_PREFIX, SECRETS_MANAGER_ID, VPC_NAME, CLOUDFRONT_PREFIX_LIST_ID, AWS_MANAGED_TASK_ROLES_LIST, GITHUB_REPO_USERNAME, GITHUB_REPO_NAME, GITHUB_REPO_BRANCH, ECR_REPO_NAME, AWS_ACCOUNT_ID, ECS_TASK_MEMORY_SIZE, ECS_TASK_CPU_SIZE, CUSTOM_HEADER, CUSTOM_HEADER_VALUE, AWS_REGION, CLOUDFRONT_GEO_RESTRICTION, DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS, GRADIO_SERVER_PORT, PUBLIC_SUBNETS_TO_USE, PUBLIC_SUBNET_CIDR_BLOCKS, PUBLIC_SUBNET_AVAILABILITY_ZONES, PRIVATE_SUBNETS_TO_USE, PRIVATE_SUBNET_CIDR_BLOCKS, PRIVATE_SUBNET_AVAILABILITY_ZONES, CODEBUILD_PROJECT_NAME, NAT_GATEWAY_EIP_NAME, ECS_SECURITY_GROUP_NAME, ALB_NAME_SECURITY_GROUP_NAME, ALB_NAME, COGNITO_USER_POOL_NAME, COGNITO_USER_POOL_CLIENT_NAME, COGNITO_USER_POOL_CLIENT_SECRET_NAME, FARGATE_TASK_DEFINITION_NAME, ECS_SERVICE_NAME, WEB_ACL_NAME, CLOUDFRONT_DISTRIBUTION_NAME, ECS_TASK_ROLE_NAME, ALB_TARGET_GROUP_NAME, S3_LOG_CONFIG_BUCKET_NAME, S3_OUTPUT_BUCKET_NAME, ACM_CERTIFICATE_ARN, CLUSTER_NAME
from cdk_functions import create_subnets, create_web_acl_with_common_rules, add_custom_policies, add_alb_https_listener_with_cert # Only keep CDK-native functions

if PUBLIC_SUBNETS_TO_USE: PUBLIC_SUBNETS_TO_USE = eval(PUBLIC_SUBNETS_TO_USE)
if PUBLIC_SUBNET_CIDR_BLOCKS: PUBLIC_SUBNET_CIDR_BLOCKS = eval(PUBLIC_SUBNET_CIDR_BLOCKS)
if PUBLIC_SUBNET_AVAILABILITY_ZONES: PUBLIC_SUBNET_AVAILABILITY_ZONES = eval(PUBLIC_SUBNET_AVAILABILITY_ZONES)

if PRIVATE_SUBNETS_TO_USE: PRIVATE_SUBNETS_TO_USE = eval(PRIVATE_SUBNETS_TO_USE)
if PRIVATE_SUBNET_CIDR_BLOCKS: PRIVATE_SUBNET_CIDR_BLOCKS = eval(PRIVATE_SUBNET_CIDR_BLOCKS)
if PRIVATE_SUBNET_AVAILABILITY_ZONES: PRIVATE_SUBNET_AVAILABILITY_ZONES = eval(PRIVATE_SUBNET_AVAILABILITY_ZONES)

if AWS_MANAGED_TASK_ROLES_LIST: AWS_MANAGED_TASK_ROLES_LIST = eval(AWS_MANAGED_TASK_ROLES_LIST)

class CdkStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = CDK_PREFIX

        # --- Helper to get context values ---
        def get_context_bool(key: str, default: bool = False) -> bool:
            return self.node.try_get_context(key) or default

        def get_context_str(key: str, default: str = None) -> str:
             return self.node.try_get_context(key) or default
        
        def get_context_dict(scope: Construct, key: str, default: dict = None) -> dict:
            return scope.node.try_get_context(key) or default


        # --- VPC and Subnets (Assuming VPC is always lookup, Subnets are created/returned by create_subnets) ---
        # --- VPC Lookup (Always lookup as per your assumption) ---
        try:
            vpc = ec2.Vpc.from_lookup(
                self,
                "VPC",
                vpc_name=VPC_NAME
            )
            print("Successfully looked up VPC")
        except Exception as e:
            raise Exception("Could not look up VPC due to:", e)


        # --- Subnet Handling (Check Context and Create/Import) ---
        try:
            checked_public_subnets_ctx = get_context_dict(self, "checked_public_subnets")
            checked_private_subnets_ctx = get_context_dict(self, "checked_private_subnets")

            if not checked_public_subnets_ctx or not checked_private_subnets_ctx:
                 # This should not happen if check_resources.py ran correctly
                raise RuntimeError("Subnet existence information not found in context. Please run check_resources.py first.")
                #print("Existing subnets not found in context. Creating all new")
                #PUBLIC_SUBNETS_TO_USE = [f"{prefix}PublicSubnet1", f"{prefix}PublicSubnet2", f"{prefix}PublicSubnet3"]
                #PRIVATE_SUBNETS_TO_USE = [f"{prefix}PrivateSubnet1", f"{prefix}PrivateSubnet2", f"{prefix}PrivateSubnet3"]

            public_subnets: List[ec2.ISubnet] = []
            private_subnets: List[ec2.ISubnet] = []
            private_route_tables:List = []

            # Handle Public Subnets
            public_subnets_to_create = []
            public_subnets_create_cidr = []
            public_subnets_create_az = []

            print("PUBLIC_SUBNET_CIDR_BLOCKS:", PUBLIC_SUBNET_CIDR_BLOCKS)
            print("PUBLIC_SUBNET_AVAILABILITY_ZONES:", PUBLIC_SUBNET_AVAILABILITY_ZONES)

            for i, subnet_name in enumerate(PUBLIC_SUBNETS_TO_USE):
                 subnet_info = checked_public_subnets_ctx.get(subnet_name)
                 if not subnet_info:
                     raise RuntimeError(f"Subnet info for '{subnet_name}' missing in context.")

                 if subnet_info["exists"]:
                     # Subnet exists, import it by ID
                     if not subnet_info["id"]:
                          raise RuntimeError(f"Subnet ID for existing subnet '{subnet_name}' missing in context.")
                     imported_subnet = ec2.Subnet.from_subnet_id(self, f"ImportedPublicSubnet{i+1}", subnet_info["id"]) # Use unique logical ID
                     public_subnets.append(imported_subnet)
                     print(f"Using existing public subnet: {subnet_name} ({subnet_info['id']})")
                 else:
                     # Subnet does not exist, add to list for creation
                     public_subnets_to_create.append(subnet_name)
                     public_subnets_create_cidr.append(PUBLIC_SUBNET_CIDR_BLOCKS[i])
                     public_subnets_create_az.append(PUBLIC_SUBNET_AVAILABILITY_ZONES[i])
                     print(f"Public subnet '{subnet_name}' does not exist. Will be created.")

            # Create the public subnets that don't exist
            if public_subnets_to_create:
                 # Call your create_subnets function for the ones that need creating
                 # Ensure your create_subnets can handle creating a subset and returns ISubnet objects
                 created_public_subnets, placeholder_route_tables = create_subnets(
                      self,
                      vpc=vpc,
                      prefix=prefix,
                      subnet_names=public_subnets_to_create,
                      cidr_blocks=public_subnets_create_cidr,
                      availability_zones=public_subnets_create_az,
                      is_public=True # Assuming create_subnets needs a flag
                 )
                 public_subnets.extend(created_public_subnets) # Add created subnets to the list
                 

            # Handle Private Subnets (Similar logic as public)
            private_subnets_to_create = []
            private_subnets_create_cidr = []
            private_subnets_create_az = []

            for i, subnet_name in enumerate(PRIVATE_SUBNETS_TO_USE):
                 subnet_info = checked_private_subnets_ctx.get(subnet_name)
                 if not subnet_info:
                     raise RuntimeError(f"Subnet info for '{subnet_name}' missing in context.")

                 if subnet_info["exists"]:
                     # Subnet exists, import it by ID
                     if not subnet_info["id"]:
                          raise RuntimeError(f"Subnet ID for existing subnet '{subnet_name}' missing in context.")
                     imported_subnet = ec2.Subnet.from_subnet_id(self, f"ImportedPrivateSubnet{i+1}", subnet_info["id"]) # Use unique logical ID
                     private_subnets.append(imported_subnet)
                     print(f"Using existing private subnet: {subnet_name} ({subnet_info['id']})")
                 else:
                     # Subnet does not exist, add to list for creation
                     private_subnets_to_create.append(subnet_name)
                     private_subnets_create_cidr.append(PRIVATE_SUBNET_CIDR_BLOCKS[i])
                     private_subnets_create_az.append(PRIVATE_SUBNET_AVAILABILITY_ZONES[i])
                     print(f"Private subnet '{subnet_name}' does not exist. Will be created.")

            # Create the private subnets that don't exist
            if private_subnets_to_create:
                 # Call your create_subnets function for the ones that need creating
                 created_private_subnets, created_private_route_tables = create_subnets(
                      self,
                      vpc=vpc,
                      prefix=prefix,
                      subnet_names=private_subnets_to_create,
                      cidr_blocks=private_subnets_create_cidr,
                      availability_zones=private_subnets_create_az,
                      is_public=False # Assuming create_subnets needs a flag
                 )
                 private_subnets.extend(created_private_subnets) # Add created subnets to the list
                 private_route_tables.extend(created_private_route_tables) # Add created subnets to the list

            # Now public_subnets and private_subnets contain a mix
            # of imported and newly created ISubnet objects.
            # You can use these lists when configuring ALB, ECS Service, etc.

        except Exception as e:
            raise Exception("Could not handle subnets due to:", e)

        # --- IAM Roles ---
        try:
            codebuild_role_name = f"{prefix}CodeBuildRole"
            if get_context_bool(f"exists:{codebuild_role_name}"):
                # If exists, lookup/import the role using ARN from context
                role_arn = get_context_str(f"arn:{codebuild_role_name}")
                if not role_arn:
                     raise ValueError(f"Context value 'arn:{codebuild_role_name}' is required if role exists.")
                codebuild_role = iam.Role.from_role_arn(self, "CodeBuildRole", role_arn=role_arn)
                print("Using existing CodeBuild role")
            else:
                # If not exists, create the role
                codebuild_role = iam.Role(
                    self, "CodeBuildRole", # Logical ID
                    role_name=codebuild_role_name, # Explicit resource name
                    assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com")
                )
                codebuild_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(f"service-role/EC2InstanceProfileForImageBuilderECRContainerBuilds"))
                print("Successfully created new CodeBuild role")

            task_role_name = ECS_TASK_ROLE_NAME
            if get_context_bool(f"exists:{task_role_name}"):
                role_arn = get_context_str(f"arn:{task_role_name}")
                if not role_arn:
                     raise ValueError(f"Context value 'arn:{task_role_name}' is required if role exists.")
                task_role = iam.Role.from_role_arn(self, "TaskRole", role_arn=role_arn)
                print("Using existing ECS task role")
            else:
                task_role = iam.Role(
                    self, "TaskRole", # Logical ID
                    role_name=task_role_name, # Explicit resource name
                    assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
                )
                for role in AWS_MANAGED_TASK_ROLES_LIST:
                    print(f"Adding {role} to policy")
                    task_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(f"{role}"))
                task_role = add_custom_policies(self, task_role) # Assuming add_custom_policies modifies the role in place or returns it
                print("Successfully created new ECS task role")

            execution_role_name = f"{prefix}ExecutionRole"
            if get_context_bool(f"exists:{execution_role_name}"):
                 role_arn = get_context_str(f"arn:{execution_role_name}")
                 if not role_arn:
                      raise ValueError(f"Context value 'arn:{execution_role_name}' is required if role exists.")
                 execution_role = iam.Role.from_role_arn(self, "ExecutionRole", role_arn=role_arn)
                 print("Using existing ECS execution role")
            else:
                 execution_role = iam.Role(
                     self, "ExecutionRole", # Logical ID
                     role_name=execution_role_name, # Explicit resource name
                     assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
                 )
                 for role in AWS_MANAGED_TASK_ROLES_LIST:
                     execution_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(f"{role}"))
                 execution_role = add_custom_policies(self, execution_role)
                 print("Successfully created new ECS execution role")

        except Exception as e:
            raise Exception("Failed at IAM role step due to:", e)

        # --- S3 Buckets ---
        try:
            log_bucket_name = S3_LOG_CONFIG_BUCKET_NAME
            if get_context_bool(f"exists:{log_bucket_name}"):
                bucket = s3.Bucket.from_bucket_name(self, "LogConfigBucket", bucket_name=log_bucket_name)
                print("Using existing S3 bucket", log_bucket_name)
            else:
                bucket = s3.Bucket(self, "LogConfigBucket", bucket_name=log_bucket_name) # Explicitly set bucket_name
                print("Created S3 bucket", log_bucket_name)

            # Add policies - this will apply to both created and imported buckets
            # CDK handles idempotent policy additions
            bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role], # Pass the role object directly
                    actions=["s3:GetObject", "s3:PutObject"],
                    resources=[f"{bucket.bucket_arn}/*"]
                )
            )
            bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role],
                    actions=["s3:ListBucket"],
                    resources=[bucket.bucket_arn]
                )
            )

            output_bucket_name = S3_OUTPUT_BUCKET_NAME
            if get_context_bool(f"exists:{output_bucket_name}"):
                 output_bucket = s3.Bucket.from_bucket_name(self, "OutputBucket", bucket_name=output_bucket_name)
                 print("Using existing Output bucket", output_bucket_name)
            else:
                 output_bucket = s3.Bucket(self, "OutputBucket", bucket_name=output_bucket_name,
                     lifecycle_rules=[
                         s3.LifecycleRule(
                             expiration=Duration.days(int(DAYS_TO_DISPLAY_WHOLE_DOCUMENT_JOBS))
                         )
                     ]
                 )
                 print("Created Output bucket:", output_bucket_name)

            # Add policies to output bucket
            output_bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role],
                    actions=["s3:GetObject", "s3:PutObject"],
                    resources=[f"{output_bucket.bucket_arn}/*"]
                )
            )
            output_bucket.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[task_role],
                    actions=["s3:ListBucket"],
                    resources=[output_bucket.bucket_arn]
                )
            )

        except Exception as e:
            raise Exception("Could not handle S3 buckets due to:", e)

        # --- Elastic Container Registry ---
        try:
            full_ecr_repo_name = f"{prefix}{ECR_REPO_NAME}".lower()
            if get_context_bool(f"exists:{full_ecr_repo_name}"):
                ecr_repo = ecr.Repository.from_repository_name(self, "ECRRepo", repository_name=full_ecr_repo_name)
                print("Using existing ECR repository")
            else:
                ecr_repo = ecr.Repository(self, "ECRRepo", repository_name=full_ecr_repo_name) # Explicitly set repository_name
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
                     raise ValueError(f"Context value 'arn:{codebuild_project_name}' is required if project exists.")
                 codebuild_project = codebuild.Project.from_project_arn(self, "CodeBuildProject", project_arn=project_arn)
                 print("Using existing CodeBuild project")
            else:
                codebuild_project = codebuild.Project(self, "CodeBuildProject", # Logical ID
                    project_name=codebuild_project_name, # Explicit resource name
                    source=codebuild.Source.git_hub(
                        owner=GITHUB_REPO_USERNAME,
                        repo=GITHUB_REPO_NAME,
                        branch_or_ref=GITHUB_REPO_BRANCH
                    ),
                    environment=codebuild.BuildEnvironment(
                        build_image=codebuild.LinuxBuildImage.STANDARD_5_0,
                        privileged=True
                    ),
                    build_spec=codebuild.BuildSpec.from_object({
                        "version": "0.2",
                        "phases": {
                            "pre_build": {
                                "commands": [
                                    "echo Logging in to Amazon ECR",
                                    "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"
                                ]
                            },
                            "build": {
                                "commands": [
                                    "echo Building the Docker image",
                                    "docker build -t $ECR_REPO_NAME:latest .",
                                    "docker tag $ECR_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_NAME:latest"
                                ]
                            },
                            "post_build": {
                                "commands": [
                                    "echo Pushing the Docker image",
                                    "docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_NAME:latest"
                                ]
                            }
                        },
                        "env": {
                            "variables": {
                                "AWS_DEFAULT_REGION":AWS_REGION,
                                "AWS_ACCOUNT_ID":AWS_ACCOUNT_ID,
                                "ECR_REPO_NAME": full_ecr_repo_name
                            }
                        }
                    })
                )
                print("Successfully created CodeBuild project", codebuild_project_name)

            # Grant permissions - applies to both created and imported project role
            ecr_repo.grant_pull_push(codebuild_project.role)

        except Exception as e:
            raise Exception("Could not handle Codebuild project due to:", e)


        # --- NAT Gateway (CDK Native Approach) ---
        try:
            nat_gateway = None # Initialize variable

            # Check if NAT Gateway exists from context
            nat_gateway_exists_ctx = get_context_bool("exists:NatGateway")

            if nat_gateway_exists_ctx:
                # If exists, import it by ID from context
                nat_gateway_id_ctx = get_context_str("id:NatGateway")
                if not nat_gateway_id_ctx:
                    raise RuntimeError("NAT Gateway ID missing in context despite 'exists:NatGateway' being true.")
                # Use NatGateway.from_nat_gateway_id (High-level construct)
                #nat_gateway = ec2.CfnNatGateway(self, "ImportedNatGateway", nat_gateway_id_ctx)
                print(f"Using existing NAT Gateway: {nat_gateway_id_ctx}")
            else:
                # If it does not exist (or context says it doesn't) AND
                # you need one (e.g., if you created new private subnets),
                # then create it.
                # A simple check: Create a NAT Gateway if any private subnets were newly created.
                if private_subnets_to_create: # Check the list from your subnet handling logic
                    # 1. Define the Elastic IP
                    nat_eip = ec2.CfnEIP(
                        self,
                        NAT_GATEWAY_EIP_NAME,
                        tags=[CfnTag(key="Name", value=f"{prefix}-nat-gateway-eip")]
                    )

                    # 2. Define the NAT Gateway in a public subnet (preferably one of the newly created ones)
                    nat_gateway_public_subnet = None
                    # Find a suitable public subnet from your public_subnets list (mix of imported and created)
                    # You might want to prioritize a newly created one or one in a specific AZ.
                    if public_subnets:
                        nat_gateway_public_subnet = public_subnets[0] # Using the first available public subnet
                    if not nat_gateway_public_subnet:
                        raise RuntimeError("Could not find a suitable public subnet to place the NAT Gateway.")

                    # Use ec2.NatGateway (High-level construct) if available and suitable
                    # Or continue using ec2.CfnNatGateway if you need specific Cfn properties
                    nat_gateway = ec2.CfnNatGateway(
                        self,
                        "CreatedNatGateway", # Logical ID
                        #vpc=vpc, # Requires VPC
                        subnet_id=nat_gateway_public_subnet.subnet_id, # Requires ISubnet object
                        allocation_ids=[nat_eip.attr_allocation_id] # Requires EIP Allocation ID
                        # High-level NatGateway often handles EIP creation implicitly, but requires `vpc` prop.
                        # If using CfnNatGateway, you explicitly link the CfnEIP's alloc ID.
                        # Let's stick to CfnNatGateway for consistency with EIP definition.
                        # nat_gateway = ec2.CfnNatGateway(...) # Re-use your existing CfnNatGateway code here
                    )
                    # If you used the high-level ec2.NatGateway:
                    # CfnOutput(self, f"{prefix}NATGatewayIdOutput", value=nat_gateway.nat_gateway_id)
                    # If you used ec2.CfnNatGateway:
                    # CfnOutput(self, f"{prefix}NATGatewayIdOutput", value=nat_gateway.ref)

                    nat_gateway_id_ctx = nat_gateway.attr_nat_gateway_id

                    print("NAT Gateway and EIP defined.")

                else:
                    print("No new private subnets created. Assuming existing NAT setup and not creating a NAT Gateway.")
                    # In this case, nat_gateway variable remains None.
                    # Your route table logic needs to handle this (e.g., don't add routes if nat_gateway is None).


            # --- 3. Add routes to the NAT Gateway in private subnet route tables ---
            # This part needs to link the NEWLY CREATED private subnet route tables
            # to the CREATED OR IMPORTED NAT Gateway.

            # Assuming create_subnets returns created_private_route_tables when is_public=False
            # Check if the 'creation_result' from private subnet handling contained route tables
            # This requires coordinating the return type of create_subnets.
            # If create_subnets returns (subnets, route_tables) for private creation:
            if private_route_tables: # Check if variable exists or is non-empty
                if not nat_gateway_id_ctx:
                    if nat_gateway: # Only add routes if a NAT Gateway was created or imported
                        # Get the ID of the NAT Gateway (works for both high-level and Cfn constructs)
                        nat_gateway_id_ctx = nat_gateway.nat_gateway_id if isinstance(nat_gateway, ec2.NatGateway) else nat_gateway.ref

                for i, route_table in enumerate(created_private_route_tables):
                    # Assuming route_table is a CfnRouteTable or IRouteTable object
                    route_table_id_ref = route_table.ref if isinstance(route_table, ec2.CfnRouteTable) else route_table.route_table_id

                    ec2.CfnRoute(
                        self,
                        f"{prefix}PrivateRouteToNat{i+1}",
                        route_table_id=route_table_id_ref,
                        destination_cidr_block="0.0.0.0/0",
                        nat_gateway_id=nat_gateway_id_ctx,
                    )
                print("Routes added to NAT Gateway for new private subnet route tables.")
            else:
                print("No new private route tables to add routes to.")


        except Exception as e:
            raise Exception("Could not handle NAT gateway/EIP or routes due to:", e)


        # --- Security Groups ---
        try:
            ecs_security_group_name = ECS_SECURITY_GROUP_NAME
            # Use CDK's from_lookup_by_name which handles lookup or throws an error if not found
            try:
                # ecs_security_group = ec2.SecurityGroup.from_lookup_by_name(
                #     self, "ECSSecurityGroup", vpc=vpc, security_group_name=ecs_security_group_name
                # )
                # print("ecs_security_group:", ecs_security_group)
                # print(f"Using existing Security Group: {ecs_security_group_name}")
            #except Exception: # If lookup fails, create
                 ecs_security_group = ec2.SecurityGroup(
                     self,
                     "ECSSecurityGroup", # Logical ID
                     security_group_name=ecs_security_group_name, # Explicit resource name
                     vpc=vpc,
                 )
                 print(f"Created Security Group: {ecs_security_group_name}")
            except Exception as e: # If lookup fails, create
                print("Failed to create ECS security group due to:", e)

            alb_security_group_name = ALB_NAME_SECURITY_GROUP_NAME
            try:
                # alb_security_group = ec2.SecurityGroup.from_lookup_by_name(
                #     self, "ALBSecurityGroup", vpc=vpc, security_group_name=alb_security_group_name
                # )
                #print(f"Using existing Security Group: {alb_security_group_name}")
            # except Exception: # If lookup fails, create
                alb_security_group = ec2.SecurityGroup(
                    self,
                    "ALBSecurityGroup", # Logical ID
                    security_group_name=alb_security_group_name, # Explicit resource name
                    vpc=vpc,
                )
                print(f"Created Security Group: {alb_security_group_name}")
            except Exception as e: # If lookup fails, create
                print("Failed to create ALB security group due to:", e)

            # Define Ingress Rules - CDK will manage adding/removing these as needed
            ec2_port_gradio_server_port = ec2.Port.tcp(int(GRADIO_SERVER_PORT)) # Ensure port is int
            ecs_security_group.add_ingress_rule(
                peer=alb_security_group,
                connection=ec2_port_gradio_server_port,
                description="ALB traffic",
            )

        except Exception as e:
            raise Exception("Could not handle security groups due to:", e)


        # --- ECS Cluster ---
        try:
            cluster_name = CLUSTER_NAME
            # Use from_cluster_attributes or from_lookup
            # try:
                # cluster = ecs.Cluster.from_cluster_attributes(
                #     self, "ECSCluster", # Logical ID
                #     cluster_name=cluster_name,
                #     vpc=vpc, # VPC is required for attributes lookup
                #     security_groups=[], # Provide if part of lookup attributes
                #     cluster_arn=f"arn:aws:ecs:{AWS_REGION}:{AWS_ACCOUNT_ID}:cluster/{cluster_name}" # Requires ARN
                # )
                # print(f"Using existing cluster {cluster_name}.")
            # except Exception: # If lookup fails, create
            cluster = ecs.Cluster(
                self,
                "ECSCluster", # Logical ID
                cluster_name=cluster_name, # Explicit resource name
                enable_fargate_capacity_providers=True,
                vpc=vpc
            )
            print("Successfully created new ECS cluster")
        except Exception as e:
            raise Exception("Could not handle ECS cluster due to:", e)

        # --- ALB ---
        try:
            load_balancer_name = ALB_NAME
            if len(load_balancer_name) > 32: load_balancer_name = load_balancer_name[-32:]
            if get_context_bool(f"exists:{load_balancer_name}"):
                 # Lookup ALB by ARN from context
                 alb_arn = get_context_str(f"arn:{load_balancer_name}")
                 if not alb_arn:
                     raise ValueError(f"Context value 'arn:{load_balancer_name}' is required if ALB exists.")
                 alb = elbv2.ApplicationLoadBalancer.from_lookup(
                     self, "ALB", # Logical ID
                     load_balancer_arn=alb_arn
                 )
                 print(f"Using existing Application Load Balancer {load_balancer_name}.")
            else:
                alb = elbv2.ApplicationLoadBalancer(
                    self,
                    "ALB", # Logical ID
                    load_balancer_name=load_balancer_name, # Explicit resource name
                    vpc=vpc,
                    internet_facing=True,
                    security_group=alb_security_group, # Link to SG
                    vpc_subnets=ec2.SubnetSelection(subnets=public_subnets), # Link to subnets
                )
                print("Successfully created new Application Load Balancer")
        except Exception as e:
            raise Exception("Could not handle application load balancer due to:", e)

        # --- Cognito User Pool ---
        try:
            user_pool_name = COGNITO_USER_POOL_NAME
            if get_context_bool(f"exists:{user_pool_name}"):
                # Lookup by ID from context
                user_pool_id = get_context_str(f"id:{user_pool_name}")
                if not user_pool_id:
                     raise ValueError(f"Context value 'id:{user_pool_name}' is required if User Pool exists.")
                user_pool = cognito.UserPool.from_user_pool_id(self, "UserPool", user_pool_id=user_pool_id)
                print(f"Using existing user pool {user_pool_id}.")
            else:
                user_pool = cognito.UserPool(self, "UserPool",
                                            user_pool_name=user_pool_name,
                                            mfa=cognito.Mfa.OFF, # Adjust as needed
                                            sign_in_aliases=cognito.SignInAliases(email=True)) # Adjust as needed
                print(f"Created new user pool {user_pool.user_pool_id}.")

            # Add a domain to the User Pool (crucial for ALB integration)
            user_pool_domain = user_pool.add_domain(
                "UserPoolDomain",
                cognito_domain=cognito.CognitoDomainOptions(                    
                    domain_prefix="redaction-cognito-1" # <-- REPLACE WITH A UNIQUE PREFIX!
                )
                
            )
            CfnOutput(self, "CognitoUserPoolLoginUrl", value=user_pool_domain.base_url())

            user_pool_client_name = COGNITO_USER_POOL_CLIENT_NAME
            if get_context_bool(f"exists:{user_pool_client_name}"):
                 # Lookup by ID from context (requires User Pool object)
                 user_pool_client_id = get_context_str(f"id:{user_pool_client_name}")
                 if not user_pool_client_id:
                     raise ValueError(f"Context value 'id:{user_pool_client_name}' is required if User Pool Client exists.")
                 user_pool_client = cognito.UserPoolClient.from_user_pool_client_id(self, "UserPoolClient", user_pool_client_id=user_pool_client_id)
                 print(f"Using existing user pool client {user_pool_client_id}.")
            else:
                 user_pool_client = cognito.UserPoolClient(self, "UserPoolClient",
                                                        auth_flows=cognito.AuthFlow(user_srp=True), # Example: enable SRP for secure sign-in
                                                        user_pool=user_pool,
                                                        generate_secret=True,
                                                        user_pool_client_name=user_pool_client_name,
                                                        supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO],
                                                        o_auth=cognito.OAuthSettings(
                                                        flows=cognito.OAuthFlows(authorization_code_grant=True),
                                                        scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE]
                                                        )
                                    )
            CfnOutput(self, "CognitoAppClientId", value=user_pool_client.user_pool_client_id)

            print(f"Created new user pool client {user_pool_client.user_pool_client_id}.")

        except Exception as e:
            raise Exception("Could not handle Cognito resources due to:", e)

        # --- Secrets Manager Secret ---
        try:
             secret_name = COGNITO_USER_POOL_CLIENT_SECRET_NAME
             if get_context_bool(f"exists:{secret_name}"):
                 # Lookup by name
                 secret = secretsmanager.Secret.from_secret_name_v2(self, "CognitoSecret", secret_name=secret_name)
                 print(f"Using existing Secret {secret_name}.")
             else:
                 secret = secretsmanager.Secret(self, "CognitoSecret", # Logical ID
                     secret_name=secret_name, # Explicit resource name
                     secret_object_value={
                         "pool_id": SecretValue.unsafe_plain_text(user_pool.user_pool_id), # Use the CDK attribute
                         "app_client_id": SecretValue.unsafe_plain_text(user_pool_client.user_pool_client_id), # Use the CDK attribute
                         "app_client_secret": user_pool_client.user_pool_client_secret # Use the CDK attribute
                     }
                     # Note: SECRETS_MANAGER_ID from config is used as the *name* of the secret here.
                     # Ensure this aligns with your intent.
                 )
                 print(f"Created new secret {secret_name}.")

        except Exception as e:
             raise Exception("Could not handle Secrets Manager secret due to:", e)


        # --- Fargate Task Definition ---
        try:
            # For task definitions, re-creating with the same logical ID creates new revisions.
            # If you want to use a *specific existing revision*, you'd need to look it up by ARN.
            # If you want to update the latest revision, defining it here is the standard.
            # Let's assume we always define it here to get revision management.
            fargate_task_definition_name = FARGATE_TASK_DEFINITION_NAME

            # Load or define task_def_params as before...
            task_definition_path = "cdk/config/task_definition.json"
            if os.path.exists(task_definition_path):
                    with open(task_definition_path) as f: # Use correct path
                        task_def_params = json.load(f)
                    # Need to ensure taskRoleArn and executionRoleArn in JSON are correct ARN strings
            else:
                    task_def_params = {}
                    task_def_params['taskRoleArn'] = task_role.role_arn # Use CDK role object ARN
                    task_def_params['executionRoleArn'] = execution_role.role_arn # Use CDK role object ARN
                    task_def_params['memory'] = ECS_TASK_MEMORY_SIZE
                    task_def_params['cpu'] = ECS_TASK_CPU_SIZE
                    # Define container_def structure as before...
                    container_def = { # Simplified for example
                        "name": full_ecr_repo_name,
                        "image": ecr_image_loc,
                        "essential": True,
                        "portMappings": [{"containerPort": int(GRADIO_SERVER_PORT), "hostPort": int(GRADIO_SERVER_PORT), "protocol": "tcp", "appProtocol": "http"}],
                        "logConfiguration": {"logDriver": "awslogs", "options": {"awslogs-group": "/ecs/" + ECS_SERVICE_NAME, "awslogs-region": AWS_REGION, "awslogs-stream-prefix": "ecs"}},
                        "environmentFiles": [{"value": bucket.bucket_arn + "/config.env", "type": "s3"}] # Correct ARN format
                    }
                    task_def_params['containerDefinitions'] = [container_def]

            

            log_group_name_from_config=task_def_params['containerDefinitions'][0]['logConfiguration']['options']['awslogs-group']

            cdk_managed_log_group = logs.LogGroup(self, "MyTaskLogGroup", # CDK Logical ID
            log_group_name=log_group_name_from_config,
            retention=logs.RetentionDays.ONE_MONTH, # Example: set retention
            # removal_policy=RemovalPolicy.DESTROY # If you want it deleted when stack is deleted
    )


            fargate_task_definition = ecs.FargateTaskDefinition(
                self,
                "FargateTaskDefinition", # Logical ID
                family=fargate_task_definition_name, # Use name as family
                cpu=int(task_def_params['cpu']),
                memory_limit_mib=int(task_def_params['memory']),
                task_role=task_role, # Pass the role object
                execution_role=execution_role, # Pass the role object
                #network_mode=ecs.NetworkMode.AWS_VPC,
                runtime_platform=ecs.RuntimePlatform(
                    cpu_architecture=ecs.CpuArchitecture.X86_64,
                    operating_system_family=ecs.OperatingSystemFamily.LINUX
                )
            )
            print("Fargate task definition defined.")

            # Add container definitions to the task definition object
            if task_def_params['containerDefinitions']:
                container_def_params = task_def_params['containerDefinitions'][0]
                container = fargate_task_definition.add_container(
                    container_def_params['name'],
                    image=ecs.ContainerImage.from_registry(container_def_params['image']),
                    # ... other container parameters ...
                    logging=ecs.LogDriver.aws_logs(
                        stream_prefix=container_def_params['logConfiguration']['options']['awslogs-stream-prefix'],
                        log_group=cdk_managed_log_group
                        ),
                        secrets={
                            "COGNITO_POOL_ID": ecs.Secret.from_secrets_manager(secret, "pool_id"),
                            "COGNITO_APP_CLIENT_ID": ecs.Secret.from_secrets_manager(secret, "app_client_id"),
                            "COGNITO_APP_CLIENT_SECRET": ecs.Secret.from_secrets_manager(secret, "app_client_secret")
                        }
                )
                for port_mapping in container_def_params['portMappings']:
                    container.add_port_mappings(
                        ecs.PortMapping(
                            container_port=port_mapping['containerPort'],
                            host_port=port_mapping['hostPort'],
                            protocol=ecs.Protocol.TCP
                        )
                    )
                if container_def_params.get('environmentFiles'):
                    for env_file_param in container_def_params['environmentFiles']:
                        # Need to parse the ARN to get the bucket object and key
                        env_file_arn_parts = env_file_param['value'].split(":::")
                        bucket_name_and_key = env_file_arn_parts[-1]
                        env_bucket_name, env_key = bucket_name_and_key.split("/", 1)

                        env_bucket_obj = s3.Bucket.from_bucket_name(
                        self,
                        f"EnvFileBucket_{container_def_params['name']}_{1}", # Unique CDK ID
                        env_bucket_name # This can be a Token, from_bucket_name handles it
                        )


        except Exception as e:
            raise Exception("Could not handle Fargate task definition due to:", e)

        # --- ECS Service ---
        try:
            ecs_service_name = ECS_SERVICE_NAME
            # Check if service exists - from_service_arn or from_service_name (needs cluster)
            try:
                 # from_service_name is useful if you have the cluster object
                 ecs_service = ecs.FargateService.from_service_attributes(
                     self, "ECSService", # Logical ID
                     cluster=cluster, # Requires the cluster object
                     service_name=ecs_service_name
                 )
                 print(f"Using existing ECS service {ecs_service_name}.")
            except Exception: # If lookup fails, create
                 ecs_service = ecs.FargateService(
                     self,
                     "ECSService", # Logical ID
                     service_name=ecs_service_name, # Explicit resource name
                     cluster=cluster,
                     task_definition=fargate_task_definition, # Link to TD
                     security_groups=[ecs_security_group], # Link to SG
                     vpc_subnets=ec2.SubnetSelection(subnets=private_subnets), # Link to subnets
                     min_healthy_percent=10
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

        # --- CLOUDFRONT DISTRIBUTION ---
        # try:
        web_acl_name = WEB_ACL_NAME
        if get_context_bool(f"exists:{web_acl_name}"):
            # Lookup WAF ACL by ARN from context
                web_acl_arn = get_context_str(f"arn:{web_acl_name}")
                if not web_acl_arn:
                    raise ValueError(f"Context value 'arn:{web_acl_name}' is required if Web ACL exists.")
                # Note: CDK's aws_wafv2 doesn't have a simple from_web_acl_arn lookup that returns a CfnWebACL.
                # You might need to use a Custom Resource or CfnRuleGroup/CfnWebACL if you need the specific properties.
                # For linking to CloudFront, the ARN is often sufficient.
                # Let's assume create_web_acl_with_common_rules handles lookup/creation and returns the object with .attr_arn
                web_acl = create_web_acl_with_common_rules(self, web_acl_name) # Assuming it takes scope and name
                print(f"Handled Cloudfront WAF web ACL {web_acl_name}.")
        else:
            web_acl = create_web_acl_with_common_rules(self, web_acl_name) # Assuming it takes scope and name
            print(f"Created Cloudfront WAF web ACL {web_acl_name}.")


        # Add ALB as CloudFront Origin
        origin = origins.LoadBalancerV2Origin(
            alb, # Use the created or looked-up ALB object
            custom_headers={CUSTOM_HEADER: CUSTOM_HEADER_VALUE},
            origin_shield_enabled=False,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
        )

        cloudfront_name = CLOUDFRONT_DISTRIBUTION_NAME
        # Check if CloudFront distribution exists - from_distribution_id (needs ID) or lookup
        # Lookup is less common as CF distribution IDs are generated.
        # Often simpler to define in the stack if managed here.
        # If you need to check existence, you need a lookup that returns the Distribution object.
        # For simplicity, let's define it in the stack.

        geo_restrict = cloudfront.GeoRestriction.allowlist(CLOUDFRONT_GEO_RESTRICTION)

        cloudfront_distribution = cloudfront.Distribution(
            self,
            "CloudFrontDistribution", # Logical ID
            comment=cloudfront_name, # Use name as comment for easier identification
            geo_restriction=geo_restrict,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origin,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER,
            ),
            web_acl_id=web_acl.attr_arn # Link to WAF ACL ARN
        )
        print(f"Cloudfront distribution {cloudfront_name} defined.")

        # except Exception as e:
        #     raise Exception("Could not handle Cloudfront distribution due to:", e)

        # --- ALB TARGET GROUPS AND LISTENERS ---
        # This section should primarily define the resources if they are managed by this stack.
        # CDK handles adding/removing targets and actions on updates.
        # If they might pre-exist outside the stack, you need lookups.
        cookie_duration = Duration.hours(12)
        target_group_name = ALB_TARGET_GROUP_NAME # Explicit resource name 
        cloudfront_distribution_url = cloudfront_distribution.domain_name

        try:
            # --- CREATING TARGET GROUPS AND ADDING THE CLOUDFRONT LISTENER RULE ---           

            target_group = elbv2.ApplicationTargetGroup(
                self,
                "AppTargetGroup", # Logical ID
                target_group_name=target_group_name, # Explicit resource name
                port=int(GRADIO_SERVER_PORT), # Ensure port is int
                protocol=elbv2.ApplicationProtocol.HTTP,
                targets=[ecs_service], # Link to ECS Service
                stickiness_cookie_duration=cookie_duration,
                vpc=vpc, # Target Groups need VPC
            )
            print(f"ALB target group {target_group_name} defined.")


            # Targets based on header - does not work consistently with Gradio apps
            # This is done unconditionally if the listener and target group objects are available
            # http_listener.add_target_groups(
            #     "Targets", # Logical ID for this target group association
            #     target_groups=[target_group],
            #     priority=1,
            #     conditions=[
            #         elbv2.ListenerCondition.http_header(
            #             CUSTOM_HEADER,
            #             [CUSTOM_HEADER_VALUE]
            #         )
            #     ],
            # )

            # First HTTP
            listener_port = 80
            # Check if Listener exists - from_listener_arn or lookup by port/ALB

            http_listener = alb.add_listener(
                "HttpListener", # Logical ID
                port=listener_port,
                open=False, # Be cautious with open=True, usually restrict source SG
            )
            print(f"ALB listener on port {listener_port} defined.")

            # Add default action to the listener
            http_listener.add_action(
                "DefaultAction", # Logical ID for the default action
                action=elbv2.ListenerAction.fixed_response(
                    status_code=403,
                    content_type="text/plain",
                    message_body="Access denied",
                ),
            )
            print("Added default action to ALB listener for HTTP.")

            # Add the Listener Rule for the specific CloudFront Host Header
            http_listener.add_action(
                "CloudFrontHostHeaderRule",
                action=elbv2.ListenerAction.forward(target_groups=[target_group],stickiness_duration=cookie_duration),
                priority=1, # Example priority. Adjust as needed. Lower is evaluated first.
                conditions=[
                    elbv2.ListenerCondition.host_headers([cloudfront_distribution_url])
                ]
            )
            
            print("Added targets to ALB listener for HTTP.")


            # Now the same for HTTPS if you have an ACM certificate
            if ACM_CERTIFICATE_ARN:
                listener_port_https = 443
                # Check if Listener exists - from_listener_arn or lookup by port/ALB

                https_listener = add_alb_https_listener_with_cert(
                self,
                "MyHttpsListener", # Logical ID for the HTTPS listener
                alb,
                acm_certificate_arn=ACM_CERTIFICATE_ARN,
                default_target_group=target_group,
                enable_cognito_auth=True, # <-- Enable Cognito Authentication
                cognito_user_pool=user_pool,
                cognito_user_pool_client=user_pool_client,
                cognito_user_pool_domain=user_pool_domain.domain_prefix, # Pass the domain prefix
                listener_open_to_internet=False
                )

                if https_listener:
                    CfnOutput(self, "HttpsListenerArn", value=https_listener.listener_arn)

                # https_listener = alb.add_listener(
                #     "HttpsListener", # Logical ID
                #     port=listener_port_https,
                #     open=False, # Be cautious with open=True, usually restrict source SG
                #     certificates=[]
                # )
                print(f"ALB listener on port {listener_port_https} defined.")

                # Add default action to the listener
                https_listener.add_action(
                    "DefaultAction", # Logical ID for the default action
                    action=elbv2.ListenerAction.fixed_response(
                        status_code=403,
                        content_type="text/plain",
                        message_body="Access denied",
                    ),
                )
                print("Added default action to ALB listener for HTTPS.")

                # Add the Listener Rule for the specific CloudFront Host Header
                https_listener.add_action(
                    "CloudFrontHostHeaderRuleHTTPS",
                    action=elbv2.ListenerAction.forward(target_groups=[target_group],stickiness_duration=cookie_duration),
                    priority=1, # Example priority. Adjust as needed. Lower is evaluated first.
                    conditions=[
                        elbv2.ListenerCondition.host_headers([cloudfront_distribution_url])
                    ]
                )
                
                print("Added targets to ALB listener for HTTPS.")            


        except Exception as e:
            raise Exception("Could not handle ALB target groups and listeners due to:", e)


        # --- Outputs ---
        CfnOutput(self, "CloudFrontDistributionURL",
                  value=cloudfront_distribution.domain_name)
        CfnOutput(self, "CognitoPoolId",
                  value=user_pool.user_pool_id)
        # Add other outputs if needed
        CfnOutput(self, "ALBName", value=alb.load_balancer_name)
        CfnOutput(self, "ECRRepoUri", value=ecr_repo.repository_uri)