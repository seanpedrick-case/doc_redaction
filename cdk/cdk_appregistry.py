"""AWS Console myApplications (Service Catalog AppRegistry) integration."""

from aws_cdk import App, Environment
from aws_cdk.aws_servicecatalogappregistry_alpha import (
    ApplicationAssociator,
    TargetApplication,
)


def register_doc_redaction_application(
    app: App,
    *,
    aws_account_id: str,
    aws_region: str,
    application_name: str,
    application_description: str,
    appregistry_stack_name: str,
    attribute_group_name: str,
    repository_url: str,
    cdk_prefix: str,
    use_cloudfront: str,
    alb_dns_name: str | None = None,
) -> ApplicationAssociator:
    """
    Register regional CDK stacks with AWS Console myApplications.

    Only stacks in ``aws_region`` are associated (phase 1). Cross-region stacks
    such as RedactionStackCloudfront (us-east-1) are not included.

    ``alb_dns_name`` must be a plain string (e.g. from pre-check context). Do not
    pass a CloudFormation token from RedactionStack or synth will fail with a
    dependency cycle against the associator stack.
    """
    associator = ApplicationAssociator(
        app,
        "DocRedactionAppRegistry",
        applications=[
            TargetApplication.create_application_stack(
                application_name=application_name,
                application_description=application_description,
                stack_name=appregistry_stack_name,
                env=Environment(account=aws_account_id, region=aws_region),
            )
        ],
    )

    attributes = {
        "repository": repository_url,
        "cdkPrefix": cdk_prefix,
        "awsRegion": aws_region,
        "useCloudFront": use_cloudfront,
        "cloudFrontInAppRegistry": "false",
        "cloudFrontNote": (
            "CloudFront/WAF (RedactionStackCloudfront) is in us-east-1 and is "
            "not linked to this myApplications entry in phase 1. View it in "
            "CloudFormation (us-east-1) or the CloudFront console."
        ),
    }
    if alb_dns_name:
        attributes["albDnsName"] = alb_dns_name

    associator.app_registry_application.add_attribute_group(
        "DocRedactionAttributeGroup",
        attribute_group_name=attribute_group_name,
        description="doc_redaction deployment metadata",
        attributes=attributes,
    )

    return associator
