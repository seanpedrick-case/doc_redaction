# Phase 0: Express Service Connect validation (dev account)

Service Connect for Express is applied in `post_cdk_build_quickstart.py` (not during
`cdk deploy`). Express primary containers do not define named `portMappings` at create
time, so the post-deploy step registers a task-definition revision with
`port-{port}` and then calls `ecs:UpdateService` with `serviceConnectConfiguration`.
Use this checklist to confirm behaviour in a dev account before relying on Pi Express demos.

## Prerequisites

- Main Express stack deployed (`USE_ECS_EXPRESS_MODE=True`).
- Optional: deploy with `ENABLE_PI_AGENT_EXPRESS_SERVICE=True` and run
  `python post_cdk_build_quickstart.py` (Service Connect is configured there).

## Manual validation (without Pi CDK flag)

1. Note cluster name, main Express **service name**, and task security group from stack outputs.
2. Ensure the cluster has a default Cloud Map namespace (CDK creates one when Pi Express is enabled).
3. Update the main Express service (server):

```bash
aws ecs update-service \
  --cluster <CLUSTER_NAME> \
  --service <ECS_EXPRESS_SERVICE_NAME> \
  --force-new-deployment \
  --service-connect-configuration '{
    "enabled": true,
    "namespace": "<ECS_SERVICE_CONNECT_NAMESPACE>",
    "services": [{
      "portName": "port-7860",
      "discoveryName": "redaction",
      "clientAliases": [{"port": 7860, "dnsName": "redaction"}]
    }]
  }'
```

4. Deploy a second Express service (Pi image, port **7862**) or use CDK Pi Express.
5. Update the Pi Express service (client only):

```bash
aws ecs update-service \
  --cluster <CLUSTER_NAME> \
  --service <ECS_PI_EXPRESS_SERVICE_NAME> \
  --force-new-deployment \
  --service-connect-configuration '{
    "enabled": true,
    "namespace": "<ECS_SERVICE_CONNECT_NAMESPACE>"
  }'
```

6. Exec into a Pi task (ECS Exec enabled on the service):

```bash
curl -sS -o /dev/null -w "%{http_code}\n" http://redaction:7860/
```

Expect HTTP **200** (or Gradio redirect) without Cognito.

7. Optional: run a minimal `gradio_client` predict against `/doc_redact` from the Pi task.

## Exit criteria

- Service Connect DNS `redaction` resolves inside the Pi task network namespace.
- Gradio responds on port 7860 without ALB Cognito.
- If `update-service` fails or `portName` is rejected, stop and use legacy Fargate Pi +
  `ENABLE_ECS_SERVICE_CONNECT` until AWS/CDK support is confirmed (plan Phase 2 hybrid).

## CDK deploy path

With `ENABLE_PI_AGENT_EXPRESS_SERVICE=True`, run `post_cdk_build_quickstart.py` after
`cdk deploy`. Check its output for Service Connect / task-definition errors.
