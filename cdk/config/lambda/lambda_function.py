"""
S3-triggered Lambda: start a one-shot ECS Fargate task for doc_redaction direct mode.

Upload a job ``.env`` under ENV_PREFIX on the output bucket; the Lambda merges optional
default params, sets RUN_DIRECT_MODE=True, and calls ecs.run_task with container overrides.
"""

import os
import urllib.parse

import boto3

ecs = boto3.client("ecs")
s3 = boto3.client("s3")

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET", "")
INPUT_PREFIX = os.environ.get("INPUT_PREFIX", "input/")
CONFIG_PREFIX = os.environ.get("CONFIG_PREFIX", "")
ENV_PREFIX = os.environ.get("ENV_PREFIX", f"{INPUT_PREFIX}config/")
ENV_SUFFIX = os.environ.get("ENV_SUFFIX", ".env")
DEFAULT_PARAMS_KEY = os.environ.get("DEFAULT_PARAMS_KEY", "").strip()
CLUSTER = os.environ.get("ECS_CLUSTER", "")
TASK_DEF = os.environ.get("ECS_TASK_DEF", "")
SUBNETS = [s.strip() for s in os.environ.get("SUBNETS", "").split(",") if s.strip()]
SECURITY_GROUPS = [
    s.strip() for s in os.environ.get("SECURITY_GROUPS", "").split(",") if s.strip()
]
CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "")
DEFAULT_DIRECT_MODE_TASK = os.environ.get("DEFAULT_DIRECT_MODE_TASK", "redact")
ASSIGN_PUBLIC_IP = os.environ.get("ECS_ASSIGN_PUBLIC_IP", "false").lower() == "true"
APP_CONFIG_ENV_KEY = os.environ.get("APP_CONFIG_ENV_KEY", "app_config.env").strip()

_DEFAULT_INPUT_KEYS = ("DIRECT_MODE_INPUT_FILE",)
_DEFAULT_CONFIG_KEYS = (
    "DENY_LIST_PATH",
    "ALLOW_LIST_PATH",
    "WHOLE_PAGE_REDACTION_LIST_PATH",
    "DENY_LIST_FILE",
    "ALLOW_LIST_FILE",
    "REDACT_WHOLE_PAGE_FILE",
)
S3_PREFIX_INPUT_KEYS = tuple(
    k.strip()
    for k in os.environ.get(
        "S3_PREFIX_INPUT_KEYS", ",".join(_DEFAULT_INPUT_KEYS)
    ).split(",")
    if k.strip()
)
S3_PREFIX_CONFIG_KEYS = tuple(
    k.strip()
    for k in os.environ.get(
        "S3_PREFIX_CONFIG_KEYS", ",".join(_DEFAULT_CONFIG_KEYS)
    ).split(",")
    if k.strip()
)

_TASK_HINTS = (
    ("combine_review", "combine_review_pdfs"),
    ("deduplicate", "deduplicate"),
    ("summarise", "summarise"),
    ("summarize", "summarise"),
    ("textract", "textract"),
    ("redact", "redact"),
)


def _key_matches(key: str) -> bool:
    return key.startswith(ENV_PREFIX) and key.endswith(ENV_SUFFIX)


def _s3_uri(bucket: str, prefix: str, relative_path: str) -> str:
    rel = relative_path.lstrip("/")
    base = prefix or ""
    if base and not base.endswith("/"):
        base += "/"
    return f"s3://{bucket}/{base}{rel}"


def _maybe_prefix_value(key: str, value: str, input_base: str, config_base: str) -> str:
    if not value:
        return value
    if value.strip().upper().startswith("S3://"):
        return value

    if key in S3_PREFIX_INPUT_KEYS:
        if "," in value:
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return ",".join(
                _maybe_prefix_value(key, p, input_base, config_base) for p in parts
            )
        return input_base + value.lstrip("/")

    if key in S3_PREFIX_CONFIG_KEYS:
        return config_base + value.lstrip("/")

    return value


def _parse_dotenv(
    dotenv_bytes: bytes,
    output_bucket: str,
    input_prefix: str,
    config_bucket: str,
    config_prefix: str,
) -> dict:
    """Parse a basic .env and apply S3 URI prefixes to configured keys."""
    env = {}
    text = dotenv_bytes.decode("utf-8", errors="replace")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        env[key] = value

    input_base = f"s3://{output_bucket}/{input_prefix}"
    config_base = f"s3://{config_bucket}/{config_prefix}"

    for key in list(env.keys()):
        env[key] = _maybe_prefix_value(key, env[key], input_base, config_base)

    return env


def _derive_task_from_key(key: str) -> str | None:
    """Optional: infer DIRECT_MODE_TASK from job filename."""
    basename = key.split("/")[-1].lower()
    for hint, task in _TASK_HINTS:
        if hint in basename:
            return task
    return None


def _build_environment_array(*env_dicts):
    """Merge dictionaries left→right; later dicts win."""
    merged = {}
    for d in env_dicts:
        if d:
            merged.update(d)
    return [{"name": k, "value": str(v)} for k, v in merged.items()]


def _load_app_config_env() -> dict:
    """Base app_config.env from the config/logs bucket (replaces ECS environmentFiles)."""
    if not APP_CONFIG_ENV_KEY:
        return {}
    bucket = CONFIG_BUCKET
    if not bucket:
        return {}
    try:
        obj = s3.get_object(Bucket=bucket, Key=APP_CONFIG_ENV_KEY)
    except s3.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            print(f"app_config env not found at s3://{bucket}/{APP_CONFIG_ENV_KEY}")
            return {}
        raise
    return _parse_dotenv(
        obj["Body"].read(),
        bucket,
        INPUT_PREFIX,
        bucket,
        CONFIG_PREFIX,
    )


def _load_default_env(output_bucket: str) -> dict:
    if not DEFAULT_PARAMS_KEY:
        return {}
    bucket = OUTPUT_BUCKET or output_bucket
    if not bucket:
        return {}
    try:
        obj = s3.get_object(Bucket=bucket, Key=DEFAULT_PARAMS_KEY)
    except s3.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            print(f"Default params not found at s3://{bucket}/{DEFAULT_PARAMS_KEY}")
            return {}
        raise
    return _parse_dotenv(
        obj["Body"].read(),
        bucket,
        INPUT_PREFIX,
        CONFIG_BUCKET or bucket,
        CONFIG_PREFIX,
    )


def _required_runtime_env(merged: dict) -> dict:
    """Always-on overrides; do not clobber job-specific DIRECT_MODE_TASK."""
    runtime = {"RUN_DIRECT_MODE": "True"}
    if "DIRECT_MODE_TASK" not in merged:
        runtime["DIRECT_MODE_TASK"] = DEFAULT_DIRECT_MODE_TASK
    if CONFIG_BUCKET:
        runtime["DOCUMENT_REDACTION_BUCKET"] = CONFIG_BUCKET
    elif OUTPUT_BUCKET:
        runtime["DOCUMENT_REDACTION_BUCKET"] = OUTPUT_BUCKET
    return runtime


def lambda_handler(event, context):
    runs = []
    default_file_env = {}
    app_config_env = _load_app_config_env()

    if DEFAULT_PARAMS_KEY:
        try:
            default_file_env = _load_default_env(
                event.get("Records", [{}])[0]
                .get("s3", {})
                .get("bucket", {})
                .get("name", OUTPUT_BUCKET)
            )
        except Exception as exc:
            print(f"Could not load default params: {exc}")

    for record in event.get("Records", []):
        s3rec = record.get("s3", {})
        bucket = s3rec.get("bucket", {}).get("name")
        raw_key = s3rec.get("object", {}).get("key")
        if not bucket or not raw_key:
            print("Missing bucket or key in S3 event record, skipping.")
            continue

        key = urllib.parse.unquote_plus(raw_key)

        if not _key_matches(key):
            print(f"Key does not match filter: {key}, skipping.")
            continue

        obj = s3.get_object(Bucket=bucket, Key=key)
        file_env = _parse_dotenv(
            obj["Body"].read(),
            bucket,
            INPUT_PREFIX,
            CONFIG_BUCKET or bucket,
            CONFIG_PREFIX,
        )

        derived = {}
        if "DIRECT_MODE_TASK" not in file_env:
            hinted = _derive_task_from_key(key)
            if hinted:
                derived["DIRECT_MODE_TASK"] = hinted

        merged_for_required = {}
        for d in (app_config_env, default_file_env, derived, file_env):
            if d:
                merged_for_required.update(d)
        environment = _build_environment_array(
            app_config_env,
            default_file_env,
            derived,
            file_env,
            _required_runtime_env(merged_for_required),
        )

        print(f"Starting batch task for s3://{bucket}/{key} ({len(environment)} env vars)")

        response = ecs.run_task(
            cluster=CLUSTER,
            launchType="FARGATE",
            taskDefinition=TASK_DEF,
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": SUBNETS,
                    "securityGroups": SECURITY_GROUPS,
                    "assignPublicIp": "ENABLED" if ASSIGN_PUBLIC_IP else "DISABLED",
                }
            },
            overrides={
                "containerOverrides": [
                    {
                        "name": CONTAINER_NAME,
                        "environment": environment,
                    }
                ]
            },
        )

        runs.append(
            {
                "bucket": bucket,
                "key": key,
                "taskArns": [t["taskArn"] for t in response.get("tasks", [])],
                "failures": response.get("failures", []),
                "envCount": len(environment),
            }
        )

    return {"runs": runs}
