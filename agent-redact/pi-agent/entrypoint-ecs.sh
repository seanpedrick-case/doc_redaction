#!/bin/bash
# ECS Fargate: ephemeral volume mounts are root-owned; chown then drop to user (image USER).
set -euo pipefail

for dir in /tmp/agent-coding /tmp/agent-logs /tmp/agent-usage /tmp/agent-feedback \
    /home/user/app/workspace /tmp/gradio /tmp/agent-sessions; do
  mkdir -p "$dir"
  chown -R user:user "$dir"
done

cd /workspace/doc_redaction
exec su -s /bin/bash user -c "$*"
