# syntax=docker/dockerfile:1
# Pi agent Gradio UI for Hugging Face Docker Space (remote doc_redaction backend).
# Build from monorepo root: docker build -f agent-redact/pi-agent/Dockerfile .
# Root .dockerignore must allow config/*.example into the context (secrets stay gitignored).

FROM node:22-bookworm-slim

ENV NODE_ENV=production
ENV DEBIAN_FRONTEND=noninteractive
ENV NPM_CONFIG_LOGLEVEL=warn
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace/doc_redaction:/workspace/doc_redaction/agent-redact/pi
ENV PI_DEPLOYMENT_PROFILE=hf-space
ENV PI_DEFAULT_PROVIDER=google-gemini
ENV PI_DEFAULT_MODEL=gemini-flash-lite-latest
ENV DOC_REDACTION_GRADIO_URL=https://seanpedrickcase-document-redaction.hf.space
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PI_WORKSPACE_DIR=/home/user/app/workspace
ENV PI_WORKDIR=/workspace/doc_redaction
ENV PI_UPLOAD_ROOT=/tmp/gradio
ENV PI_SESSION_DIR=/tmp/pi-sessions
ENV PI_OFFLINE=1
ENV PI_SKIP_VERSION_CHECK=1
ENV PI_GRADIO_SHOW_EXAMPLES=true
ENV HOME=/home/node

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    curl \
    ca-certificates \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent

COPY agent-redact/requirements_pi_agent.txt /tmp/requirements_pi_agent.txt
RUN pip3 install --no-cache-dir --break-system-packages \
    -r /tmp/requirements_pi_agent.txt \
    && rm /tmp/requirements_pi_agent.txt

WORKDIR /workspace/doc_redaction

COPY agent-redact/pi agent-redact/pi
COPY skills skills
COPY tools tools
# Committed template only (see sync-manifest.txt); runtime secrets come from S3/env on ECS.
COPY config/pi_agent.env.example config/pi_agent.env.example
COPY intros intros
COPY AGENTS.md AGENTS.md
COPY doc_redaction/example_data doc_redaction/example_data

RUN test -f doc_redaction/example_data/example_of_emails_sent_to_a_professor_before_applying.pdf \
    && test -f doc_redaction/example_data/graduate-job-example-cover-letter.pdf \
    && ! head -1 doc_redaction/example_data/example_of_emails_sent_to_a_professor_before_applying.pdf \
        | grep -q "^version https://git-lfs.github.com/spec/v1"

RUN mkdir -p /home/node/.pi/agent /home/user/app/workspace /tmp/gradio /tmp/pi-sessions \
    && chown -R node:node /home/node/.pi /home/user/app /tmp/gradio /tmp/pi-sessions /workspace

USER node

RUN pi --version

EXPOSE 7860

CMD ["bash", "-c", "python3 agent-redact/pi/pi_agent_config.py && exec python3 agent-redact/pi/gradio_app.py"]
