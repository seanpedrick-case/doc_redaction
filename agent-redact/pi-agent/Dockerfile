# syntax=docker/dockerfile:1
# Pi agent image (dev + production). Build from monorepo root:
#   docker build -f agent-redact/pi-agent/Dockerfile --target dev .
#   docker build -f agent-redact/pi-agent/Dockerfile --target runtime .
# Root .dockerignore must allow config/*.example into the context (secrets stay gitignored).
#
# Targets:
#   dev     — docker-compose: Pi CLI + Python deps; app tree bind-mounted at runtime.
#   runtime — HF Space / AWS ECS: baked agent-redact tree, non-root user, named volumes.

# ===================================================================
# Stage 1: Pi CLI (Node) — isolated so the runtime base stays Python 3.12
# ===================================================================
FROM node:22-bookworm-slim AS pi-cli

ENV NPM_CONFIG_PREFIX=/opt/pi
ENV PATH="/opt/pi/bin:${PATH}"

RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent

# ===================================================================
# Stage 2: Shared Python base (aligned with main app Dockerfile)
# ===================================================================
FROM public.ecr.aws/docker/library/python:3.12.13-slim-trixie AS pi-base

ENV NODE_ENV=production
ENV DEBIAN_FRONTEND=noninteractive
ENV NPM_CONFIG_LOGLEVEL=warn
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_HOME=/home/user
ENV PI_WORKDIR=/workspace/doc_redaction
ENV PYTHONPATH=${PI_WORKDIR}:${PI_WORKDIR}/agent-redact/pi
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV MPLCONFIGDIR=/tmp/matplotlib_cache/
ENV XDG_CACHE_HOME=/tmp/xdg_cache/user_1000
ENV PATH="/opt/pi/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    curl \
    ca-certificates \
    procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=pi-cli /opt/pi /opt/pi
COPY --from=pi-cli /usr/local/bin/node /usr/local/bin/node

COPY agent-redact/requirements_pi_agent.txt /tmp/requirements_pi_agent.txt
RUN pip install --no-cache-dir -r /tmp/requirements_pi_agent.txt \
    && rm /tmp/requirements_pi_agent.txt

# ===================================================================
# Stage 3: Dev — thin image for docker-compose (repo bind-mounted)
# ===================================================================
FROM pi-base AS dev

ENV HOME=${APP_HOME}
ENV PI_WORKSPACE_DIR=${APP_HOME}/app/workspace
ENV PI_UPLOAD_ROOT=/tmp/gradio
ENV PI_SESSION_DIR=${APP_HOME}/.pi/agent/sessions

RUN useradd -m -u 1000 user \
    && mkdir -p \
        ${APP_HOME}/app/workspace \
        ${APP_HOME}/.pi/agent/sessions \
        ${PI_WORKDIR} \
        /tmp/gradio \
        /tmp/matplotlib_cache \
        ${XDG_CACHE_HOME} \
    && chown -R user:user ${APP_HOME} ${PI_WORKDIR} \
    && chown user:user /tmp/gradio /tmp/matplotlib_cache ${XDG_CACHE_HOME} \
    && chmod 1777 /tmp/gradio /tmp/matplotlib_cache \
    && chmod 700 ${XDG_CACHE_HOME}

WORKDIR ${PI_WORKDIR}

USER user

RUN pi --version

# Compose overrides entrypoint with agent-redact/pi/start.sh on the bind mount.

# ===================================================================
# Stage 4: Runtime — baked app for Hugging Face Space and AWS ECS
# ===================================================================
FROM pi-base AS runtime

ENV PI_DEPLOYMENT_PROFILE=hf-space
ENV PI_DEFAULT_PROVIDER=google-gemini
ENV PI_DEFAULT_MODEL=gemini-flash-lite-latest
ENV DOC_REDACTION_GRADIO_URL=https://seanpedrickcase-document-redaction.hf.space
ENV GRADIO_SERVER_PORT=7860
ENV HOME=${APP_HOME}
ENV PI_WORKSPACE_DIR=${APP_HOME}/app/workspace
ENV PI_WORKDIR=/workspace/doc_redaction
ENV PI_UPLOAD_ROOT=/tmp/gradio
ENV PI_SESSION_DIR=/tmp/pi-sessions
ENV PI_OFFLINE=1
ENV PI_SKIP_VERSION_CHECK=1
ENV PI_GRADIO_SHOW_EXAMPLES=true

WORKDIR ${PI_WORKDIR}

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

RUN useradd -m -u 1000 user \
    && mkdir -p \
        ${APP_HOME}/app/workspace \
        ${APP_HOME}/.pi/agent \
        /tmp/gradio \
        /tmp/pi-sessions \
        /tmp/matplotlib_cache \
        ${XDG_CACHE_HOME} \
    && chown user:user \
        ${APP_HOME}/app/workspace \
        ${APP_HOME}/.pi \
        /tmp/gradio \
        /tmp/pi-sessions \
        /tmp/matplotlib_cache \
        ${XDG_CACHE_HOME} \
    && chmod 755 ${APP_HOME}/app/workspace ${APP_HOME}/.pi \
    && chmod 1777 /tmp/gradio /tmp/pi-sessions /tmp/matplotlib_cache \
    && chmod 700 ${XDG_CACHE_HOME} \
    && chown -R root:root ${PI_WORKDIR} \
    && find ${PI_WORKDIR} -type d -exec chmod 755 {} \; \
    && find ${PI_WORKDIR} -type f -exec chmod 644 {} \; \
    && mkdir -p ${APP_HOME}/app \
    && chown user:user ${APP_HOME}/app

COPY agent-redact/pi-agent/entrypoint-ecs.sh /usr/local/bin/entrypoint-ecs.sh
COPY agent-redact/pi-agent/entrypoint.sh ${APP_HOME}/app/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint-ecs.sh ${APP_HOME}/app/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint-ecs.sh ${APP_HOME}/app/entrypoint.sh

# Writable paths only via runtime mounts (read-only root FS friendly).
VOLUME ["${APP_HOME}/app/workspace"]
VOLUME ["/tmp/gradio"]
VOLUME ["/tmp/pi-sessions"]
VOLUME ["/tmp/matplotlib_cache"]
VOLUME ["${XDG_CACHE_HOME}"]
VOLUME ["/tmp"]
VOLUME ["/var/tmp"]

USER user

RUN pi --version

EXPOSE 7860

ENTRYPOINT ["/home/user/app/entrypoint.sh"]
