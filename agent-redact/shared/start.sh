#!/usr/bin/env bash
# Start Gradio Pi chat UI in the background; keep container alive for `docker compose exec pi-agent pi`.
set -euo pipefail

export HOME="${HOME:-/home/user}"
export AGENT_WORKDIR="${AGENT_WORKDIR:-/workspace/doc_redaction}"
export PYTHONPATH="${AGENT_WORKDIR}:${AGENT_WORKDIR}/agent-redact:${AGENT_WORKDIR}/agent-redact/shared:${AGENT_WORKDIR}/agent-redact/pi:${AGENT_WORKDIR}/agent-redact/agentcore:${PYTHONPATH:-}"

cd "$AGENT_WORKDIR"

export APP_TYPE="${APP_TYPE:-agent}"
# Config file renamed agent.env (legacy: pi_agent.env). Prefer the new name.
if [ -z "${APP_CONFIG_PATH:-}" ]; then
  if [ -f "$AGENT_WORKDIR/config/agent.env" ]; then
    export APP_CONFIG_PATH="$AGENT_WORKDIR/config/agent.env"
  elif [ -f "$AGENT_WORKDIR/config/pi_agent.env" ]; then
    export APP_CONFIG_PATH="$AGENT_WORKDIR/config/pi_agent.env"
  else
    export APP_CONFIG_PATH="$AGENT_WORKDIR/config/agent.env"
  fi
fi

mkdir -p "${AGENT_WORKSPACE_DIR:-/home/user/app/workspace}"

if [ -z "${AGENT_INSTALL_EVAL_DEPS:-}" ] && [ -f "${APP_CONFIG_PATH:-}" ]; then
  _eval_flag="$(grep -E '^[[:space:]]*AGENT_INSTALL_EVAL_DEPS=' "${APP_CONFIG_PATH}" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '\r' | xargs || true)"
  case "${_eval_flag}" in
    1|true|True|yes|Yes|on|On) export AGENT_INSTALL_EVAL_DEPS=1 ;;
  esac
  unset _eval_flag
fi

# Optional DeepEval (post-run eval_summary.json). Prefer baking with INSTALL_AGENT_EVAL=1
# at image build; this path installs at container start when deps are missing.
_install_eval_deps_if_requested() {
  case "${AGENT_INSTALL_EVAL_DEPS:-}" in
    1|true|True|yes|Yes|on|On) ;;
    *) return 0 ;;
  esac
  if python3 -c "import deepeval" 2>/dev/null; then
    return 0
  fi
  echo "AGENT_INSTALL_EVAL_DEPS enabled — installing agent-redact/requirements_agent_eval.txt ..."
  pip install --user -r "${AGENT_WORKDIR}/agent-redact/requirements_agent_eval.txt"
}
_install_eval_deps_if_requested

python3 agent-redact/pi/pi_agent_config.py

if [ "${RUN_FASTAPI:-False}" = "True" ]; then
  exec uvicorn gradio_app:app \
    --app-dir agent-redact/shared \
    --host "${GRADIO_SERVER_NAME:-0.0.0.0}" \
    --port "${AGENT_GRADIO_PORT:-${GRADIO_SERVER_PORT:-7862}}"
else
  python3 agent-redact/shared/gradio_app.py &
fi

wait -n
