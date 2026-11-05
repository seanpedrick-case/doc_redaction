#!/usr/bin/env bash
# Local agent regression (Linux/macOS). Requires Docker compose stack + config/agent.env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GOLD="${GOLD:-${FIXTURE:-agent-redact/eval/gold/example_live}}"
ORCHESTRATOR="${ORCHESTRATOR:-langgraph}"
GATES_ONLY="${GATES_ONLY:-0}"
PROVIDER="${PROVIDER:-}"
MODEL="${MODEL:-}"

if [[ ! -f config/agent.env ]]; then
  echo "warning: config/agent.env not found — copy from config/agent.env.example" >&2
fi

args=(
  agent-redact/eval/run_agent_regression.py
  --gold "$GOLD"
  --orchestrator "$ORCHESTRATOR"
)
if [[ "$GATES_ONLY" == "1" ]]; then
  args+=(--gates-only)
fi
if [[ -n "$PROVIDER" ]]; then
  args+=(--provider "$PROVIDER")
fi
if [[ -n "$MODEL" ]]; then
  args+=(--model "$MODEL")
fi

echo "Running: python ${args[*]}"
exec python "${args[@]}"
