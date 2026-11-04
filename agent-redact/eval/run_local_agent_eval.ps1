# Local agent regression (Windows). Requires Docker compose stack + config/agent.env.
param(
    [string]$Fixture = "agent-redact/eval/fixtures/example_live",
    [ValidateSet("langgraph", "pi", "both")]
    [string]$Orchestrator = "langgraph",
    [switch]$GatesOnly,
    [string]$Provider = "",
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $RepoRoot

if (-not (Test-Path "config/agent.env")) {
    Write-Warning "config/agent.env not found — copy from config/agent.env.example and set AGENT_* / LLM URLs."
}

$args = @(
    "agent-redact/eval/run_agent_regression.py",
    "--fixture", $Fixture,
    "--orchestrator", $Orchestrator
)
if ($GatesOnly) { $args += "--gates-only" }
if ($Provider) { $args += @("--provider", $Provider) }
if ($Model) { $args += @("--model", $Model) }

Write-Host "Running: python $($args -join ' ')"
python @args
exit $LASTEXITCODE
