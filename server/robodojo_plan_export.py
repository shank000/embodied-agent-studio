"""Portable official-client scripts; export never starts a simulator or policy."""
import shlex

from .robodojo import EvaluationPlan, REVISION, plan


def export_plan(spec):
    spec = spec if isinstance(spec, EvaluationPlan) else EvaluationPlan.model_validate(spec)
    prepared = plan(spec)
    # Use the same validated arguments as the plan. Its local source path must
    # not leak into a script intended to run from a different Linux checkout.
    commands = []
    for row in prepared['commands']:
        arguments = ['run_client', *row['argv'][2:]]
        # The pinned shell client constructs ws://host:port without adding
        # IPv6 brackets (the Python native runner handles this separately).
        if ':' in spec.policy_host:
            arguments[arguments.index('--policy-host')+1] = f'[{spec.policy_host}]'
        commands.append(shlex.join(arguments))
    script = f'''#!/usr/bin/env bash
# EmbodiWorks / RoboDojo official client plan
# Run from the prepared RoboDojo repository, or set ROBODOJO_ROOT.
# First inspect without starting Isaac Sim: bash this-file.sh --dry-run
# Start the selected external XPolicyLab policy server before real execution.
# Results and native videos remain in eval_result/RoboDojo; import _result.json
# in EmbodiWorks after completion. This script does not upload or delete data.
set -euo pipefail
extra=()
if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
  echo "Usage: bash this-file.sh [--dry-run]" >&2; exit 2
fi
if [[ $# -eq 1 ]]; then extra=(--dry-run); fi
cd -- "${{ROBODOJO_ROOT:-$PWD}}"
if [[ ! -f scripts/robodojo.sh ]]; then
  echo "Run this script from the RoboDojo source root, or set ROBODOJO_ROOT." >&2; exit 2
fi
if [[ "$(git rev-parse HEAD)" != {shlex.quote(REVISION)} ]]; then
  echo "RoboDojo revision differs from this plan: {REVISION}." >&2; exit 2
fi
command -v timeout >/dev/null || {{ echo "GNU timeout (coreutils) is required." >&2; exit 2; }}
SECONDS=0
job=0
# The official CLI leaves inherited EVAL_NUM unchanged for --eval-num native.
export EVAL_NUM={shlex.quote(str(spec.episodes))}
run_client() {{
  local remaining=$(({spec.max_wall_time_s} - SECONDS))
  if (( remaining <= 0 )); then echo "Evaluation wall-time budget exhausted." >&2; return 124; fi
  job=$((job + 1))
  export ROBODOJO_RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$-$job"
  timeout --signal=TERM --kill-after=15s "${{remaining}}s" bash scripts/robodojo.sh "$@" "${{extra[@]}}"
}}
{chr(10).join(commands)}
echo "Client commands finished. Inspect native terminal results in eval_result/RoboDojo."
'''
    return {'filename': prepared['plan_id']+'.sh', 'script': script,
            'plan_id': prepared['plan_id'], 'revision': REVISION,
            'execution': 'manual_linux', 'ready': prepared['ready']}
