"""Adapt the upstream episode CLI to the workbench's native, immutable evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main(program):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--video-out', required=True)
    parser.add_argument('--suite')
    args = parser.parse_args()
    job_path = Path(os.environ['HARNESSPAI_WORKBENCH_JOB'])
    job = json.loads(job_path.read_text(encoding='utf-8'))
    if (job_path.parent / 'cancel.requested').exists():
        raise RuntimeError('用户停止了程序演化。')
    # Each rollout freezes this generation before any candidate code is imported.
    job['source_sha256'] = hashlib.sha256(Path(program).read_bytes()).hexdigest()
    job['cancel_file'] = str(job_path.parent / 'cancel.requested')
    seed = job['evolution_seeds'][args.episode_start]
    output = Path(args.video_out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'episode-job.json'
    frozen.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
    sys.argv = [sys.argv[0], '--seed', str(seed), '--episode-start', str(args.episode_start),
                '--video-out', str(output), '--program', str(program), '--job', str(frozen),
                '--deadline', str(time.monotonic() + job['timeout_s'])]
    from tools.harnesspai_episode import main as episode
    return episode()
