"""Verify official HarnessPAI rollout execution against EmbodiWorks MuJoCo.

Run with the isolated HarnessPAI Python (3.10–3.12). This is an integration
smoke test, not a task-solving policy or a reproduction of paper success rates.
The native child uses the workbench Python and writes independent evidence.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = '3af7d561d531ce3e3a3884a3f801c41b8b98e2fe'


def episode(args):
    from server.simulation import Simulation
    from server.task_verification import environment_success
    from server.video import SimulationVideo

    output = Path(args.video_out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sim = Simulation()
    video = SimulationVideo(output / 'trajectory.mp4', enabled=True, fps=20)
    try:
        sim.command({'action': 'scene', 'scene_id': 'drawer', 'task_id': 'open_drawer', 'seed': args.seed})
        before = sim.observe()
        for camera in before['cameras']:
            (output / (camera['name'] + '-before.jpg')).write_bytes(base64.b64decode(camera['image_base64']))
        target = {**before['tcp'], 'status': 'move', 'gripper_open': 1., 'duration_s': .5,
                  'explanation': 'Integration validation: move the real robot TCP up by one centimetre.'}
        target['position_m'] = list(target['position_m'])
        target['position_m'][2] += .01
        result = sim.command({'action': 'move_tcp', 'generation': before['generation'], 'pose': target})
        for _ in range(1500):
            sim.step()
            if video.due(float(sim.data.time)):
                video.capture(sim.render_rgb(), float(sim.data.time))
            if sim.task['status'] not in ('running', 'settling'):
                break
        motion_status = sim.task.copy()
        sim.command({'action': 'stop'})
        verification = sim.verify_task(sim.generation)
        after = sim.observe()
        for camera in after['cameras']:
            (output / (camera['name'] + '-after.jpg')).write_bytes(base64.b64decode(camera['image_base64']))
        evidence = {'backend': sim.state()['backend'], 'before_tcp': before['tcp'], 'action': target,
                    'execution': result, 'motion_status': motion_status,
                    'after_tcp': after['tcp'], 'evaluation': verification,
                    'seed': args.seed, 'sim_time_s': float(sim.data.time)}
        (output / 'native_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
        diagnostic = {'success': environment_success(sim.state()), 'termination': 'smoke_program_finished',
                      'scope': 'native integration smoke; no language, vision or action model',
                      'native_source': verification.get('source')}
        (output / 'diagnostics.jsonl').write_text(json.dumps(diagnostic) + '\n', encoding='utf-8')
    finally:
        video.finalize()
        sim.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--video-out', required=True)
    parser.add_argument('--workbench-python', default=str(ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')))
    args = parser.parse_args()
    if args.episode:
        return episode(args)
    if os.name == 'nt' and not sys.flags.utf8_mode:
        parser.error('On Windows, launch the isolated Python with -X utf8 so upstream subprocess logs decode correctly.')
    from harnesspai.execution.rollout import run_rollout
    result = run_rollout('tools.harnesspai_smoke', seed=args.seed, episode_start=args.episode_start,
                         video_out_dir=args.video_out, python_exe=args.workbench_python,
                         extra_args=['--episode'], cwd=ROOT, timeout=90, require_diagnostics=True,
                         env={'PYTHONUTF8': '1'})
    report = {key: result[key] for key in ('process_ok', 'task_success', 'success', 'exit_code', 'video_path', 'termination')}
    report['upstream_revision'] = REVISION
    output = Path(args.video_out)
    (output / 'harnesspai_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if not result['process_ok']:
        print(result['stderr'], file=sys.stderr)
        return 1
    if result['task_success'] is not False or not result['video_path']:
        raise RuntimeError('Expected an unfinished drawer task with native video; this is not a success benchmark.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
