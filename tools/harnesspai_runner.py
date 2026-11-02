"""Run in the isolated upstream Python; native physics uses the workbench Python."""
import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Publish complete snapshots despite short Windows reader/scanner locks.

    Windows readers may deny delete-sharing, so replace can return access denied
    while another process reads the previous snapshot. Keep that snapshot intact;
    retry only Windows access/sharing locks for at most one second. Persistent
    permission errors still propagate to the caller with the original cause.
    """
    path = Path(path)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                prefix='.'+path.name+'.', suffix='.pending', dir=path.parent, delete=False) as stream:
            pending = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
        deadline = time.monotonic() + 1.0
        while True:
            try:
                pending.replace(path)
                return
            except OSError as exc:
                if os.name != 'nt' or getattr(exc, 'winerror', None) not in (5, 32, 33):
                    raise
                if time.monotonic() >= deadline:
                    exc.add_note(f'写入 {path.name} 时文件持续被占用或无替换权限；旧快照保留。请关闭占用该文件的程序或检查目录权限。')
                    raise
                time.sleep(.02)
    finally:
        if pending is not None:
            with suppress(OSError): pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--promote', action='store_true')
    args = parser.parse_args()
    folder = Path(args.run_dir).resolve()
    job = json.loads((folder / 'job.json').read_text(encoding='utf-8'))
    # Nothing in the adapter requires or modifies the upstream data/model trees.
    os.environ['HARNESSPAI_RUNS_DIR'] = str(folder / 'unused-upstream-runs')
    from harnesspai.evolution.candidate import prepare_candidate_workspace
    from harnesspai.execution.rollout import run_rollout
    candidate = folder / 'candidate' / 'workbench_candidate' / 'program.py'
    if args.promote:
        from harnesspai.code_library.promote import promote
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        rows = summary.get('harnesspai', {}).get('rows', [])
        if (summary.get('kind') != 'harnesspai' or summary.get('status') != 'verified' or len(job['seeds']) < 2
                or (job.get('mode') in ('auto','evolve') and not summary.get('harnesspai', {}).get('upstream_outcome', {}).get('converged'))
                or len(set(job['seeds'])) != len(job['seeds']) or [row.get('seed') for row in rows] != job['seeds']
                or not all(row.get('success') is True and row.get('process_ok') is True
                           and row.get('evaluation', {}).get('source') == 'mujoco_privileged_state'
                           and row.get('evaluation', {}).get('success') is True for row in rows)
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != job['source_sha256']):
            raise ValueError('候选程序缺少完整验证证据或内容已变化。')
        destination = Path(job.get('library_root', ROOT / 'methods' / 'harnesspai')) / folder.name
        if destination.exists(): raise FileExistsError('已验证版本目录已存在；不覆盖已有程序。')
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.promote-', dir=destination.parent))
        if not promote([str(candidate)], staging): raise RuntimeError('官方程序保存工具未完成写入；候选验证记录已保留。')
        write_json(staging / 'manifest.json', {key: job[key] for key in
            ('name', 'scene_id', 'task_id', 'seeds', 'source_sha256', 'upstream_revision')} | {'run_id': folder.name,
            **{key:job[key] for key in ('primitive_contract','evolution_seeds') if key in job},
            'scope': '通过这些种子、任务和 MuJoCo/Panda 配置的本地验证；不代表跨任务泛化。'})
        staging.rename(destination)
        return 0
    source = folder / 'source'
    package = source / 'workbench_candidate'
    package.mkdir(parents=True, exist_ok=False)
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'program.py').write_bytes((folder / 'program.py').read_bytes())
    sys.path.insert(0, str(source))
    candidate_root, targets, candidate_env = prepare_candidate_workspace(folder,
        [str(package / 'program.py')], 'workbench_candidate.program')
    candidate_env.update(PYTHONUTF8='1', PYTHONPATH=os.pathsep.join([str(candidate_root), str(ROOT)]))
    report = {'rows': [], 'message': '候选程序副本已创建。'}
    write_json(folder / 'progress.json', report)
    for seed in job['seeds']:
        if (folder / 'cancel.requested').exists(): break
        output = folder / 'episodes' / f'seed-{seed}'
        deadline = time.monotonic() + job['timeout_s']
        report.update(current_seed=seed, deadline_monotonic=deadline,
                      message=f'正在运行种子 {seed}；已完成 {len(report["rows"])}/{len(job["seeds"])}。')
        write_json(folder / 'progress.json', report)
        started = time.monotonic()
        result = run_rollout('tools.harnesspai_episode', seed=seed, video_out_dir=output,
            python_exe=job['workbench_python'], cwd=ROOT, timeout=job['timeout_s'], require_diagnostics=True,
            env=candidate_env, extra_args=['--program', targets[0], '--job', str(folder / 'job.json'), '--deadline', str(deadline)])
        native = json.loads((output / 'result.json').read_text(encoding='utf-8')) if (output / 'result.json').is_file() else {}
        valid = (result['process_ok'] and result['task_success'] is True and native.get('success') is True
                 and (native.get('evaluation') or {}).get('source') == 'mujoco_privileged_state'
                 and (native.get('evaluation') or {}).get('success') is True)
        row = {**native, 'seed': seed, 'process_ok': result['process_ok'], 'exit_code': result['exit_code'],
               'success': bool(valid), 'elapsed_s': time.monotonic()-started,
               'message': native.get('message') or (result['stderr'][-5000:] if not result['process_ok'] else None)}
        if not native: row['message'] = row['message'] or '缺少原生验收结果；不能认定成功。'
        report['rows'].append(row)
        write_json(folder / 'progress.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
