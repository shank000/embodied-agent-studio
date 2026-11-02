"""Official HarnessPAI evolution with workbench model transport and native episodes.

The upstream edit/probe/batch/analyze algorithm is executed unchanged. Runtime,
model transport and scoped memory roots are explicit host adapters. This local
MuJoCo profile is not the paper's LIBERO/pi0.5 benchmark profile.
"""
import argparse
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from tools.harnesspai_runner import ROOT, write_json


def apply_files(result, cwd, allowed_program=None, memory_root=None, *, require_change=True):
    """Validate the whole edit before writing; model cannot edit the evaluator."""
    replacements, seen = [], set()
    for item in result.get('files', []):
        path = Path(item['path'])
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        permitted = allowed_program and path == allowed_program.resolve()
        if memory_root and path.is_relative_to(memory_root.resolve()) and path.suffix in ('.md', '.json'):
            permitted = True
        if not permitted or path in seen:
            raise ValueError(f'模型试图修改未授权文件或重复路径：{item["path"]}')
        content = item['content'].replace('\r\n', '\n').replace('\r', '\n')
        if path == allowed_program:
            tree = ast.parse(content)
            compile(tree, str(path), 'exec')
            if not any(isinstance(node, ast.FunctionDef) and node.name == 'run' for node in tree.body):
                raise ValueError('改写后缺少 def run(ctx) 入口。')
            if any(isinstance(node, ast.Attribute) and node.attr.startswith('_') for node in ast.walk(tree)):
                raise ValueError('演化程序不得访问私有属性；使用公开观察与控制接口。')
        if path.suffix == '.json': json.loads(content)
        replacements.append((path, content)); seen.add(path)
    if allowed_program and require_change and not any(path == allowed_program and
            content != allowed_program.read_text(encoding='utf-8') for path, content in replacements):
        raise ValueError('编码模型未返回候选程序的有效修改，已停止重复验证。模型说明：' +
                         str(result.get('explanation') or '未提供原因。')[:3000])
    for path, content in replacements:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + '.pending')
        pending.write_text(content, encoding='utf-8', newline='')
        pending.replace(path)


def observation_evidence(folder, batches):
    """Only evolution observations are supplied; holdout episodes stay withheld."""
    if not batches: return [], []
    rows = batches[-1]['rows']
    row = next((item for item in rows if not item.get('success')), rows[0] if rows else None)
    if not row or not row.get('record_path'): return [], []
    root = folder / row['artifact_dir'] / row['record_path']
    events = root / 'events.jsonl'
    if not events.is_file(): return [], []
    initial, final, motions = None, None, []
    for line in events.read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        data = item.get('data', item)
        if item.get('event') in ('observation', 'observation_after'):
            observation = data.get('observation')
            if observation:
                if initial is None: initial = observation
                final = observation
        if item.get('event') == 'motion_feedback': motions.append(data)
    context, images = [], []
    for phase, observation in [('initial', initial), ('final', final)]:
        if not observation or (phase == 'final' and final == initial): continue
        context.append({'phase': phase, 'seed': row['seed'], 'observation': observation})
        for camera in observation.get('cameras', []):
            path = root / camera['image_ref']
            images.append({'name': phase + '_' + camera['name'], 'path': path.relative_to(folder).as_posix()})
    if motions: context.append({'motion_feedback': motions[-20:]})
    return context, images


async def run(folder):
    job_path = folder / 'job.json'
    job = json.loads(job_path.read_text(encoding='utf-8'))
    os.environ['HARNESSPAI_RUNS_DIR'] = str(folder / 'unused-upstream-runs')
    from harnesspai import config
    # Process-local host adapters; never rewrite the vendor checkout or its config.
    config.GRAPH_LIBRARY_DIR = folder / 'graphs'
    memory_home = Path(job['memory_root'])
    config.SKILL_LIBRARY_DIR = folder / 'memory' / 'skills'
    config.CODE_LIBRARY_DIR = folder / 'memory' / 'code'
    latest = memory_home / 'latest.json'
    if latest.is_file():
        memory_id = json.loads(latest.read_text(encoding='utf-8'))['run_id']
        saved = (memory_home / 'versions' / memory_id).resolve()
        if saved.parent != (memory_home / 'versions').resolve(): raise ValueError('经验库版本路径无效。')
        shutil.copytree(saved / 'skills', config.SKILL_LIBRARY_DIR)
    config.models = lambda: {'coder': {'model': job['coder']['model'], 'drop_proxy_env': []},
                            'vlm': {'model': 'workbench-configured-model'}}
    from harnesspai.evolution import loop
    from harnesspai.envs.registry import Runtime
    from harnesspai.execution.rollout import run_rollout_batch
    from harnesspai.execution import rollout as rollout_module
    from harnesspai.memory.skill import graph_skill as gs
    gs.BENCHMARKS = (*gs.BENCHMARKS, 'mujoco')
    from harnesspai.memory.skill import writeback
    upstream_rollout = rollout_module.run_rollout
    def timed_rollout(*args, **kwargs):
        started = time.monotonic()
        result = upstream_rollout(*args, **kwargs)
        result['elapsed_s'] = time.monotonic() - started
        return result
    rollout_module.run_rollout = timed_rollout

    source = folder / 'source' / 'workbench_candidate'
    source.mkdir(parents=True, exist_ok=False)
    (source / '__init__.py').write_text('', encoding='utf-8')
    (source / 'program.py').write_bytes((folder / 'program.py').read_bytes())
    (source / '__main__.py').write_text(
        f'import sys\nsys.path.insert(0, {str(ROOT)!r})\n'
        'from pathlib import Path\nfrom tools.harnesspai_entry import main\n'
        'raise SystemExit(main(Path(__file__).with_name("program.py")))\n', encoding='utf-8')
    sys.path.insert(0, str(source.parent))
    graph_path = folder / 'graphs' / 'mujoco' / 'task.json'
    graph_path.parent.mkdir(parents=True)
    write_json(graph_path, {'task': job['task_instruction'], 'task_name': job['task_id'],
        'nodes': [{'id': 's0', 'type': 'state', 'label': 'observe robot'},
                  {'id': 'a1', 'type': 'action', 'label': job['task_id']},
                  {'id': 's1', 'type': 'state', 'label': job['task_instruction'], 'terminal': True}],
        'start': 's0', 'edges': [{'from': 's0', 'to': 'a1'}, {'from': 'a1', 'to': 's1'}]})
    candidate = folder / 'evolution' / 'candidate' / 'workbench_candidate' / 'program.py'
    report = {'mode': job.get('mode','evolve'), 'source_origin':job.get('source_origin'), 'rows': [], 'batches': [], 'model_calls': 0, 'training': False,
              'scope': 'official-evolution/workbench-mujoco', 'stage': 'baseline'}
    def progress(message):
        report.update(message=message, deadline_monotonic=None)
        write_json(folder / 'progress.json', report)
    def cancelled():
        if (folder / 'cancel.requested').exists(): raise RuntimeError('用户停止了程序演化。')

    async def model(prompt, *, cwd, sandbox='read-only', **kwargs):
        cancelled()
        cwd = Path(cwd).resolve()
        editable = sandbox != 'read-only'
        memory_root = config.SKILL_LIBRARY_DIR / 'mujoco' if cwd.is_relative_to(config.SKILL_LIBRARY_DIR) else None
        allowed = candidate if editable and memory_root is None else None
        context = []
        if candidate.is_file(): context.append(f'CURRENT PROGRAM: {candidate}\n{candidate.read_text(encoding="utf-8")}')
        # Public interface and seed-independent example; no privileged state access.
        context.append('HOST EXECUTION CONTRACT AND PUBLIC INTERFACE:\n' +
                       (ROOT / 'docs/HARNESSPAI_PROGRAM_API.md').read_text(encoding='utf-8'))
        context.append('CONFIGURED INFERENCE PRIMITIVES (credentials held by host):\n'+json.dumps({
            'action_model':job.get('action_model'),'tools':job.get('tools',[])},ensure_ascii=False)+
            '\nUse ctx.policy(instruction) and ctx.execute_policy(output) for a configured action model; '
            'ctx.tool(name,instruction,payload) for configured perception/decision tools. '
            'Do not create direct HTTP clients or hardcode service URLs in candidate code. '
            'If no action model is configured, use public RGB/robot observations and ctx.move_tcp.')
        context.append('REACH-UP API EXAMPLE ONLY, not a solution for other tasks:\n' +
                       (ROOT / 'examples/harnesspai/reach_up.py').read_text(encoding='utf-8'))
        context.append('ctx.observe() returns the current tcp (including position_m), three cameras and robot state. '
                       "The gripper opening is observation['gripper']['target_open_fraction'], NOT a tcp field. "
                       'ctx.move_tcp(pose) uses the exact pose fields shown in the example. '
                       'ctx.note(text) writes a log. No private fields. '
                       f'At most {job["max_steps"]} moves and {job["sim_seconds"]} simulated seconds per episode. '
                       'Do not include imports from capx/LIBERO; this profile runs workbench MuJoCo.')
        skill_cards = gs.skill_cards_for_phase('workbench', job['task_id'],
            graph=json.loads(graph_path.read_text(encoding='utf-8')), bench='mujoco')
        if skill_cards:
            context.append('RETRIEVED VERIFIED SKILLS (official graph-node lookup):\n' + skill_cards[:18000])
            report['retrieved_memory'] = json.loads(latest.read_text(encoding='utf-8'))['run_id'] if latest.is_file() else None
        feedback = []
        for recent in report['batches'][-2:]:
            selected = [row for row in recent['rows'] if not row.get('success')] or recent['rows']
            feedback.append({'batch': recent['path'], 'success_rate': recent['success_rate'],
                'episode_count': len(recent['rows']), 'evidence_sample_max': 5,
                'sample': [{key: row.get(key) for key in ('seed', 'success', 'message', 'evaluation', 'steps')}
                           for row in selected[:5]]})
        context.append('RECENT NATIVE FEEDBACK (holdout results excluded):\n' + json.dumps(feedback, ensure_ascii=False))
        observations, images = observation_evidence(folder, report['batches'])
        context.append('ACTUAL PUBLIC OBSERVATIONS AND MOTION FEEDBACK:\n' + json.dumps(observations, ensure_ascii=False))
        if memory_root:
            for path in sorted(memory_root.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.md'):
                    context.append(f'FILE: {path}\n{path.read_text(encoding="utf-8")[:12000]}')
        context.append('ALLOWED REPLACEMENTS: ' + (str(memory_root) + '/**/*.md or *.json' if memory_root else str(allowed) if allowed else 'NONE; files must be []'))
        report['model_calls'] += 1
        call_id = report['model_calls']
        progress('模型正在整理经验。' if memory_root else '模型正在改写候选程序。' if editable else '模型正在诊断失败并提出下一轮修改。')
        write_json(folder / 'model-request.json', {'id': call_id, 'editable': editable,
            'role': 'memory' if memory_root else 'edit' if editable else 'diagnosis',
            'images': images,
            'prompt': 'HOST CONTRACT: Return JSON replacement files; the host writes files and executes tests. '
                      'You do not directly edit files or run commands. The upstream workflow below is reference context; '
                      'use def run(ctx) and the host API, not upstream entrypoints.\n\n' + prompt +
                      '\n\nWORKBENCH ADAPTER CONTEXT (authoritative execution contract):\n' + '\n\n'.join(context)})
        response_path = folder / f'model-response-{call_id}.json'
        deadline = time.monotonic() + job['coder']['timeout_s'] + 30
        while not response_path.exists():
            cancelled()
            if time.monotonic() > deadline: raise TimeoutError('平台模型调用未在预算内返回。')
            await asyncio.sleep(.2)
        response = json.loads(response_path.read_text(encoding='utf-8'))
        if response.get('error'): raise RuntimeError(response['error'])
        result = response['result']
        report['last_model_response'] = {'id': call_id, 'role': 'memory' if memory_root else 'edit' if editable else 'diagnosis',
            'explanation': result.get('explanation'), 'file_count': len(result.get('files', []))}
        progress('正在检查模型返回的修改。')
        apply_files(result, cwd, allowed, memory_root if editable else None,
                    require_change=not report['batches'] or report['batches'][-1]['success_rate'] < 1)
        text = result['explanation']
        return {'ok': True, 'last_text': text, 'events': [
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': text}},
            {'type': 'turn.completed', 'usage': response.get('usage') or {}}]}

    runtime_env = {'PYTHONUTF8': '1', 'HARNESSPAI_WORKBENCH_JOB': str(job_path),
                   'PYTHONPATH': str(ROOT)}
    loop.bench_runtime = lambda bench: Runtime(bench, job['workbench_python'], runtime_env, ())
    loop.run_codex = model
    writeback.run_codex = model

    def batch(task_module, **kwargs):
        cancelled()
        kwargs['timeout'] = job['timeout_s'] + 5
        env = kwargs.setdefault('env', {}).copy()
        env.update(PYTHONUTF8='1', HARNESSPAI_WORKBENCH_JOB=str(job_path))
        env['PYTHONPATH'] = os.pathsep.join([str(kwargs['cwd']), str(ROOT), env.get('PYTHONPATH', '')])
        kwargs['env'] = env
        progress('正在执行固定程序并读取环境验收结果。')
        started = time.monotonic()
        result = run_rollout_batch(task_module, **kwargs)
        rows = []
        for item in result.get('episode_results', []):
            output = Path(item['video_out_dir'])
            native_path = output / 'result.json'
            native = json.loads(native_path.read_text(encoding='utf-8')) if native_path.exists() else {}
            native_success = (native.get('success') is True and item.get('process_ok') is True and
                              native.get('evaluation', {}).get('source') == 'mujoco_privileged_state' and
                              native.get('evaluation', {}).get('success') is True)
            item['success'] = item['task_success'] = native_success
            rows.append({**native, 'seed': job['evolution_seeds'][item['episode']],
                         'success': native_success, 'process_ok': item.get('process_ok'),
                         'elapsed_s': item.get('elapsed_s'),
                         'message': native.get('message') or item.get('stderr'),
                         'artifact_dir': output.relative_to(folder).as_posix()})
        result['success'] = result['task_success'] = bool(rows) and all(row['success'] for row in rows)
        result['success_count'] = sum(row['success'] for row in rows)
        result['success_rate'] = result['success_count'] / len(rows) if rows else 0
        result['failed_episodes'] = [item['episode'] for item in result.get('episode_results', []) if not item['success']]
        report['batches'].append({'path': Path(kwargs['video_out_dir']).relative_to(folder).as_posix(),
            'stage': report['stage'], 'elapsed_s': time.monotonic() - started,
            'success_rate': result['success_rate'], 'rows': rows})
        progress(f'本批次通过 {result["success_count"]}/{len(rows)}；未改动模型权重。')
        return result
    loop.run_rollout_batch = batch
    # Baseline is evaluated on evolution seeds; holdout evidence is never fed back.
    baseline = batch('workbench_candidate', seed=0, episode_start=0, episode_end=len(job['evolution_seeds']),
          video_out_dir=folder / 'baseline', env=runtime_env,
          cwd=source.parent, python_exe=job['workbench_python'], require_diagnostics=True)
    reused = job.get('mode') == 'auto' and baseline['success']
    if reused:
        candidate = source/'program.py'
        outcome = {'converged':True,'reused_without_evolution':True,'rounds_run':0}
        progress('已有程序通过当前开发种子复测，直接进入最终验证；不调用编码模型。')
    else:
        report['stage'] = 'evolution'
        outcome = await loop.evolve(job['task_instruction'], targets=[str(source / 'program.py')],
            task_module='workbench_candidate', graph=str(graph_path), suite='workbench',
            rounds=job['evolution_rounds'], episode_start=0, episode_end=len(job['evolution_seeds']),
            run_dir=folder / 'evolution', benchmark='mujoco', skill_writeback=False,
            use_video_diagnosis=False, codex_model=job['coder']['model'], smoke_retries=1,
            probe_max_attempts=2, codex_timeout=job['coder']['timeout_s'],
            codex_synth_timeout=job['coder']['timeout_s'])
    report['upstream_outcome'] = outcome
    cancelled()
    # A fresh, independent phase-1 validation run freezes the final source.
    report['stage'] = 'holdout'
    final_source = candidate.read_bytes()
    job.update(initial_source_sha256=job['source_sha256'], source_sha256=hashlib.sha256(final_source).hexdigest())
    write_json(job_path, job)
    frozen = folder / 'candidate' / 'workbench_candidate'
    frozen.mkdir(parents=True)
    (frozen / 'program.py').write_bytes(final_source)
    from harnesspai.execution.rollout import run_rollout
    for seed in job['seeds']:
        cancelled()
        output = folder / 'episodes' / f'seed-{seed}'
        deadline = time.monotonic() + job['timeout_s']
        progress(f'保留种子 {seed}：最终验证；此结果不会回传编码模型。')
        started = time.monotonic()
        result = run_rollout('tools.harnesspai_episode', seed=seed, video_out_dir=output,
            python_exe=job['workbench_python'], cwd=ROOT, timeout=job['timeout_s'], require_diagnostics=True,
            env=runtime_env, extra_args=['--program', str(frozen / 'program.py'), '--job', str(job_path), '--deadline', str(deadline)])
        native = json.loads((output / 'result.json').read_text(encoding='utf-8')) if (output / 'result.json').exists() else {}
        valid = result['process_ok'] and native.get('success') is True and native.get('evaluation', {}).get('success') is True
        report['rows'].append({**native, 'seed': seed, 'success': bool(valid), 'process_ok': result['process_ok'],
                              'elapsed_s': time.monotonic()-started, 'message': native.get('message') or result.get('stderr')})
        progress(f'已完成 {len(report["rows"])}/{len(job["seeds"])} 个保留种子。')
    # Only native-passed programs may teach reusable repair skills.
    if not reused and outcome.get('converged') and all(row['success'] for row in report['rows']):
        report['stage'] = 'memory'
        skill = await writeback.write_skills(suite='workbench', task_name=job['task_id'],
            graph_path=str(graph_path), run_dir=folder / 'evolution', targets=[str(source / 'program.py')],
            promoted=[str(frozen / 'program.py')], codex_model=job['coder']['model'], bench='mujoco')
        cards = list(config.SKILL_LIBRARY_DIR.glob('mujoco/graphs/*/nodes/*/*.md'))
        if skill.get('ok') and not cards:
            skill.update(ok=False, error='模型未生成可检索的经验卡；未更新共享经验库。')
        if skill.get('ok'):
            try:
                saved = memory_home / 'versions' / folder.name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(config.SKILL_LIBRARY_DIR, saved / 'skills')
                write_json(memory_home / 'latest.json', {'run_id': folder.name})
                skill['version'] = folder.name
            except OSError as exc:
                skill.update(ok=False, error=f'经验版本保存失败；旧库保持不变：{type(exc).__name__}: {exc}')
        report['skill_writeback'] = skill
    report['stage'] = 'complete'
    progress('程序演化与保留种子验证完成。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    asyncio.run(run(Path(args.run_dir).resolve()))


if __name__ == '__main__': main()
