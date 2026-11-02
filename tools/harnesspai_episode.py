"""Native episode adapter. The candidate never supplies the success verdict."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from threading import Timer
import time
import traceback
from uuid import uuid4


class ProgramContext:
    def __init__(self, simulation, log, job):
        self._sim, self._log, self._job = simulation, log, job
        self._steps = 0
        self.seed = simulation.seed
        self.task = {'scene_id': job['scene_id'], 'task_id': job['task_id']}
        self.capabilities = {'action_model': deepcopy(job.get('action_model')), 'tools': deepcopy(job.get('tools',[]))}
        self._policy_calls = 0
        self._policy_executions = 0

    def _primitive(self, kind, instruction, *, name=None, payload=None):
        from tools.harnesspai_runner import write_json
        self._check_cancel()
        config = self._job.get('action_model') if kind == 'policy' else next(
            (tool for tool in self._job.get('tools',[]) if tool['name'] == name), None)
        if not config: raise ValueError('未配置动作模型或工具：'+str(name or kind))
        if not isinstance(instruction,str) or not 1 <= len(instruction) <= 2000: raise ValueError('工具指令须为 1–2000 字符。')
        if kind == 'policy': self._policy_calls += 1
        directory = Path(self._job['broker_dir'])
        directory.mkdir(exist_ok=True)
        call = uuid4().hex
        request_path, response_path = directory/f'{call}.request.json', directory/f'{call}.response.json'
        deadline = min(self._job['episode_deadline']-.5, time.monotonic()+config['timeout_s']+10)
        write_json(request_path, {'kind':kind,'name':name,'instruction':instruction,'input':payload or {},
            'observation':self.observe(),'seed':self.seed,'reset':self._policy_calls==1,
            'step':self._steps+1,'deadline_monotonic':deadline})
        while not response_path.exists():
            self._check_cancel()
            if time.monotonic() >= deadline: raise TimeoutError('动作模型或工具调用超时；未执行新的动作。')
            time.sleep(.05)
        response = json.loads(response_path.read_text(encoding='utf-8'))
        if response.get('error'): raise RuntimeError(response['error'])
        return response['result']

    def policy(self, instruction):
        return self._primitive('policy', instruction)

    def tool(self, name, instruction, payload=None):
        return self._primitive('tool', instruction, name=name, payload=payload)

    def execute_policy(self, output):
        from server.policy import PolicyOutput
        result = PolicyOutput.model_validate(output)
        if result.action_type == 'tcp_pose' and result.pose.status != 'move':
            raise ValueError('策略返回 done/blocked；程序须自行处理，完成情况仍由环境验收。')
        self._policy_executions += 1
        if result.action_type == 'tcp_pose': return self.move_tcp(result.pose.model_dump())
        return self._execute({'action':'policy_chunk','actions':result.actions,'joint_names':result.joint_names,
                              'dt_s':result.dt_s,'robot_id':result.robot_id}, policy_output=result.model_dump())

    def observe(self):
        # Simulation.observe deliberately excludes evaluator/scene ground truth.
        self._check_cancel()
        return self._sim.observe()

    def budget(self):
        """Execution limits only; never expose evaluator or object ground truth."""
        self._check_cancel()
        return {'actions_used': self._steps, 'actions_remaining': max(0,self._job['max_steps']-self._steps),
                'sim_time_s': float(self._sim.data.time),
                'sim_remaining_s': max(0.,self._job['sim_seconds']-float(self._sim.data.time)),
                'wall_remaining_s': max(0.,self._job['episode_deadline']-time.monotonic())}

    def _check_cancel(self):
        if Path(self._job['cancel_file']).exists(): raise RuntimeError('用户停止了程序验证。')

    def note(self, message):
        self._log.write('program_note', {'round': self._steps, 'message': str(message)[:10000]})

    def _capture(self):
        self._check_cancel()
        t = float(self._sim.data.time)
        if self._log.video_due(t): self._log.capture_video_frame(self._sim.render_rgb(), t)

    def move_tcp(self, pose):
        return self._execute({'action':'move_tcp','pose':pose})

    def _execute(self, command, policy_output=None):
        from server.task_verification import environment_success
        from server.rounds import evaluation_snapshot, transition_evidence
        if self._steps >= self._job['max_steps']: raise RuntimeError('达到程序动作步数预算。')
        self._steps += 1
        step, sim, log = self._steps, self._sim, self._log
        started = time.monotonic()
        before = sim.state()
        evaluation = evaluation_snapshot
        log.write('round_start', {'round': step, 'decision_id': f'{log.id}:round:{step}', 'before': evaluation(before)})
        error = None
        try:
            observation = self.observe()
            saved = log.save_observation(observation, step)
            log.write('observation', {'round': step, 'observation': saved})
            if policy_output is None: log.write('action', {'round': step, 'action': deepcopy(command['pose'])})
            else: log.write('policy_action', {'round':step,'output':policy_output})
            if float(sim.data.time) >= self._job['sim_seconds']: raise RuntimeError('达到仿真时间预算。')
            result = sim.command({**command, 'generation': observation['generation']})
            while sim.task['status'] in ('running', 'settling'):
                if float(sim.data.time) >= self._job['sim_seconds']: raise RuntimeError('达到仿真时间预算。')
                sim.step()
                self._capture()
            motion = deepcopy(sim.task)
            log.write('motion_feedback', {'round': step, 'motion': motion, 'tcp': self.observe()['tcp']})
            log.write_trajectory(sim.trajectory_frame())
            return {'execution': result, 'motion': motion, 'tcp': self.observe()['tcp']}
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            sim.command({'action': 'stop'})
            after = sim.state()
            try:
                saved = log.save_observation(sim.observe(), step, phase='after')
                log.write('observation_after', {'round': step, 'observation': saved})
            finally:
                elapsed = time.monotonic()-started
                log.write('round_end', {'round': step, 'decision_id': f'{log.id}:round:{step}',
                    'status': 'error' if error else 'complete', 'error': error,
                    'elapsed_s': elapsed, 'after': evaluation(after),
                    **transition_evidence(evaluation(before), evaluation(after), elapsed)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--episode-start', type=int, default=0)
    parser.add_argument('--video-out', required=True)
    parser.add_argument('--program', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    # Windows venv launchers can outlive subprocess.run's direct-child kill.
    # The actual native interpreter exits before the outer rollout deadline,
    # including when candidate code sleeps without calling context methods.
    def expired():
        try: os.write(2, b'\n[workbench] TIMEOUT: native program exceeded its wall-clock budget.\n')
        finally: os._exit(124)
    watchdog = Timer(max(.001, args.deadline - time.monotonic() - .25), expired)
    watchdog.daemon = True
    watchdog.start()
    output = Path(args.video_out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    job = json.loads(Path(args.job).read_text(encoding='utf-8'))
    job['episode_deadline'] = args.deadline
    job.setdefault('cancel_file', str(Path(args.job).parent / 'cancel.requested'))
    from server.simulation import Simulation
    from server.runlog import RunLog
    from server.task_verification import environment_success
    from server.scenes import TASKS
    log = RunLog('program', {'model': 'HarnessPAI · '+job['name'], 'seed': args.seed,
        'scene_id': job['scene_id'], 'task_id': job['task_id'], 'program_sha256': job['source_sha256'],
        'backend': Simulation.backend_identity(), 'robot': 'panda', 'task': TASKS[job['task_id']]['instruction'],
        'external_telemetry': True}, directory=output / 'records', record_video=job['record_video'])
    log._call_costs.append({'status': 'unavailable', 'total_calls': 0, 'missing_calls': 1,
        'call_count_incomplete': True, 'warnings': ['自定义程序内部的外部模型请求尚未接入用量计量。']})
    (output / 'record.json').write_text(json.dumps({'run_id': log.id}), encoding='utf-8')
    sim, ctx, error, success, evaluation = None, None, None, False, None
    try:
        program = Path(args.program)
        if hashlib.sha256(program.read_bytes()).hexdigest() != job['source_sha256']:
            raise ValueError('候选程序与冻结快照不一致。')
        sim = Simulation(model_id='panda', scene_id=job['scene_id'], seed=args.seed, task_id=job['task_id'])
        log.save_checkpoint(sim.checkpoint())
        ctx = ProgramContext(sim, log, job)
        ctx._capture()
        log.write('observation', {'round': 0, 'observation': log.save_observation(ctx.observe(), 0)})
        spec = importlib.util.spec_from_file_location('frozen_workbench_program', program)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        returned = module.run(ctx)
        if hasattr(returned, '__await__'): raise ValueError('run(ctx) 必须为同步函数。')
        if job.get('action_model') and (ctx._policy_calls == 0 or ctx._policy_executions == 0):
            raise ValueError('已选择动作模型，但程序没有调用并执行该模型的策略输出；不能把纯 IK 运行标为模型实验。')
        if hashlib.sha256(program.read_bytes()).hexdigest() != job['source_sha256']:
            raise ValueError('程序执行期间修改了候选源文件，验证结果无效。')
        sim.command({'action': 'stop'})
        evaluation = sim.verify_task(sim.generation)
        success = environment_success(sim.state())
        log.write('completion_verification', {'accepted': success, 'evaluation': evaluation})
        log.save_checkpoint(sim.checkpoint(), 'final')
        ctx._capture()
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
        if sim:
            sim.command({'action': 'stop'})
            evaluation = sim.state().get('evaluation')
    finally:
        if sim: sim.close()
        result = {'status': 'error' if error else 'verified' if success else 'unverified',
                  'success': success and not error, 'message': error, 'evaluation': evaluation,
                  'backend': Simulation.backend_identity(),
                  'steps': ctx._steps if ctx else 0, 'termination': 'program_error' if error else 'native_evaluation'}
        result['policy_calls'] = ctx._policy_calls if ctx else 0
        result['policy_executions'] = ctx._policy_executions if ctx else 0
        log.finish({'status': result['status'], 'message': error, 'metrics': {'success': result['success']},
                    'evaluation': evaluation})
        result['record_path'] = f'records/{log.id}'
        result['video'] = deepcopy(log.artifacts['video'])
        if result['video'].get('path'):
            result['video']['path'] = result['record_path'] + '/' + result['video']['path']
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        (output / 'diagnostics.jsonl').write_text(json.dumps({'success': result['success'],
            'termination': result['termination'], 'evaluation': evaluation, 'steps': result['steps'],
            'seed': args.seed, 'episode': args.episode_start, 'requested_episode': args.episode_start})+'\n', encoding='utf-8')
        watchdog.cancel()
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
