"""Environment verification of completion proposals, separate from policy inputs."""
import asyncio
from copy import deepcopy
from .simulators import EVALUATORS


def trusted_evaluation(evaluation, snapshot):
    engine_id = (snapshot.get('backend') or {}).get('engine_id', 'mujoco')
    expected = EVALUATORS.get(engine_id)
    return bool(expected and evaluation.get('source') == expected[0])


def environment_success(snapshot):
    evaluation = snapshot.get('evaluation') or {}
    return (snapshot.get('ready') is True
            and trusted_evaluation(evaluation, snapshot)
            and evaluation.get('success') is True)


def environment_failure(snapshot):
    evaluation = snapshot.get('evaluation') or {}
    return (snapshot.get('ready') is True
            and trusted_evaluation(evaluation, snapshot)
            and ((evaluation.get('task_graph') or {}).get('terminal_failure') is True
                 or (evaluation.get('terminated') is True and evaluation.get('success') is False)))


def environment_stop(snapshot, *, evaluator_stop=True):
    """Honor native episode endings even when optional success stopping is disabled."""
    evaluation = snapshot.get('evaluation') or {}
    if snapshot.get('ready') is not True or not trusted_evaluation(evaluation, snapshot):
        return None
    ended = evaluation.get('terminated') is True or evaluation.get('truncated') is True
    if evaluation.get('success') is True and (evaluator_stop or ended):
        return {'status': 'verified', 'message': '场景评价器已确认任务成功。',
                'termination_reason': evaluation.get('termination_reason') or 'task_success'}
    if environment_failure(snapshot):
        reason = ((evaluation.get('task_graph') or {}).get('failure_reason')
                  or evaluation.get('termination_reason') or 'task_failure')
        return {'status': 'failed', 'message': f'环境判定任务已失败，本次实验结束。原因：{reason}',
                'termination_reason': reason}
    if evaluation.get('truncated') is True:
        reason = evaluation.get('termination_reason') or 'environment_limit'
        return {'status': 'limit', 'message': f'原生环境已达到本回合上限，请重置环境后开始新任务。原因：{reason}',
                'termination_reason': reason}
    return None


def public_feedback(value):
    """Keep proprioceptive execution feedback; withhold evaluator ground truth."""
    if isinstance(value, dict):
        return {key: public_feedback(item) for key, item in value.items()
                if key not in ('evaluation', 'verification', 'task_graph', 'reward_components', 'evaluator_state')}
    if isinstance(value, list):
        return [public_feedback(item) for item in value]
    return value


async def verify_completion(controller, worker, log, step, explanation, generation):
    controller.state.update(status='verifying', message='正在依据仿真环境实际状态验收任务。')
    evaluation = await worker.execute({'action': 'verify_task', 'generation': generation})
    if not controller._accepting or asyncio.current_task().cancelling():
        raise asyncio.CancelledError
    with worker.lock: snapshot = worker.snapshot.copy()
    accepted = (trusted_evaluation(evaluation, snapshot)
                and evaluation.get('success') is True)
    message = ('环境评价器已确认任务成功。' if accepted else
               '环境未确认任务完成，请根据最新观察继续操作；模型声明不作为成功依据。')
    verification = {'step': step, 'model_claim': explanation, 'accepted': accepted,
                    'message': message, 'evaluation': deepcopy(evaluation)}
    log.write('completion_verification', verification)
    controller.state.update(verification=verification,
                            status='verified' if accepted else 'replan', message=message)
    return accepted
