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


def public_feedback(value):
    """Keep proprioceptive execution feedback; withhold evaluator ground truth."""
    if isinstance(value, dict):
        return {key: public_feedback(item) for key, item in value.items()
                if key not in ('evaluation', 'verification')}
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
