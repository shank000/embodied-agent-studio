"""Environment verification of completion proposals, separate from policy inputs."""
import asyncio
from copy import deepcopy


def environment_success(snapshot):
    evaluation = snapshot.get('evaluation') or {}
    return (snapshot.get('ready') is True
            and evaluation.get('source') == 'mujoco_privileged_state'
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
    controller.state.update(status='verifying', message='正在依据 MuJoCo 实际状态验收任务。')
    evaluation = await worker.execute({'action': 'verify_task', 'generation': generation})
    if not controller._accepting or asyncio.current_task().cancelling():
        raise asyncio.CancelledError
    accepted = (evaluation.get('source') == 'mujoco_privileged_state'
                and evaluation.get('success') is True)
    message = ('环境评价器已确认任务成功。' if accepted else
               '环境未确认任务完成，请根据最新三路图像继续操作；模型声明不作为成功依据。')
    verification = {'step': step, 'model_claim': explanation, 'accepted': accepted,
                    'message': message, 'evaluation': deepcopy(evaluation)}
    log.write('completion_verification', verification)
    controller.state.update(verification=verification,
                            status='verified' if accepted else 'replan', message=message)
    return accepted
