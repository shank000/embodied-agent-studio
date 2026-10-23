"""Controller completion contracts with explicit model/worker boundary fixtures.

These tests cover orchestration and privacy. Native task geometry is checked by
the task evaluator tests; no inference service or user experiment is used here.
"""
import asyncio
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from threading import RLock
import unittest
from unittest.mock import patch

from PIL import Image

from server.composition import CompositionController, PlannerOutput
from server.control import PoseAction
from server.policy import PolicyConfig, PolicyController, PolicyStart
from server.policy_service import InferenceRequest


class CompletionWorker:
    """Closed drawer until an explicitly configured verification boundary accepts."""
    def __init__(self, *, accept_on_verification=False):
        self.lock = RLock()
        self.accept_on_verification = accept_on_verification
        self.commands = []
        self.observations = 0
        self.pose = {'frame': 'world', 'position_m': [.2, 0., .3],
                     'quaternion_wxyz': [1., 0., 0., 0.]}
        self.snapshot = {'model': 'panda', 'scene_id': 'drawer', 'seed': 42, 'ready': True,
            'time': 0., 'generation': 1, 'task': {'status': 'complete'}, 'paused': True,
            'evaluation': {'task_id': 'open_drawer', 'success': False, 'currently_satisfied': False,
                           'source': 'mujoco_privileged_state', 'version': 'fixture-v1',
                           'values': {'drawer_open_m': 0.}}, 'tcp_pose': deepcopy(self.pose)}
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'gray').save(image, 'JPEG')
        self.jpeg = base64.b64encode(image.getvalue()).decode()

    async def execute(self, command):
        await asyncio.sleep(0)
        action = command['action']
        self.commands.append(deepcopy({key: value for key, value in command.items() if key != 'log'}))
        if action == 'checkpoint': return {'state_id': 'completion-worker-contract-fixture'}
        if action == 'observe':
            self.observations += 1
            return {'time': self.snapshot['time'], 'robot': 'panda', 'generation': self.snapshot['generation'],
                'tcp': deepcopy(self.pose), 'limits': {'translation_per_round_m': .12},
                'cameras': [{'name': name, 'image_base64': self.jpeg} for name in ('front', 'side', 'wrist')]}
        if action == 'verify_task':
            if command['generation'] != self.snapshot['generation']:
                raise ValueError('Fixture scene generation changed before verification.')
            if self.accept_on_verification:
                self.snapshot['evaluation'].update(success=True, currently_satisfied=True,
                    values={'drawer_open_m': .14})
            self.snapshot['time'] += .3
            return deepcopy(self.snapshot['evaluation'])
        if action in ('stop', 'pause', 'record_start', 'record_stop', 'run_limits'): return None
        raise AssertionError(f'Unexpected worker command: {action}')


def pose(status):
    return PoseAction(status=status, frame='world', position_m=[.2, 0., .3],
        quaternion_wxyz=[1., 0., 0., 0.], gripper_open=1., duration_s=.5,
        explanation='Explicit local completion boundary fixture.')


def policy_output(status):
    return {'protocol': 'embodied-policy-v1', 'action_type': 'tcp_pose', 'pose': pose(status).model_dump()}


def plan(status='plan'):
    return PlannerOutput(status=status, subgoal='Open the drawer.', visual_summary='Local image fixture.',
        skill='policy', direction_world=[0., 0., 0.], gripper_open=1., expires_after_steps=3,
        explanation='Explicit local planner completion boundary fixture.')


class CompletionControllerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.logs = tempfile.TemporaryDirectory(prefix='completion-controller-tests-')
        self.addCleanup(self.logs.cleanup)
        self.log_patch = patch('server.runlog.RUNS', Path(self.logs.name))
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def events(self, controller):
        return [json.loads(line) for line in (Path(self.logs.name) / controller.state['run_id'] /
                'events.jsonl').read_text(encoding='utf-8').splitlines()]

    def assert_policy_request_contract(self, bodies):
        fields = {'protocol', 'model', 'device', 'task', 'observation', 'seed', 'reset', 'horizon', 'dt_s'}
        for body in bodies:
            self.assertEqual(set(body), fields, 'Existing v1 services must not receive new top-level fields.')
            self.assertEqual(InferenceRequest.model_validate(body).task, body['task'])
        self.assertEqual(bodies[0]['task'], 'Open the drawer.')
        self.assertEqual(bodies[1]['task'], 'Open the drawer.\n环境验收反馈：环境尚未确认任务完成，请根据当前三路图像和机器人状态继续操作。')

    def policy_controller(self, worker, chunks=3):
        controller = PolicyController(worker)
        controller.configure(PolicyConfig(kind='vla', model='local-completion-fixture',
            endpoint='http://127.0.0.1:9/predict', max_chunks=chunks))
        return controller

    def composition_controller(self, worker, mode):
        recipe = {'name': 'Completion contract fixture', 'mode': 'hybrid' if mode.startswith('hybrid') else mode,
            'agent': {'provider': 'custom', 'model': 'local-completion-fixture',
                      'base_url': 'http://127.0.0.1:9/agent'},
            'decision': {'kind': 'custom', 'url': 'http://127.0.0.1:9/choose'},
            'policy': {'kind': 'vla', 'model': 'local-completion-fixture',
                       'endpoint': 'http://127.0.0.1:9/predict'}}
        if mode == 'hybrid_policy': recipe['skills'] = {'candidate_source': 'policy'}
        return CompositionController(worker, lambda _: recipe)

    def assert_unverified(self, controller, worker, observations):
        self.assertEqual(worker.observations, observations)
        self.assertFalse(controller.state['metrics']['success'], controller.state)
        self.assertFalse(any(command['action'] in ('move_tcp', 'policy_chunk') for command in worker.commands))
        checks = [event['data'] for event in self.events(controller) if event['event'] == 'completion_verification']
        self.assertTrue(checks)
        self.assertTrue(all(not check['accepted'] for check in checks))

    async def test_direct_policy_rejects_done_and_reobserves_with_generic_feedback(self):
        worker = CompletionWorker(); controller = self.policy_controller(worker)
        bodies = []
        async def boundary(config, body, client=None):
            bodies.append(deepcopy(body))
            return policy_output('done' if len(bodies) == 1 else 'blocked')
        with patch('server.policy.predict_policy', boundary):
            await controller.start(PolicyStart(kind='vla', task='Open the drawer.'))
            await controller.task
        self.assertEqual(controller.state['status'], 'blocked', controller.state)
        self.assert_unverified(controller, worker, 2)
        self.assert_policy_request_contract(bodies)
        feedback = bodies[1]['task']
        self.assertNotIn('drawer_open_m', json.dumps(feedback))
        self.assertNotIn('evaluation', json.dumps(feedback))

    async def test_direct_policy_repeated_done_respects_chunk_budget(self):
        worker = CompletionWorker(); controller = self.policy_controller(worker, chunks=2)
        async def boundary(*args, **kwargs): return policy_output('done')
        with patch('server.policy.predict_policy', boundary):
            await controller.start(PolicyStart(kind='vla', task='Open the drawer.'))
            await controller.task
        self.assertEqual(controller.state['status'], 'limit', controller.state)
        self.assert_unverified(controller, worker, 2)
        self.assertEqual(len([command for command in worker.commands if command['action'] == 'verify_task']), 2)

    async def test_direct_policy_only_accepts_independent_environment_verification(self):
        worker = CompletionWorker(accept_on_verification=True); controller = self.policy_controller(worker)
        async def boundary(*args, **kwargs): return policy_output('done')
        with patch('server.policy.predict_policy', boundary):
            await controller.start(PolicyStart(kind='vla', task='Open the drawer.'))
            await controller.task
        self.assertEqual(controller.state['status'], 'verified', controller.state)
        self.assertTrue(controller.state['metrics']['success'])
        self.assertEqual(worker.observations, 1)
        check = next(event['data'] for event in self.events(controller) if event['event'] == 'completion_verification')
        self.assertTrue(check['accepted'])

    async def test_all_composition_sources_reject_done_and_continue_instead_of_reporting_success(self):
        for mode in ('agent', 'policy', 'hybrid_planner', 'hybrid_policy'):
            with self.subTest(mode=mode):
                worker = CompletionWorker(); controller = self.composition_controller(worker, mode)
                prompts = []; bodies = []
                async def agent_boundary(config, prompt, cameras, **kwargs):
                    prompts.append(json.loads(prompt))
                    if mode == 'hybrid_policy': return plan(), None
                    status = 'done' if len(prompts) == 1 else 'blocked'
                    return (plan(status) if mode == 'hybrid_planner' else pose(status)), None
                async def policy_boundary(config, body, client=None):
                    bodies.append(deepcopy(body))
                    return policy_output('done' if len(bodies) == 1 else 'blocked')
                with patch('server.agent.api_action', agent_boundary), patch('server.composition.predict_policy', policy_boundary):
                    await controller.start({'method_id': 'fixture', 'task': 'Open the drawer.', 'max_steps': 3})
                    await controller.task
                self.assertEqual(controller.state['status'], 'blocked', controller.state)
                self.assert_unverified(controller, worker, 2)
                if mode.startswith('hybrid'):
                    self.assertEqual(prompts[1]['planning_trigger'], 'completion_rejected')
                if bodies:
                    self.assert_policy_request_contract(bodies)
                    feedback = bodies[1]['task']
                else:
                    feedback = prompts[1]['recent_feedback']
                    self.assertTrue(any(row['result']['status'] == 'completion_rejected' for row in feedback))
                self.assertNotIn('drawer_open_m', json.dumps(feedback))
                self.assertNotIn('evaluation', json.dumps(feedback))

    async def test_composition_done_at_step_budget_is_unverified_limit(self):
        worker = CompletionWorker(); controller = self.composition_controller(worker, 'agent')
        async def boundary(*args, **kwargs): return pose('done'), None
        with patch('server.agent.api_action', boundary):
            await controller.start({'method_id': 'fixture', 'task': 'Open the drawer.', 'max_steps': 2})
            await controller.task
        self.assertEqual(controller.state['status'], 'limit', controller.state)
        self.assert_unverified(controller, worker, 2)
        self.assertIn('步数上限', controller.state['message'])

    async def test_composition_completion_requires_environment_evidence_for_every_source(self):
        for mode in ('agent', 'policy', 'hybrid_planner', 'hybrid_policy'):
            with self.subTest(mode=mode):
                worker = CompletionWorker(accept_on_verification=True)
                controller = self.composition_controller(worker, mode)
                async def agent_boundary(*args, **kwargs):
                    if mode == 'hybrid_policy': return plan(), None
                    return (plan('done') if mode == 'hybrid_planner' else pose('done')), None
                async def policy_boundary(*args, **kwargs): return policy_output('done')
                with patch('server.agent.api_action', agent_boundary), patch('server.composition.predict_policy', policy_boundary):
                    await controller.start({'method_id': 'fixture', 'task': 'Open the drawer.', 'max_steps': 2})
                    await controller.task
                self.assertEqual(controller.state['status'], 'verified', controller.state)
                self.assertTrue(controller.state['metrics']['success'])
                check = next(event['data'] for event in self.events(controller) if event['event'] == 'completion_verification')
                self.assertTrue(check['accepted'])

    async def test_composition_stale_done_cannot_verify_a_new_scene(self):
        worker = CompletionWorker(); controller = self.composition_controller(worker, 'agent')
        async def boundary(*args, **kwargs):
            worker.snapshot['generation'] += 1
            return pose('done'), None
        with patch('server.agent.api_action', boundary):
            await controller.start({'method_id': 'fixture', 'task': 'Open the drawer.', 'max_steps': 2})
            await controller.task
        self.assertEqual(controller.state['status'], 'error', controller.state)
        self.assertFalse(any(command['action'] == 'verify_task' for command in worker.commands))
        self.assertFalse(controller.state['metrics']['success'])

    async def test_policy_cancelled_inference_cannot_verify_a_late_done(self):
        worker = CompletionWorker(); controller = self.policy_controller(worker)
        entered = asyncio.Event()
        async def boundary(*args, **kwargs):
            entered.set()
            try: await asyncio.Event().wait()
            except asyncio.CancelledError: pass
            return policy_output('done')
        with patch('server.policy.predict_policy', boundary):
            await controller.start(PolicyStart(kind='vla', task='Open the drawer.'))
            await asyncio.wait_for(entered.wait(), 5)
            await controller.stop()
        self.assertEqual(controller.state['status'], 'stopped', controller.state)
        self.assertFalse(any(command['action'] == 'verify_task' for command in worker.commands))
        self.assertFalse(controller.state['metrics']['success'])


if __name__ == '__main__': unittest.main()
