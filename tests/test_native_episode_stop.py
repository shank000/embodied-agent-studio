"""Real controller loops must not infer or repair after a native episode ends."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.actions import NativeAction
from server.agent import AgentConfig, AgentController, AgentStart
from server.composition import CompositionController
from server.policy import PolicyConfig, PolicyController, PolicyStart
from server.task_verification import environment_stop
from tests.test_native_agent import NativeWorker, SPACE


class EndingWorker(NativeWorker):
    def __init__(self, ending, *, already_ended=False):
        super().__init__()
        self.ending = ending
        if already_ended:
            self.snapshot['evaluation'].update(ending)

    async def execute(self, command):
        if command['action'] in ('native_action', 'policy_chunk'):
            evaluation = self.snapshot['evaluation']
            if evaluation.get('terminated') or evaluation.get('truncated'):
                raise AssertionError('Controller attempted another action after native episode end.')
            self.commands.append(command['action'])
            self.moves += 1
            self.snapshot['time'] += .1
            evaluation.update(self.ending)
            return {'controller': 'fixture_native', 'frames': 1, 'limited_targets': 0}
        return await super().execute(command)


class NativeEpisodeStopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='native-episode-ending-')
        self.addCleanup(self.folder.cleanup)
        self.log_patch = patch('server.runlog.RUNS', Path(self.folder.name))
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    async def episode(self, kind, ending, *, already_ended=False, evaluator_stop=True):
        worker = EndingWorker(ending, already_ended=already_ended)
        calls = []
        agent_settings = {'provider': 'custom', 'model': 'termination-fixture',
                          'base_url': 'http://127.0.0.1:9/agent', 'max_rounds': 3}
        policy_settings = {'kind': 'vla', 'model': 'termination-fixture',
                           'endpoint': 'http://127.0.0.1:9/predict', 'max_chunks': 3}

        async def agent_boundary(*args, **kwargs):
            calls.append('agent')
            return NativeAction(status='move', action=[.1, 0., 0.], steps=2,
                                explanation='Explicit controller boundary fixture.'), None

        async def policy_boundary(*args, **kwargs):
            calls.append('policy')
            return {'protocol': 'embodied-policy-v1', 'action_type': 'native',
                    'actions': [[.1, 0., 0.]], 'action_names': SPACE['names'], 'dt_s': .05}

        if kind == 'agent':
            controller = AgentController(worker)
            controller.configure(AgentConfig(**agent_settings))
            request = AgentStart(task='Native ending fixture.', max_sim_time_s=100,
                                 evaluator_stop=evaluator_stop)
        elif kind == 'policy':
            controller = PolicyController(worker)
            controller.configure(PolicyConfig(**policy_settings))
            request = PolicyStart(kind='vla', task='Native ending fixture.', max_sim_time_s=100)
        else:
            recipe = {'name': 'Native ending fixture', 'mode': kind.removeprefix('composition_'),
                      'agent': agent_settings, 'policy': policy_settings}
            controller = CompositionController(worker, lambda _: recipe)
            request = {'method_id': 'fixture', 'task': 'Native ending fixture.',
                       'max_steps': 3, 'max_sim_time_s': 100}
        with patch('server.agent.api_action', agent_boundary), \
             patch('server.policy.predict_policy', policy_boundary), \
             patch('server.composition.predict_policy', policy_boundary):
            await controller.start(request)
            await asyncio.wait_for(controller.task, 5)
        events = [json.loads(line) for line in controller.run_log.path.read_text(encoding='utf-8').splitlines()]
        self.assertFalse(controller.state['active'])
        self.assertEqual(worker.moves, 0 if already_ended else 1)
        self.assertEqual(len(calls), worker.moves, controller.state)
        self.assertFalse(any(event['event'] in ('error', 'recovery_start', 'repair_start') for event in events))
        self.assertEqual(controller.state['metrics']['evaluation'], worker.snapshot['evaluation'])
        return controller, events

    async def test_sdk_truncation_stops_all_controllers_and_retains_evidence(self):
        ending = {'success': False, 'terminated': False, 'truncated': True,
                  'termination_reason': 'native_timeout', 'failure_reasons': ['Native step budget reached.']}
        for kind in ('agent', 'policy', 'composition_agent', 'composition_policy'):
            for already_ended in (False, True):
                with self.subTest(kind=kind, already_ended=already_ended):
                    controller, events = await self.episode(kind, ending, already_ended=already_ended)
                    self.assertEqual(controller.state['status'], 'limit', controller.state)
                    self.assertEqual(controller.state['termination_reason'], 'native_timeout')
                    self.assertIn('native_timeout', controller.state['message'])
                    rounds = [event['data'] for event in events if event['event'] == 'round_end']
                    self.assertEqual(len(rounds), 0 if already_ended else 1)
                    if rounds:
                        self.assertTrue(rounds[0]['truncated'])
                        self.assertFalse(rounds[0]['terminated'])
                        self.assertEqual(rounds[0]['termination_reason'], 'native_timeout')

    async def test_native_terminated_success_overrides_optional_evaluator_stop(self):
        for already_ended in (False, True):
            with self.subTest(already_ended=already_ended):
                controller, _ = await self.episode('agent', {'success': True, 'terminated': True,
                    'truncated': False, 'termination_reason': 'native_success'},
                    already_ended=already_ended, evaluator_stop=False)
                self.assertEqual(controller.state['status'], 'verified', controller.state)
                self.assertEqual(controller.state['termination_reason'], 'native_success')

    async def test_success_wins_when_goal_reached_at_native_budget(self):
        for kind in ('agent', 'policy', 'composition_agent', 'composition_policy'):
            with self.subTest(kind=kind):
                controller, events = await self.episode(kind, {'success': True, 'terminated': True,
                    'truncated': True, 'termination_reason': 'native_success'})
                self.assertEqual(controller.state['status'], 'verified', controller.state)
                row = next(event['data'] for event in events if event['event'] == 'round_end')
                self.assertTrue(row['terminated'])
                self.assertFalse(row['truncated'])

    def test_optional_success_stop_and_untrusted_endings_remain_distinct(self):
        state = {'ready': True, 'backend': {'engine_id': 'mujoco'},
                 'evaluation': {'source': 'mujoco_privileged_state', 'success': True}}
        self.assertIsNone(environment_stop(state, evaluator_stop=False))
        self.assertEqual(environment_stop(state)['status'], 'verified')
        for mutation in ({'source': 'model_claim', 'truncated': True}, {'terminated': True}):
            unknown = deepcopy(state)
            unknown['evaluation'].update(mutation)
            if mutation.get('terminated'): unknown['ready'] = False
            self.assertIsNone(environment_stop(unknown))


if __name__ == '__main__':
    unittest.main()
