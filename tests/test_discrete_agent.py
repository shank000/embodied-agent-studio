"""Discrete Agent completion/budget and actual camera API boundary regressions.

These tests replace environment and inference boundaries. Native SDK execution,
goal semantics, and rendering are checked by the opt-in integrations separately.
"""
import asyncio
import base64
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from server.actions import EnvironmentAction
from server.agent import AgentConfig, AgentController, AgentStart
from server.simulators import backend_identity
from tests.test_agent_round_limits import DeterministicWorker


class DiscreteWorker(DeterministicWorker):
    def __init__(self, success_after=None):
        super().__init__(success_after=success_after)
        self.snapshot.update(backend=backend_identity('alfworld', 'boundary-fixture'),
            model='alfworld_agent', scene_id='alfworld_unseen', time=7., steps=7,
            timestep=1., tcp_pose=None, task={'status': 'idle'})
        self.snapshot['evaluation'] = {'source': 'alfworld_native_task', 'success': False}

    async def execute(self, command):
        kind = command['action']
        if kind == 'observe':
            await asyncio.sleep(0)
            self.commands.append(kind)
            return {'backend': deepcopy(self.snapshot['backend']), 'time': self.snapshot['time'],
                'robot': 'alfworld_agent', 'generation': 1, 'tcp': None, 'cameras': [],
                'environment': {'observation_text': 'A local orchestration boundary.', 'admissible_commands': ['look']}}
        if kind == 'environment_action':
            await asyncio.sleep(0)
            self.commands.append(kind)
            if command['environment_action'] != 'look': raise AssertionError('Unexpected environment action')
            self.moves += 1
            self.snapshot['time'] += 1
            self.snapshot['steps'] += 1
            self.snapshot['evaluation']['success'] = self.success_after is not None and self.moves >= self.success_after
            return {'environment_action': 'look', 'step_count': self.snapshot['steps']}
        return await super().execute(command)


class DiscreteAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='discrete-agent-boundary-')
        self.addCleanup(self.folder.cleanup)
        log_patch = patch('server.runlog.RUNS', Path(self.folder.name))
        log_patch.start(); self.addCleanup(log_patch.stop)

    async def episode(self, *, success_after=None, max_rounds=None, steps=1, evaluator_stop=True, one_round=False):
        worker = DiscreteWorker(success_after)
        agent = AgentController(worker)
        agent.configure(AgentConfig(provider='custom', model='discrete-boundary',
                                    base_url='http://127.0.0.1:9/agent', max_rounds=max_rounds))
        calls = []

        async def inference(config, prompt, cameras, **options):
            calls.append(json.loads(prompt))
            self.assertIs(options['response_type'], EnvironmentAction)
            self.assertEqual(cameras, [])
            return EnvironmentAction(status='move', action='look', explanation='Explicit local discrete fixture.'), None

        with patch('server.agent.api_action', inference):
            await agent.start(AgentStart(task='Complete the fixture environment task.', max_environment_steps=steps,
                max_sim_time_s=100., evaluator_stop=evaluator_stop, one_round=one_round))
            await asyncio.wait_for(agent.task, 5)
        self.assertFalse(agent.state['active'])
        self.assertIsNone(agent.state['metrics']['sim_time_s'])
        self.assertEqual(agent.state['metrics']['environment_steps'], worker.moves)
        summary = json.loads((Path(self.folder.name)/agent.state['run_id']/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['metrics']['environment_steps'], worker.moves)
        self.assertIsNone(summary['metrics']['sim_time_s'])
        return agent, worker, calls

    async def test_success_on_last_environment_step_is_verified_before_budget_limit(self):
        agent, worker, calls = await self.episode(success_after=1, steps=1)
        self.assertEqual(agent.state['status'], 'verified', agent.state)
        self.assertTrue(agent.state['metrics']['success'])
        self.assertEqual(worker.moves, 1)
        self.assertEqual(len(calls), 1)

    async def test_success_on_last_round_or_one_round_button_is_verified(self):
        for max_rounds, one_round in ((1, False), (None, True)):
            with self.subTest(max_rounds=max_rounds, one_round=one_round):
                agent, worker, calls = await self.episode(success_after=1, steps=10,
                                                         max_rounds=max_rounds, one_round=one_round)
                self.assertEqual(agent.state['status'], 'verified', agent.state)
                self.assertTrue(agent.state['metrics']['success'])
                self.assertEqual((worker.moves, len(calls)), (1, 1))

    async def test_unsuccessful_last_step_remains_budget_limit(self):
        agent, worker, calls = await self.episode(steps=1)
        self.assertEqual(agent.state['status'], 'limit', agent.state)
        self.assertFalse(agent.state['metrics']['success'])
        self.assertEqual((worker.moves, len(calls)), (1, 1))

    async def test_disabled_evaluator_stop_does_not_auto_finish_on_native_success(self):
        agent, worker, calls = await self.episode(success_after=1, steps=10, max_rounds=2, evaluator_stop=False)
        self.assertEqual(agent.state['status'], 'limit', agent.state)
        self.assertTrue(agent.state['metrics']['success'])
        self.assertEqual((worker.moves, len(calls)), (2, 2))


class ObservationCameraApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_routes_use_actual_camera_set_for_agent_and_policy(self):
        application = importlib.import_module('server.app')
        jpeg = DeterministicWorker().jpeg
        fixtures = [('not_captured', None, set()),
                    ('alfworld', {'backend': backend_identity('alfworld', 'fixture'), 'cameras': []}, set()),
                    ('ai2thor', {'backend': backend_identity('ai2thor', 'fixture'),
                                'cameras': [{'name': 'front', 'image_base64': jpeg}]}, {'front'})]
        for source in ('agent', 'policy'):
            for label, observation, available in fixtures:
                with self.subTest(source=source, observation=label):
                    with patch.object(application, source, SimpleNamespace(observation=observation)):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                                     base_url='http://127.0.0.1:8000') as client:
                            for name in ('front', 'side', 'wrist'):
                                response = await client.get(f'/api/{source}/camera/{name}')
                                self.assertEqual(response.status_code, 200 if name in available else 404, response.text)
                                if name in available:
                                    self.assertEqual(response.headers['content-type'], 'image/jpeg')
                                    self.assertEqual(response.content, base64.b64decode(jpeg))
