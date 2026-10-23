"""Agent round budgets and stop paths without a model service or native simulation."""
import asyncio
import base64
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
from threading import RLock
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from server.agent import AgentConfig, AgentController, AgentStart
from server.control import PoseAction
from server.telemetry import track_inference


class DeterministicWorker:
    """A completed-motion boundary; physics correctness belongs to native tests."""
    def __init__(self, *, reject_moves=False, time_per_move=.01, success_after=None):
        self.lock = RLock()
        self.reject_moves = reject_moves
        self.time_per_move = time_per_move
        self.success_after = success_after
        self.commands = []
        self.moves = 0
        self.pose = {'frame': 'world', 'position_m': [.2, 0., .3], 'quaternion_wxyz': [1., 0., 0., 0.]}
        self.snapshot = {'model': 'fixture', 'scene_id': 'fixture', 'seed': 42, 'ready': True,
                         'time': 0., 'timestep': .002, 'generation': 1, 'task': {'status': 'complete'},
                         'evaluation': {'success': False, 'source': 'mujoco_privileged_state'},
                         'tcp_pose': deepcopy(self.pose)}
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'gray').save(image, 'JPEG')
        self.jpeg = base64.b64encode(image.getvalue()).decode()

    async def execute(self, command):
        await asyncio.sleep(0)
        kind = command['action']
        self.commands.append(kind)
        if kind == 'checkpoint': return {'state_id': 'deterministic-worker-fixture'}
        if kind == 'observe':
            return {'time': self.snapshot['time'], 'robot': 'fixture', 'generation': 1,
                    'tcp': deepcopy(self.pose), 'cameras': [
                        {'name': name, 'image_base64': self.jpeg} for name in ('front', 'side', 'wrist')]}
        if kind == 'move_tcp':
            if self.reject_moves: raise HTTPException(400, 'Deterministic unreachable target.')
            self.moves += 1
            self.snapshot['time'] += self.time_per_move
            self.snapshot['evaluation']['success'] = self.success_after is not None and self.moves >= self.success_after
            return {'joint_targets_rad': []}
        if kind == 'run_limits':
            self.deadline = self.snapshot['time'] + command['duration_s']
            return None
        if kind == 'verify_task':
            duration = min(.3, self.deadline - self.snapshot['time'])
            steps = math.floor((duration + 1e-9) / self.snapshot['timestep'])
            self.snapshot['time'] += steps * self.snapshot['timestep']
            return deepcopy(self.snapshot['evaluation'])
        if kind in ('stop', 'pause', 'record_start', 'record_stop', 'run_limits'): return None
        raise AssertionError(f'Unexpected worker command: {kind}')


class AgentRoundLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='agent-round-budget-')
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.log_patch = patch('server.runlog.RUNS', self.root)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def controller(self, max_rounds=None, **worker_options):
        worker = DeterministicWorker(**worker_options)
        agent = AgentController(worker)
        agent.configure(AgentConfig(provider='custom', model='round-fixture',
                                    base_url='http://127.0.0.1:9/agent', max_rounds=max_rounds))
        return agent, worker

    def events(self, agent):
        return [json.loads(line) for line in
                (self.root/agent.state['run_id']/'events.jsonl').read_text(encoding='utf-8').splitlines()]

    async def finish(self, agent, *, done_after=None, **request_options):
        rounds = []

        @track_inference('agent_inference')
        async def boundary(config, prompt, cameras):
            rounds.append(agent.state['round'])
            pose = json.loads(prompt)['observation']['tcp']
            return PoseAction(status='done' if done_after is not None and len(rounds) >= done_after else 'move',
                              **pose, gripper_open=1., duration_s=.5, explanation='Explicit round budget fixture.'), None

        with patch('server.agent.api_action', boundary):
            await agent.start(AgentStart(task='Round budget fixture.', **request_options))
            await asyncio.wait_for(agent.task, 15)
        self.assertFalse(agent.state['active'])
        self.assertEqual(rounds, list(range(1, len(rounds)+1)))
        self.assertEqual(agent.state['metrics']['inference_count'], len(rounds))
        return rounds

    def test_optional_budget_normalizes_zero_and_has_no_artificial_maximum(self):
        self.assertIsNone(AgentConfig().max_rounds)
        for value in (None, 0):
            with self.subTest(value=value): self.assertIsNone(AgentConfig(max_rounds=value).max_rounds)
        for value in (1, 31, 10000):
            with self.subTest(value=value): self.assertEqual(AgentConfig(max_rounds=value).max_rounds, value)
        for value in (-1, True, 1.5, '31'):
            with self.subTest(value=value), self.assertRaises(ValidationError): AgentConfig(max_rounds=value)

    async def test_unlimited_verifies_completion_after_old_thirty_round_boundary(self):
        agent, worker = self.controller(success_after=31)
        rounds = await self.finish(agent, done_after=32, evaluator_stop=False)
        self.assertEqual(len(rounds), 32)
        self.assertEqual(worker.moves, 31)
        self.assertEqual(agent.state['status'], 'verified', agent.state)
        self.assertIsNone(agent.state['max_rounds'])

    async def test_positive_budget_above_thirty_stops_exactly_at_requested_round(self):
        agent, worker = self.controller(33)
        rounds = await self.finish(agent)
        self.assertEqual(len(rounds), 33)
        self.assertEqual(worker.moves, 33)
        self.assertEqual(agent.state['status'], 'limit', agent.state)
        self.assertIn('轮数上限', agent.state['message'])

    async def test_explicit_one_round_and_one_round_button_keep_their_contract(self):
        for budget, one_round, expected in ((1, False, 'limit'), (None, True, 'round_complete'),
                                            (10000, True, 'round_complete')):
            with self.subTest(budget=budget, one_round=one_round):
                agent, worker = self.controller(budget)
                self.assertEqual(await self.finish(agent, one_round=one_round), [1])
                self.assertEqual(worker.moves, 1)
                self.assertEqual(agent.state['status'], expected, agent.state)
                self.assertEqual(agent.state['max_rounds'], 1)

    async def test_unlimited_manual_stop_records_cancelled_call_and_never_moves(self):
        agent, worker = self.controller()
        entered = asyncio.Event()

        @track_inference('agent_inference')
        async def boundary(config, prompt, cameras):
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError('A cancelled model call must not resume.')

        with patch('server.agent.api_action', boundary):
            await agent.start(AgentStart(task='Cancel unlimited fixture.'))
            await asyncio.wait_for(entered.wait(), 5)
            await agent.stop()
        self.assertFalse(agent.state['active'])
        self.assertEqual(agent.state['status'], 'stopped', agent.state)
        self.assertEqual(worker.moves, 0)
        self.assertIn('record_stop', worker.commands)
        events = self.events(agent)
        self.assertEqual([event['data']['status'] for event in events if event['event'] == 'inference'], ['cancelled'])
        summary = json.loads((self.root/agent.state['run_id']/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'stopped')
        self.assertIsNotNone(summary['ended_at'])

    async def test_rejected_targets_advance_rounds_and_respect_finite_budget(self):
        agent, worker = self.controller(3, reject_moves=True)
        self.assertEqual(await self.finish(agent), [1, 2, 3])
        self.assertEqual(worker.moves, 0)
        self.assertEqual(worker.snapshot['time'], 0.)
        self.assertEqual(agent.state['status'], 'limit', agent.state)
        self.assertTrue(all('IK/动作被拒绝' in item['result'] for item in agent.state['history']))

    async def test_unlimited_keeps_simulation_time_and_evaluator_stops(self):
        for evaluator_stop, expected in ((False, 'limit'), (True, 'verified')):
            with self.subTest(evaluator_stop=evaluator_stop):
                agent, worker = self.controller(time_per_move=1., success_after=1)
                self.assertEqual(await self.finish(agent, max_sim_time_s=1., evaluator_stop=evaluator_stop), [1])
                self.assertEqual(worker.moves, 1)
                self.assertEqual(agent.state['status'], expected, agent.state)
                self.assertEqual(agent.state['metrics']['sim_time_s'], 1.)

    async def test_unlimited_repeated_done_stops_when_less_than_one_physics_step_remains(self):
        agent, worker = self.controller()
        rounds = await self.finish(agent, done_after=1, max_sim_time_s=1.001)
        self.assertEqual(rounds, [1, 2, 3, 4])
        self.assertEqual(worker.commands.count('verify_task'), 4)
        self.assertEqual(worker.moves, 0)
        self.assertAlmostEqual(worker.snapshot['time'], 1.)
        self.assertEqual(agent.state['status'], 'limit', agent.state)
        self.assertFalse(agent.state['metrics']['success'])


if __name__ == '__main__': unittest.main()
