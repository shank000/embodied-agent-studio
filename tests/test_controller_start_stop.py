"""Accepted starts are recorded before an immediate stop; no simulator or inference."""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from threading import RLock
import unittest
from unittest.mock import patch

from server.agent import AgentConfig, AgentController, AgentStart
from server.policy import PolicyConfig, PolicyController, PolicyStart
from server.comparison import ComparisonController, ComparisonStart
from server.composition import CompositionController, CompositionStart
from server.research import ResearchController, ResearchExperiment
from server.simulators import backend_identity
from server.analysis import analyze_run
from server.runlog import RunLog


class PreparingWorker:
    def __init__(self, block_action):
        self.lock = RLock(); self.block_action = block_action; self.blocked = False
        self.commands = []
        self.snapshot = {'ready': True, 'backend': backend_identity('mujoco', 'startup-contract-fixture'),
            'session_id': 'startup-contract-fixture', 'model': 'panda', 'scene_id': 'tabletop',
            'seed': 42, 'generation': 1, 'time': 0., 'steps': 0,
            'evaluation': {'source': 'mujoco_privileged_state', 'success': False}}

    async def execute(self, command):
        self.commands.append(command['action'])
        if command['action'] == self.block_action and not self.blocked:
            self.blocked = True
            await asyncio.Event().wait()
        return None


class ControllerStartStopTests(unittest.IsolatedAsyncioTestCase):
    def make_controller(self, kind, stack):
        worker = PreparingWorker('stop' if kind == 'comparison' else 'load' if kind == 'research' else 'checkpoint')
        agent = AgentController(worker)
        agent.config = AgentConfig(provider='custom', model='fixture-agent', base_url='http://127.0.0.1:9999')
        policy = PolicyController(worker)
        policy.configure(PolicyConfig(kind='vla', model='fixture-vla', endpoint='http://127.0.0.1:9998'))
        recipe = {'name': 'fixture method', 'mode': 'agent', 'agent': agent.config.model_dump(exclude={'api_key'})}
        composition = CompositionController(worker, method_resolver=lambda key: {'recipe': recipe})
        if kind == 'agent': return agent, AgentStart(task='startup test'), worker
        if kind == 'policy': return policy, PolicyStart(kind='vla', task='startup test'), worker
        if kind == 'composition': return composition, CompositionStart(method_id='fixture', task='startup test'), worker
        if kind == 'comparison':
            return ComparisonController(worker, agent, policy), ComparisonStart(
                robot='panda', task_id='reach_up', methods=['agent', 'vla']), worker
        protocol = {'spec': {'engine_id': 'mujoco', 'robot': 'panda', 'scene_id': 'tabletop',
            'task_id': 'reach_up', 'development_seeds': [42], 'schedule_seed': 0},
            'contract': {'fixture': True}, 'task': {'instruction': 'startup test'}}
        stack.enter_context(patch('server.research.get_protocol', return_value=protocol))
        stack.enter_context(patch('server.research.evaluation_contract', return_value=protocol['contract']))
        stack.enter_context(patch('server.research.verify_method', return_value={'recipe': recipe}))
        stack.enter_context(patch('server.research.get_method', return_value={'recipe': recipe}))
        return ResearchController(worker, composition), ResearchExperiment(
            protocol_id='fixture-protocol', method_ids=['fixture-a', 'fixture-b']), worker

    async def test_immediate_stop_after_accepted_start_archives_exactly_one_stopped_run(self):
        for kind in ('agent', 'policy', 'composition', 'comparison', 'research'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                stack.enter_context(patch('server.runlog.RUNS', root))
                controller, request, worker = self.make_controller(kind, stack)
                accepted = await controller.start(request)
                self.assertTrue(accepted['active']); self.assertTrue(worker.blocked)
                self.assertIsNotNone(accepted.get('run_id'), 'Acknowledged starts already own a durable record')
                task = controller.task
                await controller.stop()
                self.assertTrue(task.done()); self.assertIsNone(controller.task)
                self.assertFalse(controller.state['active']); self.assertEqual(controller.state['status'], 'stopped')
                summaries = list(root.glob('*/summary.json')); self.assertEqual(len(summaries), 1)
                record = json.loads(summaries[0].read_text(encoding='utf-8'))
                self.assertEqual(record['status'], 'stopped')
                self.assertEqual(record['backend']['engine_id'], 'mujoco')
                self.assertIsNotNone(record.get('settings'))
                self.assertEqual(record['timing']['inference_calls'], 0)
                events = [json.loads(line) for line in (summaries[0].parent/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertFalse(any(event['event'] in ('inference', 'action', 'model_output') for event in events))
                self.assertEqual(sum(event['event'] == 'finish' for event in events), 1)
                self.assertFalse(any(action in ('observe', 'move_tcp', 'policy_chunk') for action in worker.commands))
                analysis = analyze_run(record['run_id'])
                self.assertEqual(analysis['status'], 'stopped')
                self.assertEqual(analysis['total']['calls'], 0)
                self.assertFalse((analysis.get('metrics') or {}).get('success'))
                self.assertEqual(analysis['cost']['amount_usd'], 0.)
                self.assertEqual(analysis['total']['output_tokens'], 0)

    async def test_initial_logging_failure_is_not_acknowledged_as_an_active_experiment(self):
        for kind in ('agent', 'policy', 'composition', 'comparison', 'research'):
            with self.subTest(kind=kind), ExitStack() as stack:
                controller, request, worker = self.make_controller(kind, stack)
                module = 'server.research' if kind == 'research' else 'server.'+kind
                stack.enter_context(patch(module+'.RunLog', side_effect=OSError('fixture archive is not writable')))
                with self.assertRaisesRegex(OSError, 'archive is not writable'):
                    await controller.start(request)
                self.assertFalse(controller.state['active'])
                self.assertEqual(controller.state['status'], 'error')
                self.assertTrue(controller.task.done())
                await controller.stop()
                self.assertIsNone(controller.task)

    async def test_legacy_or_missing_child_batch_usage_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory, patch('server.runlog.RUNS', Path(directory)):
            old = RunLog('comparison', {'task': 'legacy usage fixture'})
            old.metadata.pop('telemetry_schema_version')
            old.finish({'status': 'stopped', 'rows': [{'status': 'stopped', 'run_id': None}]})
            # Old files did not have the marker in their start event either.
            events = [json.loads(line) for line in old.path.read_text(encoding='utf-8').splitlines()]
            for event in events: event.get('data', {}).pop('telemetry_schema_version', None)
            old.path.write_text(''.join(json.dumps(event)+'\n' for event in events), encoding='utf-8')
            self.assertIsNone(analyze_run(old.id)['total']['output_tokens'])
            modern = RunLog('comparison', {'task': 'missing child fixture'})
            modern.finish({'status': 'stopped', 'rows': [{'status': 'stopped',
                'run_id': '20260101T000000-0000000000'}]})
            analysis = analyze_run(modern.id)
            self.assertIsNone(analysis['total']['output_tokens'])
            self.assertNotEqual(analysis['cost']['status'], 'complete')
