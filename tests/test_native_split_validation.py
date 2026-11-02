"""Native pool identity and the pre-inference split gate, without GPU dependencies."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from threading import RLock
import unittest
from unittest.mock import patch

import numpy as np

from server.backends.libero_backend import native_initial_state_identity
from server.research import ResearchController, ResearchExperiment, native_split_validation, evaluation_contract


def spec():
    return {'engine_id': 'libero', 'robot': 'panda', 'scene_id': 'libero_goal', 'task_id': 'libero_goal:0',
            'development_seeds': [0], 'validation_seeds': [1], 'test_seeds': [2],
            'schedule_seed': 0, 'max_sim_time_s': 10, 'max_wall_time_s': 30, 'max_steps': 2}


class NativeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'official_fixture.pruned_init'; self.path.write_bytes(b'official-state-file-fixture')
        self.states = np.asarray([[0., 1.], [1., 2.], [2., 3.]])
        self.origin = native_initial_state_identity(self.path, self.states, task_id='libero_goal:0', scene_id='libero_goal', seed=0)
        self.checkpoint = {'backend': {'engine_id': 'libero'}, 'scene_id': 'libero_goal',
                           'task_id': 'libero_goal:0', 'seed': 0, 'native_initial_state': self.origin}

    def test_pool_reports_source_content_and_exact_index_aliases(self):
        self.assertEqual(self.origin['file_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(self.origin['state_count'], 3)
        check = native_split_validation(spec(), self.checkpoint)
        self.assertEqual(check['status'], 'passed')
        aliased = {**spec(), 'validation_seeds': [3]}
        check = native_split_validation(aliased, self.checkpoint)
        self.assertEqual(check['status'], 'conflict')
        conflict = check['conflicts'][0]
        self.assertEqual((conflict['first']['split'], conflict['second']['split']), ('development', 'validation'))
        self.assertEqual((conflict['first']['seed'], conflict['second']['seed']), (0, 3))

    def test_same_content_at_different_indices_is_also_detected(self):
        repeated = np.asarray([[0., 1.], [0., 1.], [2., 3.]])
        origin = native_initial_state_identity(self.path, repeated, task_id='libero_goal:0', scene_id='libero_goal', seed=0)
        check = native_split_validation(spec(), {**self.checkpoint, 'native_initial_state': origin})
        self.assertEqual(check['status'], 'conflict')
        self.assertNotEqual(check['conflicts'][0]['first']['index'], check['conflicts'][0]['second']['index'])

    def test_task_identity_and_old_checkpoints_do_not_claim_verified_splits(self):
        first = native_split_validation(spec(), self.checkpoint)
        different = {**spec(), 'task_id': 'libero_goal:1'}
        self.assertEqual(native_split_validation(different, self.checkpoint)['status'], 'unverified')
        other_checkpoint = deepcopy(self.checkpoint); other_checkpoint['task_id'] = 'libero_goal:1'
        other_checkpoint['native_initial_state']['task_id'] = 'libero_goal:1'
        second = native_split_validation(different, other_checkpoint)
        self.assertNotEqual(first['splits']['development'][0]['native_initial_state_id'],
                            second['splits']['development'][0]['native_initial_state_id'])
        legacy = {key: value for key, value in self.checkpoint.items() if key != 'native_initial_state'}
        self.assertEqual(native_split_validation(spec(), legacy)['status'], 'unverified')
        self.assertEqual(native_split_validation({**spec(), 'engine_id': 'mujoco'}, legacy)['status'], 'not_applicable')
        self.assertEqual(self.checkpoint['native_initial_state'], self.origin)

    def test_malformed_pool_cannot_be_used_as_split_evidence(self):
        for updates in ({'state_count': True}, {'state_count': 0}, {'state_sha256_by_index': []},
                        {'state_index': 2}, {'state_sha256': 'wrong'}, {'file_sha256': 'unknown'}):
            checkpoint = {**self.checkpoint, 'native_initial_state': {**self.origin, **updates}}
            self.assertEqual(native_split_validation(spec(), checkpoint)['status'], 'unverified')

    def test_task_graph_code_is_part_of_mujoco_frozen_contract(self):
        self.assertIn('server/task_graph.py', evaluation_contract()['files'])


class PreInferenceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_test_alias_stops_even_development_before_model_call(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'fixture.pruned_init'; path.write_bytes(b'fixture-native-pool')
            origin = native_initial_state_identity(path, np.asarray([[0.], [1.], [2.]]),
                task_id='libero_goal:0', scene_id='libero_goal', seed=0)
            class Worker:
                lock = RLock()
                snapshot = {'backend': {'engine_id': 'libero'}}
                async def execute(self, command):
                    if command['action'] == 'checkpoint':
                        return {'scene_id': 'libero_goal', 'task_id': 'libero_goal:0', 'seed': 0,
                                'native_initial_state': origin, 'state_id': 'native-checkpoint'}
                    return None
            class Runner:
                task = None
                started = 0
                def snapshot(self): return {'active': False}
                async def start(self, request): self.started += 1; raise AssertionError('Must reject before inference')
                async def stop(self): pass
            protocol = {'spec': {**spec(), 'test_seeds': [4]}, 'task': {'instruction': 'native split contract'}}
            request = ResearchExperiment(protocol_id='fixture', method_ids=['first', 'second'])
            runner = Runner(); controller = ResearchController(Worker(), runner)
            controller.state = {'active': True, 'status': 'preparing', 'rows': [
                {'method_id': name, 'method_name': name, 'seed': 0, 'status': 'queued', 'metrics': None}
                for name in request.method_ids]}
            with patch('server.runlog.RUNS', Path(root)/'runs'), patch('server.research.get_method', return_value={'recipe': {}}):
                await controller.run(request, protocol)
                self.assertEqual(runner.started, 0)
                self.assertEqual(controller.state['status'], 'error')
                self.assertEqual(controller.state['native_split_validation']['status'], 'conflict')
                events = [json.loads(line) for line in controller.run_log.path.read_text(encoding='utf-8').splitlines()]
                self.assertTrue(any(event['event'] == 'native_split_validation' for event in events))
                self.assertFalse(any(event['event'] == 'inference_start' for event in events))
                summary = json.loads((controller.run_log.folder/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['native_split_validation']['conflicts'][0]['second']['split'], 'test')


if __name__ == '__main__': unittest.main()
