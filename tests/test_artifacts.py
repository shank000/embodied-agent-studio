"""Version registration and actual MuJoCo image/state persistence; no remote inference."""
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from server import artifacts, runlog
from server.artifacts import MethodRecipe, get_method, get_recipe, list_methods, register_method, verify_method
from server.runlog import RunLog, safe_artifact_path
from server.simulation import Simulation


class MethodVersionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='robot-method-test-')
        self.root = Path(self.temporary.name)
        (self.root / 'server').mkdir()
        (self.root / 'server' / 'agent.py').write_text('SYSTEM_PROMPT = "original prompt"\n', encoding='utf-8')
        (self.root / 'server' / 'control.py').write_text('CONTROL_VERSION = 1\n', encoding='utf-8')
        (self.root / 'requirements.txt').write_text('mujoco==3.13.0\n', encoding='utf-8')
        self.patches = [patch.object(artifacts, 'ROOT', self.root),
                        patch.object(artifacts, 'METHODS', self.root / 'methods')]
        for item in self.patches: item.start()
        self.recipe = {'name': 'baseline', 'mode': 'agent',
                       'agent': {'provider': 'codex', 'model': 'gpt-6-astra', 'max_rounds': 3}}

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temporary.cleanup()

    def test_same_canonical_recipe_is_atomic_and_child_keeps_parent(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            versions = list(pool.map(register_method, [self.recipe] * 4))
        baseline = versions[0]
        self.assertTrue(all(version == baseline for version in versions))
        self.assertEqual(baseline['id'], baseline['method_id'])
        reordered = json.loads(json.dumps(self.recipe, sort_keys=True))
        self.assertEqual(register_method(reordered), baseline)
        self.assertEqual(verify_method(baseline['id']), baseline)
        child = register_method({**self.recipe, 'parent_id': baseline['id'], 'change_summary': 'Reduce action duration.',
                                 'skills': {'duration_s': .7}})
        self.assertNotEqual(child['id'], baseline['id'])
        self.assertEqual(child['recipe']['evolution_generation'], 1)
        self.assertEqual(get_recipe(child['id'])['parent_id'], baseline['id'])
        self.assertEqual(get_method(baseline['id']), baseline)
        self.assertEqual(len(list_methods()), 2)
        with self.assertRaises(FileNotFoundError):
            register_method({**self.recipe, 'parent_id': 'method-' + '0' * 24})
        with self.assertRaises(ValueError):
            register_method({**self.recipe, 'parent_id': baseline['id'], 'evolution_generation': 8})

    def test_code_prompt_and_environment_materials_change_identity_and_block_old_runtime(self):
        baseline = register_method(self.recipe)
        material_root = self.root / 'methods' / baseline['id'] / 'materials'
        self.assertEqual((material_root / 'prompts' / 'agent-system.txt').read_text(encoding='utf-8'), 'original prompt')
        self.assertEqual(baseline['model_revision_status'], 'declared_unverified')
        (self.root / 'server' / 'control.py').write_text('CONTROL_VERSION = 2\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '登记新版本'): verify_method(baseline['id'])
        controller_change = register_method(self.recipe)
        self.assertNotEqual(controller_change['id'], baseline['id'])
        self.assertIn('CONTROL_VERSION = 1', (material_root / 'server' / 'control.py').read_text(encoding='utf-8'))
        (self.root / 'server' / 'agent.py').write_text('SYSTEM_PROMPT = "changed prompt"\n', encoding='utf-8')
        prompt_change = register_method(self.recipe)
        self.assertNotEqual(prompt_change['id'], controller_change['id'])
        (self.root / 'requirements.txt').write_text('mujoco==3.14.0\n', encoding='utf-8')
        environment_change = register_method(self.recipe)
        self.assertNotEqual(environment_change['id'], prompt_change['id'])
        self.assertEqual(verify_method(environment_change['id']), environment_change)

    def test_secret_fields_url_credentials_and_registry_escape_are_rejected(self):
        unsafe = [
            {**self.recipe, 'agent': {'api_key': 'private'}},
            {**self.recipe, 'skills': {'nested': [{'decision_api_key': 'private'}]}},
            {**self.recipe, 'skills': {'decision_api_keys': {'0': 'private'}}},
            {**self.recipe, 'agent': {'base_url': 'https://user:private@example.com/v1'}},
            {**self.recipe, 'agent': {'base_url': 'https://example.com/v1?api_key=private'}},
            {**self.recipe, 'controller': {'gain': float('nan')}},
            {**self.recipe, 'parent_id': '../../outside'},
        ]
        for recipe in unsafe:
            with self.subTest(recipe=recipe), self.assertRaises(ValueError): register_method(recipe)
        for method_id in ('../../outside', 'method-../outside', '/absolute', 'C:\\outside'):
            with self.subTest(method_id=method_id), self.assertRaises(ValueError): get_method(method_id)
        self.assertFalse((self.root / 'methods').exists())
        hybrid = MethodRecipe(name='hybrid', mode='hybrid', agent=self.recipe['agent'],
                              decision={'kind': 'typesafe', 'url': 'https://decision.example/predict',
                                        'model': 'decision-model', 'vision': False})
        self.assertEqual(hybrid.decision.kind, 'typesafe')
        self.assertEqual(hybrid.decision.question_name, 'next_action')
        with self.assertRaises(ValueError):
            MethodRecipe(name='missing-planner', mode='hybrid', decision=hybrid.decision)

    def test_multiple_decision_recipes_have_explicit_strategies_and_positive_weights(self):
        decision = {'kind': 'custom', 'url': 'http://127.0.0.1:8011/choose', 'model': 'local-fixture'}
        base = {'name': 'hybrid decision fixtures', 'mode': 'hybrid', 'agent': self.recipe['agent']}
        legacy = MethodRecipe(**base, decision=decision)
        self.assertEqual(legacy.decision.weight, 1.)
        self.assertEqual(legacy.decisions, [])
        self.assertEqual(legacy.decision_strategy, 'single')
        modern = MethodRecipe(**base, decisions=[decision])
        self.assertIsNone(modern.decision)
        self.assertEqual(len(modern.decisions), 1)
        registered = register_method(modern)
        self.assertEqual(get_method(registered['id']), registered)
        self.assertEqual(registered['recipe']['decisions'][0]['weight'], 1.)
        for strategy in ('cascade', 'weighted_vote'):
            configured = MethodRecipe(**base, decisions=[decision, {**decision, 'weight': 2.}], decision_strategy=strategy)
            self.assertEqual(configured.decisions[1].weight, 2.)
        invalid = [
            {'decision': decision, 'decisions': [decision]},
            {'decisions': [decision, decision]},
            {'decision_strategy': 'cascade', 'decisions': [decision]},
            {'decision_strategy': 'weighted_vote', 'decisions': []},
            {'decision_strategy': 'cascade', 'decisions': [decision] * 9},
            {'decision': {**decision, 'weight': 0}},
            {'decision': {**decision, 'weight': -1}},
            {'decision': {**decision, 'weight': 101}},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError): MethodRecipe(**base, **values)
        for values in ({'decision': decision}, {'decisions': [decision]}, {'decision_strategy': 'cascade'}):
            with self.subTest(mode='agent', values=values), self.assertRaises(ValueError):
                MethodRecipe(**self.recipe, **values)


class RunArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation('so101', render=True, scene_id='drawer', seed=50)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='robot-run-artifact-test-')
        self.root = Path(self.temporary.name)
        self.run_patch = patch.object(runlog, 'RUNS', self.root / 'runs')
        self.run_patch.start()
        self.sim.reset()

    def tearDown(self):
        self.run_patch.stop()
        self.temporary.cleanup()

    def test_real_images_checkpoint_and_buffered_trajectory_survive_finish_and_replay(self):
        sim = self.sim
        log = RunLog('agent', {'task': 'persist observations', 'api_key': 'known-private'}, ['known-private'])
        initial = sim.checkpoint()
        checkpoint_ref = log.save_checkpoint(initial)
        observation = sim.observe()
        saved = log.save_observation(observation, 1)
        self.assertEqual([camera['name'] for camera in saved['cameras']], ['front', 'side', 'wrist'])
        for original, camera in zip(observation['cameras'], saved['cameras']):
            self.assertIn('image_base64', original)
            self.assertNotIn('image_base64', camera)
            raw = safe_artifact_path(log.id, camera['image_ref']).read_bytes()
            self.assertEqual(raw, base64.b64decode(original['image_base64']))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), camera['image_sha256'])
        for index in range(4):
            sim.step()
            frame = {'index': index, 'time': float(sim.data.time), 'integration_state': sim.checkpoint()['state'],
                     'ctrl': sim.data.ctrl.tolist(), 'qpos': sim.data.qpos.tolist(), 'qvel': sim.data.qvel.tolist(),
                     'contacts': [], 'evaluation': sim.evaluator.snapshot(), 'api_key': 'known-private'}
            log.write_trajectory(frame)
        log.flush_trajectory()
        trajectory_path = safe_artifact_path(log.id, 'trajectory.jsonl')
        self.assertEqual(len(trajectory_path.read_text(encoding='utf-8').splitlines()), 4)
        log.write('model_output', {'content': 'known-private public output', 'reasoning_content': 'hidden thoughts'})
        log.finish({'status': 'complete'})
        summary = json.loads((log.folder / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['artifacts']['trajectory_frames'], 4)
        self.assertEqual(summary['artifacts']['observation_count'], 1)
        self.assertEqual(summary['artifacts']['checkpoints']['initial']['path'], checkpoint_ref)
        self.assertEqual(json.loads(safe_artifact_path(log.id, checkpoint_ref).read_text(encoding='utf-8')), initial)
        self.assertEqual(len(safe_artifact_path(log.id, summary['artifacts']['observation_manifest']).read_text(encoding='utf-8').splitlines()), 1)
        frames = [json.loads(line) for line in trajectory_path.read_text(encoding='utf-8').splitlines()]
        for frame in frames:
            mujoco.mj_setState(sim.model, sim.data, np.asarray(frame['integration_state']), mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(sim.model, sim.data)
            np.testing.assert_array_equal(sim.data.qpos, frame['qpos'])
            np.testing.assert_array_equal(sim.data.ctrl, frame['ctrl'])
            self.assertEqual(sim.data.time, frame['time'])
        for path in log.folder.rglob('*'):
            if path.suffix in ('.json', '.jsonl'):
                text = path.read_text(encoding='utf-8')
                self.assertNotIn('known-private', text)
                self.assertNotIn('hidden thoughts', text)
                self.assertNotIn('image_base64', text)
        with self.assertRaises(ValueError): log.write_trajectory(frames[-1])

    def test_artifact_names_and_duplicate_writes_cannot_escape_or_overwrite(self):
        log = RunLog('test', {})
        observation = self.sim.observe()
        log.save_observation(observation, 1)
        original = safe_artifact_path(log.id, 'observations/000001/front.jpg').read_bytes()
        with self.assertRaises(FileExistsError): log.save_observation(observation, 1)
        self.assertEqual(safe_artifact_path(log.id, 'observations/000001/front.jpg').read_bytes(), original)
        with self.assertRaises(ValueError): log.save_observation(observation, '../outside')
        unsafe_observation = copy.deepcopy(observation)
        unsafe_observation['cameras'][0]['name'] = '../../outside'
        with self.assertRaises(ValueError): log.save_observation(unsafe_observation, 2)
        with self.assertRaises(ValueError): log.save_checkpoint({}, '../outside')
        log.save_checkpoint({'state': [1.]})
        with self.assertRaises(FileExistsError): log.save_checkpoint({'state': [2.]})
        for path in ('../outside', '/absolute', 'C:\\outside', 'checkpoints/../../outside', 'checkpoints\\initial.json', 'checkpoints/initial.json:extra'):
            with self.subTest(path=path), self.assertRaises(ValueError): safe_artifact_path(log.id, path)
        with self.assertRaises(ValueError): safe_artifact_path('../outside', 'initial.json')
        log.finish({'status': 'complete'})

    def test_legacy_logs_remain_readable_without_artifact_metadata(self):
        log = RunLog('legacy', {'task': 'old run'})
        legacy = {**log.metadata, 'status': 'complete'}
        (log.folder / 'summary.json').write_text(json.dumps(legacy), encoding='utf-8')
        listed = runlog.list_runs()
        self.assertEqual(listed[0]['run_id'], log.id)
        self.assertNotIn('artifacts', listed[0])


if __name__ == '__main__':
    unittest.main()
