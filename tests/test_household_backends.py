"""Household SDK contracts and opt-in real ALFWorld/AI2-THOR regressions."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import zipfile

from server.backends.alfworld_backend import ALFWorldSimulation
from server.backends.thor_backend import ThorSimulation, decode_action


class HouseholdActionContracts(unittest.TestCase):
    def test_unity_debug_actions_and_force_parameters_are_rejected(self):
        for action in ('TeleportFull', 'SetObjectPoses', '{"action":"OpenObject","objectId":"x","forceAction":true}',
                       '{"action":"TeleportFull","objectId":"x"}', '[]', 'null'):
            with self.subTest(action=action), self.assertRaises(ValueError): decode_action(action)

    def test_native_navigation_and_named_object_actions_are_preserved(self):
        self.assertEqual(decode_action('MoveAhead'), {'action': 'MoveAhead'})
        handle = 'object_0123456789abcdef01234567'
        self.assertEqual(decode_action(json.dumps({'action': 'OpenObject', 'objectId': handle})),
                         {'action': 'OpenObject', 'objectId': handle})
        with self.assertRaisesRegex(ValueError, '不接受原生'):
            decode_action('{"action":"OpenObject","objectId":"Cabinet|-01.55|+00.50|+00.38"}')

    def test_public_evaluation_describes_task_criterion_and_keeps_native_checks(self):
        # No SDK starts here: this regression covers the public metadata consumed
        # by the scene card; native success semantics have their own opt-in tests.
        for implementation, task_id in ((ALFWorldSimulation, 'pick_and_place_simple'), (ThorSimulation, 'thor_open_object')):
            with self.subTest(engine=implementation.engine_id):
                sim = object.__new__(implementation)
                sim.task_id, sim.task_name = task_id, 'Current SDK task'
                sim.native_criterion = sim.tasks[task_id]['criterion']
                sim.ever_success = False
                sim.native_success = lambda: False
                evaluation = sim.evaluation()
                self.assertEqual(evaluation['name'], sim.tasks[task_id]['name'])
                self.assertEqual(evaluation['instruction'], sim.tasks[task_id]['instruction'])
                self.assertEqual(evaluation['criterion'], sim.tasks[task_id]['criterion'])
                self.assertEqual(evaluation['source'], sim.evaluator_source)
                self.assertFalse(evaluation['checks'][0]['passed'])
                self.assertEqual(sim.time_unit, 'environment_step')


class ThorPolicyBoundaryContracts(unittest.TestCase):
    def fixture(self):
        # SDK-shaped metadata validates the public policy boundary without
        # loading Unity; native transition semantics are tested separately.
        sim = object.__new__(ThorSimulation)
        sim.scene_id, sim.task_id = 'FloorPlan1', 'thor_open_object'
        sim.target_id, sim.target_type = 'Cabinet|-01.55|+00.50|+00.38', 'Cabinet'
        hidden = 'Mug|+01.23|+00.44|-02.10'
        sim._id_to_handle, sim._handle_to_id, sim.physics_steps = {}, {}, 0
        metadata = {'objects': [
            {'objectId': sim.target_id, 'objectType': 'Cabinet', 'visible': True, 'distance': .314,
             'position': {'x': -1.55, 'y': .5, 'z': .38}, 'openable': True, 'isOpen': False},
            {'objectId': hidden, 'objectType': 'Mug', 'visible': False, 'distance': 3.14159, 'pickupable': True}],
            'inventoryObjects': [{'objectId': hidden, 'objectType': 'Mug'}],
            'lastAction': 'OpenObject', 'lastActionSuccess': False,
            'errorMessage': f'{sim.target_id} is occluded by {hidden}'}
        sim.native_event = SimpleNamespace(metadata=metadata)
        sim._register_object_handles(); sim._update_observation()
        return sim

    def test_goal_visible_inventory_commands_and_error_hide_native_coordinates(self):
        sim = self.fixture()
        environment = sim.environment()
        public = json.dumps(environment)
        for native in sim._id_to_handle: self.assertNotIn(native, public)
        self.assertNotIn('|', public)
        self.assertNotIn('distance', public)
        self.assertNotIn('position', public)
        text = json.loads(environment['observation_text'])
        self.assertEqual(text['visible_objects'][0]['objectId'], sim._object_handle(sim.target_id))
        self.assertTrue(text['inventory'][0]['objectId'].startswith('object_'))
        self.assertIn(sim._object_handle(sim.target_id), environment['task_name'])
        self.assertTrue(any(json.loads(command).get('objectId') == sim._object_handle(sim.target_id)
                            for command in environment['admissible_commands'] if command.startswith('{')))
        expected = sim._id_to_handle.copy()
        sim._id_to_handle, sim._handle_to_id = {}, {}
        sim.native_event.metadata['objects'].reverse(); sim._register_object_handles()
        self.assertEqual(sim._id_to_handle, expected)

    def test_only_known_visible_opaque_handles_are_translated_for_sdk(self):
        sim = self.fixture()
        sim.controller = Mock()
        sim.controller.step.return_value = sim.native_event
        handle = sim._object_handle(sim.target_id)
        result = sim._execute(json.dumps({'action': 'OpenObject', 'objectId': handle}))
        sim.controller.step.assert_called_once_with(action='OpenObject', objectId=sim.target_id)
        self.assertNotIn(sim.target_id, json.dumps(result))
        for object_id in (sim.target_id, 'object_000000000000000000000000'):
            with self.subTest(object_id=object_id), self.assertRaises(ValueError):
                sim._execute(json.dumps({'action': 'OpenObject', 'objectId': object_id}))
        self.assertEqual(sim.controller.step.call_count, 1)


class HouseholdAssetInstallation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1]/'tools/setup-household.py'
        spec = importlib.util.spec_from_file_location('household_installer', source)
        cls.installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.installer)

    def test_only_official_unseen_game_paths_are_extracted_and_user_files_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); archive = root/'official.zip'; destination = root/'data'
            name = 'json_2.1.1/valid_unseen/pick_and_place_simple-Mug-None-Sink-1/trial_1/game.tw-pddl'
            game = {'pddl_domain': 'domain', 'pddl_problem': 'problem', 'grammar': 'grammar', 'solvable': True}
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr(name, json.dumps(game))
                output.writestr(name.replace('valid_unseen', 'train'), 'training data')
                output.writestr('../escaped.txt', 'do not extract')
                output.writestr('/absolute/file', 'do not extract')
                output.writestr('json_2.1.1/valid_unseen/a/../../outside/game.tw-pddl', 'do not extract')
            self.assertEqual(self.installer.extract_alfworld_games(archive, destination), 1)
            target = destination/name
            self.assertEqual(json.loads(target.read_text()), game)
            self.assertEqual(len(list(destination.rglob('*tw-pddl'))), 1)
            self.assertFalse((root/'escaped.txt').exists())
            target.write_text('User modified task')
            self.installer.extract_alfworld_games(archive, destination)
            self.assertEqual(target.read_text(), 'User modified task')

    def test_empty_or_wrong_archives_do_not_claim_prepared_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'wrong.zip'
            with zipfile.ZipFile(archive, 'w') as output: output.writestr('notes.txt', 'not a task')
            with self.assertRaisesRegex(ValueError, '没有官方'):
                self.installer.extract_alfworld_games(archive, Path(folder)/'data')


@unittest.skipUnless(os.environ.get('EMBODI_TEST_ALFWORLD') == '1', 'Requires the isolated ALFWorld SDK and official valid_unseen assets')
class NativeALFWorldTests(unittest.TestCase):
    def setUp(self):
        from server.backends.alfworld_backend import ALFWorldSimulation
        self.sim = ALFWorldSimulation()
        self.addCleanup(self.sim.close)

    def test_text_observation_has_native_actions_and_no_fabricated_cameras(self):
        observation = self.sim.observe()
        self.assertEqual(observation['cameras'], [])
        self.assertEqual(observation['time_unit'], 'environment_step')
        self.assertEqual(self.sim.state()['time_unit'], 'environment_step')
        self.assertIsNone(observation['tcp'])
        self.assertIn('Your task is to:', observation['environment']['observation_text'])
        self.assertIn('look', observation['environment']['admissible_commands'])
        self.assertNotIn('evaluation', observation)
        self.assertNotIn('walkthrough', json.dumps(observation))
        self.assertNotIn('pddl_domain', json.dumps(observation))
        self.assertFalse(self.sim.verify_task(self.sim.generation)['success'])
        self.assertEqual(self.sim.time, 0.)
        with self.assertRaises(ValueError): self.sim.step()
        with self.assertRaises(ValueError): self.sim.command({'action': 'move_tcp'})
        with self.assertRaises(ValueError): self.sim.command({'action': 'environment_action', 'generation': self.sim.generation+1, 'environment_action': 'look'})

    def test_native_transition_recording_and_replay(self):
        checkpoint = self.sim.checkpoint()
        action = self.sim.admissible_commands()[0]
        response = self.sim.command({'action': 'environment_action', 'generation': self.sim.generation, 'environment_action': action})
        self.assertEqual(response['step_count'], 1)
        self.assertNotEqual(self.sim.observation_text, checkpoint['environment']['observation_text'])
        frame = self.sim.trajectory_frame()
        self.sim.restore(checkpoint)
        self.assertEqual(self.sim.time, 0.)
        self.sim.replay_frame(frame)
        self.assertEqual(self.sim.time, 1.)
        self.assertEqual(self.sim.environment(), frame['environment'])
        bad = {**checkpoint, 'backend': {**checkpoint['backend'], 'engine_id': 'mujoco'}}
        with self.assertRaises(ValueError): self.sim.restore(bad)

    def test_official_walkthrough_solves_all_six_task_families(self):
        from server.backends.alfworld_backend import TASKS
        for task_id in TASKS:
            with self.subTest(task=task_id):
                self.sim.command({'action': 'scene', 'task_id': task_id, 'seed': 42})
                game = self.sim.data_path/self.sim.episode_id
                walkthrough = json.loads(game.read_text())['walkthrough']
                self.assertGreater(len(walkthrough), 0)
                # Expert actions are test input only, never provided to a policy.
                for command in walkthrough: self.sim.execute(command)
                evaluation = self.sim.verify_task(self.sim.generation)
                self.assertEqual(evaluation['source'], 'alfworld_native_task')
                self.assertTrue(evaluation['success'])

    def test_text_preview_is_labeled_and_seed_selects_reproducible_official_episode(self):
        episode, observation = self.sim.episode_id, self.sim.observation_text
        image = self.sim.render_rgb()
        self.assertEqual(image.shape, (640, 960, 3))
        self.sim.reset()
        self.assertEqual((self.sim.episode_id, self.sim.observation_text), (episode, observation))


@unittest.skipUnless(os.environ.get('EMBODI_TEST_THOR') == '1', 'Requires the isolated AI2-THOR SDK, Unity assets and usable display')
class NativeThorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.backends.thor_backend import ThorSimulation
        cls.sim = ThorSimulation()

    @classmethod
    def tearDownClass(cls): cls.sim.close()

    def setUp(self): self.sim.reset()

    def test_native_camera_and_one_action_transition(self):
        before = self.sim.render_rgb()
        observation = self.sim.observe()
        self.assertEqual(observation['time_unit'], 'environment_step')
        self.assertEqual(self.sim.state()['time_unit'], 'environment_step')
        self.assertEqual([cam['name'] for cam in observation['cameras']], ['front'])
        self.assertIsNone(observation['tcp'])
        self.assertNotIn('evaluation', observation)
        public = json.dumps(observation['environment'])
        self.assertNotIn('|', public)
        self.assertNotIn('distance', public)
        self.assertNotIn('position', public)
        self.assertEqual(self.sim.state()['backend']['state_schema'], 'ai2thor-integration-v2')
        self.assertFalse(self.sim.verify_task(self.sim.generation)['success'])
        self.sim.execute('RotateRight')
        self.assertEqual(self.sim.time, 1.)
        self.assertGreater(float(abs(before.astype(float)-self.sim.render_rgb()).mean()), 1.)

    def test_metadata_is_the_completion_authority(self):
        self.assertEqual(self.sim.verify_task(self.sim.generation)['source'], 'ai2thor_native_task')
        self.assertFalse(self.sim.native_success())
        with self.assertRaises(ValueError): self.sim.execute('TeleportFull')
        with self.assertRaises(ValueError): self.sim.execute('{"action":"OpenObject","objectId":"missing"}')
        self.assertEqual(self.sim.time, 0.)

    def test_real_open_action_changes_native_success(self):
        # Positioning is a privileged test fixture, never exposed as an actor
        # action. The actual task transition still uses normal OpenObject.
        sim = self.sim
        target_handle = sim._object_handle(sim.target_id)
        if target_handle not in {obj['objectId'] for obj in sim._visible_objects()}:
            event = sim.controller.step(action='GetInteractablePoses', objectId=sim.target_id,
                                        horizons=[0, 30, 60], rotations=[0, 90, 180, 270])
            poses = event.metadata['actionReturn']
            self.assertTrue(poses)
            sim.native_event = sim.controller.step(action='TeleportFull', **poses[0])
            self.assertTrue(sim.native_event.metadata['lastActionSuccess'])
            sim._update_observation()
        action = json.dumps({'action': 'OpenObject', 'objectId': target_handle})
        self.assertFalse(sim.verify_task(sim.generation)['success'])
        result = sim.execute(action)
        self.assertTrue(result['last_action_success'], result.get('error'))
        self.assertTrue(sim.verify_task(sim.generation)['success'])

    def test_native_navigation_checkpoint_reexecutes_and_validates(self):
        self.sim.execute('RotateRight')
        checkpoint = self.sim.checkpoint()
        self.sim.execute('RotateLeft')
        legacy = {**checkpoint, 'backend': {**checkpoint['backend'], 'state_schema': 'ai2thor-integration-v1'}}
        with self.assertRaisesRegex(ValueError, '格式'):
            self.sim.restore(legacy)
        result = self.sim.restore(checkpoint)
        self.assertEqual(result['restore_mode'], 'native_action_replay')
        self.assertEqual(self.sim.environment(), checkpoint['environment'])


if __name__ == '__main__': unittest.main()
