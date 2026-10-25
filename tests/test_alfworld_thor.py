"""ALFWorld visual contracts; native cases opt in to the isolated Unity 2.1 SDK.

EMBODI_TEST_ALFWORLD_THOR=1 enables real frames, action replay and all six
official task families. Expert trajectories are evaluator test fixtures, never
actor observations or evidence of model policy quality.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from server.backends.alfworld_thor_backend import ALFWorldThorSimulation, TASKS
from server.backends import alfworld_thor_assets as assets


def trajectory(task='pick_and_place_simple', room=1):
    return {'task_type': task, 'scene': {'scene_num': room, 'floor_plan': f'FloorPlan{room}',
        'object_poses': [{'objectName': 'fixture mug', 'position': {'x': 4, 'y': 1, 'z': 5}}],
        'object_toggles': [], 'dirty_and_empty': False,
        'init_action': {'action': 'TeleportFull', 'x': 1, 'y': 1, 'z': 2, 'rotation': 90, 'horizon': 0}},
        'turk_annotations': {'anns': [{'task_desc': 'Put the mug in the cabinet.'}]},
        'plan': {'high_pddl': [{'private_expert': 'never exposed'}], 'low_actions': []},
        'pddl_params': {'private_goal': 'never exposed'}}


class NativeControllerFixture:
    """External SDK boundary fixture; no claim of simulated physics."""
    def __init__(self):
        self.calls = []; self.won = False; self.fail_action = None
        self.return_pre_postcondition_event = False
        self.reset_event('Initialize')

    def reset_event(self, action):
        objects = [
            {'objectId': 'Mug|+01.00|+02.00|+03.00', 'objectType': 'Mug', 'visible': True,
             'pickupable': True, 'isPickedUp': False, 'position': {'x': 1}, 'distance': .5},
            {'objectId': 'Cabinet|+04.00|+05.00|+06.00', 'objectType': 'Cabinet', 'visible': True,
             'openable': True, 'isOpen': True, 'receptacle': True},
            {'objectId': 'Faucet|+07.00|+08.00|+09.00', 'objectType': 'Faucet', 'visible': True,
             'toggleable': True, 'isToggled': False},
            {'objectId': 'Hidden|+10.00|+11.00|+12.00', 'objectType': 'HiddenFixture', 'visible': False,
             'openable': True, 'isOpen': False}]
        self.last_event = SimpleNamespace(metadata={'objects': objects, 'inventoryObjects': [],
            'lastAction': action, 'lastActionSuccess': True, 'errorMessage': ''},
            frame=np.arange(16*12*3, dtype=np.uint8).reshape(12, 16, 3))

    def reset(self, scene): self.calls.append(('reset', scene)); self.reset_event('Initialize'); self.won = False
    def restore_scene(self, poses, toggles, dirty): self.calls.append(('restore_scene', deepcopy(poses), toggles, dirty))
    def set_task(self, traj, args, reward_type): self.calls.append(('set_task', deepcopy(traj), args.reward_config, reward_type))
    def get_goal_satisfied(self): return self.won
    def get_goal_conditions_met(self): return (int(self.won), 1)
    def stop(self): self.calls.append(('stop',))

    def step(self, action):
        self.calls.append(('step', deepcopy(action)))
        if self.fail_action: raise self.fail_action
        event = deepcopy(self.last_event); event.metadata['lastAction'] = action['action']
        if action['action'] == 'PickupObject':
            event.metadata['inventoryObjects'] = [{'objectId': action['objectId'], 'objectType': 'Mug'}]
        elif action['action'] == 'PutObject': event.metadata['inventoryObjects'] = []
        elif action['action'] == 'RotateRight': event.frame = np.flip(event.frame, axis=1).copy()
        self.last_event = event
        if self.return_pre_postcondition_event:
            final = deepcopy(event); final.metadata['lastAction'] = 'CleanObject'; final.frame[:] = 201
            self.last_event = final
        return event


class ALFWorldVisualContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.game = self.root/'json_2.1.1/valid_unseen/pick_and_place_simple-Mug-None-Cabinet-1/trial_0/game.tw-pddl'
        self.game.parent.mkdir(parents=True)
        self.game.write_text(json.dumps({'solvable': True}), encoding='utf-8')
        self.traj = trajectory()
        self.game.with_name('traj_data.json').write_text(json.dumps(self.traj), encoding='utf-8')
        self.controller = NativeControllerFixture()
        self.version = patch('server.backends.alfworld_backend.version', return_value='fixture-alfworld')
        self.version.start(); self.addCleanup(self.version.stop)
        self.native_version = patch('server.backends.alfworld_thor_backend.version', return_value='2.1.0')
        self.native_version.start(); self.addCleanup(self.native_version.stop)
        self.package = patch.dict('sys.modules', {'alfworld': SimpleNamespace(__file__=str(self.root/'alfworld/__init__.py'))})
        self.package.start(); self.addCleanup(self.package.stop)

    def open(self):
        sim = ALFWorldThorSimulation(data_path=self.root, controller=self.controller)
        self.addCleanup(sim.close)
        return sim

    def test_official_scene_and_task_are_loaded_and_policy_observation_is_bounded(self):
        sim = self.open()
        self.assertEqual(self.controller.calls[0], ('reset', 'FloorPlan1'))
        self.assertEqual(self.controller.calls[1][1], self.traj['scene']['object_poses'])
        self.assertEqual(self.controller.calls[2], ('step', self.traj['scene']['init_action']))
        self.assertEqual(self.controller.calls[3][1], self.traj)
        self.assertEqual(sim.episode_id, self.game.relative_to(self.root).as_posix())
        observation = sim.observe()
        self.assertEqual(len(observation['cameras']), 1)
        self.assertEqual(observation['cameras'][0]['source'], 'alfworld.ThorEnv.last_event.frame')
        self.assertEqual(observation['environment']['task_name'], 'Put the mug in the cabinet.')
        public = json.dumps(observation)
        for forbidden in ('pddl_params', 'private_expert', 'private_goal', 'HiddenFixture', 'position', 'distance', 'goal_conditions', '|+01.00'):
            self.assertNotIn(forbidden, public)
        self.assertIn('rgb_and_visible_object_state', json.dumps(sim.state()['backend']))
        self.assertFalse(sim.verify_task(sim.generation)['success'])
        self.controller.won = True
        self.assertTrue(sim.verify_task(sim.generation)['success'])
        self.assertEqual(sim.verify_task(sim.generation)['values'], {'goal_conditions_met': 1, 'goal_conditions_total': 1})

    def test_put_translates_visible_target_and_inventory_to_the_native_21_signature(self):
        sim = self.open()
        mug = sim._object_handle('Mug|+01.00|+02.00|+03.00')
        cabinet = sim._object_handle('Cabinet|+04.00|+05.00|+06.00')
        sim.execute(json.dumps({'action': 'PickupObject', 'objectId': mug}))
        sim.execute(json.dumps({'action': 'PutObject', 'objectId': cabinet}))
        self.assertEqual(self.controller.calls[-1], ('step', {'action': 'PutObject',
            'objectId': 'Mug|+01.00|+02.00|+03.00', 'receptacleObjectId': 'Cabinet|+04.00|+05.00|+06.00',
            'placeStationary': True}))
        with self.assertRaisesRegex(ValueError, '持有'): sim.execute(json.dumps({'action': 'PutObject', 'objectId': cabinet}))
        self.assertEqual(sim.physics_steps, 2)

    def test_force_teleport_hidden_native_and_stale_actions_cannot_mutate_the_world(self):
        sim = self.open(); count = len(self.controller.calls)
        hidden = sim._object_handle('Hidden|+10.00|+11.00|+12.00')
        cabinet = sim._object_handle('Cabinet|+04.00|+05.00|+06.00')
        for action in ('TeleportFull', json.dumps({'action':'OpenObject','objectId': hidden}),
                       json.dumps({'action':'OpenObject','objectId':'Cabinet|+04.00|+05.00|+06.00'}),
                       json.dumps({'action':'OpenObject','objectId':cabinet,'forceAction':True})):
            with self.subTest(action=action), self.assertRaises(ValueError): sim.execute(action)
        with self.assertRaisesRegex(ValueError, '旧动作'):
            sim.command({'action':'environment_action','environment_action':'RotateRight','generation':sim.generation-1})
        self.assertEqual(len(self.controller.calls), count); self.assertEqual(sim.physics_steps, 0)

    def test_final_official_postcondition_event_drives_observation_and_video(self):
        sim = self.open(); self.controller.return_pre_postcondition_event = True
        sim.execute(json.dumps({'action':'ToggleObjectOn','objectId':sim._object_handle('Faucet|+07.00|+08.00|+09.00')}))
        self.assertEqual(json.loads(sim.observation_text)['last_action'], 'CleanObject')
        self.assertTrue((sim.render_rgb() == 201).all())

    def test_partial_native_execution_error_is_fatal_and_preserves_cause(self):
        sim = self.open(); self.controller.fail_action = ValueError('native fixture failed after movement')
        with self.assertRaisesRegex(RuntimeError, 'native fixture failed') as failure: sim.execute('RotateRight')
        self.assertIsInstance(failure.exception.__cause__, ValueError)

    def test_missing_visual_asset_does_not_fall_back_to_a_text_or_unrelated_thor_scene(self):
        self.game.with_name('traj_data.json').unlink()
        with self.assertRaisesRegex(ValueError, '缺少官方视觉场景'): self.open()
        self.assertFalse(self.controller.calls)

    def test_foreign_sdk_is_rejected_before_constructing_a_native_controller(self):
        with patch('server.backends.alfworld_thor_backend.version', return_value='5.0.0'):
            with self.assertRaisesRegex(RuntimeError, 'ai2thor==2.1.0'):
                ALFWorldThorSimulation(data_path=self.root)

    def test_controller_uses_the_isolated_executable_and_never_the_sdk_default_cache(self):
        sim = self.open()
        constructor = Mock(return_value=self.controller)
        executable = self.root/'isolated-official-build'
        with patch.object(assets, 'executable_path', return_value=executable), patch.dict('sys.modules', {
                'alfworld.env.thor_env': SimpleNamespace(ThorEnv=constructor)}):
            self.assertIs(sim._create_controller(), self.controller)
        self.assertEqual(constructor.call_args.kwargs['build_path'], str(executable))

    def test_checkpoint_replay_preserves_episode_and_rejects_cross_environment(self):
        sim = self.open(); sim.execute('RotateRight'); checkpoint = sim.checkpoint()
        sim.execute('RotateRight'); restored = sim.restore(checkpoint)
        self.assertEqual(restored['restore_mode'], 'native_action_replay')
        self.assertEqual(sim.environment(), checkpoint['environment'])
        changed = {**checkpoint, 'backend': {**checkpoint['backend'], 'engine_id': 'alfworld'}}
        count = len(self.controller.calls)
        with self.assertRaises(ValueError): sim.restore(changed)
        self.assertEqual(len(self.controller.calls), count)


class ALFWorldBuildIsolationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root/'isolated'
        self.build = 'thor-fixture-Linux64'
        self.archive = self.root/'official.zip'
        self.url = 'https://example.invalid/official.zip'

    def archive_with(self, entries):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for name, value in entries.items(): archive.writestr(name, value)
        checksum = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        return patch.object(assets, '_build_spec', return_value=(self.build, self.url, checksum, Path(self.build)))

    def test_verified_install_preserves_other_releases_and_reuses_its_own_build(self):
        other = self.root/'.ai2thor/releases/new-sdk/important-build'
        other.parent.mkdir(parents=True); other.write_text('preserve')
        with self.archive_with({self.build: b'fixture executable', self.build+'_Data/data': b'fixture data'}), patch.object(assets, 'urlopen') as network:
            executable = assets.prepare_build(self.archive, self.cache)
            self.assertEqual(executable.read_bytes(), b'fixture executable')
            self.assertEqual(assets.executable_path(self.cache), executable)
            self.assertEqual(assets.prepare_build(self.archive, self.cache), executable)
            network.assert_not_called()
        self.assertEqual(other.read_text(), 'preserve')
        self.assertTrue(self.archive.is_file())

    def test_missing_build_and_existing_incomplete_directory_are_reported_without_mutation(self):
        with self.archive_with({self.build: b'exe'}):
            with self.assertRaisesRegex(RuntimeError, '尚未安装'): assets.executable_path(self.cache)
            incomplete = self.cache/self.build; incomplete.mkdir(parents=True)
            sentinel = incomplete/'keep'; sentinel.write_text('existing user work')
            with self.assertRaisesRegex(RuntimeError, '已保留原文件'): assets.prepare_build(self.archive, self.cache)
            self.assertEqual(sentinel.read_text(), 'existing user work')

    def test_checksum_mismatch_is_rejected_before_install(self):
        with self.archive_with({self.build: b'exe'}):
            self.archive.write_bytes(b'corrupt archive')
            with self.assertRaisesRegex(RuntimeError, 'SHA256'): assets.prepare_build(self.archive, self.cache)
        self.assertFalse((self.cache/self.build).exists())
        self.assertEqual(self.archive.read_bytes(), b'corrupt archive')

    def test_path_traversal_is_rejected_before_extracting_any_member(self):
        with self.archive_with({self.build: b'exe', '../outside': b'escape'}):
            with self.assertRaisesRegex(RuntimeError, '越界路径'): assets.prepare_build(self.archive, self.cache)
        self.assertFalse((self.cache/self.build).exists())
        self.assertFalse((self.root/'outside').exists())
        self.assertFalse((self.cache/'outside').exists())

    def test_https_download_is_streamed_verified_and_cached(self):
        with self.archive_with({self.build: b'exe'}):
            response = io.BytesIO(self.archive.read_bytes()); response.geturl = lambda: self.url
            with patch.object(assets, 'urlopen', return_value=response) as network:
                executable = assets.prepare_build(cache_root=self.cache)
            network.assert_called_once_with(self.url, timeout=120)
        self.assertTrue(executable.is_file()); self.assertTrue((self.cache/(self.build+'.zip')).is_file())

    def test_sdk_metadata_keeps_the_official_digest_and_upgrades_url_to_https(self):
        metadata = {'Linux': {'url': 'http://official.invalid/builds/thor-201909061227-Linux64.zip', 'sha256': 'a'*64}}
        with patch.object(assets, 'version', return_value='2.1.0'), patch.object(assets.platform, 'system', return_value='Linux'), patch.dict('sys.modules', {'ai2thor._builds': SimpleNamespace(BUILDS=metadata)}):
            build, url, checksum, relative = assets._build_spec()
        self.assertEqual(url, 'https://official.invalid/builds/thor-201909061227-Linux64.zip')
        self.assertEqual(checksum, 'a'*64); self.assertEqual(relative, Path(build))


@unittest.skipUnless(os.getenv('EMBODI_TEST_ALFWORLD_THOR') == '1', 'Requires isolated official ALFWorld visual SDK/Unity 2.1 and episode assets')
class NativeALFWorldVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.sim = ALFWorldThorSimulation()

    @classmethod
    def tearDownClass(cls): cls.sim.close()

    def setUp(self):
        self.sim.command({'action':'scene','scene_id':'alfworld_thor_unseen','task_id':'pick_and_place_simple','seed':42})

    def test_native_rgb_changes_after_navigation_and_no_expert_or_scene_state_reaches_policy(self):
        sim = self.sim; before = sim.render_rgb(); observation = sim.observe()
        self.assertEqual(before.shape, (480, 640, 3)); self.assertGreater(float(before.std()), 5.)
        self.assertEqual(len(observation['cameras']), 1)
        self.assertIsNone(observation['tcp']); self.assertEqual(observation['time_unit'], 'environment_step')
        public = json.dumps(observation['environment'])
        for forbidden in ('pddl_params', 'high_pddl', 'low_actions', 'position', 'distance', 'goal_conditions', '|'):
            self.assertNotIn(forbidden, public)
        self.assertFalse(sim.native_success())
        sim.execute('RotateRight')
        self.assertGreater(float(np.abs(before.astype(float)-sim.render_rgb()).mean()), 1.)
        self.assertEqual(sim.physics_steps, 1)

    def test_native_navigation_checkpoint_replay_and_stale_action_rejection(self):
        sim = self.sim; sim.execute('RotateRight'); checkpoint = sim.checkpoint(); generation = sim.generation
        sim.execute('RotateLeft'); result = sim.restore(checkpoint)
        self.assertEqual(result['restore_mode'], 'native_action_replay')
        self.assertEqual(sim.environment(), checkpoint['environment'])
        with self.assertRaisesRegex(ValueError, '旧动作'):
            sim.command({'action':'environment_action','environment_action':'RotateRight','generation':generation})

    def test_each_official_task_family_loads_and_uses_its_native_goal(self):
        for task in TASKS:
            with self.subTest(task=task):
                self.sim.command({'action':'scene','scene_id':'alfworld_thor_unseen','task_id':task,'seed':42})
                self.assertEqual(self.sim.trajectory_data['task_type'], task)
                evaluation = self.sim.verify_task(self.sim.generation)
                self.assertEqual(evaluation['source'], 'alfworld_thor_native_task')
                self.assertFalse(evaluation['success'])
                self.assertEqual(evaluation['success'], bool(self.sim.controller.get_goal_satisfied()))
                self.assertGreater(evaluation['values']['goal_conditions_total'], 0)
                self.assertGreater(float(self.sim.render_rgb().std()), 5.)

    def test_original_official_demonstration_satisfies_the_native_task(self):
        # Replay privileged original actions solely to validate task setup and
        # the official positive success path. They are never sent to a model.
        sim = self.sim
        for step in sim.trajectory_data['plan']['low_actions']:
            sim.controller.step(dict(step['api_action']))
        sim.native_event = sim.controller.last_event; sim._update_observation()
        self.assertTrue(sim.controller.get_goal_satisfied())
        self.assertTrue(sim.verify_task(sim.generation)['success'])

    def test_public_low_level_interface_completes_pick_and_place(self):
        # A fixed reference trajectory exercises the same public action bridge
        # as a policy. Remove expert force flags and use current opaque handles.
        # This validates the integration, not autonomous policy performance.
        sim = self.sim
        for index, step in enumerate(sim.trajectory_data['plan']['low_actions']):
            native = step['api_action']; action = native['action']
            target = native.get('receptacleObjectId' if action == 'PutObject' else 'objectId')
            command = action if target is None else json.dumps({'action': action, 'objectId': sim._object_handle(target)})
            result = sim.command({'action':'environment_action', 'environment_action':command, 'generation':sim.generation})
            self.assertTrue(result['last_action_success'], f'public step {index} {action}: {result.get("error")}')
        self.assertTrue(sim.verify_task(sim.generation)['success'])


if __name__ == '__main__': unittest.main()
