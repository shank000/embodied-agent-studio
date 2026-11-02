"""SDK-free adapter contracts plus explicit opt-in native acceptance.

Fixture tests establish the adapter/recording contract, not physical capability.
Run EMBODI_TEST_OMNIGIBSON=1 in the main workbench Python environment. The real
test launches the configured Python 3.11 SDK through the production IPC bridge.
It never calls a model or writes user experiment data.
"""
import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from server.backends.omnigibson_backend import OmniGibsonSimulation, CAPABILITIES
from server.backends.omnigibson_catalog import catalog, resolve_selection, DEFAULT_SCENE, DEFAULT_TASK
from server.backends.omnigibson_native import NativeSession, camera_alias, single, MANUAL_EPISODE_MAX_STEPS
from server.runlog import RunLog


class SessionFixture:
    """Deterministic external-protocol fixture; deliberately contains no physics."""
    def __init__(self, model, scene, task, seed, render=True):
        self.timestep, self.closed = 1/30, False
        self.camera_map = {'head_native': None, 'wrist_native': None}
        self.action_spec = {'type': 'native', 'robot_id': 'robot0', 'names': ['base_0', 'arm_left_0', 'gripper_right_0'],
            'low': [-1.]*3, 'high': [1.]*3, 'dimension': 3, 'dt_s': self.timestep,
            'unit': 'normalized_controller_command', 'control_groups': [], 'grasping_mode': 'physical', 'max_repeat_steps': 120}
        self.initial_state = {'scene_model': scene, 'activity_name': task, 'activity_instance_id': seed,
                              'test_fixture': True}
        self.reset()

    def reset(self):
        self.elapsed, self.steps, self.reward = 0., 0, 0.
        self.actions = []
        self.success = self.terminated = self.truncated = False
        self.q = 0.

    def hold_action(self): return [0., .2, -.5]
    def set_budget(self, duration): self.native_budget_s = duration
    def advance(self, action):
        self.actions.append(list(action)); self.steps += 1; self.elapsed = self.steps*self.timestep
    def evaluate(self):
        return {'success': self.success, 'terminated': self.terminated, 'truncated': self.truncated,
                'q_score': self.q, 'reward': self.reward, 'goal_status': {'satisfied': [], 'unsatisfied': [0, 1]}}
    def instruction(self): return 'Fixture task: arrange household objects.'
    def robot_state(self):
        pose = {'frame': 'world', 'position_m': [0., 0., 1.], 'quaternion_wxyz': [1., 0., 0., 0.]}
        return {'id': 'robot0', 'name': 'Fixture R1Pro', 'active': True, 'tcp': pose,
            'joints': [{'name': 'fixture_joint', 'position': .2, 'velocity': 0., 'unit': 'rad', 'position_rad': .2, 'velocity_rad_s': 0.}],
            'end_effectors': {'left': pose, 'right': pose}, 'proprioception': [.2, 0.],
            'gripper': {'unit': 'native_controller_command'}}
    def images(self):
        return {name: {'rgb': np.full((12, 16, 3), 40+i*100, dtype=np.uint8),
            'depth': np.ones((12, 16), dtype=np.float32)*(i+1), 'native_name': '/robot/' + name,
            'intrinsics': np.eye(3).tolist(), 'position_world_m': [0., 0., 1.],
            'quaternion_xyzw': [0., 0., 0., 1.], 'depth_measurement': 'distance_to_camera'}
            for i, name in enumerate(self.camera_map)}
    def viewer_rgb(self): return np.full((12, 16, 3), 225, dtype=np.uint8)
    def camera(self, payload): self.last_camera = payload
    def close(self): self.closed = True


class OmniGibsonCatalogTests(unittest.TestCase):
    def template(self, root, scene, task, definition, instance, *, split='scenes'):
        path = root / '2026-challenge-task-instances' / split / scene / 'json' / f'{scene}_task_{task}_{definition}_{instance}_template.json'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}', encoding='utf-8')
        return path

    def test_installed_instances_are_discovered_without_sdk_or_hidden_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.template(root, 'kitchen', 'setting_table', 0, 4)
            self.template(root, 'kitchen', 'setting_table', 0, 9)
            self.template(root, 'kitchen', 'setting_table', 1, 2)
            self.template(root, 'secret_house', 'secret_activity', 0, 0, split='scene_test/private')
            models, scenes, tasks = catalog(root)
            self.assertIn('r1pro', models)
            self.assertEqual(scenes['kitchen']['instances']['setting_table'], [4, 9])
            self.assertIn('setting_table__definition_1', tasks)
            self.assertNotIn('secret_house', scenes)
            self.assertFalse(tasks['setting_table']['native_validated'])
            self.assertEqual(resolve_selection('r1pro', 'kitchen', 'setting_table', 1, root)['activity_instance_id'], 9)
            for seed in (2, -1, True):
                with self.assertRaises(ValueError): resolve_selection('r1pro', 'kitchen', 'setting_table', seed, root)

    def test_example_is_not_claimed_installed_or_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            _, scenes, tasks = catalog(directory)
            self.assertEqual(scenes[DEFAULT_SCENE]['instances'][DEFAULT_TASK], [])
            self.assertEqual(tasks[DEFAULT_TASK]['catalog_evidence'], 'upstream_example')
            with self.assertRaisesRegex(ValueError, '预采样实例'):
                resolve_selection('r1pro', DEFAULT_SCENE, DEFAULT_TASK, 0, directory)

    def test_legacy_asset_template_is_not_listed_as_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root/'behavior-1k-assets/scenes/legacy_house/json/legacy_house_task_setting_table_0_0_template.json'
            legacy.parent.mkdir(parents=True)
            legacy.write_text('{}', encoding='utf-8')
            _, scenes, tasks = catalog(root)
            self.assertNotIn('legacy_house', scenes)
            self.assertNotIn('setting_table', tasks)


class OmniGibsonAdapterTests(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('server.backends.omnigibson_backend.NativeSession', SessionFixture)
        self.patcher.start()
        self.sim = OmniGibsonSimulation()

    def tearDown(self): self.sim.close(); self.patcher.stop()

    def action(self, steps=3):
        return {'action': 'native_action', 'generation': self.sim.generation,
            'native_action': {'status': 'move', 'action': [.1, .3, -.4], 'steps': steps, 'explanation': 'fixture action'}}

    def test_native_observation_preserves_sensor_modalities_and_excludes_evaluation(self):
        state, obs = self.sim.state(), self.sim.observe()
        self.assertEqual(state['backend']['action_mode'], 'native')
        self.assertEqual(state['backend']['camera_names'], ['head_native', 'wrist_native'])
        self.assertEqual(obs['native_action_spec']['names'], ['base_0', 'arm_left_0', 'gripper_right_0'])
        self.assertNotIn('evaluation', obs)
        self.assertNotIn('goal_status', json.dumps(obs))
        self.assertEqual(len(obs['cameras']), 2)
        depth = np.load(io.BytesIO(base64.b64decode(obs['cameras'][1]['depth']['data_base64'])), allow_pickle=False)
        self.assertEqual(depth.dtype, np.float32); self.assertTrue(np.all(depth == 2))
        json.dumps(state, allow_nan=False)

    def test_native_actions_execute_incrementally_and_stop_discards_pending_steps(self):
        result = self.sim.command(self.action(10))
        self.assertEqual(result['native_steps'], 10)
        self.assertEqual(self.sim.session.actions, [])
        self.sim.step(); self.sim.step()
        self.assertEqual(len(self.sim.session.actions), 2)
        self.assertAlmostEqual(self.sim.time, 2/30)
        self.sim.command({'action': 'stop'})
        self.assertFalse(self.sim.pending); self.assertTrue(self.sim.paused)
        self.assertEqual(self.sim.task['status'], 'cancelled')

    def test_action_validation_rejects_stale_oversized_nonfinite_or_done(self):
        for edit in ({'action': [0., 0.]}, {'action': [2., 0., 0.]}, {'action': [float('nan'), 0., 0.]},
                     {'action': [True, 0., 0.]}, {'steps': 121}, {'steps': True}, {'status': 'done'}):
            payload = self.action(); payload['native_action'].update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError): self.sim.command(payload)
        with self.assertRaises(ValueError): self.sim.command({**self.action(), 'generation': 999})
        self.assertEqual(self.sim.session.actions, [])

    def test_policy_chunk_uses_named_controller_commands_and_native_cadence(self):
        payload = {'action': 'policy_chunk', 'generation': self.sim.generation, 'action_space': 'native',
            'action_names': self.sim.action_space['names'], 'actions': [[.1, .2, .3], [-.1, -.2, -.3]],
            'dt_s': 2/30, 'robot_id': 'robot0'}
        self.sim.command(payload)
        while not self.sim.paused: self.sim.step()
        self.assertEqual(len(self.sim.session.actions), 4)
        self.assertEqual(self.sim.session.actions[0], self.sim.session.actions[1])
        self.assertEqual(self.sim.task['status'], 'complete')
        for change in ({'action_space': 'joint'}, {'action_names': ['x']*3}, {'dt_s': .05}, {'robot_id': 'arm1'}):
            with self.subTest(change=change), self.assertRaises(ValueError): self.sim.command({**payload, **change})

    def test_native_q_is_not_full_success_and_verification_does_not_move(self):
        self.sim.session.q = .7
        result = self.sim.verify_task(self.sim.generation)
        self.assertFalse(result['success']); self.assertEqual(result['q_score'], .7)
        self.assertEqual(self.sim.session.actions, [])
        self.sim.session.success = self.sim.session.terminated = True
        self.sim.session.q = 1.
        self.assertTrue(self.sim.verify_task(self.sim.generation)['success'])
        with self.assertRaises(ValueError): self.sim.command(self.action())

    def test_native_failure_and_timeout_preserve_distinct_endings(self):
        self.sim.command(self.action())
        self.sim.session.terminated = True
        self.sim.step()
        self.assertTrue(self.sim.paused); self.assertEqual(self.sim.task['status'], 'failed')
        self.assertTrue(self.sim.state()['evaluation']['terminated'])
        self.assertFalse(self.sim.state()['evaluation']['truncated'])
        self.sim.session.reset(); self.sim._reset_state()
        self.sim.command(self.action()); self.sim.session.truncated = True; self.sim.step()
        value = self.sim.state()['evaluation']
        self.assertFalse(value['terminated']); self.assertTrue(value['truncated'])
        self.assertEqual(self.sim.task['status'], 'limit')

    def test_run_budget_does_not_execute_rest_of_action(self):
        self.sim.command({'action': 'run_limits', 'duration_s': 2/30, 'evaluator_stop': True})
        self.sim.command(self.action(10))
        while not self.sim.paused: self.sim.step()
        self.assertEqual(len(self.sim.session.actions), 2)
        self.assertEqual(self.sim.task['status'], 'limit')
        with self.assertRaises(ValueError): self.sim.command(self.action())

    def test_free_view_and_native_camera_share_genuine_render_path(self):
        self.assertTrue(np.all(self.sim.render_rgb() == 225))
        self.sim.command({'action': 'view', 'view': 'head_native'})
        self.assertTrue(np.all(self.sim.render_rgb() == 40))
        self.sim.command({'action': 'camera', 'pan_dy': .2})
        self.assertEqual(self.sim.view, 'perspective')
        self.assertEqual(self.sim.session.last_camera['pan_dy'], .2)

    def test_metadata_checkpoint_does_not_claim_physics_restore(self):
        checkpoint = self.sim.checkpoint()
        self.assertFalse(checkpoint['restorable']); self.assertIn('state_id', checkpoint)
        self.assertNotIn('restore', CAPABILITIES)
        self.assertFalse(self.sim.trajectory_frame()['replay_supported'])
        with self.assertRaises(ValueError): self.sim.command({'action': 'restore', 'checkpoint': checkpoint})

    def test_observation_roundtrip_uses_production_log_and_typed_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            log = RunLog('omnigibson-contract', {'backend': self.sim.backend}, directory=Path(directory))
            saved = log.save_observation(self.sim.observe(), 1)
            camera = saved['cameras'][0]
            self.assertTrue((log.folder/camera['image_ref']).is_file())
            depth = camera['depth']
            self.assertNotIn('data_base64', depth)
            stored = np.load(log.folder/depth['depth_ref'], allow_pickle=False)
            self.assertEqual(stored.dtype, np.float32); self.assertEqual(stored.shape, (12, 16))
            log.finish({'status': 'test_complete'})


class OmniGibsonNativeBoundaryTests(unittest.TestCase):
    def test_native_reset_restores_manual_budget_before_resetting_sdk(self):
        session = NativeSession.__new__(NativeSession)
        timeout = SimpleNamespace(_max_steps=300)
        observed = []
        session.env = SimpleNamespace(task=SimpleNamespace(_termination_conditions={'timeout': timeout}))
        def reset():
            observed.append(timeout._max_steps)
            raise RuntimeError('stop before unrelated SDK initialization')
        session.env.reset = reset
        with self.assertRaisesRegex(RuntimeError, 'unrelated SDK'):
            session.reset()
        self.assertEqual(observed, [MANUAL_EPISODE_MAX_STEPS])

    def test_native_batch_shape_is_explicit_and_ambiguous_batch_rejected(self):
        self.assertEqual(single([{'rgb': 'frame'}], 'observation'), {'rgb': 'frame'})
        with self.assertRaises(RuntimeError): single([{}, {}], 'observation')
        self.assertNotEqual(camera_alias('/robot/a:b'), camera_alias('/robot/a/b'))
        self.assertNotIn('/', camera_alias('/robot/a:b'))

    def test_native_evaluation_calls_original_q_function_with_initial_masks(self):
        session = NativeSession.__new__(NativeSession)
        seen = []
        task = SimpleNamespace(compiled_task=SimpleNamespace(check_goal=lambda fn: (False, {'satisfied': [0], 'unsatisfied': [1]})),
            _evaluate_predicate=lambda *args: False, get_goal_option_satisfaction=lambda index: [[True, False]])
        session.env = SimpleNamespace(task=task)
        session.initial_options = [[False, False]]
        session.compute_q_score = lambda success, now, initial: seen.append((success, now, initial)) or .5
        session.reward, session.steps = .5, 2
        session.native_step_limit = 100
        session.terminated, session.truncated, session.last_info = False, True, {'done': {'termination_conditions': {'timeout': {'done': True}}}}
        value = session.evaluate()
        self.assertEqual(seen, [(False, [[True, False]], [[False, False]])])
        self.assertEqual(value['q_score'], .5); self.assertTrue(value['truncated']); self.assertFalse(value['terminated'])


@unittest.skipUnless(os.getenv('EMBODI_TEST_OMNIGIBSON') == '1', 'Native OmniGibson requires the configured licensed SDK, assets and RTX host')
class OmniGibsonNativeAcceptanceTests(unittest.TestCase):
    def test_real_task_rgb_depth_action_evaluation_and_log(self):
        from server.backends.process_backend import ProcessBackend
        from server.video import find_ffmpeg
        sim = ProcessBackend('omnigibson')
        try:
            before = sim.command({'action': 'observe'})
            self.assertTrue(before['cameras'])
            self.assertTrue(all(camera['depth']['unit'] == 'm' for camera in before['cameras']))
            command = [item['target'] for item in before['actuators']]
            with tempfile.TemporaryDirectory() as directory:
                log = RunLog('omnigibson-native', {'backend': sim.state()['backend']}, directory=Path(directory),
                             record_video=bool(find_ffmpeg()))
                log.save_observation(before, 1)
                log.save_checkpoint(sim.checkpoint())
                def capture():
                    frame = sim.trajectory_frame()
                    log.write_trajectory(frame)
                    if log.video_due(frame['time']):
                        log.capture_video_frame(sim.render_rgb(), frame['time'])
                sim.command({'action': 'run_limits', 'duration_s': 5., 'evaluator_stop': True})
                sim.command({'action': 'native_action', 'generation': sim.generation,
                    'native_action': {'status': 'move', 'action': command, 'steps': 2, 'explanation': 'native acceptance hold'}})
                while not sim.paused:
                    sim.advance(2, on_step=capture, recording=log.video_sampling())
                self.assertGreater(sim.state()['time'], 0)
                evaluated = sim.verify_task(sim.generation)
                self.assertGreaterEqual(evaluated['q_score'], 0); self.assertLessEqual(evaluated['q_score'], 1)
                log.save_observation(sim.command({'action': 'observe'}), 1, phase='after')
                log.finish({'status': 'native_acceptance', 'evaluation': evaluated})
                self.assertGreater(log.artifacts['trajectory_frames'], 0)
                if find_ffmpeg(): self.assertEqual(log.artifacts['video']['status'], 'saved')
        finally:
            sim.close()


if __name__ == '__main__': unittest.main()
