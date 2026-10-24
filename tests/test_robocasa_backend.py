"""Official RoboCasa365 contracts and opt-in native Windows/Linux acceptance.

EMBODI_TEST_ROBOCASA365=1 uses one actual kitchen for the entire native class.
No policy model, demonstration dataset, or external API is required.
"""
import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from server.backends.robocasa_catalog import ATOMIC, COMPOSITE, MODELS, SCENES, TASKS
from server.backends.robocasa_backend import ACTION_NAMES, RoboCasaSimulation, primitive_state, restore_primitives


class RoboCasaContracts(unittest.TestCase):
    def test_nonempty_native_contact_history_roundtrips_without_future_entries(self):
        state = SimpleNamespace(board_contact_positions=[np.array([.1, .2]), np.array([.3, .4])],
            table_contact_positions=[np.array([.5, .6])], counters={'contact_count': np.int64(2)}, sdk_handle=object())
        saved = json.loads(json.dumps(primitive_state(state), allow_nan=False))
        self.assertNotIn('sdk_handle', saved)
        state.board_contact_positions.append(np.array([9., 9.]))
        state.table_contact_positions[0][:] = 10.
        state.counters['contact_count'] = np.int64(99)
        restore_primitives(state, saved)
        np.testing.assert_array_equal(state.board_contact_positions, [[.1, .2], [.3, .4]])
        np.testing.assert_array_equal(state.table_contact_positions, [[.5, .6]])
        self.assertEqual(state.counters['contact_count'], 2)
        self.assertIsInstance(state.board_contact_positions[0], np.ndarray)
        self.assertIsInstance(state.counters['contact_count'], np.int64)
        goals = SimpleNamespace(goal_pos=np.array([.1, .2, .3]), ref_pos=np.empty((0, 3)), goal_ori=None)
        saved = json.loads(json.dumps(primitive_state(goals)))
        goals.goal_pos = None; goals.ref_pos = np.empty((0,)); goals.goal_ori = np.eye(3)
        restore_primitives(goals, saved)
        self.assertIsInstance(goals.goal_pos, np.ndarray)
        self.assertEqual(goals.ref_pos.shape, (0, 3))
        self.assertIsNone(goals.goal_ori)
        toaster = SimpleNamespace(_state={0: {'button': np.float64(.5)}, 1: {'button': np.float64(.7)}})
        saved = json.loads(json.dumps(primitive_state(toaster)))
        toaster._state[0]['button'] = 99.
        restore_primitives(toaster, saved)
        self.assertEqual(toaster._state[0]['button'], .5)
        self.assertIsInstance(next(iter(toaster._state)), int)
        long_history = SimpleNamespace(table_contact_positions=[np.array([i*.001, i*.002]) for i in range(1000)])
        encoded = json.dumps(primitive_state(long_history))
        self.assertGreater(len(encoded), 50000)
        long_history.table_contact_positions.append(np.array([99., 99.]))
        restore_primitives(long_history, json.loads(encoded))
        self.assertEqual(len(long_history.table_contact_positions), 1000)
        np.testing.assert_array_equal(long_history.table_contact_positions[-1], [.999, 1.998])

    def test_catalog_is_365_unique_official_tasks_and_imports_no_sdk(self):
        self.assertEqual((len(ATOMIC), len(COMPOSITE), len(TASKS)), (65, 300, 365))
        self.assertEqual(set(SCENES['robocasa_atomic']['tasks']) | set(SCENES['robocasa_composite']['tasks']), set(TASKS))
        self.assertIn('NavigateKitchen', TASKS)
        self.assertIn('OpenDrawer', TASKS)
        probe = subprocess.run([sys.executable, '-B', '-c',
            'import sys; import server.backends.robocasa_catalog; assert not ({"robocasa","robosuite","mujoco","torch"} & set(sys.modules))'],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_native_commands_reject_joint_angles_and_stale_generation(self):
        sim = RoboCasaSimulation.__new__(RoboCasaSimulation)
        sim.generation, sim.motion = 2, None
        with self.assertRaisesRegex(ValueError, '场景'):
            sim.policy_chunk({'generation': 1})
        with self.assertRaisesRegex(ValueError, '12'):
            sim.policy_chunk({'generation': 2, 'joint_names': ['joint1']})
        payload = {'generation': 2, 'joint_names': ACTION_NAMES,
                   'actions': [[0.]*12], 'dt_s': .05}
        payload['actions'][0][7] = 2.
        with self.assertRaisesRegex(ValueError, r'\[-1,1\]'):
            sim.policy_chunk(payload)
        payload['actions'][0][7] = 0.
        payload['dt_s'] = .075
        with self.assertRaisesRegex(ValueError, '20Hz'):
            sim.policy_chunk(payload)
        self.assertIsNone(sim.motion)


@unittest.skipUnless(os.getenv('EMBODI_TEST_ROBOCASA365') == '1', 'Opt-in official native RoboCasa365 kitchen')
class RoboCasaNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        started = time.perf_counter()
        cls.sim = RoboCasaSimulation(task_id='NavigateKitchen')
        cls.initial_observation = cls.sim.observe()
        json.dumps(cls.initial_observation, allow_nan=False)
        cls.initial_checkpoint = cls.sim.checkpoint()
        cls.startup_s = time.perf_counter()-started
        print(f'Native RoboCasa365 startup: {cls.startup_s:.3f}s', flush=True)

    @classmethod
    def tearDownClass(cls): cls.sim.close()

    def setUp(self): self.sim.restore(self.initial_checkpoint)

    def test_a_construct_and_reset_observe_finite_three_native_cameras(self):
        sim = self.sim
        observation = sim.observe()
        json.dumps(observation, allow_nan=False)
        # Official Kitchen.reset settles ten 20Hz steps before returning.
        self.assertAlmostEqual(observation['time'], .5)
        self.assertEqual(sim.state()['initial_time_s'], observation['time'])
        self.assertEqual([c['name'] for c in observation['cameras']], ['front', 'side', 'wrist'])
        self.assertNotIn('evaluation', observation)
        self.assertTrue(observation['task_language'])
        self.assertEqual(len({c['image_base64'] for c in observation['cameras']}), 3)
        for camera in observation['cameras']:
            with Image.open(io.BytesIO(base64.b64decode(camera['image_base64']))) as image:
                self.assertEqual(image.size, (640, 426))
                self.assertGreater(np.asarray(image).std(), 5.)
        self.assertTrue(all(a['unit'] == 'normalized_command' and a['actual'] is None for a in sim.state()['actuators']))
        capture = os.getenv('EMBODI_NATIVE_CAPTURE_DIR')
        if capture:
            folder = Path(capture); folder.mkdir(parents=True, exist_ok=True)
            (folder/'viewer-initial.jpg').write_bytes(sim.render())
            for camera in observation['cameras']:
                (folder/(camera['name']+'-initial.jpg')).write_bytes(base64.b64decode(camera['image_base64']))

    def test_b_absolute_world_tcp_two_cm_and_hold_mobile_base(self):
        sim = self.sim
        initial = sim.tcp_pose()
        base = np.asarray(sim.robot_states()[0]['mobile_base']['position_world_m'])
        target = deepcopy(initial); target['position_m'][2] += .02
        pose = dict(status='move', **target, gripper_open=1., duration_s=1., explanation='Native controller acceptance.')
        sim.move_tcp({'generation': sim.generation, 'pose': pose})
        for _ in range(65):
            sim.step()
            if sim.task['status'] in ('complete', 'failed'): break
        self.assertEqual(sim.task['status'], 'complete', sim.task)
        error = np.linalg.norm(np.asarray(sim.tcp_pose()['position_m'])-target['position_m'])
        self.assertLess(error, .01)
        self.assertLess(np.linalg.norm(np.asarray(sim.robot_states()[0]['mobile_base']['position_world_m'])-base), .01)
        self.assertFalse(sim.evaluation['success'])

    def test_c_native_base_velocity_and_torso_actions(self):
        sim = self.sim
        before = sim.robot_states()[0]
        action = np.zeros(12); action[7] = .4; action[10] = .1; action[-1] = 1.
        result = sim.policy_chunk({'generation': sim.generation, 'joint_names': ACTION_NAMES,
                                  'actions': [action.tolist()]*20, 'dt_s': .05})
        self.assertEqual(result['unit'], 'normalized_command')
        for _ in range(20): sim.step()
        after = sim.robot_states()[0]
        self.assertGreater(np.linalg.norm(np.asarray(after['mobile_base']['position_world_m'])-before['mobile_base']['position_world_m']), .01)
        self.assertGreater(np.max(np.abs(np.asarray(after['torso']['position_m'])-before['torso']['position_m'])), .001)
        self.assertTrue(np.isfinite(sim.sim.data.qpos).all())
        self.assertEqual(sim.task['status'], 'complete')

    def test_d_checkpoint_restore_replay_and_cross_engine_rejection(self):
        sim = self.sim
        checkpoint = sim.checkpoint(); qpos = sim.sim.data.qpos.copy()
        rgba = sim.sim.model.geom_rgba.copy()
        sim.sim.model.geom_rgba[0] = [0.1, 0.2, 0.3, 0.4]
        sim.step(); sim.restore(checkpoint)
        np.testing.assert_allclose(sim.sim.data.qpos, qpos, atol=1e-12)
        np.testing.assert_allclose(sim.sim.model.geom_rgba, rgba, atol=1e-12)
        self.assertEqual(sim.time, checkpoint['time'])
        frame = sim.trajectory_frame(); sim.step(); sim.replay_frame(frame)
        self.assertEqual(sim.playback['time'], frame['time'])
        bad = deepcopy(checkpoint); bad['backend']['engine_id'] = 'mujoco'
        with self.assertRaisesRegex(ValueError, '后端'):
            sim.restore(bad)

    def test_d_native_gripper_command_survives_idle_and_stopped_settling(self):
        sim = self.sim
        action = np.zeros(12); action[6] = 1.; action[-1] = -1.
        sim.policy_chunk({'generation': sim.generation, 'joint_names': ACTION_NAMES,
                          'actions': [action.tolist()], 'dt_s': .05})
        sim.step(); native_command = sim.gripper.current_action.copy()
        for _ in range(10): sim.step()
        np.testing.assert_allclose(sim.gripper.current_action, native_command, atol=1e-12)
        for cancel in ('stop', 'settle'):
            pose = dict(status='move', **sim.tcp_pose(), gripper_open=.5,
                        duration_s=.5, explanation='Cancel during settling acceptance.')
            sim.move_tcp({'generation': sim.generation, 'pose': pose})
            # Exercise the valid settling state through public cancellation;
            # numerical controller convergence is covered independently above.
            sim.task['status'] = 'settling'; sim.motion = None
            sim.command({'action': cancel})
            sim.step()
            self.assertIsNone(sim.pose_goal)
            self.assertNotEqual(sim.task['status'], 'settling')
            self.assertTrue(np.isfinite(sim.sim.data.qpos).all())

    def test_e_seed_repeats_initial_scene_and_success_uses_native_state(self):
        sim = self.sim
        first = sim.sim.data.qpos.copy()
        initial_meta = sim.env.get_ep_meta()
        fingerprint = sim.model_fingerprint
        sim.step(); sim.reset()
        np.testing.assert_allclose(sim.sim.data.qpos, first, atol=1e-10)
        self.assertEqual(sim.env.get_ep_meta()['layout_id'], initial_meta['layout_id'])
        self.assertEqual(sim.env.get_ep_meta()['style_id'], initial_meta['style_id'])
        self.assertEqual(sim.model_fingerprint, fingerprint)
        self.assertEqual(sim.evaluation['success'], bool(sim.env._check_success()))
        self.assertFalse(sim.verify_task(sim.generation)['success'])
        values = sim.evaluation['values']
        self.assertGreater(values['base_distance_m'], .2)

    def test_f_official_drawer_pickplace_and_composite_scene_samples(self):
        sim = self.sim
        for task in ('OpenDrawer', 'PickPlaceCounterToCabinet', 'TurnOffSimmeredSauceHeat'):
            with self.subTest(task=task):
                scene = TASKS[task]['scene_id']
                sim.command({'action': 'scene', 'scene_id': scene, 'task_id': task, 'seed': 42})
                observation = sim.observe(); json.dumps(observation, allow_nan=False)
                self.assertEqual(len(observation['cameras']), 3)
                self.assertGreater(sim.render_rgb().std(), 5., 'Free viewer must show the scene, not start behind a wall')
                self.assertEqual(sim.evaluation['success'], bool(sim.env._check_success()))
                self.assertTrue(sim.evaluation['instruction'])
                checkpoint = sim.checkpoint()
                if task == 'OpenDrawer':
                    self.assertFalse(sim.evaluation['success'])
                    # Validate the native predicate using privileged physical
                    # state; this is evaluator calibration, not a policy win.
                    sim.env.drawer.set_door_state(min=1., max=1., env=sim.env)
                    sim.sim.forward(); sim._evaluate()
                    self.assertTrue(sim.verify_task(sim.generation)['success'])
                    sim.restore(checkpoint)
                    self.assertFalse(sim.evaluation['success'])
                elif task == 'TurnOffSimmeredSauceHeat':
                    i = sim.sim.model.geom_name2id('saucepan_liquid')
                    before = sim.sim.model.geom_rgba[i].copy(); tick = sim.env.task_timestep
                    for _ in range(5): sim.step()
                    self.assertGreater(sim.env.task_timestep, tick)
                    self.assertGreater(np.max(np.abs(sim.sim.model.geom_rgba[i]-before)), 1e-6)
                    sim.restore(checkpoint)
                    np.testing.assert_allclose(sim.sim.model.geom_rgba[i], before, atol=1e-12)
                    self.assertEqual(sim.env.task_timestep, tick)
                else:
                    for _ in range(5): sim.step()
                    self.assertTrue(np.isfinite(sim.sim.data.qpos).all())
                folder = os.getenv('EMBODI_NATIVE_CAPTURE_DIR')
                if folder:
                    target = Path(folder)/task; target.mkdir(parents=True, exist_ok=True)
                    (target/'viewer.jpg').write_bytes(sim.render())
                    for camera in sim.observe()['cameras']:
                        (target/(camera['name']+'.jpg')).write_bytes(base64.b64decode(camera['image_base64']))
        sim.command({'action': 'scene', 'scene_id': 'robocasa_atomic', 'task_id': 'NavigateKitchen', 'seed': 42})

    def test_z_failed_native_replacement_is_fatal_not_false_ready(self):
        from unittest.mock import patch
        sim = self.sim
        with patch.object(sim, '_create_env', side_effect=FileNotFoundError('fixture deliberately unavailable')):
            with self.assertRaisesRegex(RuntimeError, '未就绪'):
                sim.command({'action': 'scene', 'scene_id': 'robocasa_atomic', 'task_id': 'OpenDrawer'})
        self.assertIsNone(sim.env)


@unittest.skipUnless(os.getenv('EMBODI_TEST_ROBOCASA365_WORKBENCH') == '1', 'Opt-in native workbench episode')
class RoboCasaWorkbenchIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_visual_episode_rejects_false_done_and_retains_artifacts(self):
        # Reuse the existing real-worker acceptance contract. Only model
        # inference is substituted; physical actions/rendering are native.
        from tests.test_robot_backends_integration import RobotBackendsIntegrationTests
        await RobotBackendsIntegrationTests._episode(self, 'robocasa365', 'robocasa_atomic',
            'NavigateKitchen', False, 'robocasa_official_state', initial_time_s=.5)


if __name__ == '__main__': unittest.main(verbosity=2)
