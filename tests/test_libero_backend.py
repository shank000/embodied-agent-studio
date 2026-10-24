"""Optional native LIBERO acceptance checks (no demonstration datasets needed)."""
import base64
import copy
import io
import os
import unittest

import numpy as np
from PIL import Image

from server.backends.libero_catalog import TASK_MAP, catalog


class LiberoCatalogTests(unittest.TestCase):
    def test_official_suites_and_task_ids(self):
        scenes, tasks = catalog()
        self.assertEqual(sum(map(len, TASK_MAP.values())), 130)
        self.assertEqual(len(tasks), 130)
        self.assertEqual(set(scenes), {'libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'})
        self.assertEqual(tasks['libero_goal:0']['instruction'], 'open the middle drawer of the cabinet')
        self.assertEqual(scenes['libero_90']['tasks'][-1], 'libero_90:89')


NATIVE = os.environ.get('EMBODI_TEST_LIBERO') == '1'


@unittest.skipUnless(NATIVE, 'Run with EMBODI_TEST_LIBERO=1 in the isolated .venv-libero environment')
class LiberoNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.backends.libero_backend import LiberoSimulation
        cls.sim = LiberoSimulation(scene_id='libero_goal', task_id='libero_goal:0')

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.command({'action': 'reset'})

    def test_three_native_cameras_without_privileged_observations(self):
        sim = self.sim
        before = sim.checkpoint()
        obs = sim.observe()
        self.assertEqual([v['name'] for v in obs['cameras']], ['front', 'side', 'wrist'])
        self.assertEqual(len({v['image_base64'] for v in obs['cameras']}), 3)
        self.assertEqual([v['index'] for v in obs['actuators']], list(range(8)))
        self.assertEqual(obs['actuators'][-1]['unit'], 'command')
        self.assertLess(obs['actuators'][-1]['actual'], .05)
        for camera in obs['cameras']:
            image = Image.open(io.BytesIO(base64.b64decode(camera['image_base64'])))
            self.assertEqual(image.size, (640, 426))
            self.assertGreater(np.asarray(image).std(), 10.)
            self.assertEqual(camera['time'], sim.time)
        self.assertNotIn('evaluation', obs)
        self.assertNotIn('state', obs)
        self.assertEqual(before, sim.checkpoint())
        self.assertEqual(Image.open(io.BytesIO(sim.render())).size, (960, 640))

    def test_native_absolute_pose_controller_moves_tcp(self):
        sim = self.sim
        before = sim.tcp_pose()
        target = (np.array(before['position_m'])+[0, 0, .02]).tolist()
        sim.command({'action': 'move_tcp', 'generation': sim.generation, 'pose': {
            'status': 'move', 'frame': 'world', 'position_m': target,
            'quaternion_wxyz': before['quaternion_wxyz'], 'gripper_open': 1., 'duration_s': 1.,
            'explanation': 'Native OSC regression'}})
        for _ in range(65):
            sim.step()
            if sim.task['status'] not in ('running', 'settling'): break
        self.assertEqual(sim.mode, 'OSC_POSE')
        self.assertFalse(sim.robot.controller.use_delta)
        self.assertEqual(sim.task['status'], 'complete', sim.task)
        self.assertLess(np.linalg.norm(np.asarray(sim.tcp_pose()['position_m'])-target), .008)

    def test_bddl_success_uses_actual_drawer_joint(self):
        sim = self.sim
        sim.command({'action': 'pause', 'value': True})
        self.assertFalse(sim.verify_task(sim.generation)['success'])
        goal = sim.native.parsed_problem['goal_state'][0]
        self.assertEqual(goal[0].lower(), 'open')
        state = sim.native.object_states_dict[goal[1]]
        # Construct a positive environment state through the official state
        # object's fixture joint. This tests the judge, not policy ability.
        obj = sim.native.get_object(state.parent_name)
        opening = float(np.mean(obj.object_properties['articulation']['default_open_ranges']))
        for joint in sim.native.object_sites_dict[goal[1]].joints:
            sim.sim.data.set_joint_qpos(joint, opening)
        sim.sim.forward()
        self.assertTrue(sim.env.check_success(), goal)
        result = sim.verify_task(sim.generation)
        self.assertTrue(result['success'], result)
        self.assertEqual(result['source'], 'libero_bddl_state')
        sim.command({'action': 'reset'})
        self.assertFalse(sim.verify_task(sim.generation)['success'])

    def test_checkpoint_replay_and_backend_boundary(self):
        sim = self.sim
        checkpoint, frame = sim.checkpoint(), sim.trajectory_frame()
        for _ in range(3): sim.step()
        sim.restore(checkpoint)
        np.testing.assert_array_equal(sim.checkpoint()['state'], checkpoint['state'])
        self.assertEqual(sim.ctrl.tolist(), checkpoint['ctrl'])
        sim.replay_frame(frame)
        self.assertIsNotNone(sim.playback)
        for key, value in [('engine_id', 'mujoco'), ('engine_version', 'other'), ('state_schema', 'other')]:
            invalid = copy.deepcopy(checkpoint); invalid['backend'][key] = value
            before = sim.checkpoint()
            with self.assertRaises(ValueError): sim.restore(invalid)
            self.assertEqual(before, sim.checkpoint())

    def test_policy_native_joint_controller_and_seeded_initial_state(self):
        from server.backends.libero_backend import JOINTS
        sim = self.sim
        before = np.asarray(sim.sim.data.qpos[sim.robot._ref_joint_pos_indexes]).copy()
        target = np.r_[before+[.04, 0, 0, 0, 0, 0, 0], 255.]
        result = sim.command({'action': 'policy_chunk', 'generation': sim.generation,
            'joint_names': JOINTS, 'actions': [target.tolist()], 'dt_s': .5})
        self.assertEqual(result['frames'], 1)
        for _ in range(60):
            sim.step()
            if sim.task['status'] not in ('running', 'settling'): break
        self.assertEqual(sim.mode, 'JOINT_POSITION')
        self.assertEqual(sim.task['status'], 'complete', sim.task)
        self.assertGreater(float(sim.sim.data.qpos[sim.robot._ref_joint_pos_indexes[0]]-before[0]), .01)
        sim.command({'action': 'scene', 'scene_id': 'libero_goal', 'task_id': 'libero_goal:0', 'seed': 7})
        first = sim.checkpoint()['state']
        sim.command({'action': 'reset'})
        np.testing.assert_array_equal(first, sim.checkpoint()['state'])
        with self.assertRaises(ValueError): sim.command({'action': 'scene', 'scene_id': 'drawer'})
