"""Multi-arm regression checks against upstream Panda physics, IK and rendering."""
import base64
import io
import unittest

import mujoco
import numpy as np
from PIL import Image

from server.scenes import SCENES
from server.simulation import Simulation


class MultiArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation('dual_panda', render=False, scene_id='cooperative')

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        if (self.sim.model_id, self.sim.scene_id, self.sim.task_id) != ('dual_panda', 'cooperative', 'sort_red_blue'):
            self.sim.load('dual_panda', 'cooperative', task_id='sort_red_blue')
        self.sim.reset(log=False)
        self.sim.select_robot('arm1')

    def step_until_complete(self):
        for _ in range(3000):
            self.sim.step()
            if self.sim.task['status'] in ('complete', 'failed'):
                break
        self.assertEqual(self.sim.task['status'], 'complete', self.sim.task)

    def pose_action(self, robot_id):
        robot = next(r for r in self.sim.robot_states() if r['id'] == robot_id)
        pose = robot['tcp']
        return {'status': 'move', 'frame': 'world', 'robot_id': robot_id,
                'position_m': [*pose['position_m'][:2], pose['position_m'][2]+.02],
                'quaternion_wxyz': pose['quaternion_wxyz'], 'gripper_open': 1.,
                'duration_s': .7, 'explanation': '抬高指定臂，其他臂保持控制目标。'}

    def test_all_scenes_start_and_settle_without_robot_fixture_or_cross_arm_overlap(self):
        sim = self.sim
        for platform in ('dual_panda', 'triple_panda'):
            for scene in SCENES:
                for seed in (17, 42):
                    with self.subTest(platform=platform, scene=scene, seed=seed):
                        sim.load(platform, scene, seed)
                        for settled in (False, True):
                            if settled:
                                sim.command({'action': 'settle'})
                            self.assertTrue(np.isfinite(sim.data.qpos).all())
                            overlaps = []
                            for contact in sim.data.contact:
                                if contact.dist >= -.001: continue
                                bodies = [int(sim.model.geom_bodyid[g]) for g in contact.geom]
                                names = [sim.model.body(body).name or '' for body in bodies]
                                arms = [name.split('_')[0] if name.startswith('arm') else None for name in names]
                                if arms[0] != arms[1] and any(arms):
                                    overlaps.append((names, contact.dist))
                            self.assertFalse(overlaps, overlaps)

    def test_two_and_three_namespaced_arms_have_independent_dynamics(self):
        for platform, count in [('dual_panda', 2), ('triple_panda', 3)]:
            with self.subTest(platform=platform):
                self.sim.load(platform, 'cooperative')
                sim = self.sim
                self.assertEqual(sim.model.nu, 8*count)
                self.assertEqual(sim.model.ntendon, count)
                self.assertEqual(sim.model.neq, count)
                self.assertEqual(sim.model.nmesh, 67, 'Panda meshes are shared, not duplicated')
                self.assertEqual(len(sim.state()['robots']), count)
                for number in range(1, count+1):
                    sim.select_robot(f'arm{number}')
                    before = sim.data.ctrl.copy()
                    qpos_before = sim.data.qpos.copy()
                    sim.instruction('关节1转到10度')
                    sim.instruction('闭合夹爪')
                    np.testing.assert_array_equal(sim.data.qpos, qpos_before)
                    own = sim.robots[sim.active_robot]['indices']
                    other = np.setdiff1d(np.arange(sim.model.nu), own)
                    np.testing.assert_array_equal(sim.data.ctrl[other], before[other])
                    for _ in range(800): sim.step()
                    self.assertAlmostEqual(sim.data.qpos[sim.actuators[0]['qpos_index']], np.deg2rad(10), delta=.02)
                    self.assertAlmostEqual(sim.data.joint(f'arm{number}_finger_joint1').qpos[0], 0, delta=.002)
                    self.assertAlmostEqual(sim.data.joint(f'arm{number}_finger_joint2').qpos[0], 0, delta=.002)
                sim.reset()
                np.testing.assert_array_equal(sim.data.ctrl, sim.home_ctrl)
                for number in range(1, count+1):
                    self.assertAlmostEqual(sim.data.joint(f'arm{number}_finger_joint1').qpos[0], .04)
                    self.assertAlmostEqual(sim.data.joint(f'arm{number}_finger_joint2').qpos[0], .04)

    def test_second_arm_ik_is_routed_atomically_without_moving_first_arm(self):
        sim = self.sim
        for _ in range(500): sim.step()
        before = sim.data.ctrl.copy()
        qpos = sim.data.qpos.copy()
        generation = sim.generation
        action = self.pose_action('arm2')
        result = sim.move_tcp({'generation': generation, 'pose': action})
        self.assertEqual(result['robot_id'], 'arm2')
        self.assertEqual(sim.active_robot, 'arm2')
        self.assertEqual(sim.generation, generation)
        np.testing.assert_array_equal(sim.data.qpos, qpos, 'IK must not teleport either robot')
        np.testing.assert_array_equal(sim.motion['poses'][0]['ctrl'][:8], before[:8])
        self.step_until_complete()
        np.testing.assert_array_equal(sim.data.ctrl[:8], before[:8])
        np.testing.assert_allclose(sim.data.site_xpos[sim.site_id], action['position_m'], atol=.006)
        np.testing.assert_allclose(sim.data.qpos[:7], qpos[:7], atol=.02)

    def test_failed_route_and_external_selection_preserve_state_and_stale_guard(self):
        sim = self.sim
        before = sim.checkpoint()
        bad = self.pose_action('arm2'); bad['position_m'][0] += 1.
        with self.assertRaisesRegex(ValueError, '0.120'):
            sim.move_tcp({'generation': sim.generation, 'pose': bad})
        self.assertEqual(sim.checkpoint(), before)
        for robot_id in ('missing', 'arm3'):
            with self.assertRaises(ValueError): sim.select_robot(robot_id)
        sim.command({'action': 'select_robot', 'robot_id': 'arm2'})
        self.assertTrue(sim.paused)
        self.assertGreater(sim.generation, 0)
        self.assertEqual(sim.checkpoint()['state'], before['state'])
        with self.assertRaisesRegex(ValueError, '失效'):
            sim.move_tcp({'generation': sim.generation-1, 'pose': self.pose_action('arm2')})
        with self.assertRaisesRegex(ValueError, '当前控制臂'):
            sim.set_control(0, .1)

    def test_policy_chunks_and_home_only_control_selected_arm(self):
        sim = self.sim
        sim.set_control(0, .1)
        original = sim.data.ctrl.copy()
        robot = sim.robots['arm2']
        target = original[robot['indices']].copy(); target[0] = .18; target[-1] = 230.
        result = sim.policy_chunk({'generation': sim.generation, 'robot_id': 'arm2',
            'actions': [target.tolist()], 'joint_names': [a['name'] for a in robot['actuators']], 'dt_s': .2})
        self.assertEqual(len(result['executed_targets'][0]), 8)
        self.step_until_complete()
        np.testing.assert_array_equal(sim.data.ctrl[:8], original[:8])
        np.testing.assert_allclose(sim.data.ctrl[8:], target)
        sim.home()
        self.step_until_complete()
        np.testing.assert_array_equal(sim.data.ctrl[:8], original[:8])
        np.testing.assert_array_equal(sim.data.ctrl[8:], sim.home_ctrl[8:])

    def test_checkpoint_and_trajectory_restore_all_arms_and_active_wrist(self):
        sim = self.sim
        sim.select_robot('arm2'); sim.set_view('wrist')
        sim.instruction('关节1转到10度')
        for _ in range(500): sim.step()
        saved = sim.checkpoint(); frame = sim.trajectory_frame()
        for _ in range(20): sim.step()
        expected = sim.data.qpos.copy()
        sim.select_robot('arm1'); sim.reset()
        sim.restore(saved)
        self.assertEqual(sim.active_robot, 'arm2')
        self.assertEqual(sim.cam.fixedcamid, sim.model.camera('arm2_wrist_cam').id)
        for _ in range(20): sim.step()
        np.testing.assert_array_equal(sim.data.qpos, expected)
        sim.select_robot('arm1')
        sim.replay_frame(frame)
        self.assertEqual(sim.active_robot, 'arm2')
        np.testing.assert_array_equal(sim.data.qpos, frame['qpos'])
        np.testing.assert_array_equal(sim.data.ctrl, frame['ctrl'])
        self.assertTrue(sim.paused)

    def test_evaluator_sees_contacts_from_inactive_arm_and_both_task_objects(self):
        sim = self.sim
        body = sim.model.body('arm2_left_finger').id
        self.assertIn(body, sim.evaluator.robot_bodies)
        geom = int(sim.model.body_geomadr[body])
        red = sim.model.joint('block_red_joint').id
        adr = int(sim.model.jnt_qposadr[red])
        sim.data.qpos[adr:adr+3] = sim.data.geom_xpos[geom]
        mujoco.mj_forward(sim.model, sim.data)
        self.assertTrue(sim.evaluator.touches_robot(sim, 'block_red'))
        sim.reset()
        for name, xy in [('red', [.5, .25]), ('blue', [.17, .26])]:
            joint = sim.model.joint(f'block_{name}_joint').id
            adr = sim.model.jnt_qposadr[joint]
            sim.data.qpos[adr:adr+3] = [*xy, .04]
            if name == 'red':
                mujoco.mj_forward(sim.model, sim.data)
                for _ in range(160): sim.evaluator.update(sim)
                self.assertFalse(sim.evaluator.success, 'One placed object is not the two-object task')
        for _ in range(400): sim.step()
        self.assertTrue(sim.evaluator.success, sim.evaluator.snapshot())

    def test_three_synchronized_images_use_selected_wrist_and_all_proprioception(self):
        sim = self.sim
        sim.renderer = mujoco.Renderer(sim.model, height=640, width=960)
        try:
            first = sim.observe()
            physics = sim.checkpoint()['state']
            sim.select_robot('arm2')
            second = sim.observe()
            self.assertEqual(sim.checkpoint()['state'], physics)
            self.assertEqual(len(second['cameras']), 3)
            self.assertEqual(len(second['robots']), 2)
            self.assertEqual(second['active_robot'], 'arm2')
            self.assertEqual(second['tcp'], second['robots'][1]['tcp'])
            for camera in second['cameras']:
                self.assertEqual(camera['time'], second['time'])
                image = Image.open(io.BytesIO(base64.b64decode(camera['image_base64'])))
                self.assertEqual(image.size, (640, 426))
                self.assertGreater(np.asarray(image).std(), 10)
            for index in (0, 1):
                before, after = first['cameras'][index], second['cameras'][index]
                self.assertEqual(before['position_world_m'], after['position_world_m'])
                self.assertEqual(before['rotation_camera_to_world'], after['rotation_camera_to_world'])
                # Driver rounding before JPEG can differ by 1–2 levels in a few pixels.
                pixels = [np.asarray(Image.open(io.BytesIO(base64.b64decode(c['image_base64']))), dtype=float)
                          for c in (before, after)]
                self.assertLess(np.abs(pixels[0]-pixels[1]).mean(), .01)
            self.assertNotEqual(first['cameras'][2]['position_world_m'], second['cameras'][2]['position_world_m'])
            camera_id = sim.model.camera('arm2_wrist_cam').id
            np.testing.assert_allclose(second['cameras'][2]['position_world_m'], sim.data.cam_xpos[camera_id])
        finally:
            sim.renderer.close(); sim.renderer = None


if __name__ == '__main__':
    unittest.main()
