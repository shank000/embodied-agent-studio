"""SO-101 controls and collision diagnostics against real MuJoCo physics."""
import math
import unittest

import mujoco
import numpy as np

from server.control import rotation_error, tcp_pose
from server.scenes import SCENES
from server.simulation import Simulation


class SO101PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation('so101', render=False)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def advance(self, seconds):
        for _ in range(math.ceil(seconds / self.sim.model.opt.timestep)):
            self.sim.step()

    def assert_robot_clear_of_scene(self):
        sim = self.sim
        penetrations = []
        for contact in sim.data.contact:
            bodies = [int(sim.model.geom_bodyid[g]) for g in contact.geom]
            robot = [body in sim.evaluator.robot_bodies for body in bodies]
            if robot[0] == robot[1] or contact.dist >= -.001 * sim.config['scale']:
                continue
            names = [sim.model.geom(int(g)).name or sim.model.body(body).name
                     for g, body in zip(contact.geom, bodies)]
            penetrations.append((names, float(contact.dist)))
        self.assertFalse(penetrations, penetrations)

    def assert_arm_at_targets(self):
        for actuator in self.sim.actuators[:self.sim.config['arm_dof']]:
            actual = self.sim.data.qpos[actuator['qpos_index']]
            target = self.sim.data.ctrl[actuator['index']]
            self.assertAlmostEqual(actual, target, delta=.025, msg=actuator['name'])

    def test_controls_work_in_each_scene_without_initial_fixture_overlap(self):
        sim = self.sim
        for scene in SCENES:
            for seed in (42, 50):
                with self.subTest(scene=scene, seed=seed):
                    sim.load('so101', scene, seed)
                    self.assert_robot_clear_of_scene()
                    sim.command({'action': 'settle'})
                    self.assert_robot_clear_of_scene()

                    sim.command({'action': 'instruction', 'text': '关节1转到30度'})
                    self.advance(2.)
                    self.assertAlmostEqual(sim.data.qpos[sim.actuators[0]['qpos_index']],
                                           math.pi / 6, delta=.02)
                    sim.command({'action': 'home'})
                    self.advance(5.6)
                    self.assertEqual(sim.task['status'], 'complete', sim.task)
                    self.assert_arm_at_targets()

                    gripper = sim.actuators[-1]
                    for instruction, target in (('闭合夹爪', sim.config['gripper_closed']),
                                                ('张开夹爪', sim.config['gripper_open'])):
                        sim.command({'action': 'instruction', 'text': instruction})
                        self.advance(2.)
                        self.assertAlmostEqual(sim.data.qpos[gripper['qpos_index']],
                                               target, delta=.01)

                    sim.command({'action': 'demo'})
                    self.advance(15.2)
                    self.assertEqual(sim.task['status'], 'complete', sim.task)
                    self.assert_arm_at_targets()
                    self.assertAlmostEqual(sim.data.qpos[gripper['qpos_index']],
                                           sim.config['gripper_open'], delta=.01)
                    self.assert_robot_clear_of_scene()

    def test_drawer_tcp_lift_reaches_existing_position_and_orientation_tolerances(self):
        sim = self.sim
        for seed in (42, 50):
            with self.subTest(seed=seed):
                sim.load('so101', 'drawer', seed)
                sim.command({'action': 'settle'})
                initial = tcp_pose(sim.model, sim.data, sim.site_id)
                target = np.asarray(initial['position_m']) + [0., 0., .02]
                before = sim.data.qpos.copy()
                sim.command({'action': 'move_tcp', 'generation': sim.generation,
                             'pose': {'status': 'move', 'frame': 'world',
                                      'position_m': target.tolist(),
                                      'quaternion_wxyz': initial['quaternion_wxyz'],
                                      'gripper_open': 1., 'duration_s': 1.2,
                                      'explanation': '保持姿态，将末端抬高 2 cm。'}})
                np.testing.assert_array_equal(sim.data.qpos, before)
                self.advance(4.4)
                self.assertEqual(sim.task['status'], 'complete', sim.task)
                actual = tcp_pose(sim.model, sim.data, sim.site_id)
                self.assertLess(np.linalg.norm(np.asarray(actual['position_m']) - target),
                                .006 * sim.config['scale'])
                self.assertLess(np.linalg.norm(rotation_error(initial['quaternion_wxyz'],
                                                              sim.data.site_xmat[sim.site_id])), .05)
                self.assert_robot_clear_of_scene()

    def test_real_collision_failure_reports_joint_deviation_and_contact_objects(self):
        sim = self.sim
        sim.load('so101', 'drawer', 50)
        # Reproduce the previous colliding initial pose through the normal reset path.
        sim.home_ctrl[1] = -.5
        sim.reset()
        self.assertTrue(any(contact.dist < -.01 and
                            'cabinet_top' in [sim.model.geom(int(g)).name for g in contact.geom]
                            for contact in sim.data.contact))
        sim.command({'action': 'home'})
        self.advance(5.6)
        self.assertEqual(sim.task['status'], 'failed', sim.task)
        errors = sim.task['joint_errors_rad']
        self.assertGreater(max(abs(value) for value in errors.values()), .025)
        for actuator in sim.actuators[:sim.config['arm_dof']]:
            self.assertAlmostEqual(errors[actuator['name']],
                                   sim.data.qpos[actuator['qpos_index']] -
                                   sim.data.ctrl[actuator['index']], delta=.001)
        self.assertTrue(any(contact['robot_body'] == 'gripper' and
                            contact['object'] == 'cabinet_top'
                            for contact in sim.task['contacts']), sim.task)
        self.assertIn('偏差', sim.task['message'])
        self.assertIn('cabinet_top', sim.task['message'])
        self.assertIn('gripper', sim.task['message'])


if __name__ == '__main__':
    unittest.main()
