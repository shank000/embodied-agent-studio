"""Real optional SAPIEN/ManiSkill regressions; run in .venv-maniskill."""
import base64
import copy
import importlib.util
import io
import time
import unittest

import numpy as np
from PIL import Image

HAS_MANISKILL = importlib.util.find_spec('mani_skill') is not None


@unittest.skipUnless(HAS_MANISKILL, 'ManiSkill is installed in the separate .venv-maniskill environment')
class ManiSkillBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.backends.maniskill_backend import ManiSkillSimulation
        started = time.monotonic()
        cls.sim = ManiSkillSimulation()
        print(f'ManiSkill backend startup_s={time.monotonic()-started:.3f}', flush=True)

    @classmethod
    def tearDownClass(cls): cls.sim.close()

    def setUp(self):
        self.sim.command({'action': 'scene', 'scene_id': 'pick_cube', 'task_id': 'pick_cube', 'seed': 42})
        self.sim.command({'action': 'settle'})

    def test_native_joint_actuation_and_gravity(self):
        from mani_skill.utils.structs import Pose
        from server.backends.maniskill_backend import array
        sim = self.sim
        sim.env.cube.set_pose(Pose.create_from_pq([.1, -.1, .2]))
        initial = array(sim.robot.get_qpos())[0, 0]
        sim.command({'action': 'control', 'index': 0, 'value': float(initial+.2)})
        for _ in range(30): sim.step()
        self.assertAlmostEqual(float(array(sim.robot.get_qpos())[0, 0]), float(initial+.2), delta=.025)
        self.assertAlmostEqual(float(array(sim.env.cube.pose.p)[0, 2]), .02, delta=.005)
        self.assertEqual(sim.state()['backend']['engine_id'], 'maniskill')
        self.assertEqual(sim.state()['compute_backend'], 'physx_cpu')

    def test_three_native_cameras_without_privileged_observation(self):
        before = self.sim.checkpoint()
        obs = self.sim.observe()
        self.assertEqual([c['name'] for c in obs['cameras']], ['front', 'side', 'wrist'])
        self.assertEqual(len({c['image_base64'] for c in obs['cameras']}), 3)
        for camera in obs['cameras']:
            self.assertEqual(camera['time'], before['time'])
            image = Image.open(io.BytesIO(base64.b64decode(camera['image_base64'])))
            self.assertEqual(image.size, (640, 426))
            self.assertGreater(np.asarray(image).std(), 10.)
            self.assertIn('intrinsic_cv', camera['calibration'])
        self.assertEqual(before, self.sim.checkpoint())
        for key in ('evaluation', 'checkpoint', 'goal_pos', 'obj_pose', 'is_grasped'): self.assertNotIn(key, obs)
        self.assertEqual(Image.open(io.BytesIO(self.sim.render())).size, (960, 640))

    def test_native_ik_move_and_generation_guard(self):
        sim = self.sim
        pose = sim.tcp_pose(); target = np.array(pose['position_m'])+[0, 0, .02]
        action = {'status': 'move', 'frame': 'world', 'position_m': target.tolist(),
                  'quaternion_wxyz': pose['quaternion_wxyz'], 'gripper_open': 1.,
                  'duration_s': 1., 'explanation': 'SAPIEN native IK regression'}
        with self.assertRaisesRegex(ValueError, '失效'): sim.move_tcp({'generation': sim.generation-1, 'pose': action})
        sim.move_tcp({'generation': sim.generation, 'pose': action})
        for _ in range(100):
            sim.step()
            if sim.task['status'] not in ('running', 'settling'): break
        self.assertEqual(sim.task['status'], 'complete', sim.task)
        self.assertLess(np.linalg.norm(np.array(sim.tcp_pose()['position_m'])-target), .006)

    def test_official_success_and_independent_verification(self):
        from mani_skill.utils.structs import Pose
        from server.backends.maniskill_backend import ManiSkillEvaluator, array
        sim = self.sim
        result = sim.verify_task(sim.generation)
        self.assertFalse(result['success'])
        self.assertEqual(result['source'], 'maniskill_official_evaluate')
        self.assertAlmostEqual(result['verification_sim_time_s'], .3, places=6)
        # Put goal on the settled object in the privileged test setup. The
        # evaluator must still require native evaluate() and elapsed physics.
        sim.env.goal_site.set_pose(Pose.create_from_pq(array(sim.env.cube.pose.p)))
        sim.evaluator = ManiSkillEvaluator(sim)
        self.assertFalse(sim.evaluator.success)
        self.assertTrue(bool(array(sim.env.evaluate()['success'])[0]))
        self.assertTrue(sim.verify_task(sim.generation)['success'])
        sim.env.goal_site.set_pose(Pose.create_from_pq([.5, .5, .3]))
        sim.evaluator.update(sim)
        self.assertFalse(sim.evaluator.success)
        self.assertTrue(sim.evaluator.ever_success)

    def test_checkpoint_and_foreign_identity_rejection(self):
        from server.backends.maniskill_backend import array
        sim = self.sim; checkpoint = sim.checkpoint(); frame = sim.trajectory_frame()
        sim.command({'action': 'control', 'index': 0, 'value': .25})
        for _ in range(15): sim.step()
        sim.restore(checkpoint)
        np.testing.assert_allclose(array(sim.robot.get_qpos())[0], frame['qpos'], rtol=0, atol=1e-6)
        self.assertEqual(sim.ctrl.tolist(), checkpoint['ctrl'])
        sim.replay_frame(frame)
        self.assertEqual(sim.playback['time'], checkpoint['time'])
        for key, value in [('engine_id', 'mujoco'), ('engine_version', 'invalid'), ('state_schema', 'unknown')]:
            invalid = copy.deepcopy(checkpoint); invalid['backend'][key] = value
            before = sim.checkpoint()
            with self.assertRaises(ValueError): sim.restore(invalid)
            self.assertEqual(before, sim.checkpoint())

    def test_official_push_task_seed_reset_and_policy_limits(self):
        from server.backends.maniskill_backend import JOINTS, array
        sim = self.sim
        sim.command({'action': 'scene', 'scene_id': 'push_cube', 'task_id': 'push_cube', 'seed': 123})
        first = sim.checkpoint()['state']
        sim.command({'action': 'reset'})
        self.assertEqual(first, sim.checkpoint()['state'])
        self.assertEqual(sim.evaluator.snapshot()['values']['official_env_id'], 'PushCube-v1')
        self.assertEqual(sim.evaluator.snapshot()['currently_satisfied'], bool(array(sim.env.evaluate()['success'])[0]))
        result = sim.policy_chunk({'generation': sim.generation, 'joint_names': JOINTS, 'actions': [[10.]*8], 'dt_s': .05})
        self.assertGreater(result['limited_targets'], 0)
        for _ in range(3): sim.step()
        self.assertEqual(sim.task['status'], 'complete')
        sim.command({'action': 'stop'})
        with self.assertRaises(ValueError): sim.command({'action': 'scene', 'scene_id': 'drawer'})
        self.assertEqual(sim.scene_id, 'push_cube')


if __name__ == '__main__': unittest.main()
