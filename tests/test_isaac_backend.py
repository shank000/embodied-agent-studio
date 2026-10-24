"""Optional real Isaac Lab/Newton regressions, run in .venv-isaac."""
import base64
import copy
import importlib.util
import io
import time
import unittest

import numpy as np
from PIL import Image

HAS_ISAAC = importlib.util.find_spec('isaaclab') is not None


class IsaacCatalogTests(unittest.TestCase):
    def test_quaternion_convention_and_catalog(self):
        from server.backends.isaac_backend import MODELS, SCENES, TASKS, wxyz, xyzw
        self.assertEqual(wxyz([0, 0, 0, 1]), [1, 0, 0, 0])
        self.assertEqual(xyzw([1, 0, 0, 0]), [0, 0, 0, 1])
        self.assertEqual(list(MODELS), ['panda'])
        self.assertEqual(SCENES['isaac_tabletop']['tasks'], list(TASKS))

    @unittest.skipIf(importlib.util.find_spec('isaacsim') is not None, 'Full Isaac Sim is installed')
    def test_full_sim_never_falls_back_to_newton(self):
        from server.backends.isaac_backend import IsaacSimSimulation
        with self.assertRaisesRegex(ImportError, 'Isaac Sim'):
            IsaacSimSimulation()


@unittest.skipUnless(HAS_ISAAC, 'Isaac Lab is installed in the separate .venv-isaac environment')
class IsaacBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.backends.isaac_backend import IsaacSimulation
        started = time.monotonic()
        cls.sim = IsaacSimulation(task_id='isaac_reach_up')
        print(f'Isaac backend startup_s={time.monotonic()-started:.3f}', flush=True)
        # This is the user's first-load path: no scene command, settle, or
        # physics step is allowed to conceal uninitialized state/calibration.
        from server.backends.isaac_backend import array
        cls.initial_q = cls.sim._q().copy()
        cls.default_q = array(cls.sim.robot.data.default_joint_pos)[0].copy()
        cls.initial_observation = cls.sim.observe()
        cls.initial_state = cls.sim.state()

    @classmethod
    def tearDownClass(cls): cls.sim.close()

    def setUp(self):
        self.sim.command({'action': 'scene', 'scene_id': 'isaac_tabletop', 'task_id': 'isaac_reach_up', 'seed': 42})
        self.sim.command({'action': 'settle'})

    def test_constructed_state_and_observation_are_ready_without_settle(self):
        import json
        np.testing.assert_allclose(self.initial_q, self.default_q, rtol=0, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(self.initial_q))), .5)
        self.assertEqual(self.initial_state['time'], 0.)
        self.assertTrue(self.initial_state['paused'])
        json.dumps(self.initial_state, allow_nan=False)
        json.dumps(self.initial_observation, allow_nan=False)
        self.assertEqual(len(self.initial_observation['cameras']), 3)
        for camera in self.initial_observation['cameras']:
            self.assertTrue(np.isfinite(camera['intrinsic_cv']).all())
            self.assertTrue(np.isfinite(camera['quaternion_world_xyzw']).all())

    def test_actual_joint_drive_gravity_and_identity(self):
        from server.backends.isaac_backend import array
        sim = self.sim
        pose = sim.red.data.root_pose_w.torch.clone(); pose[0, 2] = .2
        sim.red.write_root_pose_to_sim_index(root_pose=pose)
        initial = sim._q()[sim.joint_ids[0]]
        sim.command({'action': 'control', 'index': 0, 'value': float(initial+.15)})
        for _ in range(150): sim.step()
        self.assertAlmostEqual(float(sim._q()[sim.joint_ids[0]]), float(initial+.15), delta=.035)
        self.assertAlmostEqual(float(array(sim.red.data.root_pose_w)[0, 2]), .024, delta=.008)
        self.assertEqual(sim.state()['backend']['engine_id'], 'isaac')
        self.assertEqual(sim.state()['compute_backend'], 'newton_mjwarp')
        sim.command({'action': 'control', 'index': 7, 'value': 0.})
        for _ in range(100): sim.step()
        self.assertLess(float(np.max(sim._q()[sim.finger_ids])), .006)
        sim.command({'action': 'control', 'index': 7, 'value': 255.})
        for _ in range(100): sim.step()
        np.testing.assert_allclose(sim._q()[sim.finger_ids], [.04, .04], rtol=0, atol=.006)

    def test_native_cameras_and_unprivileged_observation(self):
        before = self.sim.checkpoint(); obs = self.sim.observe()
        self.assertEqual([c['name'] for c in obs['cameras']], ['front', 'side', 'wrist'])
        self.assertEqual(len({c['image_base64'] for c in obs['cameras']}), 3)
        for camera in obs['cameras']:
            image = Image.open(io.BytesIO(base64.b64decode(camera['image_base64'])))
            self.assertEqual(image.size, (640, 426))
            self.assertGreater(np.asarray(image).std(), 8.)
            self.assertEqual(camera['time'], before['time'])
        self.assertEqual(before, self.sim.checkpoint())
        for key in ('evaluation', 'checkpoint', 'blocks', 'object_pose'): self.assertNotIn(key, obs)
        self.assertEqual(Image.open(io.BytesIO(self.sim.render())).size, (960, 640))

    def test_official_differential_ik_and_privileged_verification(self):
        sim = self.sim
        pose = sim.tcp_pose(); target = np.asarray(pose['position_m'])+[0, 0, .02]
        sim.move_tcp({'generation': sim.generation, 'pose': {
            'status': 'move', 'frame': 'world', 'position_m': target.tolist(),
            'quaternion_wxyz': pose['quaternion_wxyz'], 'gripper_open': 1., 'duration_s': 1.,
            'explanation': 'Isaac DifferentialIK regression'}})
        for _ in range(450):
            sim.step()
            if sim.task['status'] not in ('running', 'settling'): break
        self.assertEqual(sim.task['status'], 'complete', sim.task)
        self.assertLess(np.linalg.norm(np.asarray(sim.tcp_pose()['position_m'])-target), .006)
        sim.paused = True
        result = sim.verify_task(sim.generation)
        self.assertTrue(result['success'], result)
        self.assertEqual(result['source'], 'isaaclab_newton_state')

    def test_world_jacobian_matches_native_finite_difference(self):
        from server.backends.isaac_backend import array
        sim = self.sim; joint = sim.joint_ids[1]
        original = sim.robot.data.joint_pos.torch.clone()
        indices = [i+sim.robot.num_base_dofs for i in sim.joint_ids]
        analytic = array(sim.robot.data.body_link_jacobian_w)[0, sim.jacobian_index, :3, indices[1]]
        positions = []
        try:
            for delta in (.001, -.001):
                changed = original.clone(); changed[0, joint] += delta
                sim.robot.write_joint_position_to_sim_index(position=changed)
                positions.append(array(sim.robot.data.body_pose_w)[0, sim.hand_index, :3].copy())
            measured = (positions[0]-positions[1])/.002
            np.testing.assert_allclose(analytic, measured, rtol=.02, atol=.003)
        finally:
            sim.robot.write_joint_position_to_sim_index(position=original)

    def test_actor_completion_cannot_override_environment(self):
        sim = self.sim
        result = sim.verify_task(sim.generation)
        self.assertFalse(result['success'])
        self.assertAlmostEqual(result['verification_sim_time_s'], .3, places=6)
        sim.command({'action': 'scene', 'scene_id': 'isaac_tabletop', 'task_id': 'isaac_lift_red', 'seed': 42})
        for _ in range(100): sim.step()
        sim.paused = True
        self.assertFalse(sim.verify_task(sim.generation)['success'])

    def test_policy_chunk_limits_and_native_execution(self):
        from server.backends.isaac_backend import JOINTS
        sim = self.sim; before = sim._q().copy()
        result = sim.policy_chunk({'generation': sim.generation, 'joint_names': JOINTS,
                                   'actions': [[10.]*8], 'dt_s': .1})
        self.assertGreater(result['limited_targets'], 0)
        for _ in range(30): sim.step()
        self.assertEqual(sim.task['status'], 'complete')
        self.assertGreater(float(np.max(np.abs(sim._q()-before))), .01)
        sim.command({'action': 'stop'})
        with self.assertRaisesRegex(ValueError, '失效'):
            sim.policy_chunk({'generation': sim.generation-1, 'joint_names': JOINTS, 'actions': [[0.]*8], 'dt_s': .1})

    def test_checkpoint_replay_foreign_identity_and_seed(self):
        sim = self.sim; checkpoint = sim.checkpoint(); frame = sim.trajectory_frame()
        sim.command({'action': 'control', 'index': 0, 'value': .2})
        for _ in range(25): sim.step()
        sim.restore(checkpoint)
        np.testing.assert_allclose(sim._q(), frame['qpos'], rtol=0, atol=1e-5)
        self.assertEqual(sim.ctrl.tolist(), checkpoint['ctrl'])
        sim.replay_frame(frame); self.assertEqual(sim.playback['time'], frame['time'])
        for key, value in [('engine_id', 'mujoco'), ('engine_version', 'invalid'), ('state_schema', 'unknown')]:
            invalid = copy.deepcopy(checkpoint); invalid['backend'][key] = value
            before = sim.checkpoint()
            with self.assertRaises(ValueError): sim.restore(invalid)
            self.assertEqual(before, sim.checkpoint())
        sim.command({'action': 'scene', 'seed': 123}); first = sim.checkpoint()['state']
        sim.command({'action': 'reset'}); self.assertEqual(first, sim.checkpoint()['state'])


if __name__ == '__main__': unittest.main()
