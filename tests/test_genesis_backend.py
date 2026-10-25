"""Real optional Genesis regressions; run with .venv-genesis's Python."""
import base64
import copy
import importlib.util
import io
import time
import unittest

import numpy as np
from PIL import Image

HAS_GENESIS = importlib.util.find_spec('genesis') is not None


@unittest.skipUnless(HAS_GENESIS, 'Genesis is installed in the separate .venv-genesis environment')
class GenesisBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.backends.genesis_backend import GenesisSimulation
        started = time.monotonic()
        cls.sim = GenesisSimulation(task_id='reach_up')
        print(f'Genesis backend startup_s={time.monotonic()-started:.3f}', flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.command({'action': 'scene', 'scene_id': 'tabletop', 'task_id': 'reach_up', 'seed': 42})
        self.sim.command({'action': 'settle'})

    def test_native_gravity_and_joint_actuation(self):
        from server.backends.genesis_backend import array
        sim = self.sim
        sim.blocks['red'].set_pos((.4, -.22, .2))
        sim.command({'action': 'control', 'index': 0, 'value': .3})
        for _ in range(120): sim.step()
        self.assertAlmostEqual(float(array(sim.blocks['red'].get_pos())[2]), .024, delta=.004)
        self.assertAlmostEqual(float(array(sim.robot.get_dofs_position())[0]), .3, delta=.025)
        self.assertEqual(sim.state()['backend']['engine_id'], 'genesis')

    def test_three_actual_cameras_same_physics_state(self):
        before = self.sim.checkpoint()
        obs = self.sim.observe()
        self.assertEqual([c['name'] for c in obs['cameras']], ['front', 'side', 'wrist'])
        self.assertEqual(len({c['image_base64'] for c in obs['cameras']}), 3)
        for camera in obs['cameras']:
            self.assertEqual(camera['time'], before['time'])
            image = Image.open(io.BytesIO(base64.b64decode(camera['image_base64'])))
            self.assertEqual(image.size, (640, 426))
            self.assertGreater(np.asarray(image).std(), 10.)
        self.assertEqual(before, self.sim.checkpoint())
        self.assertNotIn('blocks', obs)
        self.assertNotIn('evaluation', obs)
        self.assertNotIn('checkpoint', obs)
        self.assertEqual(Image.open(io.BytesIO(self.sim.render())).size, (960, 640))

    def test_native_ik_and_environment_completion(self):
        sim = self.sim
        pose = sim.tcp_pose()
        target = np.array(pose['position_m'])+[0, 0, .02]
        sim.command({'action': 'move_tcp', 'generation': sim.generation, 'pose': {
            'status': 'move', 'frame': 'world', 'position_m': target.tolist(),
            'quaternion_wxyz': pose['quaternion_wxyz'], 'gripper_open': 1., 'duration_s': 1.,
            'explanation': 'native IK regression'}})
        for _ in range(450):
            sim.step()
            if sim.task['status'] not in ('running', 'settling'): break
        self.assertEqual(sim.task['status'], 'complete', sim.task)
        self.assertLess(np.linalg.norm(np.array(sim.tcp_pose()['position_m'])-target), .006)
        sim.command({'action': 'pause', 'value': True})
        result = sim.verify_task(sim.generation)
        self.assertTrue(result['success'], result)
        self.assertEqual(result['source'], 'genesis_privileged_state')

    def test_checkpoint_replay_and_identity_validation(self):
        sim = self.sim
        checkpoint = sim.checkpoint(); frame = sim.trajectory_frame()
        recorded_steps = sim.state()['steps']
        sim.command({'action': 'control', 'index': 0, 'value': .35})
        for _ in range(40): sim.step()
        sim.restore(checkpoint)
        self.assertEqual(sim.checkpoint()['state'], checkpoint['state'])
        self.assertEqual(sim.ctrl.tolist(), checkpoint['ctrl'])
        self.assertEqual(sim.time, checkpoint['time'])
        self.assertEqual(sim.state()['steps'], recorded_steps)
        sim.replay_frame(frame)
        self.assertIsNotNone(sim.state()['playback'])
        self.assertEqual(sim.state()['steps'], recorded_steps)
        branches = []
        for _ in range(2):
            sim.restore(checkpoint)
            sim.command({'action': 'control', 'index': 0, 'value': .2})
            for _ in range(50): sim.step()
            self.assertEqual(sim.state()['steps'], recorded_steps+50)
            branches.append(sim.trajectory_frame())
        # Native SimState resets constraint caches. We require physically stable
        # paired rollouts at 1e-5 tolerance, not a bit-for-bit warm-start claim.
        for field in ('qpos', 'qvel'):
            np.testing.assert_allclose(branches[0][field], branches[1][field], rtol=0, atol=1e-5)
        for key, value in [('engine_id', 'mujoco'), ('engine_version', '0.0'), ('state_schema', 'unknown')]:
            invalid = copy.deepcopy(checkpoint); invalid['backend'][key] = value
            before = sim.checkpoint()
            with self.assertRaises(ValueError): sim.restore(invalid)
            self.assertEqual(sim.checkpoint(), before)
        sim.reset()
        initial = sim.checkpoint()
        for _ in range(3): sim.step()
        sim.restore(initial)
        self.assertEqual((sim.state()['steps'], sim.time), (0, 0.))
        sim.step()
        self.assertEqual((sim.state()['steps'], sim.time), (1, sim.timestep))

    def test_bin_success_requires_full_containment_and_stable_release(self):
        from server.backends.genesis_backend import GenesisTaskEvaluator
        sim = self.sim
        sim.task_id = 'place_red'
        sim.blocks['red'].set_pos((.5, .25, .04))
        sim.evaluator = GenesisTaskEvaluator(sim)
        for _ in range(100): sim.step()
        self.assertTrue(sim.evaluator.success, sim.evaluator.snapshot())
        sim.blocks['red'].set_pos((.35, .25, .04))
        sim.evaluator.update(sim)
        self.assertFalse(sim.evaluator.success)
        self.assertTrue(sim.evaluator.ever_success)
        checks = {c['key']: c for c in sim.evaluator.snapshot()['checks']}
        self.assertFalse(checks['red_inside_bin_xy']['passed'])

    def test_model_done_cannot_override_failed_physical_condition(self):
        sim = self.sim
        for _ in range(40): sim.evaluator.update(sim)
        self.assertEqual(sim.evaluator.hold_s, 0.)
        result = sim.verify_task(sim.generation)
        self.assertFalse(result['success'])
        self.assertIn('末端距抬高目标的误差', result['failure_reasons'])
        self.assertAlmostEqual(result['verification_sim_time_s'], .3, places=6)

    def test_policy_limits_seeds_and_unsupported_scene(self):
        from server.backends.genesis_backend import JOINTS
        sim = self.sim
        result = sim.command({'action': 'policy_chunk', 'generation': sim.generation,
            'joint_names': JOINTS, 'actions': [[10.]*8], 'dt_s': .02})
        self.assertGreater(result['limited_targets'], 0)
        sim.command({'action': 'stop'})
        sim.command({'action': 'scene', 'scene_id': 'tabletop', 'seed': 123})
        first = sim.checkpoint()['state']
        sim.command({'action': 'reset'})
        self.assertEqual(first, sim.checkpoint()['state'])
        with self.assertRaisesRegex(ValueError, 'Genesis 当前支持'): sim.command({'action': 'scene', 'scene_id': 'drawer'})
        self.assertEqual(sim.scene_id, 'tabletop')


if __name__ == '__main__': unittest.main()
