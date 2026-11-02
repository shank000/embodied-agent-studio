"""Adapter contracts with a small synthetic session; not native physics proof.

Set EMBODIWORKS_ROBODOJO_NATIVE_TEST=1 only in the configured Linux SDK runtime
to additionally exercise a real load, three camera capture and control step.
"""
from copy import deepcopy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from server.backends.robodojo_backend import RoboDojoSimulation, CAMERAS, CAPABILITIES, TASKS
from server.backends.robodojo_native import NativeSession, HostPolicyBoundary, seed_parts


class SessionFixture:
    """A deterministic protocol fixture, deliberately no imitation physics."""
    def __init__(self, task_id, seed):
        self.closed = False
        self.reset(task_id, seed)

    def reset(self, task_id, seed):
        self.timestep, self.elapsed = .04, 0.
        eval_seed, layout_index = seed_parts(seed)
        self.initial_state = {'task_id': task_id, 'eval_seed': eval_seed, 'layout_index': layout_index,
                              'layout_sha256': 'synthetic-test-layout'}
        self.robots = [{'id': f'arm{i+1}', 'side': side, 'joint_names': [f'joint{j+1}' for j in range(6)],
            'qpos': [0.] * 6, 'qvel': [0.] * 6, 'limits': [[-2., 2.]] * 6,
            'pose': [.3, .3 if i == 0 else -.3, .5, 1., 0., 0., 0.], 'gripper_actual': 0.,
            'gripper_open': 1., 'gripper_target': 1.} for i, side in enumerate(('left', 'right'))]
        self.actions = []
        self.terminal = self.success = self.unstable = False
        self.fail_ik = False
        self.home_action = self.hold_action()

    def robot_states(self): return deepcopy(self.robots)

    def hold_action(self):
        return {key: value for r in self.robots for key, value in
                ((r['side'] + '_arm_joint_state', r['qpos'].copy()), (r['side'] + '_ee_joint_state', [r['gripper_target']]))}

    def evaluate(self):
        return {'terminal': self.terminal, 'success': self.terminal and self.success and not self.unstable,
                'score': None, 'unstable': self.unstable, 'action_count': len(self.actions), 'action_limit': 100,
                'instruction': 'official-layout fixture instruction'}

    def advance(self, action):
        if self.fail_ik: raise ValueError('fixture IK failure')
        self.actions.append(deepcopy(action))
        for r in self.robots:
            if r['side'] + '_ee_pose' in action: r['pose'] = action[r['side'] + '_ee_pose'][:]
            else: r['qpos'] = action[r['side'] + '_arm_joint_state'][:]
            r['gripper_target'] = r['gripper_open'] = action[r['side'] + '_ee_joint_state'][0]
        self.elapsed += self.timestep

    def observe(self):
        return {'vision': {name: {'color': np.full((24, 32, 3), i * 80, dtype=np.uint8)}
                           for i, name in enumerate(CAMERAS.values())},
                'instruction': 'official-layout fixture instruction',
                'privileged_object_pose': 'must not leak', 'state': {'native_internals': 'must not leak'}}

    def close(self): self.closed = True


class RoboDojoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch('server.backends.robodojo_backend.NativeSession', SessionFixture)
        self.patch.start()
        self.sim = RoboDojoSimulation()

    def tearDown(self):
        self.sim.close(); self.patch.stop()

    def pose(self, robot='arm2'):
        return {'generation': self.sim.generation, 'pose': {'status': 'move', 'frame': 'world',
                'position_m': [.35, -.3 if robot == 'arm2' else .3, .5], 'quaternion_wxyz': [1., 0., 0., 0.],
                'gripper_open': .5, 'duration_s': .5, 'explanation': 'fixture motion', 'robot_id': robot}}

    def test_catalog_and_public_protocol(self):
        self.assertEqual(len(TASKS), 54)
        state = self.sim.state()
        self.assertTrue(state['ready']); self.assertGreater(state['timestep'], 0)
        self.assertEqual(len(state['actuators']), 14)
        self.assertEqual(state['observation_cameras'], list(CAMERAS))
        self.assertEqual([v['id'] for v in state['view_presets']], list(CAMERAS))
        self.assertEqual(state['backend']['engine_id'], 'robodojo')
        self.assertNotIn('camera', CAPABILITIES); self.assertNotIn('restore', CAPABILITIES)
        observation = self.sim.observe()
        self.assertEqual(observation['task_instruction'], 'official-layout fixture instruction')
        self.assertEqual([c['name'] for c in observation['cameras']], list(CAMERAS))
        self.assertEqual([c['width'] for c in observation['cameras']], [32] * 3)
        self.assertNotIn('evaluation', observation)
        self.assertNotIn('privileged_object_pose', json.dumps(observation))
        self.assertNotIn('native_internals', json.dumps(observation))
        self.assertEqual(len({c['image_base64'] for c in observation['cameras']}), 3)
        self.sim.command({'action': 'view', 'view': 'right_wrist'})
        self.assertTrue(np.all(self.sim.render_rgb() == 160))
        with self.assertRaisesRegex(ValueError, '未实现'): self.sim.command({'action': 'camera', 'dx': 3})

    def test_selected_right_arm_interpolates_with_other_arm_joint_hold(self):
        before = self.sim.session.robot_states()[0]
        result = self.sim.move_tcp(self.pose())
        self.assertEqual(result['native_control_steps'], 13)
        first = self.sim.pending[0]
        self.assertGreater(first['right_ee_pose'][0], .3)
        self.assertLess(first['right_ee_pose'][0], .35)
        self.assertIn('left_arm_joint_state', first)
        self.assertNotIn('left_ee_pose', first)
        while not self.sim.paused: self.sim.step()
        self.assertEqual(self.sim.session.robot_states()[0], before)
        self.assertEqual(self.sim.task['status'], 'complete')
        self.assertAlmostEqual(self.sim.time, .52)
        self.assertFalse(self.sim.verify_task(self.sim.generation)['success'])

    def test_policy_names_units_cadence_and_joint_limits(self):
        self.sim.command({'action': 'select_robot', 'robot_id': 'arm2'})
        names = [a['name'] for a in self.sim.observe()['actuators']]
        payload = {'generation': self.sim.generation, 'robot_id': 'arm2', 'joint_names': names,
                   'actions': [[.1] * 6 + [.2]], 'dt_s': .08}
        result = self.sim.policy_chunk(payload)
        self.assertEqual(result['frames'], 1)
        while not self.sim.paused: self.sim.step()
        self.assertEqual(len(self.sim.session.actions), 2)
        self.assertEqual(self.sim.session.actions[0]['right_arm_joint_state'], [.1] * 6)
        self.assertEqual(self.sim.session.actions[0]['left_arm_joint_state'], [0.] * 6)
        with self.assertRaisesRegex(ValueError, '正整数倍'): self.sim.policy_chunk({**payload, 'dt_s': .05})
        with self.assertRaisesRegex(ValueError, '真实关节范围'): self.sim.policy_chunk({**payload, 'actions': [[3.] * 7]})
        with self.assertRaisesRegex(ValueError, '完全一致'): self.sim.policy_chunk({**payload, 'joint_names': ['wrong'] * 7})

    def test_terminal_success_is_environment_owned_and_verification_does_not_step(self):
        self.sim.session.success = True  # native initialization uses True before terminal
        self.assertFalse(self.sim.verify_task(self.sim.generation)['success'])
        self.sim.session.terminal = True
        self.assertTrue(self.sim.verify_task(self.sim.generation)['success'])
        self.assertEqual(self.sim.time, 0)
        self.assertEqual(self.sim.session.actions, [])
        with self.assertRaisesRegex(ValueError, '已终止'): self.sim.move_tcp(self.pose())

    def test_terminal_failure_stops_motion_and_is_visible(self):
        self.sim.move_tcp(self.pose())
        self.sim.session.terminal = True
        self.sim.step()
        self.assertTrue(self.sim.paused)
        self.assertEqual(self.sim.task['status'], 'failed')
        self.assertTrue(self.sim.state()['evaluation']['terminated'])
        self.assertFalse(self.sim.state()['evaluation']['success'])
        self.assertFalse(self.sim.pending)

    def test_ik_failure_is_reported_without_mutation(self):
        self.sim.session.fail_ik = True
        self.sim.move_tcp(self.pose()); self.sim.step()
        self.assertEqual(self.sim.time, 0)
        self.assertEqual(self.sim.task['status'], 'failed')
        self.assertIn('fixture IK failure', self.sim.task['title'])
        self.assertTrue(self.sim.paused); self.assertFalse(self.sim.pending)

    def test_snapshot_is_explicitly_nonrestorable_and_seed_mapping_is_bijective(self):
        self.sim.command({'action': 'scene', 'scene_id': 'push_T', 'seed': 42})
        checkpoint = self.sim.checkpoint()
        self.assertFalse(checkpoint['restorable'])
        self.assertEqual(checkpoint['initial_state']['eval_seed'], 0)
        self.assertEqual(checkpoint['initial_state']['layout_index'], 14)
        self.assertEqual(checkpoint['state_id'], self.sim.checkpoint()['state_id'])
        self.assertEqual(len({seed_parts(i) for i in range(150)}), 150)
        with self.assertRaises(ValueError): self.sim.command({'action': 'scene', 'scene_id': 'push_T', 'seed': 75})
        with self.assertRaisesRegex(ValueError, '不支持'): self.sim.command({'action': 'restore', 'checkpoint': checkpoint})
        with self.assertRaisesRegex(ValueError, '不支持'): self.sim.replay_frame({})
        self.assertFalse(self.sim.trajectory_frame()['replay_supported'])

    def test_missing_native_camera_does_not_substitute_placeholder(self):
        with patch.object(self.sim.session, 'observe', return_value={'vision': {}}):
            with self.assertRaisesRegex(RuntimeError, '未返回 RGB'): self.sim.render_rgb()

    def test_failed_reset_closes_native_session_and_invalidates_worker(self):
        session = self.sim.session
        with patch.object(session, 'reset', side_effect=ValueError('missing native layout')):
            with self.assertRaisesRegex(RuntimeError, 'missing native layout'):
                self.sim.command({'action': 'reset'})
        self.assertTrue(session.closed); self.assertIsNone(self.sim.session)


class NativeBoundaryTests(unittest.TestCase):
    def test_policy_boundary_only_accepts_reset(self):
        boundary = HostPolicyBoundary()
        self.assertIsNone(boundary.call(func_name='reset'))
        with self.assertRaises(RuntimeError): boundary.call(func_name='inference')

    def test_official_ik_failure_is_atomic_and_success_not_resolved_twice(self):
        session = NativeSession.__new__(NativeSession)
        robot = SimpleNamespace(arm_name='left_arm')
        calls = []
        session._robots, session.elapsed, session.timestep = [robot], 0., .04
        def take(action): calls.append(deepcopy(action)); session.env.take_action_cnt[0] += 1
        manager = SimpleNamespace(solve_ik=lambda **kwargs: {'status': 'Fail'})
        session.env = SimpleNamespace(robot_manager=manager, take_action_cnt=[0], take_action=take)
        action = {'left_ee_pose': [.3, .1, .4, 1., 0., 0., 0.], 'left_ee_joint_state': [.2]}
        with self.assertRaisesRegex(ValueError, 'IK 无解'): session.advance(action)
        self.assertEqual(calls, []); self.assertEqual(session.elapsed, 0)
        with patch.object(manager, 'solve_ik', return_value={'status': 'Success', 'joint_value': [0.] * 6}) as ik:
            session.advance(action)
            self.assertEqual(ik.call_count, 1)
        self.assertNotIn('left_ee_pose', calls[0])
        self.assertEqual(calls[0]['left_arm_joint_state'], [0.] * 6)
        self.assertEqual(calls[0]['left_ee_joint_state'], [.2])
        self.assertAlmostEqual(session.elapsed, .04)

    def test_native_initial_success_flag_is_not_task_success_and_unstable_is_excluded(self):
        session = NativeSession.__new__(NativeSession)
        session.env = SimpleNamespace(is_episode_end=lambda: None, end_flag=[False], success=[True],
                                      unstable_envs=set(), take_action_cnt=[0], step_lim=100,
                                      obs_manager=SimpleNamespace(instruction=['fixture']))
        self.assertFalse(session.evaluate()['success'])
        session.env.end_flag[0] = True
        self.assertTrue(session.evaluate()['success'])
        session.env.unstable_envs.add(0)
        self.assertFalse(session.evaluate()['success'])

    def test_error_after_native_physics_starts_is_fatal_not_recoverable_ik(self):
        session = NativeSession.__new__(NativeSession)
        session._robots, session.elapsed, session.timestep = [], 0., .04
        def take(action):
            session.env.take_action_cnt[0] += 1
            raise ValueError('terminal camera failure after physics')
        session.env = SimpleNamespace(robot_manager=SimpleNamespace(), take_action_cnt=[0], take_action=take)
        with self.assertRaisesRegex(RuntimeError, '物理状态可能已变化') as error:
            session.advance({})
        self.assertIsInstance(error.exception.__cause__, ValueError)
        self.assertEqual(session.env.take_action_cnt, [1])


@unittest.skipUnless(os.environ.get('EMBODIWORKS_ROBODOJO_NATIVE_TEST') == '1', 'Requires real configured Linux Isaac Sim GPU runtime/assets')
class RoboDojoNativeAcceptance(unittest.TestCase):
    def test_real_default_task_cameras_step_reset_and_close(self):
        sim = RoboDojoSimulation()
        try:
            self.assertEqual(len(sim.observe()['cameras']), 3)
            self.assertTrue(sim.render().startswith(b'\xff\xd8'))
            sim.command({'action': 'step'})
            self.assertGreater(sim.time, 0)
            self.assertIn('success', sim.verify_task(sim.generation))
            sim.command({'action': 'reset'})
            self.assertEqual(sim.time, 0.)
            self.assertEqual(len(sim.observe()['cameras']), 3)
        finally: sim.close()


if __name__ == '__main__': unittest.main()
