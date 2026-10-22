"""Pure multi-arm planning/routing contracts; no simulator or model requests."""
import asyncio
from copy import deepcopy
from threading import Lock
import unittest

import numpy as np

from server.composition import CompositionController, PlannerOutput, skill_candidates
from server.policy import PolicyOutput


def observation():
    first = {'frame': 'world', 'position_m': [.4, -.3, .4], 'quaternion_wxyz': [1., 0., 0., 0.]}
    second = {'frame': 'world', 'position_m': [.5, .3, .5], 'quaternion_wxyz': [0., 0., 0., 1.]}
    return {'generation': 7, 'time': 2., 'episode_id': 'routing-test', 'observation_id': 1,
        'active_robot': 'arm1', 'tcp': first, 'limits': {'translation_per_round_m': .12},
        'robots': [{'id': 'arm1', 'tcp': first, 'gripper': {'target_open_fraction': 1.}},
                   {'id': 'arm2', 'tcp': second, 'gripper': {'target_open_fraction': .25},
                    'limits': {'translation_per_round_m': .008}}]}


def plan(**updates):
    values = {'status': 'plan', 'subgoal': 'Move selected arm.', 'visual_summary': 'Fixture state.',
        'skill': 'translate', 'direction_world': [0., 0., 1.], 'gripper_open': .25,
        'expires_after_steps': 3, 'explanation': '规划测试。'}
    return PlannerOutput.model_validate({**values, **updates})


class MultiArmPlanningTests(unittest.TestCase):
    def test_inactive_arm_translation_uses_its_pose_limits_and_explicit_id(self):
        obs = observation(); before = deepcopy(obs)
        candidates = skill_candidates(obs, plan(robot_id='arm2'))
        self.assertEqual(len(candidates), 2)
        for candidate, dz in zip(candidates, (.004, .002)):
            pose = candidate['pose']
            self.assertEqual(pose['robot_id'], 'arm2')
            np.testing.assert_allclose(pose['position_m'], [.5, .3, .5+dz])
            self.assertEqual(pose['quaternion_wxyz'], obs['robots'][1]['tcp']['quaternion_wxyz'])
            self.assertEqual(pose['gripper_open'], .25)
        self.assertEqual(obs, before)
        with self.assertRaisesRegex(ValueError, '0.0080'):
            skill_candidates(obs, plan(robot_id='arm2'), {'translation_step_m': .01})

    def test_hold_and_gripper_target_other_arm_without_using_active_tcp(self):
        obs = observation()
        for skill in ('hold', 'gripper'):
            with self.subTest(skill=skill):
                candidate, = skill_candidates(obs, plan(skill=skill, robot_id='arm2', gripper_open=.1))
                self.assertEqual(candidate['pose']['robot_id'], 'arm2')
                self.assertEqual(candidate['pose']['position_m'], [.5, .3, .5])
                self.assertEqual(candidate['pose']['gripper_open'], .1)

    def test_default_pins_current_active_arm_and_legacy_observation_still_works(self):
        obs = observation(); obs['active_robot'] = 'arm2'
        candidates = skill_candidates(obs, plan())
        self.assertEqual(candidates[0]['pose']['robot_id'], 'arm2')
        self.assertEqual(candidates[0]['pose']['position_m'][:2], [.5, .3])
        legacy = {key: value for key, value in observation().items() if key not in ('active_robot', 'robots')}
        self.assertIsNone(skill_candidates(legacy, plan())[0]['pose']['robot_id'])
        self.assertIsNone(plan().robot_id)

    def test_unknown_arm_is_rejected_before_any_candidate_is_generated(self):
        obs = observation()
        with self.assertRaisesRegex(ValueError, 'arm3'):
            skill_candidates(obs, plan(robot_id='arm3'))
        with self.assertRaisesRegex(ValueError, '策略输出'):
            plan(skill='policy', robot_id='arm2')

    def test_policy_candidates_keep_routing_for_pose_and_action_chunks(self):
        obs = observation()
        pose = skill_candidates(obs, plan(robot_id='arm2'))[0]['pose']
        pose['robot_id'] = None
        output = PolicyOutput.model_validate({'protocol': 'embodied-policy-v1', 'action_type': 'tcp_pose',
                                             'robot_id': 'arm2', 'pose': pose})
        self.assertEqual(CompositionController._policy_candidates(output)[0]['pose']['robot_id'], 'arm2')
        pose['robot_id'] = 'arm1'
        with self.assertRaises(ValueError):
            PolicyOutput.model_validate({'protocol': 'embodied-policy-v1', 'action_type': 'tcp_pose',
                                         'robot_id': 'arm2', 'pose': pose})
        output = PolicyOutput.model_validate({'protocol': 'embodied-policy-v1', 'action_type': 'actuator_targets',
            'robot_id': 'arm2', 'joint_names': ['arm2_joint1'], 'actions': [[.1], [.2]], 'dt_s': .05})
        for candidate in CompositionController._policy_candidates(output):
            self.assertEqual(candidate['robot_id'], 'arm2')
            self.assertEqual(candidate['joint_names'], ['arm2_joint1'])

    def test_executor_sends_selected_arm_and_accepts_same_generation_completion(self):
        class Worker:
            def __init__(self, obs):
                self.lock = Lock(); self.commands = []
                self.snapshot = {'ready': True, 'generation': obs['generation'], 'time': obs['time'],
                                 'task': {'status': 'complete'}, 'tcp_pose': obs['robots'][1]['tcp']}

            async def execute(self, command):
                self.commands.append(command)
                return {'robot_id': 'arm2'}

        async def check():
            obs = observation()
            pose_candidate = skill_candidates(obs, plan(robot_id='arm2'))[0]
            chunk_candidate = {'action_type': 'actuator_targets', 'robot_id': 'arm2',
                               'joint_names': ['arm2_joint1'], 'actions': [[.1]], 'dt_s': .05}
            for candidate in (pose_candidate, chunk_candidate):
                worker = Worker(obs)
                controller = CompositionController(worker)
                controller._accepting = True; controller.observation_id = 1
                controller.state.update(active=True, episode_id='routing-test')
                result = await controller._execute(candidate, obs)
                self.assertEqual(result['status'], 'complete')
                command = worker.commands[0]
                self.assertEqual(command['generation'], 7)
                self.assertEqual(command.get('robot_id') or command['pose']['robot_id'], 'arm2')
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
