"""Whole-robot controller contracts; no native SDK or paid inference is invoked."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from server.actions import NativeAction
from server.agent import AgentConfig, AgentController, AgentStart, action_options, observation_text
from server.composition import NativePlannerOutput, skill_candidates, CompositionController, CompositionStart
from server.policy import PolicyOutput, PolicyStart, policy_chunk_command, validate_policy_timestep, PolicyConfig
from tests.test_agent_round_limits import DeterministicWorker


CAMERAS = ['head', 'left_wrist', 'right_wrist']
SPACE = {'type': 'native', 'names': ['base_x', 'arm_0', 'gripper'],
         'low': [-1., -1., -1.], 'high': [1., 1., 1.], 'dt_s': .05}


class NativeWorker(DeterministicWorker):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.snapshot.update(backend={'engine_id': 'omnigibson', 'action_mode': 'native',
            'evaluator_source': 'omnigibson_bddl_state', 'camera_names': CAMERAS},
            model='r1pro', scene_id='fixture', timestep=.05, action_space=SPACE)
        self.snapshot['evaluation'] = {'source': 'omnigibson_bddl_state', 'success': False,
                                       'terminated': False, 'truncated': False}

    async def execute(self, command):
        if command['action'] == 'observe':
            observation = await super().execute(command)
            observation.update(backend=deepcopy(self.snapshot['backend']), robot='r1pro',
                               native_action_spec=SPACE, action_space=SPACE)
            for camera, name in zip(observation['cameras'], CAMERAS): camera['name'] = name
            return observation
        if command['action'] == 'native_action':
            self.commands.append('native_action')
            NativeAction.model_validate(command['native_action'])
            self.moves += 1
            self.snapshot['time'] += .1
            self.snapshot['evaluation'].update(success=not self.fail, terminated=True)
            return {'controller': 'fixture_native'}
        return await super().execute(command)


class NativeAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_logging_and_native_schema_success_and_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory(prefix='native-agent-') as folder:
                worker = NativeWorker(fail=failed)
                controller = AgentController(worker)
                controller.configure(AgentConfig(provider='custom', model='fixture',
                    base_url='http://127.0.0.1:9/agent', max_rounds=2))
                async def infer(config, prompt, cameras, **options):
                    self.assertIs(options['response_type'], NativeAction)
                    observation = json.loads(prompt)['observation']
                    self.assertNotIn('evaluation', observation)
                    self.assertEqual(observation['native_action_spec'], SPACE)
                    self.assertEqual([camera['name'] for camera in cameras], CAMERAS)
                    return NativeAction(status='move', action=[.1, 0., 0.], steps=2, explanation='公开动作'), None
                with patch('server.runlog.RUNS', Path(folder)), patch('server.agent.api_action', infer):
                    await controller.start(AgentStart(task='Fixture native task.', max_sim_time_s=1200))
                    await asyncio.wait_for(controller.task, 5)
                self.assertEqual(controller.state['status'], 'failed' if failed else 'verified', controller.state)
                self.assertEqual(worker.moves, 1)
                self.assertNotIn('move_tcp', worker.commands)
                run = Path(folder)/controller.state['run_id']
                summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['metrics']['success'], not failed)
                manifest = [json.loads(line) for line in (run/'observations/manifest.jsonl').read_text().splitlines()]
                self.assertGreaterEqual(len(manifest), 1)
                self.assertEqual([camera['name'] for camera in manifest[0]['cameras']], CAMERAS)

    async def test_composition_native_command_uses_whole_robot_action(self):
        worker = NativeWorker()
        controller = CompositionController(worker)
        controller._accepting = True
        controller._assert_current = lambda observation: None
        result = await controller._execute({'action_type': 'native', 'native_action':
            {'status': 'move', 'action': [.1, 0., 0.], 'steps': 1, 'explanation': '原生动作'}}, {'generation': 1})
        self.assertEqual(result['status'], 'complete')
        self.assertIn('native_action', worker.commands)


class NativeSchemaTests(unittest.TestCase):
    def test_native_policy_names_and_command_contract(self):
        output = PolicyOutput(protocol='embodied-policy-v1', action_type='native',
            actions=[[.1, 0., 0.]], action_names=SPACE['names'], dt_s=.05)
        command = policy_chunk_command(output, 4)
        self.assertEqual(command['action_names'], SPACE['names'])
        self.assertEqual(command['action_space'], 'native')
        self.assertNotIn('joint_names', command)
        candidates = CompositionController._policy_candidates(output)
        self.assertEqual(candidates[0]['action_type'], 'native')
        for values in ({'action_names': ['a', 'a'], 'actions': [[0., 0.]]},
                       {'action_names': ['a'], 'actions': [[0., 0.]]},
                       {'action_names': ['a'], 'actions': [[float('nan')]]}):
            with self.assertRaises(ValidationError):
                PolicyOutput(protocol='embodied-policy-v1', action_type='native', **values)

    def test_native_planner_preserves_vector_and_shortens_only_ticks(self):
        plan = NativePlannerOutput(status='plan', action=[.1, 0., 0.], steps=4,
            explanation='移动底盘', subgoal='接近柜体', visual_summary='柜体在前方', skill='native', expires_after_steps=1)
        candidates = skill_candidates({'backend': {'action_mode': 'native'}}, plan)
        self.assertEqual([item['native_action']['steps'] for item in candidates], [4, 2])
        self.assertTrue(all(item['native_action']['action'] == plan.action for item in candidates))

    def test_metadata_prompt_does_not_inline_depth_array(self):
        observation = {'backend': {'action_mode': 'native'}, 'cameras': [{'name': 'head',
            'image_base64': 'jpeg', 'depth': {'encoding': 'npy-float32', 'unit': 'm', 'data_base64': 'binary-depth'}}]}
        result = observation_text('task', observation, [], None)
        self.assertNotIn('binary-depth', result)
        self.assertNotIn('jpeg', result)
        self.assertIn('npy-float32', result)
        self.assertIs(action_options(observation)['response_type'], NativeAction)

    def test_long_tasks_are_finite_and_native_cadence_is_validated(self):
        for cls, kwargs in ((AgentStart, {'task': 'task'}), (PolicyStart, {'task': 'task'}),
                            (CompositionStart, {'task': 'task', 'method_id': 'fixture'})):
            self.assertEqual(cls(**kwargs, max_sim_time_s=3600).max_sim_time_s, 3600)
            with self.assertRaises(ValidationError): cls(**kwargs, max_sim_time_s=float('inf'))
        obs = {'backend': {'action_mode': 'native'}, 'native_action_spec': SPACE}
        validate_policy_timestep(PolicyConfig(dt_s=.1), obs)
        with self.assertRaises(ValueError): validate_policy_timestep(PolicyConfig(dt_s=.07), obs)
