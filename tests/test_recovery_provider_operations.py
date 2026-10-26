"""Pre-execution recovery wiring using local HTTP fixtures, with no native physics."""
import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from server.agent import AgentConfig
from server.composition import CompositionController, CompositionStart, PlannerOutput
from server.decision import DecisionResult
from server.policy import PolicyConfig, PolicyController, PolicyStart, predict_policy
from server.runlog import CURRENT_LOG, RunLog
from tests.test_agent_round_limits import DeterministicWorker


FEEDBACK = 'Fixture recovery: return the complete structured response.'


class ProviderFixture(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        retry = len(self.server.requests) > 1
        if self.path == '/policy':
            response = {'protocol': 'embodied-policy-v1', 'action_type': 'tcp_pose'}
            if retry:
                response['pose'] = {'status': 'move', **body['observation']['tcp'], 'gripper_open': 1.,
                                    'duration_s': .5, 'explanation': 'Local policy fixture.'}
        else:
            output = {}
            if retry:
                if body['response_schema_name'] == 'robot_subgoal':
                    output = {'status': 'plan', 'subgoal': 'Hold.', 'visual_summary': 'Local fixture.',
                              'skill': 'hold', 'direction_world': [0., 0., 0.], 'gripper_open': 1.,
                              'expires_after_steps': 1, 'explanation': 'Local planner fixture.'}
                else:
                    output = {'status': 'move', **json.loads(body['prompt'])['observation']['tcp'],
                              'gripper_open': 1., 'duration_s': .5, 'explanation': 'Local agent fixture.'}
            response = {'protocol': 'embodied-agent-v1', 'output': output, 'usage': {'output_tokens': 5}}
        content = json.dumps(response).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(content)


class RecoveryProviderOperationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), ProviderFixture)
        cls.server.requests = []
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(2)

    def setUp(self):
        self.server.requests.clear()
        self.tmp = tempfile.TemporaryDirectory(prefix='recovery-provider-')
        self.directory = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        log_patch = patch('server.runlog.RUNS', self.directory)
        log_patch.start(); self.addCleanup(log_patch.stop)
        self.operations = []

    @property
    def base_url(self):
        return f'http://127.0.0.1:{self.server.server_port}'

    async def retry_schema_once(self, owner, phase, operation, observation, config):
        """Explicit coordinator boundary fixture; production coordinator has separate tests."""
        self.operations.append((phase, id(observation), owner.state.get('round', owner.state.get('step'))))
        try:
            return await operation('', config)
        except ValidationError:
            overrides = {'timeout_s': 31}
            if isinstance(config, AgentConfig): overrides['max_output_tokens'] = 1024
            return await operation(FEEDBACK, config.model_copy(update=overrides))

    def events(self, log):
        return [json.loads(line) for line in log.path.read_text(encoding='utf-8').splitlines()]

    async def test_policy_retries_parse_error_in_same_round_and_executes_only_once(self):
        worker = DeterministicWorker()
        controller = PolicyController(worker)
        controller.configure(PolicyConfig(kind='vla', model='local-fixture', endpoint=self.base_url+'/policy', max_chunks=1, timeout_s=30))
        with patch('server.recovery.invoke_with_recovery', self.retry_schema_once):
            await controller.start(PolicyStart(kind='vla', task='Local recovery fixture.', one_chunk=True))
            await asyncio.wait_for(controller.task, 5)
        self.assertEqual(controller.state['status'], 'round_complete', controller.state)
        self.assertEqual(controller.state['round'], 1)
        self.assertEqual(worker.moves, 1)
        self.assertEqual(worker.commands.count('observe'), 2)  # Retry shares input; one additional post-execution capture.
        self.assertEqual(worker.commands.count('record_start'), 1)
        self.assertEqual(worker.commands.count('record_stop'), 1)
        first, second = self.server.requests
        self.assertNotIn('recovery_feedback', first)
        self.assertEqual(second['recovery_feedback'], FEEDBACK)
        self.assertEqual({key:value for key,value in second.items() if key != 'recovery_feedback'}, first)
        self.assertEqual((first['seed'], first['reset']), (42, True))
        events = self.events(controller.run_log)
        self.assertEqual(sum(e['event'] == 'observation' for e in events), 1)
        self.assertEqual(sum(e['event'] == 'inference' for e in events), 2)
        self.assertEqual(sum(e['event'] == 'execution' for e in events), 1)
        self.assertEqual(sum(e['event'] == 'finish' for e in events), 1)

    async def test_without_coordinator_policy_keeps_original_error_behavior(self):
        worker = DeterministicWorker()
        controller = PolicyController(worker)
        controller.configure(PolicyConfig(kind='vla', model='local-fixture', endpoint=self.base_url+'/policy', max_chunks=1))
        await controller.start(PolicyStart(kind='vla', task='Local no-recovery fixture.', one_chunk=True))
        await asyncio.wait_for(controller.task, 5)
        self.assertEqual(controller.state['status'], 'error')
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(worker.moves, 0)

    async def test_manual_paused_policy_configuration_reaches_the_retried_request(self):
        worker = DeterministicWorker()
        controller = PolicyController(worker)
        controller.configure(PolicyConfig(kind='vla', model='old-fixture', endpoint=self.base_url+'/policy', max_chunks=1))
        corrected = PolicyConfig(kind='vla', model='corrected-fixture', endpoint=self.base_url+'/policy',
                                 horizon=3, dt_s=.1, timeout_s=31, max_chunks=1)
        async def resume(owner, phase, operation, observation, config):
            try:
                return await operation('', config)
            except ValidationError:
                with self.assertRaises(ValueError): owner.configure(corrected)
                owner.state['status'] = 'repair_paused'
                owner.configure(corrected)
                return await operation(FEEDBACK, owner.configs['vla'])
        with patch('server.recovery.invoke_with_recovery', resume):
            await controller.start(PolicyStart(kind='vla', task='Manual recovery fixture.', one_chunk=True))
            await asyncio.wait_for(controller.task, 5)
        self.assertEqual(controller.state['status'], 'round_complete', controller.state)
        first, second = self.server.requests
        self.assertEqual(first['model'], 'old-fixture')
        self.assertEqual(second['model'], 'corrected-fixture')
        self.assertEqual((second['horizon'], second['dt_s']), (3, .1))
        self.assertEqual((second['reset'], second['seed'], second['observation']),
                         (first['reset'], first['seed'], first['observation']))
        self.assertEqual(worker.moves, 1)

    async def composition(self):
        worker = DeterministicWorker()
        controller = CompositionController(worker)
        observation = await worker.execute({'action': 'observe'})
        observation.update(episode_id='local-recovery', observation_id=1)
        controller._accepting = True; controller.observation_id = 1; controller.observation = observation
        controller.initial_tcp = deepcopy(observation['tcp']); controller.initial_state = worker.snapshot.copy()
        controller.policy_calls = 0
        controller.state.update(active=True, step=1, episode_id='local-recovery', task='Keep this exact task.',
                                history=[], usage={}, timings={})
        controller.log = controller.run_log = RunLog('composition', {'task': 'Local recovery fixture.'})
        return controller, observation

    async def test_composition_agent_and_planner_use_effective_config_and_feedback(self):
        for planner in (False, True):
            with self.subTest(planner=planner):
                self.server.requests.clear()
                controller, observation = await self.composition()
                config = AgentConfig(provider='custom', model='local-fixture', base_url=self.base_url+'/agent',
                                     max_output_tokens=512, timeout_s=30)
                token = CURRENT_LOG.set(controller.log)
                try:
                    with patch('server.recovery.invoke_with_recovery', self.retry_schema_once):
                        result = await controller._agent(config, observation, planner=planner, reason='episode_start')
                finally:
                    CURRENT_LOG.reset(token); controller.log.finish({'status': 'complete'})
                self.assertEqual(result.status, 'plan' if planner else 'move')
                first, second = self.server.requests
                self.assertEqual(first['model_settings']['max_output_tokens'], 512)
                self.assertEqual(second['model_settings']['max_output_tokens'], 1024)
                self.assertEqual(second['task'], 'Keep this exact task.')
                self.assertEqual(json.loads(second['prompt'])['recovery_feedback'], FEEDBACK)
                self.assertEqual(first['cameras'], second['cameras'])
                self.assertEqual(controller.worker.moves, 0)
                self.assertEqual(controller.observation_id, 1)
                self.assertEqual(controller.state['timings']['planning' if planner else 'agent_inference']['calls'], 2)

    async def test_composition_policy_freezes_reset_seed_and_logs_one_successful_output(self):
        controller, observation = await self.composition()
        config = PolicyConfig(kind='vla', model='local-fixture', endpoint=self.base_url+'/policy', timeout_s=30)
        token = CURRENT_LOG.set(controller.log)
        try:
            with patch('server.recovery.invoke_with_recovery', self.retry_schema_once):
                output = await controller._policy(config, observation, PolicyStart(kind='vla', task='Fixture.'), 1)
        finally:
            CURRENT_LOG.reset(token); controller.log.finish({'status': 'complete'})
        self.assertEqual(output.action_type, 'tcp_pose')
        self.assertEqual(controller.policy_calls, 1)
        first, second = self.server.requests
        self.assertEqual((first['seed'], first['reset']), (42, True))
        self.assertEqual({key:value for key,value in second.items() if key != 'recovery_feedback'}, first)
        self.assertEqual(second['recovery_feedback'], FEEDBACK)
        self.assertEqual(controller.worker.moves, 0)
        self.assertEqual(controller.state['timings']['policy_inference']['calls'], 2)
        events = self.events(controller.log)
        self.assertEqual(sum(e['event'] == 'policy_output' for e in events), 1)
        self.assertEqual(sum(e['event'] == 'inference' for e in events), 2)

    async def test_effective_policy_timeout_overrides_shared_http_client_default(self):
        captured = []
        def respond(request):
            captured.append(request.extensions['timeout'])
            return httpx.Response(200, json={'protocol': 'embodied-policy-v1'})
        config = PolicyConfig(kind='vla', model='timeout-fixture', endpoint=self.base_url+'/policy', timeout_s=31)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), timeout=15) as client:
            await predict_policy(config, {}, client)
        self.assertTrue(captured)
        self.assertEqual(set(captured[0].values()), {31})

    async def test_vote_recovers_only_failed_chooser_with_same_candidates_and_observation(self):
        class SkillWorker(DeterministicWorker):
            async def execute(self, command):
                result = await super().execute(command)
                if command['action'] == 'observe':
                    result['limits'] = {'translation_per_round_m': .12}
                return result

        worker = SkillWorker()
        recipe = {'mode': 'hybrid',
            'agent': {'provider': 'custom', 'model': 'planner-fixture', 'base_url': self.base_url+'/agent'},
            'decisions': [{'kind': 'custom', 'model': f'vote-{index}', 'url': self.base_url+'/decision',
                           'timeout_s': 30} for index in range(2)],
            'decision_strategy': 'weighted_vote'}
        keys = {'0': 'first-fixture-key', '1': 'second-fixture-key'}
        controller = CompositionController(worker, method_resolver=lambda _: recipe,
            runtime_credentials={'fixture': {'decision_api_keys': keys}})
        calls = {'0': [], '1': []}
        observed = []
        failures = []
        first_calls_entered = asyncio.Event()
        planner_calls = []

        class RetryOnce:
            async def call(self, owner, phase, operation, observation, config):
                if phase.startswith('decision:'):
                    observed.append((phase, observation))
                try:
                    return await operation('', config)
                except ValueError:
                    failures.append(phase)
                    return await operation(FEEDBACK, config.model_copy(update={'timeout_s': 31}))

        controller.recovery = RetryOnce()

        async def plan(config, prompt, cameras, **kwargs):
            planner_calls.append(prompt)
            return PlannerOutput(status='plan', subgoal='Hold.', visual_summary='Fixture.', skill='hold',
                direction_world=[0., 0., 0.], gripper_open=1., expires_after_steps=1,
                explanation='Keep the current pose.'), None

        async def decide(config, context, offered, *, api_key='', cameras=()):
            index = context['decision_id']
            calls[index].append({'context': deepcopy(context), 'candidates': deepcopy(offered),
                'cameras': deepcopy(cameras), 'timeout_s': config.timeout_s, 'api_key': api_key})
            if len(calls['0']) and len(calls['1']): first_calls_entered.set()
            await asyncio.wait_for(first_calls_entered.wait(), 1)
            if index == '1' and len(calls[index]) == 1:
                raise ValueError('Fixture decision response failed validation.')
            return DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
                candidate_id=offered[0]['id'], confidence=.9)

        with patch('server.agent.api_action', plan), patch('server.composition.choose_candidate', decide):
            await controller.start(CompositionStart(method_id='fixture', task='Recovery voting fixture.', one_step=True))
            await asyncio.wait_for(controller.task, 5)

        self.assertEqual(controller.state['status'], 'round_complete', controller.state)
        self.assertEqual([len(calls[index]) for index in ('0', '1')], [1, 2])
        self.assertEqual(failures, ['decision:1'])
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(worker.commands.count('observe'), 2)  # Retry shares input; one additional post-execution capture.
        self.assertEqual(worker.moves, 1)
        self.assertEqual({phase for phase, _ in observed}, {'decision:0', 'decision:1'})
        self.assertTrue(all(observation is controller.observation for _, observation in observed))
        first, retry = calls['1']
        self.assertNotIn('recovery_feedback', first['context'])
        self.assertEqual(retry['context']['recovery_feedback'], FEEDBACK)
        self.assertEqual({key: value for key, value in retry['context'].items() if key != 'recovery_feedback'},
                         first['context'])
        self.assertEqual(first['context']['episode_id'], controller.state['episode_id'])
        self.assertEqual(first['context']['observation_id'], 1)
        self.assertEqual(first['candidates'], retry['candidates'])
        self.assertEqual(first['cameras'], retry['cameras'])
        self.assertEqual((first['timeout_s'], retry['timeout_s']), (30, 31))
        self.assertEqual(calls['0'][0]['api_key'], keys['0'])
        self.assertEqual((first['api_key'], retry['api_key']), (keys['1'], keys['1']))
        events = self.events(controller.log)
        self.assertEqual(sum(event['event'] == 'decision_result' for event in events), 2)
        self.assertEqual(sum(event['event'] == 'selection' for event in events), 1)
        self.assertEqual(sum(event['event'] == 'finish' for event in events), 1)


if __name__ == '__main__':
    unittest.main()
