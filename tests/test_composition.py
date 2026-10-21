"""Real MuJoCo integration; model boundaries use explicitly labelled local fixtures."""
import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from PIL import Image

from server.app import Worker
from server.composition import CompositionController, PlannerOutput
from server.control import PoseAction
from server.decision import DecisionConfig, DecisionResult, choose_candidate, choose_decisions
from server.agent import AgentConfig, api_action


class BoundaryFixture(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append({'path': self.path, 'body': body, 'headers': dict(self.headers)})
        if self.path == '/agent':
            if body['response_schema_name'] == 'robot_subgoal':
                output = {'status': 'plan', 'subgoal': 'Raise slightly.', 'visual_summary': 'Local HTTP fixture.',
                    'skill': 'translate', 'direction_world': [0., 0., 1.], 'gripper_open': 1.,
                    'expires_after_steps': 3, 'explanation': '本机自定义 Agent 测试。'}
            else:
                output = {'status': 'move', 'frame': 'world', 'position_m': [.3, 0., .2],
                    'quaternion_wxyz': [1., 0., 0., 0.], 'gripper_open': 1., 'duration_s': .7,
                    'explanation': '本机自定义 Agent 测试。', 'hand_joint_positions_rad': None}
            result = {'protocol': 'embodied-agent-v1', 'output': output,
                      'model': 'local-agent-harness-fixture', 'usage': {'input_tokens': 5, 'output_tokens': 2}}
        elif self.path == '/policy':
            actuators = body['observation']['actuators']
            target = [actuator['target'] for actuator in actuators]; target[0] += .025
            result = {'protocol': 'embodied-policy-v1', 'action_type': 'actuator_targets',
                'joint_names': [actuator['name'] for actuator in actuators], 'actions': [target]*3, 'dt_s': .05,
                'model': 'local-policy-contract-fixture', 'summary': 'Local HTTP fixture, not a trained policy.'}
        elif self.path == '/typesafe':
            name, question = next(iter(body['questions'].items()))
            selected = next(iter(question['criteria']))
            result = {'model': 'local-typesafe-contract-fixture', 'answers': {name: {
                'type': 'choice', 'choice': selected, 'confidence': .9,
                'probabilities': {key: float(key == selected) for key in question['criteria']}}},
                'usage': {'input_tokens': 12, 'output_tokens': 1}}
        else:
            result = {'protocol': 'embodied-decision-v1', 'episode_id': body['episode_id'],
                'observation_id': body['observation_id'], 'candidate_id': body['candidates'][0]['id'],
                'confidence': .9, 'model': 'local-decision-contract-fixture'}
            if self.path == '/wrong-observation': result['observation_id'] += 1
            if self.path == '/unknown-candidate': result['candidate_id'] = 'invented-command'
        content = json.dumps(result).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(content)


class DecisionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), BoundaryFixture)
        cls.server.requests = []
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(2)

    def test_custom_requires_matching_observation_and_known_candidate(self):
        async def check():
            context = {'episode_id': 'test-episode', 'observation_id': 7}
            candidates = [{'id': 'hold', 'description': 'Hold current pose.'}]
            for path in ('wrong-observation', 'unknown-candidate'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    await choose_candidate({'kind': 'custom', 'url': f'http://127.0.0.1:{self.server.server_port}/{path}'}, context, candidates)
            result = await choose_candidate({'kind': 'custom', 'url': f'http://127.0.0.1:{self.server.server_port}/choose'}, context, candidates)
            self.assertEqual(result.candidate_id, 'hold')
        asyncio.run(check())

    def test_typesafe_choice_and_experimental_images_use_referenced_assets(self):
        async def check():
            out = io.BytesIO(); Image.new('RGB', (4, 4), 'green').save(out, 'JPEG')
            cameras = [{'name': name, 'image_base64': base64.b64encode(out.getvalue()).decode()}
                       for name in ('front', 'side', 'wrist')]
            for vision in (False, True):
                result = await choose_candidate({'kind': 'typesafe', 'url': f'http://127.0.0.1:{self.server.server_port}/typesafe',
                    'model': 'jev-latest', 'vision': vision}, {'episode_id': 'test', 'observation_id': 2,
                    'visual_summary': 'Public old summary', 'visual_summary_observation': {'time': 1.}},
                    [{'id': 'up', 'description': 'Move up.'}, {'id': 'hold', 'description': 'Hold.'}], cameras=cameras,
                    api_key='boundary-secret')
                request = self.server.requests[-1]
                self.assertEqual(result.candidate_id, 'up')
                self.assertEqual(result.usage['output_tokens'], 1)
                self.assertEqual(request['body']['questions']['next_action']['type'], 'choice')
                self.assertEqual(request['body']['state']['current_camera_images_available'], vision)
                if vision:
                    self.assertEqual(request['headers']['TypeSafe-Danger-Zone-Images'], 'true')
                    self.assertEqual(set(request['body']['assets']), set(request['body']['state']['cameras'].values()))
                else:
                    self.assertNotIn('assets', request['body'])
                    self.assertNotIn('cameras', request['body']['state'])
            with self.assertRaisesRegex(ValueError, 'API Key'):
                DecisionConfig(kind='typesafe').validate_credentials('')
        asyncio.run(check())

    def test_custom_agent_uses_full_http_endpoint_and_both_output_contracts(self):
        async def check():
            out = io.BytesIO(); Image.new('RGB', (4, 4), 'green').save(out, 'JPEG')
            cameras = [{'name': name, 'image_base64': base64.b64encode(out.getvalue()).decode()}
                       for name in ('front', 'side', 'wrist')]
            config = AgentConfig(provider='custom', model='any-local-agent',
                                base_url=f'http://127.0.0.1:{self.server.server_port}/agent')
            for response_type, schema_name in ((PoseAction, 'robot_action'), (PlannerOutput, 'robot_subgoal')):
                output, usage = await api_action(config, json.dumps({'task': 'Raise slightly.'}), cameras,
                    response_type=response_type, schema_name=schema_name)
                self.assertIsInstance(output, response_type)
                request = self.server.requests[-1]
                self.assertEqual(request['path'], '/agent')
                self.assertEqual(request['body']['protocol'], 'embodied-agent-v1')
                self.assertEqual(request['body']['task'], 'Raise slightly.')
                self.assertEqual(len(request['body']['cameras']), 3)
                self.assertEqual(request['body']['response_schema']['properties'], response_type.action_schema()['properties'])
                self.assertNotIn('Authorization', request['headers'])
                self.assertEqual(usage['output_tokens'], 2)
        asyncio.run(check())


class DecisionCombinationTests(unittest.TestCase):
    context = {'episode_id': 'combination-test', 'observation_id': 3}
    candidates = [{'id': 'up', 'description': 'Up.'}, {'id': 'down', 'description': 'Down.'}]

    def configs(self, weights=(1, 1)):
        return [DecisionConfig(kind='custom', url=f'http://127.0.0.1:{9000+i}/choose', weight=weight)
                for i, weight in enumerate(weights)]

    def test_cascade_reviews_current_choice_and_can_request_replan(self):
        async def check():
            calls = []; records = []
            async def boundary(config, context, candidates, **kwargs):
                calls.append((context, candidates, kwargs['api_key']))
                candidate = 'up' if context['decision_id'] == '0' else 'request_replan'
                return DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
                    candidate_id=candidate, confidence=.9)
            result = await choose_decisions(self.configs(), 'cascade', self.context, self.candidates,
                api_keys={'0': 'first-secret', '1': 'second-secret'}, chooser=boundary, on_result=records.append)
            self.assertEqual(result.status, 'replan')
            self.assertEqual([candidate['id'] for candidate in calls[1][1]], ['up', 'request_replan'])
            self.assertEqual(calls[1][0]['previous_selection']['candidate_id'], 'up')
            self.assertEqual([call[2] for call in calls], ['first-secret', 'second-secret'])
            self.assertEqual([record['decision_id'] for record in records], ['0', '1'])
        asyncio.run(check())

    def test_weighted_vote_is_parallel_and_conflict_requires_strict_majority(self):
        async def check(weights):
            entered = []; gate = asyncio.Event()
            async def boundary(config, context, candidates, **kwargs):
                entered.append(context['decision_id'])
                if len(entered) == 2: gate.set()
                await asyncio.wait_for(gate.wait(), 1)
                return DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
                    candidate_id='up' if context['decision_id'] == '0' else 'down', confidence=.9)
            result = await choose_decisions(self.configs(weights), 'weighted_vote', self.context,
                                            self.candidates, chooser=boundary)
            self.assertEqual(len(entered), 2)
            self.assertEqual(result.status, 'replan' if weights == (1, 1) else 'selected')
            self.assertIsNone(result.confidence, 'weighted vote share is not a calibrated probability')
            self.assertAlmostEqual(result.aggregation['vote_share'], max(weights)/sum(weights))
        asyncio.run(check((1, 1))); asyncio.run(check((2, 1)))

    def test_vote_service_error_cancels_other_calls_and_is_not_dropped(self):
        async def check():
            gate = asyncio.Event(); entered = []; cancelled = []
            async def boundary(config, context, candidates, **kwargs):
                index = context['decision_id']; entered.append(index)
                if len(entered) == 2: gate.set()
                await gate.wait()
                if index == '0': raise ValueError('Explicit decision service failure.')
                try: await asyncio.sleep(60)
                except asyncio.CancelledError: cancelled.append(index); raise
            with self.assertRaisesRegex(ValueError, 'service failure'):
                await choose_decisions(self.configs(), 'weighted_vote', self.context, self.candidates, chooser=boundary)
            self.assertEqual(cancelled, ['1'])
        asyncio.run(check())

    def test_vote_cancellation_joins_all_pending_deciders(self):
        async def check():
            gate = asyncio.Event(); entered = []; cancelled = []
            async def boundary(config, context, candidates, **kwargs):
                index = context['decision_id']; entered.append(index)
                if len(entered) == 2: gate.set()
                try: await asyncio.sleep(60)
                except asyncio.CancelledError: cancelled.append(index); raise
            task = asyncio.create_task(choose_decisions(self.configs(), 'weighted_vote', self.context, self.candidates, chooser=boundary))
            await asyncio.wait_for(gate.wait(), 1); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(set(cancelled), {'0', '1'})
        asyncio.run(check())

    def test_cascade_low_confidence_requests_replan_without_later_approval(self):
        async def check():
            calls = []
            async def boundary(config, context, candidates, **kwargs):
                calls.append(context['decision_id'])
                return DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
                                      candidate_id='up', confidence=.1)
            result = await choose_decisions(self.configs(), 'cascade', self.context, self.candidates,
                                            min_confidence=.8, chooser=boundary)
            self.assertEqual(result.status, 'replan')
            self.assertEqual(calls, ['0'])
        asyncio.run(check())


class CompositionVersionTests(unittest.TestCase):
    def test_default_start_verifies_method_before_touching_worker(self):
        controller = CompositionController(None)
        with patch('server.artifacts.verify_method', side_effect=ValueError('版本材料已变化')) as verify:
            with self.assertRaisesRegex(ValueError, '版本材料'):
                asyncio.run(controller.start({'method_id': 'method-'+'a'*24, 'task': 'test'}))
            verify.assert_called_once_with('method-'+'a'*24)
        self.assertFalse(controller.state['active'])

    def test_multi_decision_does_not_reuse_legacy_secret_or_coerce_nested_values(self):
        recipe = {'name': 'Credential fixture', 'mode': 'hybrid',
            'agent': {'provider': 'custom', 'base_url': 'http://127.0.0.1:9/agent'},
            'decisions': [{'kind': 'custom', 'url': f'http://127.0.0.1:{9000+i}/choose'} for i in range(2)],
            'decision_strategy': 'weighted_vote'}
        controller = CompositionController(None, lambda _: recipe)
        with self.assertRaises(ValueError):
            controller.configure_runtime('test', {'decision_api_keys': {'0': {'secret': 'bad-shape'}}})
        controller.configure_runtime('test', {'decision_api_key': 'not-forwarded'})
        with self.assertRaisesRegex(ValueError, '仅支持单决策器'):
            asyncio.run(controller.start({'method_id': 'test', 'task': 'test'}))
        self.assertFalse(controller.state['active'])


class CompositionPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tempfile.TemporaryDirectory(prefix='composition-fixture-logs-')
        cls.log_patch = patch('server.runlog.RUNS', Path(cls.logs.name)); cls.log_patch.start()
        cls.worker = Worker(); cls.worker.thread.start()
        assert cls.worker.ready.wait(30)
        assert cls.worker.failure is None, cls.worker.failure
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), BoundaryFixture); cls.server.requests = []
        cls.http_thread = Thread(target=cls.server.serve_forever, daemon=True); cls.http_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.http_thread.join(2)
        cls.worker.shutdown.set(); cls.worker.thread.join(5)
        cls.log_patch.stop(); cls.logs.cleanup()

    async def prepare(self):
        await self.worker.execute({'action': 'load', 'model': 'so101', 'scene_id': 'drawer', 'seed': 42})
        await self.worker.execute({'action': 'settle'})

    def controller(self, mode='hybrid'):
        recipe = {'name': 'Boundary integration fixture', 'mode': mode,
            'agent': {'provider': 'compatible', 'model': 'local-visual-fixture', 'base_url': 'http://127.0.0.1:9/v1'},
            'skills': {'translation_step_m': .005},
            'decision': {'kind': 'custom', 'url': f'http://127.0.0.1:{self.server.server_port}/choose'},
            'policy': {'kind': 'vla', 'model': 'local-policy-contract-fixture',
                       'endpoint': f'http://127.0.0.1:{self.server.server_port}/policy'}}
        return CompositionController(self.worker, lambda _: recipe,
            {'fixture-method': {'agent_api_key': 'fixture-agent-secret'}})

    @staticmethod
    def plan():
        return PlannerOutput(status='plan', subgoal='Raise the TCP slightly.', visual_summary='Three real fixture observations.',
            skill='translate', direction_world=[0., 0., 1.], gripper_open=1., expires_after_steps=4,
            explanation='执行小幅抬高。')

    def events(self, controller):
        folder = Path(self.logs.name) / controller.state['run_id']
        return folder, [json.loads(line) for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_hybrid_caches_plan_and_records_rgb_decisions_and_actual_trajectory(self):
        async def check():
            await self.prepare(); controller = self.controller(); calls = []
            async def planner(config, prompt, cameras, **options):
                calls.append(options)
                self.assertEqual(options['response_type'], PlannerOutput)
                self.assertEqual([camera['name'] for camera in cameras], ['front', 'side', 'wrist'])
                return self.plan(), {'input_tokens': 17, 'output_tokens': 8}
            with patch('server.agent.api_action', planner):
                await controller.start({'method_id': 'fixture-method', 'task': 'Raise the TCP.', 'max_steps': 2,
                    'max_sim_time_s': 10, 'max_wall_time_s': 30})
                await controller.task
            self.assertEqual(controller.state['status'], 'limit', controller.state)
            self.assertEqual(len(calls), 1, 'the slow planner must not run before every fast decision')
            self.assertEqual(controller.state['timings']['decision']['calls'], 2)
            # The local planner stub bypasses the provider boundary; only the two
            # actual HTTP decision invocations belong in provider call telemetry.
            self.assertEqual(controller.state['metrics']['inference_count'], 2)
            self.assertGreater(controller.state['metrics']['mean_inference_s'], 0)
            self.assertEqual(len(controller.state['metrics']['usage']['decision']), 2)
            self.assertIn('candidate_generation', controller.state['metrics']['timings'])
            self.assertEqual(len(controller.state['history']), 2)
            self.assertTrue(all(row['result']['status'] == 'complete' for row in controller.state['history']))
            folder, events = self.events(controller)
            self.assertEqual(len(list((folder/'observations').glob('*/*.jpg'))), 6)
            self.assertTrue((folder/'checkpoints'/'initial.json').exists())
            frames = [json.loads(line) for line in (folder/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertGreater(len(frames), 20)
            self.assertNotEqual(frames[0], frames[-1])
            self.assertEqual(sum(event['event'] == 'plan' for event in events), 1)
            self.assertEqual(sum(event['event'] == 'decision' for event in events), 2)
            self.assertNotIn('fixture-agent-secret', json.dumps(events))
            self.assertNotIn('image_base64', json.dumps(events))
            with self.worker.lock: self.assertTrue(self.worker.snapshot['paused'])
        asyncio.run(check())

    def test_policy_method_uses_original_joint_action_contract(self):
        async def check():
            await self.prepare(); controller = self.controller('policy')
            await controller.start({'method_id': 'fixture-method', 'task': 'Move slightly.', 'one_step': True})
            await controller.task
            self.assertEqual(controller.state['status'], 'round_complete', controller.state)
            result = controller.state['history'][0]['result']['execution']
            self.assertEqual(result['frames'], 3)
            self.assertEqual(len(result['executed_targets'][0]), 6)
            self.assertEqual(self.server.requests[-1]['body']['reset'], True)
        asyncio.run(check())

    def test_local_custom_main_agent_runs_hybrid_without_an_api_key(self):
        async def check():
            await self.prepare()
            recipe = {'name': 'Local agent harness fixture', 'mode': 'hybrid',
                'agent': {'provider': 'custom', 'model': 'any-local-agent',
                          'base_url': f'http://127.0.0.1:{self.server.server_port}/agent'},
                'decision': {'kind': 'custom', 'url': f'http://127.0.0.1:{self.server.server_port}/choose'},
                'skills': {'translation_step_m': .005}}
            controller = CompositionController(self.worker, lambda _: recipe)
            await controller.start({'method_id': 'local-agent-fixture', 'task': 'Raise the TCP.', 'one_step': True})
            await controller.task
            self.assertEqual(controller.state['status'], 'round_complete', controller.state)
            self.assertEqual(controller.state['history'][0]['result']['status'], 'complete')
            agent_request = next(request for request in reversed(self.server.requests) if request['path'] == '/agent')
            self.assertNotIn('Authorization', agent_request['headers'])
        asyncio.run(check())

    def test_agent_method_preserves_real_ik_rejection(self):
        async def check():
            await self.prepare(); controller = self.controller('agent')
            async def impossible(config, prompt, cameras):
                pose = json.loads(prompt)['observation']['tcp']; pose['position_m'][1] += .03
                return PoseAction(status='move', **pose, gripper_open=1., duration_s=.5, explanation='Local unreachable fixture.'), None
            with patch('server.agent.api_action', impossible):
                await controller.start({'method_id': 'fixture-method', 'task': 'Move sideways.', 'one_step': True})
                await controller.task
            self.assertEqual(controller.state['status'], 'failed', controller.state)
            result = controller.state['history'][0]['result']
            self.assertEqual(result['stage'], 'validation')
            self.assertFalse(result['submitted'])
            self.assertIn('IK', result['message'])
        asyncio.run(check())

    def test_cancelled_decision_cannot_submit_a_late_candidate(self):
        async def check():
            await self.prepare(); controller = self.controller(); started = asyncio.Event()
            async def planner(*args, **kwargs): return self.plan(), None
            async def late_result(config, context, candidates, **kwargs):
                started.set()
                try: await asyncio.sleep(60)
                except asyncio.CancelledError: pass  # Explicitly simulate a cancellation-insensitive remote boundary.
                return DecisionResult(episode_id=context['episode_id'], observation_id=context['observation_id'],
                    candidate_id=candidates[0]['id'])
            with patch('server.agent.api_action', planner), patch('server.composition.choose_candidate', late_result):
                await controller.start({'method_id': 'fixture-method', 'task': 'Raise the TCP.', 'max_steps': 2})
                await asyncio.wait_for(started.wait(), 10)
                await controller.stop()
            self.assertEqual(controller.state['status'], 'stopped', controller.state)
            _, events = self.events(controller)
            self.assertFalse(any(event['event'] in ('selection', 'transition') for event in events))
            with self.worker.lock: self.assertTrue(self.worker.snapshot['paused'])
        asyncio.run(check())

    def test_cancel_after_worker_accepts_record_start_releases_recorder_before_finish(self):
        async def check():
            await self.prepare(); controller = self.controller('agent'); started = asyncio.Event()
            original = self.worker.execute
            async def delayed_return(command):
                result = await original(command)
                if command['action'] == 'record_start':
                    started.set()
                    await asyncio.sleep(60)
                return result
            with patch.object(self.worker, 'execute', delayed_return):
                await controller.start({'method_id': 'fixture-method', 'task': 'test'})
                await asyncio.wait_for(started.wait(), 10)
                await controller.stop()
            self.assertEqual(controller.state['status'], 'stopped')
            self.assertIsNone(self.worker.recorder)
            await original({'action': 'pause', 'value': False})
            await asyncio.sleep(.1)
            self.assertIsNone(self.worker.failure)
            await original({'action': 'pause', 'value': True})
        asyncio.run(check())

    def test_scene_change_during_inference_rejects_pending_action(self):
        async def check():
            await self.prepare(); controller = self.controller('agent'); thinking = asyncio.Event(); resume = asyncio.Event()
            async def delayed(config, prompt, cameras):
                thinking.set(); await resume.wait()
                pose = json.loads(prompt)['observation']['tcp']; pose['position_m'][2] += .005
                return PoseAction(status='move', **pose, gripper_open=1., duration_s=.5, explanation='Late local fixture.'), None
            with patch('server.agent.api_action', delayed):
                await controller.start({'method_id': 'fixture-method', 'task': 'Raise the TCP.', 'one_step': True})
                await asyncio.wait_for(thinking.wait(), 10)
                await self.worker.execute({'action': 'load', 'model': 'so101', 'scene_id': 'tabletop'})
                resume.set(); await controller.task
            self.assertEqual(controller.state['status'], 'error', controller.state)
            self.assertIn('迟到', controller.state['message'])
            _, events = self.events(controller)
            self.assertFalse(any(event['event'] == 'selection' for event in events))
        asyncio.run(check())

    def test_wall_budget_and_provider_timeout_have_distinct_causes(self):
        async def check():
            for provider_timeout in (False, True):
                await self.prepare(); controller = self.controller('agent')
                async def timeout(*args, **kwargs):
                    if provider_timeout: raise TimeoutError('Local provider timed out before wall budget.')
                    await asyncio.sleep(20)
                with patch('server.agent.api_action', timeout):
                    await controller.start({'method_id': 'fixture-method', 'task': 'Wait.', 'one_step': True,
                        'max_wall_time_s': 1 if not provider_timeout else 10})
                    await controller.task
                self.assertEqual(controller.state['status'], 'error' if provider_timeout else 'limit', controller.state)
                if provider_timeout: self.assertIn('provider timed out', controller.state['message'])
                else: self.assertIn('墙钟', controller.state['message'])
        asyncio.run(check())


if __name__ == '__main__': unittest.main()
