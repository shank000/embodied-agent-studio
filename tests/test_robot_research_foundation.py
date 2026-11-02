import asyncio
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

from server.agent import AgentConfig, api_action
from server.rounds import transition_evidence, timeline
from server.runlog import RunLog, CURRENT_LOG
from server.training_data import export_transitions


def state(success=False, failed=False, potential=0, events=()):
    return {'evaluation_valid': True, 'success': success, 'clock_mode': 'physics_paused',
            'evaluation': {'task_graph': {'version': 'fixture-v1', 'potential': potential,
                'first_achievements': potential, 'terminal_failure': failed,
                'failure_reason': 'forbidden_object_moved' if failed else None, 'events': list(events)}}}


class DecisionMeasurementTests(unittest.TestCase):
    def test_stage_loss_is_recorded_and_failure_is_terminal_without_success_reward(self):
        before = state(potential=1, events=[{'id': 1, 'type': 'achieved'}])
        after = state(failed=True, events=[{'id': 1, 'type': 'achieved'}, {'id': 2, 'type': 'violation'}])
        row = transition_evidence(before, after, 2.5)
        self.assertEqual(row['reward'], 0)
        self.assertTrue(row['terminated'])
        self.assertEqual(row['termination_reason'], 'forbidden_object_moved')
        self.assertEqual(row['reward_components']['stage_potential_delta'], -1)
        self.assertEqual(row['reward_components']['elapsed_wall_s'], 2.5)
        self.assertEqual(row['stage_events'], [{'id': 2, 'type': 'violation'}])

    def test_invalid_evaluation_does_not_generate_reward_or_terminal_labels(self):
        after = state(success=True)
        after['evaluation_valid'] = False
        row = transition_evidence(state(), after, 1)
        self.assertIsNone(row['reward'])
        self.assertIsNone(row['terminated'])
        self.assertIsNone(row['reward_components']['constraint_violation'])

    def test_export_distinguishes_native_failure_from_budget_truncation(self):
        with tempfile.TemporaryDirectory() as directory, patch('server.runlog.RUNS', Path(directory)):
            for failed in (False, True):
                log = RunLog('agent', {'task': 'contract fixture'})
                log.write('round_start', {'round': 1, 'before': state()})
                log.write('observation', {'round': 1, 'observation': {'time': 0, 'cameras': []}})
                log.write('action', {'round': 1, 'action': {'status': 'move'}})
                log.write('observation_after', {'round': 1, 'observation': {'time': 1, 'cameras': []}})
                log.write('round_end', {'round': 1, 'status': 'complete', 'after': state(failed=failed),
                    **transition_evidence(state(), state(failed=failed), 2)})
                log.finish({'status': 'failed' if failed else 'limit'})
                exported = export_transitions(log.id)
                row = exported['transitions'][0]
                self.assertEqual(exported['schema_version'], 'embodied-transitions-v2')
                self.assertEqual(row['terminated'], failed)
                self.assertEqual(row['truncated'], not failed)
                self.assertTrue(row['training_eligible'])
                self.assertFalse(row['routing_training_eligible'])
                self.assertFalse(row['propensity_evaluation_eligible'])
                self.assertNotIn('task_graph', row['observation'])


class GatewayContractTests(unittest.TestCase):
    def test_compatible_gateway_headers_and_returned_route_survive_real_http(self):
        received = {}
        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                received.update(headers=dict(self.headers), body=json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                action = {'status': 'move', 'frame': 'world', 'position_m': [.3, 0, .4],
                    'quaternion_wxyz': [1, 0, 0, 0], 'gripper_open': 1, 'duration_s': 1,
                    'explanation': 'fixture', 'hand_joint_positions_rad': None}
                response = {'id': 'fixture', 'object': 'chat.completion', 'created': 1, 'model': 'gateway-alias',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(action)}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8},
                    'routing': {'resolved_model': 'chosen-model', 'resolved_provider': 'fixture-provider',
                        'router_policy_version': 'router-v2', 'selected_capability': 'plan', 'selection_probability': None},
                    'billing': {'scope': 'self'}}
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        service = ThreadingHTTPServer(('127.0.0.1', 0), Gateway)
        thread = Thread(target=service.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, patch('server.runlog.RUNS', Path(directory)):
                log = RunLog('agent', {}); log.current_round = 2
                async def call():
                    token = CURRENT_LOG.set(log)
                    try:
                        await api_action(AgentConfig(provider='compatible', model='router', api_key='fixture-key',
                            base_url=f'http://127.0.0.1:{service.server_port}/v1'), '{}', [])
                    finally: CURRENT_LOG.reset(token)
                asyncio.run(call()); log.finish({'status': 'limit'})
                record = timeline(log.id)['rounds'][0]['calls'][0]
                self.assertEqual(record['model'], 'chosen-model')
                self.assertEqual(record['routing']['router_policy_version'], 'router-v2')
                self.assertIsNone(record['routing']['selection_probability'])
                headers = {key.lower(): value for key, value in received['headers'].items()}
                self.assertEqual(headers['x-embodiworks-trace-id'], log.id)
                self.assertEqual(headers['x-embodiworks-request-id'], record['call_id'])
                self.assertEqual(record['input_tokens'], 5)
                self.assertIsNone(record['cost']['amount_usd'])
        finally:
            service.shutdown(); service.server_close(); thread.join(2)
