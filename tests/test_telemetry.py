"""Real local HTTP returns plus read-only history/cancellation accounting."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

import httpx

from server.agent import AgentConfig, api_action
from server.analysis import analyze_run, analyze_experiments
from server.decision import DecisionConfig, choose_decisions
from server.policy import PolicyConfig, predict_policy
from server.runlog import RunLog, CURRENT_LOG, model_output, list_runs
from server.telemetry import reasoning_text, usage_tokens, track_inference, effective_reasoning


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/agent':
            result = {'protocol': 'embodied-agent-v1', 'model': 'fixture-agent',
                'output': {'status': 'done', 'frame': 'world', 'position_m': [0., 0., 0.],
                    'quaternion_wxyz': [1., 0., 0., 0.], 'gripper_open': 1., 'duration_s': 1.,
                    'explanation': 'Public action.', 'hand_joint_positions_rad': None},
                'reasoning_content': 'Actual returned reasoning fixture-secret',
                'encrypted_content': 'must-not-save', 'usage': {'input_tokens': 8, 'output_tokens': 3}}
        elif self.path == '/policy':
            result = {'protocol': 'embodied-policy-v1', 'action_type': 'actuator_targets',
                'joint_names': ['joint'], 'actions': [[0.]], 'summary': 'Policy public summary.',
                'usage': {'input_tokens': 5, 'output_tokens': 2}, 'model': 'fixture-policy'}
        else:
            result = {'protocol': 'embodied-decision-v1', 'episode_id': body['episode_id'],
                'observation_id': body['observation_id'], 'candidate_id': body['candidates'][0]['id'],
                'model': 'fixture-decision', 'usage': {'input_tokens': 4, 'output_tokens': 1}}
        encoded = json.dumps(result).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(encoded)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='telemetry-fixture-')
        self.root = Path(self.tmp.name)
        self.patch = patch('server.runlog.RUNS', self.root); self.patch.start()

    def tearDown(self):
        self.patch.stop(); self.tmp.cleanup()

    def test_actual_http_agent_policy_vote_each_counted_once_and_reasoning_saved_safely(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        log = RunLog('composition', {'task': 'fixture'}, ['fixture-secret'])
        async def check():
            token = CURRENT_LOG.set(log)
            try:
                log.write('observation', {'step': 1})
                await api_action(AgentConfig(provider='custom', base_url=base+'/agent'), '{}', [])
                await predict_policy(PolicyConfig(kind='vla', endpoint=base+'/policy'), {})
                await choose_decisions([DecisionConfig(url=base+'/choose')]*2, 'weighted_vote',
                    {'episode_id': 'fixture', 'observation_id': 1}, [{'id': 'hold', 'description': 'Hold.'}])
            finally: CURRENT_LOG.reset(token)
        try: asyncio.run(check())
        finally: server.shutdown(); server.server_close(); thread.join(2)
        log.finish({'status': 'complete'})
        result = analyze_run(log.id)
        self.assertEqual(result['total']['calls'], 4)
        self.assertEqual(result['total']['output_tokens'], 7)
        self.assertEqual(result['total']['total_tokens'], 28)
        self.assertTrue(result['total']['usage_complete'])
        self.assertIn('Actual returned reasoning', result['rounds'][0]['reasoning']['text'])
        self.assertEqual(result['rounds'][1]['reasoning']['availability'], 'unavailable')
        self.assertIn('Policy public summary', result['rounds'][1]['public_text'])
        saved = (log.folder/'events.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('fixture-secret', saved); self.assertNotIn('must-not-save', saved)
        self.assertGreater(result['total']['inference_s'], 0)

    def test_cancel_error_missing_usage_and_live_timing_freeze(self):
        log = RunLog('agent', {'task': 'fixture', 'model': 'fixture-model'})
        @track_inference('agent_inference')
        async def boundary(config, entered=None):
            if entered:
                entered.set(); await asyncio.Event().wait()
            model_output({'content': 'truncated public text', 'usage': {'completion_tokens': 9}})
            raise ValueError('response truncated')
        async def check():
            token = CURRENT_LOG.set(log)
            try:
                with self.assertRaisesRegex(ValueError, 'truncated'): await boundary({'model': 'fixture-model'})
                entered = asyncio.Event(); pending = asyncio.create_task(boundary({'model': 'fixture-model'}, entered))
                await entered.wait()
                timing = log.timing_snapshot()['timing']
                self.assertEqual(timing['active_calls'], 1)
                self.assertGreater(timing['current_inference_s'], 0)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError): await pending
            finally: CURRENT_LOG.reset(token)
        asyncio.run(check()); log.finish({'status': 'stopped'})
        result = analyze_run(log.id)
        self.assertEqual([c['status'] for c in result['rounds']], ['error', 'cancelled'])
        self.assertEqual(result['rounds'][0]['output_tokens'], 9)
        self.assertIsNone(result['total']['output_tokens'])
        self.assertIsNone(result['total']['total_tokens'])
        self.assertGreater(result['total']['inference_s'], 0)
        self.assertEqual(log.timing_snapshot()['elapsed_s'], log.timing_snapshot()['elapsed_s'])

    def test_explicit_reasoning_and_malformed_optional_usage_are_not_invented(self):
        self.assertEqual(reasoning_text({'summary': 'ordinary', 'output': {'summary': [{'text': 'ordinary'}]}})['availability'], 'unavailable')
        self.assertEqual(reasoning_text({'reasoning': [{'summary': [{'type': 'summary_text', 'text': 'returned'}]}]})['text'], 'returned')
        self.assertEqual(reasoning_text({'encrypted_content': {'thinking': 'secret'}})['availability'], 'unavailable')
        counters = usage_tokens({'prompt_tokens': float('nan'), 'completion_tokens': float('inf'),
                                 'completion_tokens_details': [], 'total_tokens': True})
        self.assertTrue(all(value is None for value in counters.values()))
        self.assertEqual(effective_reasoning({'provider': 'anthropic', 'reasoning': 'high'}), 'not_requested')
        self.assertEqual(effective_reasoning({'provider': 'compatible', 'base_url': 'https://example.org/v1', 'reasoning': 'high'}), 'provider_default')
        self.assertEqual(effective_reasoning({'provider': 'openai', 'reasoning': 'high'}), 'high')

    def test_deepseek_disabled_is_off_and_invalid_usage_does_not_break_output(self):
        log = RunLog('agent', {'task': 'fixture'})
        @track_inference('agent_inference')
        async def boundary(config):
            model_output({'content': 'valid action', 'usage': {'input_tokens': 2, 'output_tokens': 1,
                'output_tokens_details': 'vendor format', 'other': float('nan')}})
            return 'valid action'
        async def check():
            token = CURRENT_LOG.set(log)
            try:
                self.assertEqual(await boundary(AgentConfig(provider='compatible', base_url='https://api.deepseek.com/v1',
                    deepseek_thinking='disabled', reasoning='high')), 'valid action')
            finally: CURRENT_LOG.reset(token)
        asyncio.run(check()); log.finish({'status': 'complete'})
        self.assertEqual(analyze_run(log.id)['reasoning'], 'off')

    def test_legacy_associates_real_action_timing_and_usage_without_modifying_history(self):
        log = RunLog('agent', {'task': 'legacy', 'model': 'legacy-model', 'settings': {'reasoning': 'high'}})
        log.metadata.pop('telemetry_schema_version')
        log.write('observation', {'round': 1}); log.write('model_output', {'content': 'action'})
        log.write('action', {'round': 1, 'latency_s': 1.25, 'usage': {'input_tokens': 12, 'output_tokens': 7}})
        log.write('observation', {'round': 2}); log.write('model_output', {'content': 'truncated'})
        log.finish({'status': 'error'})
        before = (log.folder/'events.jsonl').read_bytes()
        result = analyze_run(log.id)
        self.assertEqual(result['rounds'][0]['inference_s'], 1.25)
        self.assertEqual(result['rounds'][0]['output_tokens'], 7)
        self.assertIsNone(result['rounds'][1]['inference_s'])
        self.assertIsNone(result['total']['inference_s'])
        self.assertEqual(result['reasoning'], 'high')
        self.assertEqual(before, (log.folder/'events.jsonl').read_bytes())

    def test_full_history_terminal_leaf_groups_missing_samples_and_failed_parent_row(self):
        leaves = []
        for index in range(43):
            log = RunLog('agent', {'task': 'same', 'model': 'model', 'settings': {'reasoning': 'low'}})
            log.write('inference', {'call_id': str(index), 'round': 1, 'status': 'complete', 'model': 'model',
                'reasoning_effort': 'low', 'output_tokens': 2, 'input_tokens': 3, 'total_tokens': 5, 'inference_s': .5})
            log.finish({'status': 'error' if index == 0 else 'complete', 'metrics': {'success': index != 0}}); leaves.append(log)
        missing = RunLog('agent', {'task': 'same', 'model': 'model', 'settings': {'reasoning': 'low'}})
        missing.metadata.pop('telemetry_schema_version'); missing.write('model_output', {'content': 'no usage'})
        missing.finish({'status': 'stopped'})
        active = RunLog('agent', {'task': 'same', 'model': 'model', 'settings': {'reasoning': 'low'}})
        parent = RunLog('research_experiment', {'task': 'same'})
        parent.finish({'status': 'error', 'rows': [{'run_id': leaves[0].id, 'status': 'error', 'seed': 42},
            {'method_name': 'preflight failure', 'status': 'error', 'message': 'real error', 'seed': 43}]})
        result = analyze_experiments(task='same', model='model', kind='agent', limit=1)
        self.assertEqual(result['matched_runs'], 45)
        self.assertEqual(result['excluded_active_runs'], 1)
        group = result['groups'][0]
        self.assertEqual(group['runs'], 44); self.assertEqual(group['output_tokens'], 86)
        self.assertEqual(group['mean_output_tokens'], 2); self.assertEqual(group['missing_output_runs'], 1)
        self.assertEqual(group['token_samples'], 43); self.assertEqual(group['cancelled'], 1)
        self.assertEqual(len(list_runs(10, offset=40)), 6)
        details = analyze_run(parent.id)
        self.assertEqual(len(details['children']), 2); self.assertEqual(details['children'][1]['message'], 'real error')
        self.assertEqual(details['total']['output_tokens'], 2)
        self.assertEqual(analyze_run(active.id)['status'], 'running')

    def test_cleanup_failure_still_finalizes_summary_and_restart_unfinished_is_interrupted(self):
        class BrokenTrajectory:
            closed = False
            def flush(self): raise OSError('disk flush fixture')
            def close(self): self.closed = True
        log = RunLog('agent', {'task': 'fixture'})
        broken = BrokenTrajectory(); log._trajectory = broken
        log.finish({'status': 'stopped', 'message': 'User cancelled.'})
        summary = json.loads((log.folder/'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(broken.closed); self.assertEqual(summary['status'], 'stopped')
        self.assertIn('disk flush fixture', summary['cleanup_errors'][0]['message'])
        other = RunLog('agent', {'task': 'interrupted'})
        from server.runlog import LIVE_RUNS
        del LIVE_RUNS[other.id]
        self.assertEqual(analyze_run(other.id)['status'], 'interrupted')
        self.assertEqual(next(row for row in list_runs() if row['run_id'] == other.id)['status'], 'interrupted')

    def test_legacy_policy_duplicate_preference_preserves_later_failed_return(self):
        log = RunLog('policy', {'task': 'legacy-policy', 'model': 'fixture'})
        log.metadata.pop('telemetry_schema_version')
        log.write('observation', {'round': 1})
        first = {'summary': 'first', 'usage': {'input_tokens': 5, 'output_tokens': 2}}
        log.write('model_output', {'response': first})
        log.write('model_output', {'round': 1, 'response': first, 'latency_s': .2})
        log.write('observation', {'round': 2})
        log.write('model_output', {'response': {'summary': 'failed return', 'usage': {'input_tokens': 7, 'output_tokens': 3}}})
        log.finish({'status': 'error'})
        result = analyze_run(log.id)
        self.assertEqual(result['total']['calls'], 2)
        self.assertEqual(result['total']['output_tokens'], 5)
        self.assertEqual(result['rounds'][0]['inference_s'], .2)
        self.assertIsNone(result['rounds'][1]['inference_s'])
        self.assertIsNone(result['total']['inference_s'])

    def test_live_parent_result_reader_binds_children_without_state_poll(self):
        child = RunLog('agent', {'task': 'fixture', 'model': 'model'})
        child.write('inference', {'call_id': 'actual-return', 'round': 1, 'status': 'complete',
            'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 5, 'inference_s': .2})
        child.finish({'status': 'complete'})
        row = {'status': 'running', 'run_id': None}
        parent = RunLog('comparison', {'task': 'fixture', 'rows': [row]})
        def snapshot():
            row['run_id'] = child.id
            return {'run_id': parent.id, 'rows': [row], **parent.timing_snapshot()}
        parent.snapshot_provider = snapshot
        result = analyze_run(parent.id)
        self.assertEqual(result['total']['output_tokens'], 3)
        self.assertEqual(result['children'][0]['run_id'], child.id)
        self.assertEqual(next(run for run in list_runs() if run['run_id'] == parent.id)['rows'][0]['run_id'], child.id)

    def test_analysis_http_endpoints_validate_paths_pagination_and_unknown_history(self):
        from server.app import app
        log = RunLog('agent', {'task': 'fixture'}); log.finish({'status': 'error'})
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture') as client:
                self.assertEqual((await client.get('/api/runs?limit=1001')).status_code, 422)
                self.assertEqual(len((await client.get('/api/runs?limit=1')).json()), 1)
                response = await client.get(f'/api/runs/{log.id}/analysis')
                self.assertEqual(response.status_code, 200); self.assertEqual(response.json()['status'], 'error')
                self.assertEqual((await client.get('/api/runs/invalid-id/analysis')).status_code, 404)
                result = (await client.get('/api/experiments/analysis?task=fixture')).json()
                self.assertEqual(result['aggregation_scope'], 'leaf_runs_only')
        asyncio.run(check())


if __name__ == '__main__': unittest.main()
