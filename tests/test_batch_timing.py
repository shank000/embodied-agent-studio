"""Parent clocks count each owned child run once, including cancellation."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from server.agent import AgentConfig, AgentController
from server.app import Worker
from server.comparison import ComparisonController, ComparisonStart
from server.policy import PolicyConfig, PolicyController
from server.research import ResearchController
from server.telemetry import batch_timing


def clock(run_id, inference_s, *, active=False, current_s=0.):
    return {'run_id': run_id, 'active': active, 'timing': {
        'inference_elapsed_s': inference_s, 'current_inference_s': current_s,
        'active_calls': int(active)}}


class BatchTimingTests(unittest.TestCase):
    def test_only_bound_run_ids_count_and_duplicate_rows_do_not_double_count(self):
        rows = [{'run_id': 'finished', 'status': 'complete', 'metrics': {'inference_s': 2.}},
                {'run_id': 'finished', 'status': 'complete', 'metrics': {'inference_s': 2.}},
                {'run_id': 'current', 'status': 'running', 'metrics': None}]
        result = batch_timing({'rows': rows}, None, [clock('current', 3., active=True, current_s=1.),
                                                   clock('unrelated', 100., active=True, current_s=100.)])
        self.assertEqual(result['timing']['inference_elapsed_s'], 5.)
        self.assertEqual(result['timing']['current_inference_s'], 1.)
        self.assertEqual(result['timing']['active_calls'], 1)

    def test_finished_child_is_counted_before_parent_copies_its_metrics(self):
        row = {'run_id': 'current', 'status': 'running', 'metrics': None}
        child = clock('current', 4.)
        self.assertEqual(batch_timing({'rows': [row]}, None, [child])['timing']['inference_elapsed_s'], 4.)
        row.update(status='stopped', metrics={'inference_s': 4.})
        self.assertEqual(batch_timing({'rows': [row]}, None, [child])['timing']['inference_elapsed_s'], 4.)

    def test_reused_controllers_require_new_task_and_new_run_identity(self):
        for kind in ('comparison', 'research'):
            with self.subTest(kind=kind):
                old_task = object()
                child = SimpleNamespace(task=old_task)
                child.snapshot = lambda: clock('old-run', 10., active=True)
                parent = (ComparisonController(None, child, child) if kind == 'comparison'
                          else ResearchController(None, child))
                row = {'run_id': None, 'status': 'running', 'metrics': None}
                parent.state['rows'] = [row]
                parent._active_trial = {'row': row, 'controller': child,
                                        'previous_task': old_task, 'previous_run_id': 'old-run'}
                self.assertEqual(parent.snapshot()['timing']['inference_elapsed_s'], 0.)
                child.task = object()
                self.assertEqual(parent.snapshot()['timing']['inference_elapsed_s'], 0.)
                child.snapshot = lambda: clock('new-run', 2., active=True, current_s=.5)
                result = parent.snapshot()
                self.assertEqual(row['run_id'], 'new-run')
                self.assertEqual(result['timing']['inference_elapsed_s'], 2.)
                self.assertEqual(result['timing']['active_calls'], 1)


class TimingHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        if self.path == '/agent':
            response = {'protocol': 'embodied-agent-v1', 'model': 'local-returned-agent',
                        'output': {'status': 'done', 'position_m': [0., 0., .3],
                                   'quaternion_wxyz': [1., 0., 0., 0.], 'gripper_open': 1.,
                                   'duration_s': .5, 'explanation': 'Explicit HTTP fixture completed.'},
                        'usage': {'input_tokens': 11, 'output_tokens': 7},
                        'returned_reasoning': 'Visible fixture reasoning returned by local adapter.'}
        else:
            self.server.pending.set()
            self.server.release.wait(15)
            response = {'error': 'Cancelled policy fixture must never execute.'}
        data = json.dumps(response).encode()
        try:
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError): pass


class NativeBatchTimingTests(unittest.TestCase):
    def test_live_child_cancelled_call_and_final_parent_summary_agree(self):
        with tempfile.TemporaryDirectory(prefix='batch-timing-fixture-') as temporary, \
                patch('server.runlog.RUNS', Path(temporary)):
            server = ThreadingHTTPServer(('127.0.0.1', 0), TimingHandler)
            server.daemon_threads = True
            server.requests = []; server.pending = Event(); server.release = Event()
            http_thread = Thread(target=server.serve_forever, daemon=True); http_thread.start()
            worker = Worker(); worker.thread.start()
            try:
                self.assertTrue(worker.ready.wait(30)); self.assertIsNone(worker.failure)
                async def check():
                    base_url = f'http://127.0.0.1:{server.server_port}'
                    agent = AgentController(worker)
                    # One-call timing fixture: a model's done proposal is not an episode stop.
                    agent.configure(AgentConfig(provider='custom', model='local-agent', base_url=base_url+'/agent', max_rounds=1))
                    policy = PolicyController(worker)
                    policy.configure(PolicyConfig(kind='vla', model='local-policy', endpoint=base_url+'/policy'))
                    parent = ComparisonController(worker, agent, policy)
                    await parent.start(ComparisonStart(robot='so101', methods=['agent', 'vla'], max_sim_time_s=1))
                    try:
                        async def pending():
                            while not server.pending.is_set(): await asyncio.sleep(.01)
                        await asyncio.wait_for(pending(), 15)
                        first = parent.snapshot()
                        await asyncio.sleep(.05)
                        second = parent.snapshot()
                        self.assertEqual(second['timing']['active_calls'], 1)
                        self.assertGreater(second['timing']['inference_elapsed_s'], first['timing']['inference_elapsed_s'])
                        self.assertEqual(parent.run_log.metadata['rows'][1]['run_id'], policy.state['run_id'])
                        self.assertNotEqual(parent.state['rows'][0]['run_id'], parent.state['rows'][1]['run_id'])
                    finally:
                        await parent.stop(); server.release.set()
                    result = parent.snapshot()
                    total = sum(row['metrics']['inference_s'] for row in result['rows'])
                    self.assertGreater(total, 0.)
                    self.assertAlmostEqual(result['timing']['inference_elapsed_s'], total)
                    self.assertEqual(result['timing']['active_calls'], 0)
                    self.assertEqual(result['timing']['inference_calls'], 2)
                    self.assertEqual([row['metrics']['inference_count'] for row in result['rows']], [1, 1])
                    summary = json.loads((Path(temporary)/result['run_id']/'summary.json').read_text(encoding='utf-8'))
                    self.assertAlmostEqual(summary['timing']['inference_elapsed_s'], total)
                    from server.analysis import analyze_run
                    analysis = analyze_run(result['run_id'])
                    self.assertEqual(analysis['total']['calls'], 2)
                    self.assertAlmostEqual(analysis['total']['inference_s'], total)
                    self.assertIsNone(analysis['total']['total_tokens'])
                    self.assertEqual(analysis['total']['known_total_tokens'], 18)
                    child = analyze_run(result['rows'][1]['run_id'])
                    self.assertEqual(child['rounds'][0]['status'], 'cancelled')
                    self.assertIsNone(child['rounds'][0]['input_tokens'])
                    self.assertEqual(len(server.requests[0]['cameras']), 3)
                asyncio.run(check())
            finally:
                server.release.set(); server.shutdown(); server.server_close(); http_thread.join(2)
                worker.shutdown.set(); worker.thread.join(10)


if __name__ == '__main__': unittest.main()
