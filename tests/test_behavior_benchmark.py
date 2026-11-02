"""Official CLI/result contracts with explicit synthetic processes, never an SDK."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import patch

from server import behavior_benchmark as bench, runlog
from server.analysis import analyze_run


RESULT = {'task': 'turning_on_radio', 'instance_id': 301, 'rollout_id': 0,
          'steps': 10, 'success': False, 'q_score': {'final': 0.5},
          'time': {'simulator_steps': 10, 'simulator_time': 1/3, 'normalized_time': 0.1},
          'agent_distance': {'base': 1.5, 'left': 0.2, 'right': 0.4},
          'normalized_agent_distance': {'base': 0.3, 'left': 0.2, 'right': 0.4}}


class BehaviorBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='behavior-benchmark-contract-')
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.patch = patch.object(runlog, 'RUNS', self.root/'runs')
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def fixture(self, mode='complete'):
        """Imitate only the official CLI and JSON boundary, with real process ownership."""
        source = self.root/('source with spaces ' + mode)
        folder = source/'omnigibson/eval'
        folder.mkdir(parents=True, exist_ok=True)
        (source/'omnigibson/__init__.py').write_text('', encoding='utf-8')
        (folder/'__init__.py').write_text('', encoding='utf-8')
        (folder/'eval.py').write_text('''import json,sys,time
from pathlib import Path
def arg(name): return sys.argv[sys.argv.index(name)+1]
folder=Path(arg('--output-dir'))
(folder/'json').mkdir(parents=True)
print('Synthetic evaluator CLI fixture; no robotics SDK',flush=True)
index=int(arg('--instance-indices'))
instance=index+301 if arg('--mode')=='public_test' else index
mode=MODE
if mode!='missing':
 for rollout in range(int(arg('--num-rollouts'))):
  result=dict(RESULT,instance_id=instance,rollout_id=rollout)
  path=folder/'json'/f"turning_on_radio_{instance}_{rollout}.json"
  path.write_text('{broken' if mode=='corrupt' else json.dumps(result),encoding='utf-8')
  if mode=='partial_sleep': time.sleep(30)
if mode in ('failed','corrupt'): sys.exit(7)
'''.replace('MODE', repr(mode)).replace('RESULT', repr(RESULT)), encoding='utf-8')
        return {'kind': 'native', 'python': sys.executable, 'source_root': str(source), 'data_path': str(source/'data')}

    def patched_runtime(self, config):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(bench.omnigibson, 'runtime_config', return_value=config))
        stack.enter_context(patch.object(bench, 'readiness', return_value={'ready': True, 'checks': [], 'message': 'Synthetic contract fixture'}))
        stack.enter_context(patch.object(bench, '_check_runtime', return_value=config))
        return stack

    def test_unprepared_runtime_plan_is_available_and_start_is_rejected_without_log(self):
        with patch.object(bench.omnigibson, 'RUNTIME_FILE', self.root/'missing-runtime.json'):
            value = bench.plan({})
            self.assertFalse(value['ready'])
            self.assertEqual(value['commands'][0]['instance_id'], 301)
            self.assertFalse(value['official_leaderboard_certified'])
            controller = bench.BehaviorBenchmarkController()
            with self.assertRaisesRegex(ValueError, '尚未就绪'):
                asyncio.run(controller.start({}))
        self.assertFalse(controller.state['active'])
        self.assertFalse((self.root/'runs').exists())

    def test_plan_preserves_upstream_defaults_and_serial_one_instance_per_process(self):
        with patch.object(bench.omnigibson, 'runtime_config', return_value=None):
            value = bench.plan({'instance_indices': [0, 19], 'policy_host': '::1', 'policy_scheme': 'wss'})
        self.assertEqual(value['expected_results'], 2)
        self.assertEqual(value['grasping_mode'], 'assisted')
        for command, expected in zip(value['commands'], (301, 320)):
            argv = command['argv']
            self.assertEqual(command['instance_id'], expected)
            self.assertEqual(argv[1:4], ['-u', '-m', 'omnigibson.eval.eval'])
            self.assertEqual(argv[argv.index('--num-envs')+1], '1')
            self.assertEqual(argv[argv.index('--policy-scheme')+1], 'wss')
            self.assertNotIn('--max-steps', argv)
            self.assertNotIn('--robot-config', argv)
            self.assertNotIn('shell', argv)
        for invalid in ({'task_name': '../escape'}, {'policy_host': 'http://host/path'},
                        {'policy_host': 'host;echo'}, {'policy_port': True}, {'instance_indices': [True]},
                        {'instance_indices': [0, 0]}, {'instance_indices': [20]}, {'mode': 'hidden_test'},
                        {'num_rollouts': 0}, {'max_steps': -1}, {'max_wall_time_s': 0}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bench.BehaviorBenchmarkSpec.model_validate(invalid)

    def test_import_preserves_native_q_failure_details_and_unknown_cost(self):
        payload = deepcopy(RESULT)
        payload['failure_reason'] = 'policy_query_timeout'
        result = bench.import_result({'payload': payload, 'policy_name': 'fixture-policy'})
        self.assertEqual(result['metrics']['mean_reported_q'], 0.5)
        self.assertEqual(result['metrics']['success_rate'], 0.)
        self.assertIsNone(result['metrics']['success'])
        row = result['rows'][0]
        stored = json.loads((self.root/'runs'/result['run_id']/row['result_ref']).read_text(encoding='utf-8'))
        self.assertEqual(stored, payload)
        analysis = analyze_run(result['run_id'])
        self.assertIsNone(analysis['total']['calls'])
        self.assertIsNone(analysis['total']['total_tokens'])
        self.assertNotEqual(analysis['cost']['status'], 'complete')
        self.assertFalse(analysis['benchmark']['official_leaderboard_certified'])
        self.assertEqual(analysis['record_origin'], 'external_import')
        summary = json.loads((self.root/'runs'/result['run_id']/'summary.json').read_text(encoding='utf-8'))
        self.assertIsNone(summary['backend']['grasping_mode'])

    def test_invalid_or_duplicate_result_cannot_create_an_experiment(self):
        invalid = [[], {}, {'success': True}, RESULT|{'success': 1}, RESULT|{'q_score': {'final': True}},
                   RESULT|{'q_score': {'final': 2}}, RESULT|{'q_score': {'final': float('nan')}},
                   RESULT|{'time': {}}, RESULT|{'task': '../escape'}, RESULT|{'instance_id': 0}]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                bench.import_result({'payload': payload})
        with self.assertRaises(ValueError):
            bench.import_result({'payload': [RESULT, RESULT]})
        self.assertFalse((self.root/'runs').exists())

    def test_real_cli_boundary_archives_all_selected_results_and_stdout(self):
        async def scenario():
            config = self.fixture()
            controller = bench.BehaviorBenchmarkController()
            with self.patched_runtime(config):
                await controller.start({'instance_indices': [0, 1], 'num_rollouts': 2, 'record_video': False})
                await controller.task
            self.assertEqual(controller.state['status'], 'complete', controller.state)
            self.assertEqual(len(controller.state['rows']), 4)
            self.assertEqual(controller.state['metrics']['mean_reported_q'], 0.5)
            self.assertEqual(controller.state['completed_jobs'], 2)
            self.assertIsNone(controller.process)
            for index in (0, 1):
                output = controller.log.folder/f'benchmark/instance-{index}/stdout.log'
                self.assertIn('Synthetic evaluator', output.read_text(encoding='utf-8'))
            saved = json.loads((controller.log.folder/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['metrics']['episodes'], 4)
            self.assertEqual(saved['backend']['grasping_mode'], 'assisted')
            self.assertNotEqual(saved['cost']['status'], 'complete')
        asyncio.run(scenario())

    def test_failed_cli_retains_partial_results_and_missing_result_is_error(self):
        async def scenario():
            for mode in ('failed', 'corrupt', 'missing'):
                config = self.fixture(mode)
                controller = bench.BehaviorBenchmarkController()
                with self.patched_runtime(config):
                    await controller.start({'record_video': False})
                    await controller.task
                self.assertEqual(controller.state['status'], 'error', controller.state)
                self.assertEqual(len(controller.state['rows']), 1 if mode == 'failed' else 0)
                if mode in ('failed', 'corrupt'):
                    self.assertIn('退出 7', controller.state['message'])
                else:
                    self.assertIn('未生成', controller.state['message'])
        asyncio.run(scenario())

    def test_cancellation_preserves_completed_json_and_finishes_record(self):
        async def scenario():
            config = self.fixture('partial_sleep')
            controller = bench.BehaviorBenchmarkController()
            with self.patched_runtime(config):
                await controller.start({'num_rollouts': 2, 'record_video': False})
                path = controller.log.folder/'benchmark/instance-0/json/turning_on_radio_301_0.json'
                for _ in range(100):
                    if path.exists(): break
                    await asyncio.sleep(.03)
                self.assertTrue(path.exists())
                await controller.stop()
            self.assertEqual(controller.state['status'], 'stopped')
            self.assertFalse(controller.state['active'])
            self.assertEqual(len(controller.state['rows']), 1)
            self.assertTrue((controller.log.folder/'summary.json').exists())
            self.assertIsNone(controller.process)
        asyncio.run(scenario())

    def test_stopping_startup_or_accepted_job_before_first_tick_does_not_launch(self):
        async def scenario():
            config = self.fixture()
            controller = bench.BehaviorBenchmarkController()
            entered, release = Event(), Event()
            def blocked_check():
                entered.set()
                release.wait(4)
                return config
            with self.patched_runtime(config), patch.object(bench, '_check_runtime', side_effect=blocked_check):
                pending = asyncio.create_task(controller.start({}))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                await controller.stop()
                release.set()
                await pending
            self.assertEqual(controller.state['status'], 'stopped')
            self.assertIsNone(controller.log)
            with self.patched_runtime(config):
                await controller.start({})
                await controller.stop()
            self.assertEqual(controller.state['status'], 'stopped')
            self.assertTrue(controller.log._finished)
            self.assertIsNone(controller.process)
        asyncio.run(scenario())

    def test_wall_timeout_is_distinct_and_preserves_partial_results(self):
        async def scenario():
            config = self.fixture('partial_sleep')
            controller = bench.BehaviorBenchmarkController()
            with self.patched_runtime(config):
                await controller.start({'max_wall_time_s': 1, 'record_video': False})
                await controller.task
            self.assertEqual(controller.state['status'], 'limit')
            self.assertEqual(len(controller.state['rows']), 1)
            self.assertFalse(controller.state['active'])
        asyncio.run(scenario())

    def test_owned_process_cleanup_stops_descendant_and_leaves_unrelated_process_alive(self):
        async def scenario():
            child_script = self.root/'heartbeat.py'
            child_script.write_text('import sys,time\nfrom pathlib import Path\np=Path(sys.argv[1])\nwhile True:\n p.write_text(str(time.time()))\n time.sleep(.02)\n', encoding='utf-8')
            parent_script = self.root/'parent.py'
            parent_script.write_text('import subprocess,sys,time\nsubprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\ntime.sleep(30)\n', encoding='utf-8')
            owned_path, other_path = self.root/'owned.txt', self.root/'other.txt'
            unrelated = subprocess.Popen([sys.executable, str(child_script), str(other_path)],
                                         **({'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}))
            owned = None
            try:
                with (self.root/'owned.log').open('wb') as output:
                    owned = bench._OwnedProcess([sys.executable, str(parent_script), str(child_script), str(owned_path)],
                                                cwd=self.root, env=os.environ.copy(), stdout=output)
                    for _ in range(100):
                        if owned_path.exists() and other_path.exists(): break
                        await asyncio.sleep(.02)
                    self.assertTrue(owned_path.exists())
                    await owned.close()
                    owned = None
                    before = owned_path.read_text()
                    other_before = other_path.stat().st_mtime_ns
                    await asyncio.sleep(.15)
                    self.assertEqual(owned_path.read_text(), before)
                    self.assertNotEqual(other_path.stat().st_mtime_ns, other_before)
                    self.assertIsNone(unrelated.poll())
            finally:
                if owned:
                    await owned.close()
                unrelated.kill()
                unrelated.wait(timeout=5)
        asyncio.run(scenario())


if __name__ == '__main__':
    unittest.main()
