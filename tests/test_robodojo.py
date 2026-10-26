"""Public plan/import behavior, using explicitly synthetic terminal results; no SDK."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from server import robodojo, runlog
from server.analysis import analyze_run, analyze_experiments
from server.run_management import delete_run, restore_run


PAYLOAD={'eval_time':2,'success_rate':.5,'score':62.5,'details':{
    '0':{'layout_id':0,'success':True,'score':1.},
    '1':{'layout_id':1,'success':False,'score':.25}}}


class RoboDojoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dojo-public-contract-')
        self.root=Path(self.temp.name)
        self.patches=[patch.object(runlog,'RUNS',self.root/'runs'),
            patch.object(robodojo,'RUNTIME_FILE',self.root/'runtime.json'),
            patch.object(robodojo,'DEFAULT_SOURCE',self.root/'source')]
        for item in self.patches:item.start()
        from server.app import app
        self.app=app

    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.temp.cleanup()

    def call(self,method,path,**kwargs):
        async def request():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://127.0.0.1:8000') as client:
                return await client.request(method,'/api/benchmarks/robodojo'+path,**kwargs)
        return asyncio.run(request())

    def native_fixture(self,mode='complete'):
        """Small real process emulating only the official CLI/result boundary."""
        source=self.root/'source'
        adapter=source/'XPolicyLab/policy/demo_policy/deploy.py'
        adapter.parent.mkdir(parents=True,exist_ok=True);adapter.write_text('# synthetic policy fixture',encoding='utf-8')
        entry=source/'src/eval_client/main.py';entry.parent.mkdir(parents=True,exist_ok=True)
        entry.write_text('''import json, os, sys
from pathlib import Path
def arg(name): return sys.argv[sys.argv.index(name)+1]
root=Path('eval_result/RoboDojo')/arg('--task_name')/arg('--policy_name')/arg('--env_cfg_type')/(arg('--seed')+'_'+arg('--additional_info'))/os.environ['ROBODOJO_RUN_ID']
root.mkdir(parents=True)
print('Synthetic CLI fixture, no robotics SDK loaded',flush=True)
mode=MODE
if mode != 'missing':
 (root/'_result.json').write_text('{broken' if mode == 'corrupt_failure' else json.dumps(PAYLOAD),encoding='utf-8')
sys.exit(7 if mode in ('failed','corrupt_failure') else 0)
'''.replace('MODE',repr(mode)).replace('PAYLOAD',repr(PAYLOAD)),encoding='utf-8')
        return {'kind':'native','source_root':str(source),'python':sys.executable}

    def events(self,log,kind):
        return [r['data'] for r in map(json.loads,log.path.read_text(encoding='utf-8').splitlines()) if r['event']==kind]

    def test_unprepared_runtime_allows_catalog_and_plan_but_not_native_execution(self):
        response=self.call('GET','');self.assertEqual(response.status_code,200)
        self.assertFalse(response.json()['readiness']['ready'])
        response=self.call('POST','/plan',json={'task_ids':['stack_bowls'],'seeds':[0,2],'episodes':1})
        self.assertEqual(response.status_code,200,response.text)
        plan=response.json();self.assertFalse(plan['ready']);self.assertEqual(len(plan['commands']),2)
        self.assertEqual(plan['commands'][0]['argv'][2],'client')
        response=self.call('POST','/start',json={})
        self.assertEqual(response.status_code,400);self.assertIn('Linux GPU',response.text)
        self.assertFalse((self.root/'runs').exists())

    def test_invalid_endpoint_and_task_inputs_are_rejected(self):
        for payload in ({'policy_host':'https://host/path'},{'policy_host':'host;whoami'},
            {'task_ids':['unknown']},{'seeds':[True]},{'seeds':['1']},{'episodes':True},
            {'episodes':'4'},{'policy_port':True},{'checkpoint':'../../etc/passwd'}):
            with self.subTest(payload=payload):
                response=self.call('POST','/plan',json=payload);self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.call('POST','/plan',json={},headers={'Origin':'https://other.invalid'}).status_code,403)

    def test_terminal_import_preserves_native_metrics_unknown_usage_and_reversible_management(self):
        response=self.call('POST','/import',json={'task_id':'stack_bowls','seed':0,'policy':'fixture-policy','payload':PAYLOAD})
        self.assertEqual(response.status_code,200,response.text)
        value=response.json();identifier=value['run_id']
        self.assertEqual(value['metrics']['episodes'],2)
        self.assertEqual(value['metrics']['success_rate'],.5)
        self.assertIsNone(value['metrics']['success'])
        result=analyze_run(identifier)
        self.assertIsNone(result['total']['calls']);self.assertIsNone(result['total']['total_tokens'])
        self.assertIsNone(result['total']['inference_s'])
        self.assertNotEqual(result['cost']['status'],'complete')
        self.assertFalse(result['benchmark']['coverage']['official_overall_complete'])
        self.assertEqual(analyze_experiments()['groups'],[],'Bench scores must not be reinterpreted as failed binary robot trials')
        stored=self.root/'runs'/identifier/'benchmark/_result.json'
        self.assertEqual(json.loads(stored.read_text(encoding='utf-8')),PAYLOAD)
        delete_run(identifier);self.assertEqual(self.call('GET','').json()['history'],[])
        restore_run(identifier);self.assertEqual(len(self.call('GET','').json()['history']),1)

    def test_videos_require_matching_original_results_and_copy_only_native_video_files(self):
        folder=self.root/'native';folder.mkdir()
        (folder/'_result.json').write_text(json.dumps(PAYLOAD),encoding='utf-8')
        (folder/'episode_0000000_cam_head_success.mp4').write_bytes(b'fixture-media-not-a-real-rollout')
        (folder/'private.txt').write_text('excluded',encoding='utf-8')
        request={'task_id':'stack_bowls','seed':0,'policy':'fixture','payload':PAYLOAD,
            'artifact_dir':str(folder),'copy_videos':True}
        response=self.call('POST','/import',json=request);self.assertEqual(response.status_code,200,response.text)
        result=response.json();self.assertEqual(len(result['native_videos']),1)
        self.assertEqual(result['native_videos'][0]['camera'],'head')
        self.assertFalse((self.root/'runs'/result['run_id']/'benchmark/videos/private.txt').exists())
        (folder/'_result.json').write_text('{}',encoding='utf-8')
        response=self.call('POST','/import',json=request);self.assertEqual(response.status_code,400)

    def test_initial_success_flags_without_terminal_records_cannot_be_imported(self):
        response=self.call('POST','/import',json={'task_id':'stack_bowls','seed':0,'policy':'fixture','payload':{'success':[True]}})
        self.assertEqual(response.status_code,400)
        self.assertFalse((self.root/'runs').exists())

    def test_empty_or_shell_runtime_configs_are_actionable_and_never_executed(self):
        for config in ({},{'kind':'shell','python':'cmd /c anything'},[] ):
            with self.subTest(config=config):
                (self.root/'runtime.json').write_text(json.dumps(config),encoding='utf-8')
                self.assertFalse(robodojo.readiness()['ready'])

    def test_stop_during_startup_check_never_launches_a_late_native_job(self):
        async def scenario():
            controller=robodojo.RoboDojoController();entered=Event();release=Event()
            def delayed_plan(spec):
                entered.set();release.wait(5);return {'ready':True}
            with patch.object(robodojo,'plan',side_effect=delayed_plan):
                pending=asyncio.create_task(controller.start({}))
                self.assertTrue(await asyncio.to_thread(entered.wait,2))
                self.assertTrue(controller.snapshot()['active'])
                await controller.stop();release.set();result=await pending
                self.assertFalse(result['active']);self.assertEqual(result['status'],'stopped')
                self.assertIsNone(controller.task);self.assertIsNone(controller.process)
                self.assertIsNone(controller.log)
        asyncio.run(scenario())

    def test_accepted_job_stopped_before_first_tick_is_archived_and_unlocked(self):
        async def scenario():
            controller=robodojo.RoboDojoController()
            source=self.root/'source';adapter=source/'XPolicyLab/policy/demo_policy/deploy.py'
            adapter.parent.mkdir(parents=True);adapter.write_text('# synthetic policy fixture',encoding='utf-8')
            config={'kind':'native','source_root':str(source),'python':'fixture-python-not-executed'}
            with patch.object(robodojo,'plan',return_value={'ready':True,'commands':[]}), \
                    patch.object(robodojo,'runtime_config',return_value=config):
                state=await controller.start({});identifier=state['run_id']
                await controller.stop()
            self.assertFalse(controller.state['active']);self.assertEqual(controller.state['status'],'stopped')
            self.assertIsNone(controller.process)
            record=analyze_run(identifier)
            self.assertEqual(record['status'],'stopped')
            self.assertEqual(record['model'],'demo_policy')
            self.assertEqual(record['rounds'],[])
            self.assertIn('尚未开始',record['message'])
        asyncio.run(scenario())

    def test_native_exit_error_preserves_valid_partial_results_and_original_status(self):
        async def scenario():
            controller=robodojo.RoboDojoController();config=self.native_fixture('failed')
            with patch.object(robodojo,'plan',return_value={'ready':True,'commands':[]}),patch.object(robodojo,'runtime_config',return_value=config):
                await controller.start({'episodes':50});await controller.task
            self.assertEqual(controller.state['status'],'error')
            self.assertIn('退出 7',controller.state['message'])
            self.assertEqual(controller.state['current_task'],'stack_bowls')
            self.assertEqual(controller.state['current_seed'],0)
            self.assertEqual(controller.state['completed_jobs'],1)
            self.assertEqual(controller.state['total_jobs'],1)
            self.assertEqual(len(controller.state['rows']),1)
            child=analyze_run(controller.state['rows'][0]['run_id'])
            self.assertEqual(child['benchmark']['summary']['eval_time'],2)
            self.assertFalse(child['benchmark']['coverage']['task_seed_complete'])
            parent=analyze_run(controller.log.id)
            self.assertEqual(parent['status'],'error')
            self.assertEqual(len(parent['trials']),1)
            self.assertTrue((controller.log.folder/'benchmark/stack_bowls-0.log').is_file())
        asyncio.run(scenario())

    def test_bad_partial_result_does_not_replace_original_process_failure(self):
        async def scenario():
            controller=robodojo.RoboDojoController();config=self.native_fixture('corrupt_failure')
            with patch.object(robodojo,'plan',return_value={'ready':True,'commands':[]}),patch.object(robodojo,'runtime_config',return_value=config):
                await controller.start({});await controller.task
            self.assertEqual(controller.state['status'],'error')
            self.assertIn('退出 7',controller.state['message'])
            self.assertEqual(controller.state['rows'],[])
            self.assertEqual(len(self.events(controller.log,'archive_error')),1)
        asyncio.run(scenario())

    def test_native_exit_zero_without_results_is_not_success(self):
        async def scenario():
            controller=robodojo.RoboDojoController();config=self.native_fixture('missing')
            with patch.object(robodojo,'plan',return_value={'ready':True,'commands':[]}),patch.object(robodojo,'runtime_config',return_value=config):
                await controller.start({});await controller.task
            self.assertEqual(controller.state['status'],'error')
            self.assertIn('未生成',controller.state['message'])
            self.assertEqual(controller.state['rows'],[])
        asyncio.run(scenario())

    def test_repeated_stop_during_archival_waits_and_keeps_parent_child_link(self):
        async def scenario():
            controller=robodojo.RoboDojoController();config=self.native_fixture()
            entered=Event();release=Event();native_import=robodojo.import_result
            def delayed_import(request):
                entered.set()
                if not release.wait(5): raise AssertionError('Test did not release import')
                return native_import(request)
            with patch.object(robodojo,'plan',return_value={'ready':True,'commands':[]}),patch.object(robodojo,'runtime_config',return_value=config),patch.object(robodojo,'import_result',side_effect=delayed_import):
                await controller.start({})
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait,2))
                    first=asyncio.create_task(controller.stop());await asyncio.sleep(.03)
                    second=asyncio.create_task(controller.stop());await asyncio.sleep(.03)
                    self.assertFalse(first.done());self.assertFalse(second.done())
                    self.assertTrue(controller.state['active']);self.assertFalse(controller.log._finished)
                    release.set();await asyncio.gather(first,second)
                finally:release.set()
            self.assertEqual(controller.state['status'],'stopped')
            self.assertFalse(controller.state['active'])
            self.assertEqual(len(controller.state['rows']),1)
            self.assertEqual(len(self.events(controller.log,'benchmark_trial')),1)
            self.assertEqual(len(analyze_run(controller.log.id)['trials']),1)
            self.assertTrue(controller.log._finished)
        asyncio.run(scenario())

    def test_exhausted_wall_budget_does_not_spawn_the_next_sdk_job(self):
        async def scenario():
            controller=robodojo.RoboDojoController();config=self.native_fixture()
            times=iter([0.,0.,11.,11.])
            with patch.object(robodojo,'plan',return_value={'ready':True,'commands':[]}),patch.object(robodojo,'runtime_config',return_value=config),patch.object(robodojo,'time',SimpleNamespace(monotonic=lambda:next(times,11.))):
                await controller.start({'seeds':[0,1],'max_wall_time_s':10});await controller.task
            self.assertEqual(controller.state['status'],'error')
            self.assertIn('墙钟预算',controller.state['message'])
            self.assertEqual(controller.state['completed_jobs'],1)
            self.assertEqual(controller.state['total_jobs'],2)
            self.assertFalse((controller.log.folder/'benchmark/stack_bowls-1.log').exists())
        asyncio.run(scenario())
