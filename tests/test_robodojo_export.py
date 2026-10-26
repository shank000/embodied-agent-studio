"""Portable exports preserve official arguments and never launch SDKs."""
import asyncio
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import httpx

from server.robodojo import EvaluationPlan, REVISION
from server.robodojo_plan_export import export_plan


class RoboDojoExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dojo-export-')
        self.addCleanup(self.tmp.cleanup)
        mocked = patch('server.robodojo.readiness', return_value={
            'ready':False, 'source_root':str(Path(self.tmp.name)/'Windows source with spaces')})
        mocked.start(); self.addCleanup(mocked.stop)

    def test_export_is_portable_preserves_parameters_and_has_explicit_budget(self):
        result = export_plan(EvaluationPlan(task_ids=['stack_bowls','cover_blocks'],seeds=[0,2],
            policy_host='::1',policy_port=10001,policy_name='MyPolicy',episodes='native',max_wall_time_s=7200))
        self.assertEqual(result['execution'],'manual_linux')
        self.assertFalse(result['ready'])
        script = result['script']
        self.assertNotIn(self.tmp.name,script)
        self.assertIn('ROBODOJO_ROOT',script)
        self.assertIn(REVISION,script)
        self.assertIn('7200 - SECONDS',script)
        self.assertIn('--kill-after=15s',script)
        self.assertIn('ROBODOJO_RUN_ID',script)
        self.assertIn('export EVAL_NUM=native',script)
        commands = [shlex.split(line) for line in script.splitlines() if line.startswith('run_client ')]
        self.assertEqual(len(commands),4)
        for command in commands:
            self.assertEqual(command[:2],['run_client','client'])
            self.assertEqual(command[command.index('--policy-host')+1],'[::1]')
            self.assertEqual(command[command.index('--policy-name')+1],'MyPolicy')
            self.assertEqual(command[command.index('--eval-num')+1],'native')
        self.assertEqual([c[c.index('--seed')+1] for c in commands],['0','2','0','2'])

    def test_export_api_is_read_only_to_runtime_and_rejects_invalid_requests(self):
        from server.app import app
        async def exercise():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://127.0.0.1:8000') as client:
                path='/api/benchmarks/robodojo/export'
                with patch('server.robodojo.RoboDojoController.start',side_effect=AssertionError('Export must not execute')):
                    response=await client.post(path,json={'task_ids':['stack_bowls']})
                self.assertEqual(response.status_code,200,response.text)
                self.assertTrue(response.json()['filename'].endswith('.sh'))
                self.assertEqual((await client.post(path,json={'policy_name':'bad;command'})).status_code,422)
                self.assertEqual((await client.post(path,json={},headers={'Origin':'https://example.org'})).status_code,403)
        asyncio.run(exercise())


if __name__ == '__main__': unittest.main()
