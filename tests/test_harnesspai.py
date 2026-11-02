import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.harnesspai import HarnessPAIController, ProgramValidation, EXAMPLE, PYTHON, digest, promotion_evidence


class ProgramContractTests(unittest.TestCase):
    def test_rejects_invalid_program_task_and_seed_sets(self):
        for kwargs in ({'source': 'def run('}, {'source': 'async def run(ctx): pass'},
                       {'seeds': [42, 42]}, {'seeds': [True]}, {'seeds': [-1]},
                       {'scene_id': 'drawer', 'task_id': 'reach_up'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProgramValidation(**{'source': 'def run(ctx): pass', **kwargs})

    def test_hash_is_portable_and_does_not_execute_source_at_validation(self):
        request = ProgramValidation(source='raise RuntimeError("not during validation")\r\ndef run(ctx): pass\r\n')
        self.assertNotIn('\r', request.source)

    def test_promotion_requires_every_native_result_and_unchanged_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root/'candidate'/'workbench_candidate'/'program.py'
            candidate.parent.mkdir(parents=True)
            source = 'def run(ctx): pass\n'
            candidate.write_text(source, encoding='utf-8', newline='')
            job = {'seeds': [42, 43], 'source_sha256': digest(source)}
            (root/'job.json').write_text(json.dumps(job), encoding='utf-8')
            row = {'process_ok': True, 'success': True,
                   'evaluation': {'success': True, 'source': 'mujoco_privileged_state'}}
            summary = {'kind': 'harnesspai', 'status': 'verified', 'harnesspai': {'rows': [{**row, 'seed': i} for i in job['seeds']]}}
            (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            self.assertEqual(promotion_evidence(root)[1], candidate)
            for rows in ([{**row, 'seed': 42}],
                         [{**row, 'seed': 42}, {**row, 'seed': 43, 'success': False}],
                         [{**row, 'seed': 42}, {**row, 'seed': 43, 'process_ok': False}],
                         [{**row, 'seed': 42}, {**row, 'seed': 43, 'evaluation': {'source': 'model_claim', 'success': True}}]):
                summary['harnesspai']['rows'] = rows
                (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
                with self.assertRaises(ValueError): promotion_evidence(root)
            summary['harnesspai']['rows'] = [{**row, 'seed': i} for i in job['seeds']]
            (root/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
            candidate.write_text(source+'# changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '发生变化'): promotion_evidence(root)


class ProgramConsentTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_execution_without_local_code_opt_in(self):
        controller = HarnessPAIController()
        with self.assertRaisesRegex(ValueError, '确认执行本机代码'):
            await controller.start(ProgramValidation(source='def run(ctx): pass'))
        self.assertIsNone(controller.log)
        self.assertIsNone(controller.task)


@unittest.skipUnless(os.getenv('HARNESSPAI_INTEGRATION') == '1' and PYTHON.is_file(),
                     'Opt-in real native simulation with installed official HarnessPAI runtime')
class NativeProgramTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.patch = patch('server.runlog.RUNS', Path(self.directory.name))
        self.patch.start()
        self.controller = HarnessPAIController()

    async def asyncTearDown(self):
        await self.controller.stop()
        self.patch.stop()
        self.directory.cleanup()

    async def run_program(self, **kwargs):
        await self.controller.start(ProgramValidation(**{'source': EXAMPLE.read_text(encoding='utf-8'),
            'record_video': False, 'allow_local_code_execution': True, **kwargs}))
        await asyncio.wait_for(self.controller.task, 90)
        return self.controller.snapshot()

    async def test_native_success_snapshot_and_video(self):
        source = EXAMPLE.read_text(encoding='utf-8') + '''
_run_task = run
def run(ctx):
    before = ctx.budget()
    _run_task(ctx)
    after = ctx.budget()
    assert before['actions_used'] == 0
    assert after['actions_used'] > 0
    assert after['actions_remaining'] == before['actions_remaining'] - after['actions_used']
    assert after['sim_time_s'] > before['sim_time_s']
    assert after['sim_remaining_s'] < before['sim_remaining_s']
    assert after['wall_remaining_s'] < before['wall_remaining_s']
    assert abs(after['sim_time_s'] - ctx.observe()['time']) < 1e-9
'''
        result = await self.run_program(source=source, record_video=True)
        self.assertEqual(result['status'], 'verified')
        self.assertTrue(result['can_promote'])
        for row in result['rows']:
            self.assertTrue(row['process_ok'])
            self.assertEqual(row['evaluation']['source'], 'mujoco_privileged_state')
            self.assertGreater(row['steps'], 0)
            video = self.controller.log.folder/'episodes'/f"seed-{row['seed']}"/row['video']['path']
            self.assertGreater(video.stat().st_size, 1000)
            events = video.parent/'events.jsonl'
            records = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
            for event in [e for e in records if e['event'] == 'observation']:
                self.assertNotIn('evaluation', event['data']['observation'])
                self.assertEqual(len(event['data']['observation']['cameras']), 3)
        self.assertTrue(promotion_evidence(self.controller.log.folder))

    async def test_exit_zero_and_self_report_cannot_complete_unopened_drawer(self):
        result = await self.run_program(source='def run(ctx):\n    return {"success": True}\n',
            scene_id='drawer', task_id='open_drawer', seeds=[42])
        self.assertTrue(result['rows'][0]['process_ok'])
        self.assertFalse(result['rows'][0]['success'])
        self.assertFalse(result['can_promote'])
        self.assertEqual(result['status'], 'unverified')

    async def test_program_error_records_real_reason_and_continues_requested_seeds(self):
        result = await self.run_program(source='def run(ctx):\n    raise RuntimeError("test failure evidence")\n')
        self.assertEqual(result['status'], 'error')
        self.assertEqual([row['seed'] for row in result['rows']], [42, 43])
        self.assertTrue(all('test failure evidence' in row['message'] for row in result['rows']))
        self.assertFalse(result['can_promote'])

    async def test_stop_is_recorded_and_no_following_seed_is_launched(self):
        await self.controller.start(ProgramValidation(source='import time\ndef run(ctx):\n    while True:\n        ctx.observe()\n        time.sleep(.1)\n',
            record_video=True, allow_local_code_execution=True))
        for _ in range(100):
            if list(self.controller.log.folder.glob('episodes/seed-42/record.json')): break
            await asyncio.sleep(.05)
        await self.controller.stop()
        self.assertEqual(self.controller.state['status'], 'stopped')
        self.assertFalse(self.controller.state['active'])
        self.assertFalse((self.controller.log.folder/'episodes'/'seed-43').exists())
        self.assertTrue((self.controller.log.folder/'summary.json').exists())

    async def test_native_timeout_ends_uncooperative_python_and_preserves_error(self):
        started = asyncio.get_running_loop().time()
        result = await self.run_program(source='import time\ndef run(ctx):\n    time.sleep(60)\n',
                                        seeds=[42], timeout_s=10)
        self.assertLess(asyncio.get_running_loop().time()-started, 16)
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['rows'][0]['process_ok'])
        self.assertIn('TIMEOUT', result['rows'][0]['message'])
        self.assertFalse(result['can_promote'])

    async def test_force_stop_ends_uncooperative_process_tree(self):
        await self.controller.start(ProgramValidation(source='import time\ndef run(ctx):\n    time.sleep(60)\n',
            timeout_s=60, record_video=False, allow_local_code_execution=True))
        for _ in range(100):
            if list(self.controller.log.folder.glob('episodes/seed-42/record.json')): break
            await asyncio.sleep(.05)
        started = asyncio.get_running_loop().time()
        await self.controller.stop()
        self.assertLess(asyncio.get_running_loop().time()-started, 9)
        self.assertEqual(self.controller.state['status'], 'stopped')
        self.assertFalse(self.controller.state['active'])
        self.assertFalse((self.controller.log.folder/'episodes'/'seed-43').exists())


if __name__ == '__main__':
    unittest.main()
