import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.harnesspai import HarnessPAIController, ProgramValidation, ProgramEdit, ProgramFile, EXAMPLE, PYTHON, promotion_evidence
from tools.harnesspai_evolve import apply_files


class EvolutionContractTests(unittest.TestCase):
    def test_edit_requires_changed_code_but_diagnosis_can_be_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            program = root / 'program.py'
            source = 'def run(ctx): pass'
            program.write_text(source, encoding='utf-8')
            for files in ([], [{'path':str(program),'content':source}]):
                with self.assertRaisesRegex(ValueError, '未返回候选程序的有效修改.*missing camera'):
                    apply_files({'explanation':'missing camera', 'files':files}, root, program)
            apply_files({'explanation':'diagnosis only','files':[]}, root)
            apply_files({'explanation':'native baseline already passed','files':[]}, root, program, require_change=False)
            self.assertEqual(program.read_text(), source)

    def test_disjoint_evolution_and_holdout(self):
        with self.assertRaisesRegex(ValueError, '必须分开'):
            ProgramValidation(source='def run(ctx): pass', mode='evolve', seeds=[42], evolution_seeds=[42])

    def test_holdout_alone_does_not_promote_an_unconverged_evolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'job.json').write_text(json.dumps({'mode':'evolve','seeds':[42,43]}))
            rows=[{'seed':s,'process_ok':True,'success':True,'evaluation':{'source':'mujoco_privileged_state','success':True}} for s in (42,43)]
            (root/'summary.json').write_text(json.dumps({'kind':'harnesspai','status':'verified',
                'harnesspai':{'rows':rows,'upstream_outcome':{'converged':False}}}))
            with self.assertRaisesRegex(ValueError,'原生环境验收'): promotion_evidence(root)

    def test_edit_batch_cannot_modify_evaluator_or_partially_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            program = root / 'program.py'
            program.write_text('def run(ctx): pass', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '未授权'):
                apply_files({'files': [{'path': str(program), 'content': 'def run(ctx): ctx.note("changed")'},
                                      {'path': '../evaluation.py', 'content': 'success=True'}]}, root, program)
            self.assertEqual(program.read_text(), 'def run(ctx): pass')
            with self.assertRaisesRegex(ValueError, '私有'):
                apply_files({'files': [{'path': str(program), 'content': 'def run(ctx): ctx._sim.reset()'}]}, root, program)


@unittest.skipUnless(os.getenv('HARNESSPAI_INTEGRATION') == '1' and PYTHON.is_file(), 'Installed upstream + native physics required')
class NativeEvolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_loop_repairs_after_failure_then_checks_unseen_seeds(self):
        """Real official loop and physics; deterministic model transport tests orchestration, not model quality."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            controller = HarnessPAIController()
            async def model(config, prompt, cameras, **kwargs):
                calls.append(prompt)
                self.assertIn('image_base64', prompt)
                self.assertIn('rotation_camera_to_world', prompt)
                self.assertIn('ACTUAL PUBLIC OBSERVATIONS', prompt)
                self.assertGreaterEqual(len(cameras), 3)
                self.assertTrue(all(camera['image_base64'] for camera in cameras))
                allowed = prompt.rsplit('ALLOWED REPLACEMENTS: ', 1)[-1].strip()
                if allowed == 'NONE; files must be []' or '/**/' in allowed:
                    return ProgramEdit(explanation='Raise the TCP using the public interface; native success is required.', files=[]), None
                writes = sum('ALLOWED REPLACEMENTS: NONE' not in p and '/**/*.md' not in p for p in calls)
                source = 'def run(ctx):\n    ctx.note("first candidate deliberately fails the reach task")\n' if writes == 1 else EXAMPLE.read_text(encoding='utf-8')
                return ProgramEdit(explanation='Candidate revision for native validation.', files=[ProgramFile(path=allowed, content=source)]), None
            with patch('server.runlog.RUNS', root), patch('server.harnesspai.LIBRARY', root/'methods/harnesspai'), patch('server.harnesspai.invoke_agent_action', model), patch('server.harnesspai.validate_agent_access'):
                await controller.start(ProgramValidation(name='演化链路回归', source='def run(ctx): pass', mode='evolve',
                    seeds=[42,43], evolution_seeds=[0,1], record_video=False, allow_local_code_execution=True))
                await asyncio.wait_for(controller.task, 100)
            self.assertEqual(controller.state['status'], 'verified', controller.state)
            report = json.loads((controller.log.folder / 'summary.json').read_text(encoding='utf-8'))['harnesspai']
            self.assertFalse(report['training'])
            self.assertEqual(report['batches'][0]['success_rate'], 0)
            self.assertTrue(any('③' in p or 'ANALY' in p.upper() or 'self-review' in p.lower() for p in calls))
            self.assertEqual([row['seed'] for row in report['rows']], [42,43])
            self.assertTrue(all(row['success'] for row in report['rows']))
            self.assertTrue((controller.log.folder/'evolution/transcript.md').exists())
            promotion_evidence(controller.log.folder)

    async def test_empty_edit_stops_before_repeating_unchanged_failed_program(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = HarnessPAIController()
            calls = []
            async def model(config, prompt, cameras, **kwargs):
                calls.append(prompt)
                self.assertEqual(len(cameras), 3)  # Initial observation even for a no-op.
                self.assertIn('host applies files', kwargs['system_prompt'])
                return ProgramEdit(explanation='required service is unavailable', files=[]), None
            with patch('server.runlog.RUNS',root), patch('server.harnesspai.LIBRARY',root/'methods/harnesspai'), patch('server.harnesspai.invoke_agent_action',model), patch('server.harnesspai.validate_agent_access'):
                await controller.start(ProgramValidation(source='def run(ctx): pass', mode='evolve', task_id='lift_red',
                    evolution_seeds=[0], record_video=False, allow_local_code_execution=True))
                await asyncio.wait_for(controller.task, 30)
            self.assertEqual(controller.state['status'], 'error')
            self.assertEqual(len(calls), 1)
            self.assertIn('required service is unavailable', controller.state['message'])
            report=json.loads((controller.log.folder/'summary.json').read_text(encoding='utf-8'))['harnesspai']
            self.assertEqual(len(report['batches']),1)
            self.assertEqual(report['last_model_response']['file_count'],0)
            self.assertEqual(report['rows'],[])

    async def test_model_error_archives_native_baseline_without_leaking_credentials(self):
        from server.agent import AgentConfig
        secret='test-key-not-a-real-credential'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); controller=HarnessPAIController()
            async def failed(*args,**kwargs): raise RuntimeError('provider unavailable: '+secret)
            with patch('server.runlog.RUNS',root), patch('server.harnesspai.LIBRARY',root/'methods/harnesspai'), patch('server.harnesspai.invoke_agent_action',failed), patch('server.harnesspai.validate_agent_access'):
                await controller.start(ProgramValidation(source='def run(ctx): pass',mode='evolve', evolution_seeds=[0],
                    record_video=False,allow_local_code_execution=True,coder=AgentConfig(api_key=secret)))
                await asyncio.wait_for(controller.task,25)
            self.assertEqual(controller.state['status'],'error')
            self.assertIn('provider unavailable',controller.state['message'])
            report=json.loads((controller.log.folder/'summary.json').read_text(encoding='utf-8'))['harnesspai']
            self.assertEqual(len(report['batches']),1)
            for name in ('job.json','summary.json','events.jsonl','runner.log','model-response-1.json'):
                self.assertNotIn(secret,(controller.log.folder/name).read_text(encoding='utf-8'))
