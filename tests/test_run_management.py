"""Names and reversible deletion through public HTTP APIs; no simulation/model jobs."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from server.analysis import analyze_experiments, analyze_run
from server.run_management import batch_delete_runs, record_labels, primary_models, read_run_record
from server.runlog import RunLog, list_runs


class RunManagementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='run-management-fixture-')
        self.root = Path(self.temporary.name)
        self.patch = patch('server.runlog.RUNS', self.root / 'runs'); self.patch.start()
        from server.app import app
        self.app = app
        self.logs = []

    def tearDown(self):
        self.logs.clear()
        self.patch.stop(); self.temporary.cleanup()

    def log(self, model='fixture-primary', *, kind='agent', complete=True, **metadata):
        log = RunLog(kind, {'task': 'management fixture task', 'model': model, **metadata})
        self.logs.append(log)
        if complete:
            log.write('inference', {'call_id': 'fixture-call', 'round': 1, 'status': 'complete',
                'model': model, 'reasoning_effort': 'low', 'input_tokens': 5, 'output_tokens': 3,
                'total_tokens': 8, 'inference_s': .2})
            log.finish({'status': 'complete', 'metrics': {'success': True}})
        return log

    def call(self, method, path, **kwargs):
        async def request():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://127.0.0.1:8000') as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(request())

    def rename(self, log, name):
        return self.call('PATCH', f'/api/runs/{log.id}', json={'name': name})

    def test_default_names_use_primary_model_and_local_start_time_for_all_method_kinds(self):
        agent = self.log('agent-model')
        policy = self.log('policy-model', kind='policy')
        hybrid = self.log(None, kind='composition', recipe={'mode': 'hybrid',
            'agent': {'model': 'planner-model'}, 'policy': {'model': 'candidate-model'},
            'decision': {'model': 'decision-helper'}})
        policy_recipe = self.log(None, kind='composition', recipe={'mode': 'policy',
            'agent': {'model': 'unused-agent'}, 'policy': {'model': 'policy-primary'}})
        for log, primary in ((agent, 'agent-model'), (policy, 'policy-model'), (hybrid, 'planner-model'), (policy_recipe, 'policy-primary')):
            with self.subTest(primary=primary):
                result = self.call('GET', f'/api/runs/{log.id}/analysis').json()
                local_start = datetime.fromisoformat(log.started_at).astimezone().strftime('%Y-%m-%d %H:%M:%S')
                self.assertEqual(result['display_name'], f'{primary} · {local_start}')
                self.assertEqual(result['default_name'], result['display_name'])
                self.assertIsNone(result['custom_name']); self.assertIsNone(result['deleted_at'])
        self.assertEqual(primary_models({'recipe': {'mode': 'hybrid', 'agent': {'model': 'planner'},
            'policy': {'model': 'vla'}, 'decisions': [{'model': 'jev'}]}}), ['planner'])

    def test_declared_batch_models_include_unstarted_methods_and_legacy_child_fallback(self):
        child = self.log('child-planner')
        declared = self.log(None, kind='research_experiment', primary_models=['planned-agent', 'planned-policy', 'planned-agent'],
            rows=[{'run_id': child.id, 'status': 'complete'}, {'run_id': None, 'status': 'error'}])
        fallback = self.log(None, kind='comparison', rows=[{'run_id': child.id}, {'run_id': child.id}])
        self.assertTrue(record_labels(read_run_record(declared.id))['default_name'].startswith('planned-agent + planned-policy · '))
        self.assertTrue(record_labels(read_run_record(fallback.id))['default_name'].startswith('child-planner · '))
        self.assertNotIn('child-planner', record_labels(read_run_record(declared.id))['default_name'])

    def test_legacy_missing_start_time_uses_run_id_utc_in_local_format(self):
        run_id = '20260102T030405-0123456789'
        folder = self.root/'runs'/run_id; folder.mkdir(parents=True)
        record = {'run_id': run_id, 'kind': 'agent', 'model': 'legacy-model', 'status': 'error'}
        (folder/'summary.json').write_text(json.dumps(record), encoding='utf-8')
        expected_time = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')
        listed = self.call('GET', '/api/runs').json()[0]
        self.assertEqual(listed['display_name'], 'legacy-model · ' + expected_time)

    def test_rename_and_blank_reset_change_only_record_json_and_keep_analysis_identity(self):
        log = self.log('stable-model')
        events = (log.folder/'events.jsonl').read_bytes(); summary = (log.folder/'summary.json').read_bytes()
        name = '  第一次具身实验 <script>fixture</script>  '
        renamed = self.rename(log, name)
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()['custom_name'], name.strip())
        analysis = self.call('GET', f'/api/runs/{log.id}/analysis').json()
        self.assertEqual(analysis['display_name'], name.strip()); self.assertEqual(analysis['model'], 'stable-model')
        self.assertEqual(self.call('GET', '/api/runs').json()[0]['display_name'], name.strip())
        self.assertEqual((log.folder/'events.jsonl').read_bytes(), events)
        self.assertEqual((log.folder/'summary.json').read_bytes(), summary)
        self.assertEqual(json.loads((log.folder/'record.json').read_text(encoding='utf-8'))['name'], name.strip())
        reset = self.rename(log, '   ')
        self.assertEqual(reset.status_code, 200); self.assertIsNone(reset.json()['custom_name'])
        self.assertEqual(reset.json()['display_name'], reset.json()['default_name'])
        self.assertEqual((log.folder/'events.jsonl').read_bytes(), events)
        self.assertEqual((log.folder/'summary.json').read_bytes(), summary)

    def test_rename_active_run_survives_finish_and_active_delete_is_blocked(self):
        log = self.log(complete=False)
        self.assertEqual(self.rename(log, 'Named while running').status_code, 200)
        response = self.call('DELETE', f'/api/runs/{log.id}')
        self.assertEqual(response.status_code, 409); self.assertIn('运行', response.json()['detail'])
        self.assertFalse(json.loads((log.folder/'record.json').read_text(encoding='utf-8')).get('deleted_at'))
        log.finish({'status': 'stopped'})
        result = self.call('GET', f'/api/runs/{log.id}/analysis').json()
        self.assertEqual(result['display_name'], 'Named while running'); self.assertEqual(result['status'], 'stopped')
        self.assertEqual(self.call('DELETE', f'/api/runs/{log.id}').status_code, 200)

    def test_recycle_restore_preserves_evidence_parent_references_and_stats_count_once(self):
        child = self.log('cost-model')
        parent = self.log(None, kind='comparison', rows=[{'run_id': child.id, 'status': 'complete', 'seed': 42}])
        (child.folder/'trajectory.mp4').write_bytes(b'opaque-video-retention-fixture')
        (child.folder/'trajectory.jsonl').write_text('{"time":0}\n', encoding='utf-8')
        initial_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in child.folder.iterdir() if path.is_file()}
        registry = self.root/'registry.json'; registry.write_text(json.dumps({'experiment_id': parent.id, 'child_id': child.id}), encoding='utf-8')
        registry_bytes = registry.read_bytes(); parent_summary = (parent.folder/'summary.json').read_bytes()
        before = analyze_experiments(kind='agent', model='cost-model')['groups'][0]
        self.assertEqual(before['runs'], 1); self.assertEqual(before['output_tokens'], 3)
        deleted = self.call('DELETE', f'/api/runs/{child.id}')
        self.assertEqual(deleted.status_code, 200); self.assertIsNotNone(deleted.json()['deleted_at'])
        self.assertNotIn(child.id, [row['run_id'] for row in list_runs()])
        self.assertEqual([row['run_id'] for row in list_runs(deleted=True)], [child.id])
        self.assertEqual(analyze_experiments(kind='agent', model='cost-model')['groups'], [])
        retained_parent = analyze_run(parent.id)
        self.assertEqual(retained_parent['children'][0]['run_id'], child.id)
        self.assertIsNotNone(retained_parent['children'][0]['deleted_at'])
        self.assertEqual(retained_parent['total']['output_tokens'], 3)
        for path_name, digest in initial_hashes.items(): self.assertEqual(hashlib.sha256((child.folder/path_name).read_bytes()).hexdigest(), digest)
        self.assertEqual(registry.read_bytes(), registry_bytes); self.assertEqual((parent.folder/'summary.json').read_bytes(), parent_summary)
        # Direct access still permits auditing/recovery of logically deleted evidence.
        self.assertEqual(self.call('GET', f'/api/runs/{child.id}/download').status_code, 200)
        self.assertEqual(self.call('GET', f'/api/runs/{child.id}/analysis').status_code, 200)
        for _ in range(2): self.assertEqual(self.call('POST', f'/api/runs/{child.id}/restore').status_code, 200)
        restored = analyze_experiments(kind='agent', model='cost-model')['groups'][0]
        self.assertEqual(restored['runs'], 1); self.assertEqual(restored['output_tokens'], 3)
        self.assertEqual(list_runs(deleted=True), [])

    def test_completed_child_is_protected_while_active_parent_still_uses_it(self):
        child = self.log('completed-child')
        row = {'run_id': None, 'status': 'complete'}
        parent = self.log(None, kind='research_experiment', complete=False, rows=[row])
        def snapshot():
            row['run_id'] = child.id
            return {'rows': [row], **parent.timing_snapshot()}
        parent.snapshot_provider = snapshot
        blocked = self.call('DELETE', f'/api/runs/{child.id}')
        self.assertEqual(blocked.status_code, 409)
        self.assertFalse((child.folder/'record.json').exists())
        parent.finish({'status': 'stopped', 'rows': [row]})
        self.assertEqual(self.call('DELETE', f'/api/runs/{child.id}').status_code, 200)

    def test_trash_pagination_applies_after_filter_and_leaves_live_results_visible(self):
        logs = [self.log(f'model-{index}') for index in range(6)]
        for log in logs[:4]: self.assertEqual(self.call('DELETE', f'/api/runs/{log.id}').status_code, 200)
        visible = self.call('GET', '/api/runs?limit=1&offset=1').json()
        trash = self.call('GET', '/api/runs?deleted=true&limit=2&offset=1').json()
        self.assertEqual(len(visible), 1); self.assertEqual(len(trash), 2)
        self.assertTrue(all(row['deleted_at'] for row in trash)); self.assertFalse(visible[0]['deleted_at'])
        self.assertEqual({row['run_id'] for row in self.call('GET', '/api/runs').json()}, {log.id for log in logs[4:]})

    def test_batch_delete_reports_partial_results_without_changing_live_or_parent_evidence(self):
        completed = self.log('batch-completed')
        cancelled = self.log('batch-cancelled', complete=False)
        cancelled.finish({'status': 'cancelled'})
        active = self.log('batch-active', complete=False)
        child = self.log('protected-child')
        parent = self.log(None, kind='comparison', complete=False, rows=[{'run_id': child.id}])
        (completed.folder/'trajectory.mp4').write_bytes(b'batch-video-retention-fixture')
        evidence = {path: path.read_bytes() for log in (completed, cancelled, active, child, parent)
                    for path in log.folder.iterdir() if path.is_file()}
        missing = '20260102T030405-0000000000'
        response = self.call('POST', '/api/runs/batch-delete', json={'run_ids': [
            completed.id, active.id, cancelled.id, child.id, missing, '../outside', completed.id]})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['deleted'], [completed.id, cancelled.id])
        failures = {item['run_id']: item['detail'] for item in result['failed']}
        self.assertEqual(set(failures), {active.id, child.id, missing, '../outside'})
        self.assertIn('正在运行', failures[active.id]); self.assertIn('批量实验', failures[child.id])
        self.assertIn('不存在', failures[missing]); self.assertIn('无效日志编号', failures['../outside'])
        self.assertEqual({row['run_id'] for row in list_runs(deleted=True)}, {completed.id, cancelled.id})
        for path, content in evidence.items(): self.assertEqual(path.read_bytes(), content)
        for log in (active, child, parent): self.assertFalse((log.folder/'record.json').exists())
        self.assertEqual(analyze_experiments(model='batch-completed')['groups'], [])
        for log in (completed, cancelled):
            self.assertEqual(self.call('POST', f'/api/runs/{log.id}/restore').status_code, 200)
        self.assertEqual(list_runs(deleted=True), [])
        self.assertEqual(analyze_experiments(model='batch-completed')['groups'][0]['output_tokens'], 3)

    def test_batch_delete_rejects_invalid_request_before_mutating_records(self):
        log = self.log()
        for value in (None, [], 'not-a-list', [log.id]*1001, [log.id, 42]):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError): batch_delete_runs(value)
                self.assertFalse((log.folder/'record.json').exists())
        for payload in ({'run_ids': []}, {'run_ids': [log.id]*1001}, {'run_ids': [log.id, 42]},
                        {'run_ids': [log.id], 'unknown': True}):
            self.assertEqual(self.call('POST', '/api/runs/batch-delete', json=payload).status_code, 422)
            self.assertFalse((log.folder/'record.json').exists())
        rejected = self.call('POST', '/api/runs/batch-delete', json={'run_ids': [log.id]},
                             headers={'Origin': 'https://attacker.example'})
        self.assertEqual(rejected.status_code, 403)
        self.assertFalse((log.folder/'record.json').exists())

    def test_mutation_rejects_remote_origin_invalid_name_and_path_without_touching_evidence(self):
        log = self.log()
        for method, path, kwargs in (
            ('PATCH', f'/api/runs/{log.id}', {'json': {'name': 'remote'}}),
            ('DELETE', f'/api/runs/{log.id}', {}),
            ('POST', f'/api/runs/{log.id}/restore', {}),
        ):
            with self.subTest(method=method):
                response = self.call(method, path, headers={'Origin': 'https://attacker.example'}, **kwargs)
                self.assertEqual(response.status_code, 403)
        self.assertFalse((log.folder/'record.json').exists())
        for bad in ('a\nb', 'a\x00b', 'a\x7fb'):
            self.assertEqual(self.rename(log, bad).status_code, 400)
        for payload in ({'name': 'x'*161}, {'name': None}, {'name': 42}, {'name': 'ok', 'unknown': True}):
            self.assertEqual(self.call('PATCH', f'/api/runs/{log.id}', json=payload).status_code, 422)
        for invalid in ('invalid-id', '20260102T030405-nothex0000', '%2e%2e%2foutside'):
            # ASGI/client path normalization can land on a read-only route (405);
            # either route rejection or the run-ID validator must prevent mutation.
            self.assertIn(self.call('DELETE', '/api/runs/'+invalid).status_code, (400, 404, 405))
        missing = '20260102T030405-0123456789'
        self.assertEqual(self.call('PATCH', f'/api/runs/{missing}', json={'name': 'missing'}).status_code, 404)
        self.assertFalse((log.folder/'record.json').exists())


if __name__ == '__main__': unittest.main()
