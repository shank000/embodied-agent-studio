"""Frozen research protocols, promotion contracts and real policy episode recording."""
import asyncio
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
import zipfile
from unittest.mock import patch

import httpx

from server import artifacts, research, runlog
from server.artifacts import register_method
from server.composition import CompositionController
from server.runlog import RunLog


class IsolatedResearchCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rsi-contract-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        material_root = self.root / 'materials'
        for path in (Path(__file__).resolve().parents[1] / 'server').glob('*.py'):
            target = material_root / 'server' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        for name in ('requirements.txt', 'requirements-policy.txt', 'package.json'):
            source = Path(__file__).resolve().parents[1] / name
            (material_root / name).write_bytes(source.read_bytes())
        for item in (patch.object(artifacts, 'ROOT', material_root),
                     patch.object(artifacts, 'METHODS', self.root / 'methods'),
                     patch.object(research, 'ROOT', material_root),
                     patch.object(research, 'PROTOCOLS', self.root / 'protocols'),
                     patch.object(research, 'REGISTRY', self.root / 'methods' / 'registry.json'),
                     patch.object(runlog, 'RUNS', self.root / 'runs')):
            item.start()
            self.addCleanup(item.stop)

    def protocol(self, **overrides):
        return research.register_protocol({'name': 'statistics fixture; no learned-policy claim',
            'robot': 'so101', 'scene_id': 'tabletop', 'task_id': 'reach_up',
            'development_seeds': [42], 'validation_seeds': [43, 44, 45], 'test_seeds': [46, 47, 48],
            'min_pairs': 3, 'max_sim_time_s': 1, 'max_wall_time_s': 30, 'max_steps': 1,
            **overrides})

    def family(self, endpoint='http://127.0.0.1:9/policy'):
        parent = register_method({'name': 'parent HTTP contract fixture', 'mode': 'policy',
            'policy': {'kind': 'vla', 'model': 'local-contract-fixture', 'endpoint': endpoint,
                       'horizon': 3, 'dt_s': .05}})
        child = register_method({'name': 'child HTTP contract fixture', 'mode': 'policy',
            'parent_id': parent['id'], 'change_summary': 'Change the declared horizon.',
            'policy': {**parent['recipe']['policy'], 'horizon': 4}})
        return parent, child

    def paired_rows(self, parent, child, seeds, parent_success=False, child_success=True):
        return [{'method_id': method['id'], 'method_name': method['recipe']['name'], 'seed': seed,
                 'initial_state_id': f'statistics-only-state-{seed}', 'status': 'limit',
                 'metrics': {'success': success, 'wall_time_s': .1}}
                for seed in seeds for method, success in ((parent, parent_success), (child, child_success))]

    def experiment_summary(self, protocol, rows, split='validation', status='complete'):
        log = RunLog('research_experiment', {'task': 'statistics-only fixture',
            'protocol_id': protocol['protocol_id'], 'split': split})
        log.finish({'status': status, 'rows': rows})
        return log


class ResearchContractTests(IsolatedResearchCase):
    def test_protocol_splits_are_disjoint_and_identity_is_immutable(self):
        for invalid in ({'validation_seeds': [42, 43]}, {'test_seeds': [43, 99]},
                        {'development_seeds': [42, 42]}, {'test_seeds': [-1]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): self.protocol(**invalid)
        original = self.protocol()
        self.assertEqual(self.protocol(), original)
        changed = self.protocol(max_sim_time_s=2)
        self.assertNotEqual(original['protocol_id'], changed['protocol_id'])
        self.assertEqual(research.get_protocol(original['protocol_id']), original)
        path = research.PROTOCOLS / (original['protocol_id'] + '.json')
        tampered = copy.deepcopy(original)
        tampered['spec']['validation_seeds'] = [53, 54, 55]
        path.write_text(json.dumps(tampered), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '编号'): research.get_protocol(original['protocol_id'])

    def test_changed_physics_contract_rejects_old_protocol_before_allocating_trials(self):
        async def check():
            protocol = self.protocol()
            parent, child = self.family()
            path = research.ROOT / 'server' / 'scenes.py'
            path.write_text(path.read_text(encoding='utf-8') + '\n# changed evaluation fixture\n', encoding='utf-8')
            runner = type('IdleRunner', (), {'snapshot': lambda _: {'active': False}})()
            controller = research.ResearchController(None, runner)
            with self.assertRaisesRegex(ValueError, '环境|评价器'):
                await controller.start({'protocol_id': protocol['protocol_id'],
                                        'method_ids': [parent['id'], child['id']]})
            self.assertFalse(controller.state['active'])
            self.assertEqual(controller.state['rows'], [])
        asyncio.run(check())

    def test_all_assigned_rows_remain_in_denominator_including_errors_and_pending(self):
        rows = [{'method_id': 'fixture', 'method_name': 'fixture', 'status': status,
                 'metrics': {'success': True} if status != 'queued' else None}
                for status in ('verified', 'error', 'stopped', 'queued', 'running')]
        result = research.aggregate_rows(rows, ['fixture'])[0]
        self.assertEqual(result['assigned'], 5)
        self.assertEqual(result['completed'], 3)
        self.assertEqual(result['successes'], 1)
        self.assertEqual(result['success_rate'], .2)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['stopped'], 1)

    def test_promotion_requires_validation_direct_parent_complete_matching_pairs_and_thresholds(self):
        protocol = self.protocol(min_success_delta=.2)
        parent, child = self.family()
        rows = self.paired_rows(parent, child, protocol['spec']['validation_seeds'])
        unrelated = register_method({'name': 'unrelated policy fixture', 'mode': 'policy', 'policy': parent['recipe']['policy']})
        log = self.experiment_summary(protocol, rows)
        with self.assertRaisesRegex(ValueError, '直接子版本'):
            research.promote(log.id, unrelated['id'], parent['id'])
        for split in ('development', 'test'):
            wrong_split = self.experiment_summary(protocol, rows, split=split)
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, '验证集'):
                research.promote(wrong_split.id, child['id'], parent['id'])
        missing = rows[:-1]
        duplicated = rows + [copy.deepcopy(rows[-1])]
        mismatched = copy.deepcopy(rows)
        mismatched[-1]['initial_state_id'] = 'different-physical-state'
        pending = copy.deepcopy(rows)
        pending[-1]['status'] = 'stopped'
        tied = self.paired_rows(parent, child, protocol['spec']['validation_seeds'], child_success=False)
        extra_error = copy.deepcopy(rows)
        extra_error[-1].update(status='error', metrics={'success': False})
        for case, invalid in (('missing pair', missing), ('duplicate pair', duplicated),
                              ('different initial state', mismatched), ('stopped trial', pending),
                              ('no improvement', tied), ('extra error', extra_error)):
            rejected = self.experiment_summary(protocol, invalid)
            with self.subTest(case=case), self.assertRaises(ValueError):
                research.promote(rejected.id, child['id'], parent['id'])
        self.assertIsNone(research.read_registry()['active_method_id'])

    def test_promotion_and_rollback_append_history_without_overwriting_trial_results(self):
        protocol = self.protocol()
        parent, child = self.family()
        log = self.experiment_summary(protocol, self.paired_rows(parent, child, protocol['spec']['validation_seeds']))
        before = (log.folder / 'summary.json').read_bytes()
        promoted = research.promote(log.id, child['id'], parent['id'])
        self.assertEqual(promoted['active_method_id'], child['id'])
        record = promoted['history'][0]
        self.assertEqual(record['paired_success_delta'], 1.)
        self.assertEqual(record['acceptance_status'], 'exploratory')
        self.assertEqual((log.folder / 'summary.json').read_bytes(), before)
        self.assertEqual(json.loads((log.folder / 'selection.jsonl').read_text(encoding='utf-8'))['method_id'], child['id'])
        rolled_back = research.rollback()
        self.assertEqual(rolled_back['active_method_id'], parent['id'])
        self.assertEqual(rolled_back['history'][:1], promoted['history'])
        self.assertEqual([item['action'] for item in rolled_back['history']], ['promote', 'rollback'])
        self.assertEqual((log.folder / 'summary.json').read_bytes(), before)


class PolicyBoundary(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        controls = body['observation']['actuators']
        targets = [item['target'] for item in controls]
        targets[0] += .025
        response = {'protocol': 'embodied-policy-v1', 'action_type': 'actuator_targets',
                    'joint_names': [item['name'] for item in controls], 'actions': [targets] * 3,
                    'dt_s': .05, 'summary': 'Local HTTP contract fixture; not a trained VLA.'}
        raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(raw)


class ResearchIntegrationTests(IsolatedResearchCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = importlib.import_module('server.app')
        cls.worker = cls.app_module.Worker()
        cls.worker.thread.start()
        if not cls.worker.ready.wait(30) or cls.worker.failure:
            raise RuntimeError(cls.worker.failure or 'Native MuJoCo worker did not start.')
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), PolicyBoundary)
        cls.server.requests = []
        cls.http_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.http_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.http_thread.join(2)
        cls.worker.shutdown.set()
        cls.worker.thread.join(5)

    def app_components(self, composition=None, experiment=None):
        app = self.app_module
        agent = app.AgentController(self.worker)
        policy = app.PolicyController(self.worker)
        composition = composition or CompositionController(self.worker, method_resolver=artifacts.get_recipe)
        experiment = experiment or research.ResearchController(self.worker, composition)
        return patch.multiple(app, worker=self.worker, agent=agent, policy=policy,
                              comparison=app.ComparisonController(self.worker, agent, policy),
                              composition=composition, research=experiment)

    def test_two_policy_versions_share_real_initial_state_and_episode_can_replay_via_api(self):
        async def check():
            endpoint = f'http://127.0.0.1:{self.server.server_port}/policy'
            parent, child = self.family(endpoint)
            protocol = self.protocol()
            runner = CompositionController(self.worker, method_resolver=artifacts.get_recipe)
            controller = research.ResearchController(self.worker, runner)
            await controller.start({'protocol_id': protocol['protocol_id'], 'method_ids': [parent['id'], child['id']],
                                    'record_video': True})
            await asyncio.wait_for(controller.task, 30)
            self.assertEqual(controller.state['status'], 'complete', controller.state)
            rows = controller.state['rows']
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['initial_state_id'], rows[1]['initial_state_id'])
            self.assertTrue(all(row['status'] == 'limit' for row in rows), rows)
            self.assertTrue(all(row['metrics']['steps'] == 1 for row in rows))
            self.assertTrue(all(group['assigned'] == 1 for group in controller.state['aggregates']))
            for trial in rows:
                self.assertEqual(trial['video']['status'], 'saved', trial)
                self.assertGreater(trial['video']['frame_count'], 1)
                self.assertEqual(trial['video']['fps'], 20)
            for body in self.server.requests[-2:]:
                self.assertEqual([camera['name'] for camera in body['observation']['cameras']], ['front', 'side', 'wrist'])
                self.assertNotIn('blocks', body['observation'])
                self.assertEqual(len(body['observation']['actuators']), 6)
            row = rows[0]
            folder = runlog.run_folder(row['run_id'])
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            self.assertGreater(summary['artifacts']['trajectory_frames'], 10)
            self.assertEqual(len(list((folder / 'observations').glob('*/*.jpg'))), 6)  # Before and after images for training transitions.
            frames = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertGreater(abs(frames[-1]['qpos'][0] - frames[0]['qpos'][0]), .001)
            with self.app_components(runner, controller):
                transport = httpx.ASGITransport(app=self.app_module.app)
                async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1:8000') as client:
                    episode = await client.get(f'/api/runs/{row["run_id"]}/episode')
                    self.assertEqual(episode.status_code, 200)
                    self.assertTrue(episode.json()['available'])
                    video = episode.json()['video']
                    self.assertEqual(video, summary['artifacts']['video'])
                    clip = await client.get(f'/api/runs/{row["run_id"]}/artifact/{video["path"]}')
                    self.assertEqual(clip.status_code, 200)
                    self.assertEqual(clip.headers['content-type'], 'video/mp4')
                    self.assertIn(b'ftyp', clip.content[:32])
                    partial = await client.get(f'/api/runs/{row["run_id"]}/artifact/{video["path"]}',
                                               headers={'Range': 'bytes=0-127'})
                    self.assertEqual(partial.status_code, 206)
                    self.assertEqual(partial.content, clip.content[:128])
                    bundle = await client.get(f'/api/runs/{row["run_id"]}/bundle')
                    self.assertEqual(bundle.status_code, 200)
                    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
                        self.assertEqual(archive.read(video['path']), clip.content)
                    replay = await client.post('/api/research/replay', json={'run_id': row['run_id'], 'frame_index': len(frames) - 1})
                    self.assertEqual(replay.status_code, 200, replay.text)
                    self.assertTrue(replay.json()['ok'])
            with self.worker.lock:
                self.assertAlmostEqual(self.worker.snapshot['actuators'][0]['actual'], frames[-1]['qpos'][0], places=9)
                self.assertTrue(self.worker.snapshot['paused'])
        asyncio.run(check())

    def test_preallocated_rows_and_stopping_keep_all_trial_denominators(self):
        async def check():
            parent, child = self.family()
            protocol = self.protocol()
            class WaitingRunner:
                def __init__(self):
                    self.task = None
                    self.entered = asyncio.Event()
                def snapshot(self): return {'active': bool(self.task and not self.task.done())}
                async def start(self, request):
                    async def wait():
                        self.entered.set()
                        await asyncio.Event().wait()
                    self.task = asyncio.create_task(wait())
                async def stop(self):
                    if self.task:
                        self.task.cancel()
                        try: await self.task
                        except asyncio.CancelledError: pass
            runner = WaitingRunner()
            controller = research.ResearchController(self.worker, runner)
            state = await controller.start({'protocol_id': protocol['protocol_id'], 'split': 'validation',
                                            'method_ids': [parent['id'], child['id']]})
            self.assertEqual(len(state['rows']), 6)
            self.assertTrue(all(row['status'] == 'queued' for row in state['rows']))
            self.assertTrue(all(group['assigned'] == 3 for group in state['aggregates']))
            await asyncio.wait_for(runner.entered.wait(), 10)
            await controller.stop()
            self.assertEqual(controller.state['status'], 'stopped', controller.state)
            self.assertTrue(all(row['status'] == 'stopped' for row in controller.state['rows']))
            self.assertTrue(all(group['assigned'] == 3 and group['success_rate'] == 0 for group in controller.state['aggregates']))
        asyncio.run(check())

    def test_stopping_a_video_trial_retains_its_episode_link_and_playable_clip(self):
        async def check():
            parent, child = self.family()
            protocol = self.protocol()
            runner = CompositionController(self.worker, method_resolver=artifacts.get_recipe)
            controller = research.ResearchController(self.worker, runner)
            entered = asyncio.Event()
            async def pending_policy(*args):
                entered.set()
                await asyncio.Event().wait()
            with patch('server.composition.predict_policy', pending_policy):
                await controller.start({'protocol_id': protocol['protocol_id'],
                    'method_ids': [parent['id'], child['id']], 'record_video': True})
                await asyncio.wait_for(entered.wait(), 15)
                await controller.stop()
            self.assertEqual(controller.state['status'], 'stopped')
            recorded = [row for row in controller.state['rows'] if row.get('run_id')]
            self.assertEqual(len(recorded), 1, controller.state)
            trial = recorded[0]
            self.assertEqual(trial['status'], 'stopped')
            self.assertEqual(trial['video']['status'], 'saved')
            folder = runlog.run_folder(trial['run_id'])
            self.assertTrue((folder / trial['video']['path']).is_file())
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['artifacts']['video'], trial['video'])
            self.assertEqual(summary['status'], 'stopped')
            self.assertIsNone(self.worker.recorder)
            self.assertTrue(all(group['assigned'] == 1 and group['success_rate'] == 0
                                for group in controller.state['aggregates']))
        asyncio.run(check())

    def test_api_artifacts_reject_escape_and_unfinished_or_legacy_replay(self):
        async def check():
            unfinished = RunLog('fixture', {'task': 'unfinished fixture'})
            await self.worker.execute({'action': 'stop'})
            unfinished.save_checkpoint(await self.worker.execute({'action': 'checkpoint'}))
            unfinished.write_trajectory({'index': 0, 'time': 0., 'integration_state': []})
            unfinished.flush_trajectory()
            outside = runlog.RUNS / 'outside.json'
            outside.write_text('{"private":"outside-fixture-secret"}', encoding='utf-8')
            legacy = RunLog('fixture', {'task': 'old observation-only record'})
            legacy.finish({'status': 'complete'})
            try:
                with self.app_components():
                    transport = httpx.ASGITransport(app=self.app_module.app)
                    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1:8000') as client:
                        valid = await client.get(f'/api/runs/{unfinished.id}/artifact/checkpoints/initial.json')
                        self.assertEqual(valid.status_code, 200)
                        for suffix in ('%2E%2E/outside.json', '%2E%2E%5Coutside.json'):
                            response = await client.get(f'/api/runs/{unfinished.id}/artifact/{suffix}')
                            self.assertIn(response.status_code, (400, 404))
                            self.assertNotIn('outside-fixture-secret', response.text)
                        for log in (unfinished, legacy):
                            episode = await client.get(f'/api/runs/{log.id}/episode')
                            self.assertFalse(episode.json()['available'])
                            response = await client.post('/api/research/replay', json={'run_id': log.id})
                            self.assertEqual(response.status_code, 400, response.text)
                        response = await client.get(f'/api/runs/{unfinished.id}/bundle')
                        self.assertEqual(response.status_code, 409)
            finally:
                unfinished.finish({'status': 'stopped'})
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
