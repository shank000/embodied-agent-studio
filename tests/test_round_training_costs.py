import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import RLock
import unittest
from unittest.mock import patch

from server.rounds import measured_round, round_phase, timeline
from server.runlog import RunLog, CURRENT_LOG, model_output
from server.training_data import export_transitions
from server.experiment_costs import Expense, append_expense, void_expense, attribution
from server.telemetry import track_inference


class TextWorker:
    """Explicit adapter boundary fixture; physics is covered by controller integration tests."""
    def __init__(self):
        self.lock = RLock()
        self.snapshot = {'ready': True, 'time': 0, 'steps': 0, 'backend': {'engine_id': 'alfworld'},
                         'evaluation': {'source': 'alfworld_native_goal', 'success': False}}

    async def execute(self, command):
        if command['action'] == 'pause': return None
        if command['action'] == 'observe':
            return {'time': self.snapshot['time'], 'generation': 1, 'robot': 'text', 'cameras': [],
                    'backend': {'action_mode': 'text'}, 'environment': {'text': 'Public room description'}}
        raise ValueError('Unexpected fixture operation')


class RoundContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = patch('server.runlog.RUNS', self.root); self.patch.start()
        self.addCleanup(self.patch.stop); self.addCleanup(self.tmp.cleanup)
        self.worker = TextWorker()
        from server.simulators import EVALUATORS
        self.worker.snapshot['evaluation']['source'] = EVALUATORS['alfworld'][0]

    async def record(self, log, number, success=False):
        async with measured_round(log, self.worker, number, {'status': 'executing'}):
            saved = log.save_observation(await self.worker.execute({'action': 'observe'}), number)
            log.write('observation', {'round': number, 'observation': saved})
            round_phase(log, 'inference')
            await asyncio.sleep(.002)
            log.write('action', {'round': number, 'action': {'status': 'move', 'action': 'look'}})
            round_phase(log, 'execution')
            self.worker.snapshot['time'] += 1
            self.worker.snapshot['steps'] += 1
            self.worker.snapshot['evaluation']['success'] = success

    async def test_before_after_reward_and_terminal_semantics_are_exported(self):
        log = RunLog('agent', {'task': 'fixture'})
        await self.record(log, 1)
        await self.record(log, 2, True)
        log.finish({'status': 'verified'})
        data = export_transitions(log.id)
        self.assertEqual(data['eligible_transitions'], 2)
        self.assertEqual([row['reward'] for row in data['transitions']], [0., 1.])
        self.assertEqual([row['terminated'] for row in data['transitions']], [False, True])
        self.assertFalse(data['transitions'][-1]['truncated'])
        for row in data['transitions']:
            self.assertGreater(row['next_observation']['time'], row['observation']['time'])
            self.assertAlmostEqual(row['elapsed_s'], sum(row['phase_s'].values()), places=4)
            self.assertNotIn('evaluation', row['observation'])
        self.assertEqual(log.artifacts['observation_count'], 4)

    async def test_failed_first_inference_retains_timing_error_and_safe_partial_export(self):
        log = RunLog('agent', {}, ['secret-fixture'])
        with self.assertRaisesRegex(ValueError, 'failure'):
            async with measured_round(log, self.worker, 1, {}):
                raise ValueError('failure secret-fixture')
        log.finish({'status': 'error'})
        row = timeline(log.id)['rounds'][0]
        self.assertEqual(row['status'], 'error'); self.assertGreater(row['elapsed_s'], 0)
        self.assertNotIn('secret-fixture', json.dumps(row))
        exported = export_transitions(log.id)['transitions'][0]
        self.assertFalse(exported['training_eligible']); self.assertTrue(exported['truncated'])
        self.assertIn('next_observation', exported['missing'])

    async def test_cancelled_round_has_record_without_a_fabricated_response(self):
        log = RunLog('agent', {})
        with self.assertRaises(asyncio.CancelledError):
            async with measured_round(log, self.worker, 1, {}):
                self.assertEqual(timeline(log.id)['rounds'][0]['status'], 'running')
                raise asyncio.CancelledError()
        log.finish({'status': 'stopped'})
        self.assertEqual(timeline(log.id)['rounds'][0]['status'], 'cancelled')
        self.assertEqual(timeline(log.id)['rounds'][0]['calls'], [])

    async def test_cancellation_during_feedback_still_closes_round(self):
        log = RunLog('agent', {})
        original = self.worker.execute
        async def cancelled(command):
            if command['action'] == 'observe': raise asyncio.CancelledError()
            return await original(command)
        self.worker.execute = cancelled
        with self.assertRaises(asyncio.CancelledError):
            async with measured_round(log, self.worker, 1, {}): pass
        log.finish({'status': 'stopped'})
        row = timeline(log.id)['rounds'][0]
        self.assertEqual(row['status'], 'cancelled');self.assertIsNotNone(row['ended_at'])

    async def test_public_apis_paginate_export_and_validate_cost_entries(self):
        import httpx
        from server.app import app
        log = RunLog('agent', {'task': 'fixture'})
        await self.record(log, 1);await self.record(log, 2);log.finish({'status': 'limit'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:8000') as client:
            page = await client.get(f'/api/runs/{log.id}/rounds?limit=1&offset=1')
            self.assertEqual(page.status_code,200);self.assertEqual(page.json()['rounds'][0]['round'],1)
            self.assertEqual(page.json()['total'],2)
            export = await client.get(f'/api/runs/{log.id}/training')
            self.assertEqual(export.status_code,200);self.assertTrue(export.json()['transitions'][-1]['truncated'])
            body={'category':'compute','name':'fixture','quantity':2,'unit':'hour','unit_price_usd':.3,'source':'fixture rate'}
            response=await client.post(f'/api/runs/{log.id}/expenses',json=body)
            self.assertEqual(response.status_code,200)
            cost=(await client.get(f'/api/runs/{log.id}/analysis')).json()['cost_attribution']
            self.assertAlmostEqual(cost['known_total_usd'],.6)
            bad=await client.post(f'/api/runs/{log.id}/expenses',json={**body,'quantity':-2})
            self.assertEqual(bad.status_code,422)
            foreign=await client.post(f'/api/runs/{log.id}/expenses',json=body,headers={'Origin':'https://unrelated.example'})
            self.assertEqual(foreign.status_code,403)
            self.assertEqual((await client.post(f'/api/runs/{log.id}/expenses/{response.json()["id"]}/void')).status_code,200)

    async def test_explicit_costs_void_and_parent_child_attribution_do_not_change_run_log(self):
        log = RunLog('agent', {});log.finish({'status': 'limit'})
        original = log.path.read_bytes()
        entry = append_expense(log.id, Expense(category='compute', name='GPU', quantity=1.5,
                    unit='hour', unit_price_usd=.2, source='explicit fixture rate'))
        api = {'known_amount_usd': .1}
        result = attribution(log.id, [], api)
        self.assertAlmostEqual(result['known_total_usd'], .4)
        parent = RunLog('comparison', {});parent.finish({'status': 'complete'})
        combined = attribution(parent.id, [], api, [{'run_id':log.id,'cost_attribution':result}])
        self.assertAlmostEqual(combined['known_total_usd'], .4)
        void_expense(log.id, entry['id']);void_expense(log.id, entry['id'])
        self.assertEqual(attribution(log.id, [], api)['non_model_amount_usd'], 0)
        self.assertEqual(log.path.read_bytes(), original)
        with self.assertRaises(ValueError): Expense(category='compute', name='x',quantity=-1,unit='hour',unit_price_usd=1,source='x')

    async def test_routed_model_identity_and_public_reasoning_survive_trace(self):
        log = RunLog('agent', {}, ['secret-fixture']);token=CURRENT_LOG.set(log)
        @track_inference('agent')
        async def routed(settings):
            model_output({'model':'actual-model','reasoning_content':'public secret-fixture',
                          'content':'output','usage':{'input_tokens':5,'output_tokens':3}})
        try:
            async with measured_round(log, self.worker, 1, {}):
                await routed({'provider':'compatible','model':'router-alias','base_url':'http://localhost:9/v1'})
        finally: CURRENT_LOG.reset(token)
        log.finish({'status':'limit'})
        call = timeline(log.id)['rounds'][0]['calls'][0]
        self.assertEqual(call['requested_model'],'router-alias'); self.assertEqual(call['model'],'actual-model')
        self.assertEqual(call['output_tokens'],3); self.assertIn('public',call['reasoning']['text'])
        self.assertNotIn('secret-fixture',json.dumps(call))

    async def test_legacy_records_do_not_invent_round_time_rewards_or_next_observations(self):
        log = RunLog('agent', {})
        log.write('observation',{'round':1,'observation':{'time':0}})
        log.write('action',{'round':1,'action':{'status':'done'}})
        log.finish({'status':'model_done'})
        row = export_transitions(log.id)['transitions'][0]
        self.assertIsNone(row['elapsed_s']);self.assertIsNone(row['reward'])
        self.assertFalse(row['training_eligible'])
