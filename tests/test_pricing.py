"""Price refresh, persisted snapshots and public analysis contracts; no paid calls."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from server import pricing
from server.analysis import analyze_run
from server.runlog import RunLog, CURRENT_LOG, model_output
from server.telemetry import track_inference


def price(provider='openai', amount=2.):
    return {'provider': provider, 'currency': 'USD', 'source_url': pricing.SOURCE_URLS[provider],
            'fetched_at': datetime.now(timezone.utc).isoformat(), 'models': [
                {'model': 'fixture-model', 'input_per_million': amount, 'output_per_million': 10.,
                 'cached_input_per_million': 1.}]}


class PriceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='official-pricing-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = pricing.PriceCatalog(self.root/'catalog.json')
        self.catalog.providers['openai'] = price()
        for target, value in [('server.pricing.catalog', self.catalog), ('server.runlog.RUNS', self.root/'runs')]:
            mocked = patch(target, value); mocked.start(); self.addCleanup(mocked.stop)

    async def test_hourly_refresh_manual_override_failure_and_cached_restart(self):
        calls = []
        async def fetch(provider): calls.append(provider); return price(provider, amount=4.)
        with patch('server.pricing.fetch_official_prices', fetch):
            await self.catalog.refresh()
            await self.catalog.refresh()
            self.assertEqual(len(calls), 3)
            await self.catalog.refresh(force=True)
            self.assertEqual(len(calls), 6)
        previous = self.catalog.providers['openai']['fetched_at']
        async def failure(provider): raise ValueError('Official table fixture changed schema')
        with patch('server.pricing.fetch_official_prices', failure): await self.catalog.refresh(force=True)
        status = self.catalog.status()
        self.assertTrue(all(row['status']=='stale' for row in status['providers']))
        self.assertEqual(self.catalog.providers['openai']['fetched_at'], previous)
        self.assertEqual(self.catalog.providers['openai']['models'][0]['input_per_million'], 4.)
        restarted = pricing.PriceCatalog(self.catalog.path)
        self.assertEqual(restarted.quote({'provider':'openai','model':'fixture-model','base_url':'https://api.openai.com/v1'})['status'], 'stale')
        self.assertIn('schema', restarted.status()['providers'][0]['error'])

    async def test_call_snapshot_and_finish_cost_survive_later_official_price_change(self):
        log = RunLog('agent', {'model':'fixture-model','provider':'openai','settings':{'base_url':'https://api.openai.com/v1'}})
        @track_inference('agent_inference')
        async def boundary(settings):
            model_output({'content':'returned','usage':{'input_tokens':1000,'output_tokens':200,
                'input_tokens_details':{'cached_tokens':100,'cache_write_tokens':0}}})
            raise ValueError('Returned action failed validation')
        token = CURRENT_LOG.set(log)
        try:
            with self.assertRaisesRegex(ValueError, 'validation'):
                await boundary({'provider':'openai','model':'fixture-model','base_url':'https://api.openai.com/v1'})
        finally: CURRENT_LOG.reset(token)
        log.finish({'status':'error'})
        before = (log.folder/'events.jsonl').read_bytes()
        summary = json.loads((log.folder/'summary.json').read_text(encoding='utf-8'))
        self.assertAlmostEqual(summary['cost']['amount_usd'], .0039)
        self.catalog.providers['openai'] = price(amount=100.)
        analyzed = analyze_run(log.id)
        self.assertEqual(analyzed['cost']['basis'], 'recorded_snapshot')
        self.assertAlmostEqual(analyzed['cost']['amount_usd'], .0039)
        self.assertEqual((log.folder/'events.jsonl').read_bytes(), before)
        self.assertEqual(analyzed['rounds'][0]['status'], 'error')

    async def test_legacy_history_uses_current_price_label_without_rewriting_logs(self):
        log = RunLog('agent', {'model':'fixture-model','provider':'openai','settings':{'base_url':'https://api.openai.com/v1'}})
        log.write('model_output', {'content':'legacy','usage':{'input_tokens':1000,'output_tokens':200,
            'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}})
        log.finish({'status':'complete'})
        before = (log.folder/'events.jsonl').read_bytes()
        cost = analyze_run(log.id)['cost']
        self.assertEqual(cost['basis'], 'current_price_estimate')
        self.assertAlmostEqual(cost['amount_usd'], .004)
        self.assertEqual(before, (log.folder/'events.jsonl').read_bytes())

    async def test_parent_cost_counts_children_once_and_keeps_subscription_reference_separate(self):
        children = []
        @track_inference('agent_inference')
        async def boundary(settings):
            model_output({'usage':{'input_tokens':1000,'output_tokens':200,'cached_input_tokens':0,
                'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}})
        for provider in ('openai','codex'):
            log = RunLog('agent', {'model':'fixture-model','provider':provider})
            token = CURRENT_LOG.set(log)
            try: await boundary({'provider':provider,'model':'fixture-model','base_url':'https://api.openai.com/v1'})
            finally: CURRENT_LOG.reset(token)
            log.finish({'status':'complete'});children.append(log)
        parent = RunLog('comparison', {'rows':[{'run_id':children[0].id},{'run_id':children[0].id},{'run_id':children[1].id}]})
        parent.finish({'status':'complete'})
        cost = analyze_run(parent.id)['cost']
        self.assertEqual(cost['total_calls'], 2)
        self.assertAlmostEqual(cost['amount_usd'], .004)
        self.assertAlmostEqual(cost['reference_amount_usd'], .004)
        self.assertEqual(cost['billable_calls'], 1)
        self.assertEqual(cost['reference_calls'], 1)

    async def test_pricing_apis_return_refresh_status_and_reject_remote_mutation(self):
        from server.app import app
        async def fetch(provider): return price(provider)
        with patch('server.app.price_catalog', self.catalog), patch('server.pricing.fetch_official_prices', fetch):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture') as client:
                self.assertEqual((await client.get('/api/pricing')).json()['refresh_interval_s'], 3600)
                blocked = await client.post('/api/pricing/refresh', headers={'Origin':'https://example.org'})
                self.assertEqual(blocked.status_code, 403)
                result = await client.post('/api/pricing/refresh')
                self.assertEqual(result.status_code, 200)
                self.assertTrue(all(row['status']=='ready' for row in result.json()['providers']))

    async def test_damaged_or_missing_call_events_never_become_a_complete_zero(self):
        log = RunLog('agent', {'model':'fixture-model','provider':'openai'})
        log.finish({'status':'error'})
        (log.folder/'events.jsonl').write_text('{broken\n', encoding='utf-8')
        cost = analyze_run(log.id)['cost']
        self.assertIsNone(cost['amount_usd'])
        self.assertEqual(cost['status'], 'unavailable')
        summary_path = log.folder/'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['cost'].update(total_calls=1, priced_calls=1, billable_calls=1, amount_usd=.25, known_amount_usd=.25)
        summary_path.write_text(json.dumps(summary), encoding='utf-8')
        (log.folder/'events.jsonl').unlink()
        cost = analyze_run(log.id)['cost']
        self.assertEqual(cost['amount_usd'], .25)
        self.assertTrue(any('快照' in text for text in cost['warnings']))


if __name__ == '__main__': unittest.main()
