"""Harness/model isolation in public settings, credentials and actual trace records."""
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from server.agent import AgentConfig, AgentController
from server.analysis import analyze_run
from server.pricing import PriceCatalog
from server.runlog import CURRENT_LOG, RunLog, model_output
from server.telemetry import track_inference, effective_reasoning
from server.costing import calculate_call_cost


class HarnessAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_metadata_and_aggregate_billing_survive_live_history_and_price_refresh(self):
        usage = {'input_tokens':1000,'output_tokens':100,'cached_input_tokens':0,'cache_write_input_tokens':0,
                 'cache_read_input_tokens':0,'cache_creation_input_tokens':0}
        cli_details = {'cli_num_turns':3,'cli_total_cost_usd':.0042,
                       'cli_model_usage':{'fixture-model':{'inputTokens':800,'outputTokens':80},
                                          'helper-model':{'inputTokens':200,'outputTokens':20}}}
        cases = [('claude_code','anthropic','https://api.anthropic.com',cli_details,False),
                 ('dsh','openai','https://api.openai.com/v1',{'dsh_assistant_messages':2},False),
                 ('dsh','openai','https://api.openai.com/v1',{'dsh_assistant_messages':1},True)]
        with tempfile.TemporaryDirectory() as folder, patch('server.runlog.RUNS',Path(folder)/'runs'):
            for index,(harness,provider,base_url,details,can_price) in enumerate(cases):
                with self.subTest(harness=harness,details=details):
                    catalog = PriceCatalog(Path(folder)/f'prices-{index}.json')
                    settings = {'harness':harness,'provider':provider,'model':'fixture-model','base_url':base_url,
                                'reasoning':'high','reasoning_native':False}
                    # Per-inference identity must win over the main run's harness
                    # (for example a repair or planning call using another CLI).
                    log = RunLog('agent',{'provider':provider,'model':'fixture-model',
                                         'settings':{**settings,'harness':'workbench'}})
                    log.write('observation',{'round':1})
                    @track_inference('agent_inference')
                    async def action(config):
                        model_output({'content':'Fixture action','model':'fixture-model','usage':usage,**details})
                    with patch('server.pricing.catalog',catalog):
                        token = CURRENT_LOG.set(log)
                        try:
                            await action(settings)
                        finally:
                            CURRENT_LOG.reset(token)
                        raw = [json.loads(line)['data'] for line in log.path.read_text(encoding='utf-8').splitlines()
                               if json.loads(line)['event']=='inference']
                        self.assertEqual(len(raw),1)
                        self.assertEqual(raw[0]['harness'],harness)
                        self.assertEqual(raw[0]['reasoning_effort'],'high')
                        self.assertEqual(raw[0]['usage'],usage)
                        for field,value in details.items():
                            self.assertEqual(raw[0][field],value)
                        self.assertIsNone(raw[0]['cost']['amount_usd'])
                        if harness=='claude_code':
                            self.assertEqual(raw[0]['cost']['cli_reported_cost_usd'],.0042)
                        # Simulate the official price becoming available after the
                        # completed call. Reanalysis may price a single dsh request,
                        # but must never interpret an aggregate as one request.
                        catalog.providers[provider] = {'fetched_at':datetime.now(timezone.utc).isoformat(),
                            'models':[{'model':'fixture-model','input_per_million':1.,'output_per_million':2.,
                                       'cached_input_per_million':.5}]}
                        live = analyze_run(log.id)
                        log.finish({'status':'complete'})
                        original = log.path.read_bytes()
                        history = analyze_run(log.id)
                        self.assertEqual(log.path.read_bytes(),original,'History repricing must not rewrite source events')
                        for report in (live,history):
                            call = report['rounds'][0]
                            self.assertEqual(call['harness'],harness)
                            self.assertEqual(report['harness'],harness)
                            for field,value in details.items():
                                self.assertEqual(call[field],value)
                            if can_price:
                                self.assertEqual(report['cost']['status'],'complete')
                                self.assertAlmostEqual(report['cost']['amount_usd'],.0012)
                            else:
                                self.assertIsNone(report['cost']['amount_usd'])
                                self.assertEqual(report['cost']['known_amount_usd'],0.)
                                self.assertEqual(report['cost']['status'],'unavailable')
                                self.assertEqual(call['cost']['priced_calls'],0)
                                self.assertTrue(call['cost']['warnings'])
                            if harness=='claude_code':
                                self.assertEqual(call['cost']['cli_reported_cost_usd'],.0042)
                                self.assertTrue(any('0.004200' in text and '参考' in text for text in report['cost']['warnings']))

    async def test_aggregate_guard_also_applies_when_the_original_price_is_already_available(self):
        quote = {'billing_kind':'api','status':'ready','currency':'USD','provider':'openai',
                 'input_per_million':1.,'cached_input_per_million':.5,'output_per_million':2.}
        usage = {'input_tokens':1000,'cached_input_tokens':0,'cache_write_input_tokens':0,'output_tokens':100}
        for details in ({'harness':'claude_code','cli_num_turns':1,'cli_total_cost_usd':.003},
                        {'harness':'claude_code','cli_num_turns':4,'cli_total_cost_usd':.0042},
                        {'harness':'dsh','dsh_assistant_messages':2}):
            with self.subTest(details=details):
                result = calculate_call_cost({'usage':usage,**details},quote)
                self.assertIsNone(result['amount_usd'])
                self.assertIsNone(result['reference_amount_usd'])
                self.assertEqual(result['known_amount_usd'],0.)
                self.assertEqual(result['priced_calls'],0)
        single = calculate_call_cost({'harness':'dsh','dsh_assistant_messages':1,'usage':usage},quote)
        self.assertEqual(single['status'],'complete')
        self.assertAlmostEqual(single['amount_usd'],.0012)

    async def test_cli_fee_reference_rejects_invalid_or_non_numeric_values(self):
        for invalid in (None,True,-1,float('nan'),float('inf'),'0.001'):
            with self.subTest(value=invalid):
                cost = calculate_call_cost({'harness':'claude_code','cli_total_cost_usd':invalid},{})
                self.assertNotIn('cli_reported_cost_usd',cost)
                self.assertIsNone(cost['amount_usd'])
        zero = calculate_call_cost({'harness':'claude_code','cli_total_cost_usd':0},{})
        self.assertEqual(zero['cli_reported_cost_usd'],0)
        self.assertIsNone(zero['amount_usd'],'A reported zero is still a CLI reference, not an API bill')

    async def test_effective_reasoning_tracks_each_harness_actual_parameter_contract(self):
        for harness,provider,expected in [('claude_code','anthropic','high'),('dsh','anthropic','high'),
                                          ('dsh','compatible','high'),('workbench','anthropic','not_requested'),
                                          ('workbench','compatible','provider_default')]:
            with self.subTest(harness=harness,provider=provider):
                self.assertEqual(effective_reasoning({'harness':harness,'provider':provider,'reasoning':'high',
                                                      'reasoning_native':False,'base_url':'https://router.example/v1'}),expected)
        for harness in ('dsh','workbench'):
            self.assertEqual(effective_reasoning({'harness':harness,'provider':'compatible',
                'base_url':'https://api.deepseek.com','deepseek_thinking':'disabled','reasoning':'high'}),'off')
        self.assertEqual(effective_reasoning({'harness':'claude_code','provider':'anthropic','reasoning':None}),
                         'provider_default')

    async def test_public_config_roundtrip_keeps_harness_model_and_keys_independent(self):
        application = importlib.import_module('server.app')
        controller = AgentController(None)
        base = {'provider': 'openai', 'model': 'fixture-model', 'base_url': 'https://models.example/v1'}
        cases = [
            ({**base, 'harness': 'workbench', 'api_key': 'model-private'}, True, False),
            # Same model API, different harness; its model credential is retained.
            ({**base, 'harness': 'codex_cli', 'api_key': ''}, True, False),
            ({**base, 'harness': 'http', 'harness_url': 'https://harness.example/action',
              'api_key': 'model-private', 'harness_api_key': 'harness-private'}, True, True),
            # Saving empty fields at the same two endpoints preserves both keys.
            ({**base, 'harness': 'http', 'harness_url': 'https://harness.example/action',
              'api_key': '', 'harness_api_key': ''}, True, True),
            # The model may change independently while the harness keeps its auth.
            ({**base, 'provider': 'anthropic', 'model': 'other-model',
              'harness': 'http', 'harness_url': 'https://harness.example/action',
              'api_key': '', 'harness_api_key': ''}, False, True),
        ]
        with patch.object(application, 'agent', controller), \
                patch.object(application, 'comparison', SimpleNamespace(state={'active': False})), \
                patch.object(application, 'research_active', return_value=False), \
                patch('server.agent.codex_executable', return_value='local-fixture'), \
                patch('server.agent.invoke_agent_action', side_effect=AssertionError('Configuration cannot infer')):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                         base_url='http://127.0.0.1:8004') as client:
                for payload, model_key, harness_key in cases:
                    with self.subTest(harness=payload['harness'], provider=payload['provider']):
                        saved = await client.post('/api/agent/config', json=payload)
                        self.assertEqual(saved.status_code, 200, saved.text)
                        loaded = await client.get('/api/agent/config')
                        self.assertEqual(loaded.status_code, 200)
                        self.assertEqual(saved.json(), loaded.json())
                        public = loaded.json()
                        for field in ('harness', 'provider', 'model', 'base_url'):
                            self.assertEqual(public[field], payload[field])
                        self.assertEqual(public['harness_url'], payload.get('harness_url', ''))
                        self.assertEqual(public['has_api_key'], model_key)
                        self.assertEqual(public['has_harness_api_key'], harness_key)
                        self.assertFalse(public['forward_model_key'])
                        for key in ('api_key', 'harness_api_key'): self.assertNotIn(key, public)
                        for secret in ('model-private', 'harness-private'):
                            self.assertNotIn(secret, saved.text); self.assertNotIn(secret, loaded.text)
                        self.assertEqual(bool(controller.config.api_key.get_secret_value()), model_key)
                        self.assertEqual(controller.config.harness_api_key.get_secret_value(),
                                         'harness-private' if harness_key else '')

    async def test_invalid_harness_config_does_not_echo_keys_or_url_query(self):
        application = importlib.import_module('server.app')
        cases = [
            {'harness': 'workbench', 'provider': 'codex'},
            {'harness': 'http', 'provider': 'openai',
             'harness_url': 'https://harness.example/action?key=query-private'},
            {'harness': 'workbench', 'provider': 'openai',
             'base_url': 'https://models.example/v1?key=query-private'},
        ]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                     base_url='http://fixture') as client:
            for config in cases:
                with self.subTest(config=config):
                    response = await client.post('/api/agent/config', json={**config,
                        'api_key': 'model-private', 'harness_api_key': 'harness-private'})
                    self.assertEqual(response.status_code, 422)
                    for secret in ('model-private', 'harness-private', 'query-private'):
                        self.assertNotIn(secret, response.text)
                    self.assertTrue(response.json()['detail'])
                    for detail in response.json()['detail']:
                        self.assertEqual(set(detail), {'loc', 'msg', 'type'})

    async def test_http_harness_usage_stays_unknown_after_live_and_history_repricing(self):
        with tempfile.TemporaryDirectory() as folder, patch('server.runlog.RUNS', Path(folder)/'runs'):
            catalog = PriceCatalog(Path(folder)/'prices.json')
            catalog.providers['openai'] = {'fetched_at': datetime.now(timezone.utc).isoformat(),
                'models': [{'model': 'returned-model', 'input_per_million': 1., 'output_per_million': 2.,
                            'cached_input_per_million': .5}]}
            config = {'harness': 'http', 'harness_url': 'https://harness.example/action',
                      'provider': 'openai', 'model': 'fixture', 'base_url': 'https://api.openai.com/v1'}
            # Repair/planning can use a different harness than the parent run.
            log = RunLog('agent', {'provider': 'openai', 'model': 'fixture',
                                  'settings': {**config, 'harness': 'workbench', 'harness_url': ''}})
            @track_inference('agent_inference')
            async def action(settings):
                model_output({'model': 'returned-model', 'usage': {'input_tokens': 1000, 'output_tokens': 100,
                              'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}},
                              'content': 'public action'})
            with patch('server.pricing.catalog', catalog):
                token = CURRENT_LOG.set(log)
                try: await action({**config, 'api_key': 'model-private', 'harness_api_key': 'harness-private'})
                finally: CURRENT_LOG.reset(token)
                log.finish({'status': 'complete'})
                report = analyze_run(log.id)
            events = (log.folder/'events.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('model-private', events)
            self.assertNotIn('harness-private', events)
            calls = [json.loads(line)['data'] for line in events.splitlines()
                     if json.loads(line)['event'] == 'inference']
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['harness'], 'http')
            self.assertEqual(calls[0]['harness_url'], config['harness_url'])
            self.assertEqual(calls[0]['model'], 'returned-model')
            self.assertIsNone(calls[0]['cost']['amount_usd'])
            self.assertEqual(calls[0]['pricing']['billing_kind'], 'unpriced')
            self.assertIsNone(report['cost']['amount_usd'])
            self.assertEqual(report['cost']['status'], 'unavailable')

    async def test_public_recipe_defaults_exclude_both_keys(self):
        application = importlib.import_module('server.app')
        config = AgentConfig(harness='http', provider='openai', model='fixture',
            base_url='https://api.openai.com/v1', api_key='model-private',
            harness_url='https://harness.example/action', harness_api_key='harness-private')
        with patch.object(application, 'agent', SimpleNamespace(config=config)):
            result = await application.research_defaults()
        self.assertEqual(result['agent']['harness'], 'http')
        self.assertEqual(result['agent']['provider'], 'openai')
        self.assertNotIn('api_key', result['agent'])
        self.assertNotIn('harness_api_key', result['agent'])

    async def test_model_and_harness_runtime_keys_match_their_own_endpoints(self):
        application = importlib.import_module('server.app')
        config = AgentConfig(harness='http', provider='openai', model='fixture',
            base_url='https://api.openai.com/v1', api_key='model-private',
            harness_url='https://harness.example/action', harness_api_key='harness-private')
        captured = {}
        composition = SimpleNamespace(runtime_credentials={}, configure_runtime=lambda name, values: captured.update(values))
        for changes, expected in (({}, {'agent_api_key': 'model-private', 'agent_harness_api_key': 'harness-private'}),
                                  ({'harness_url': 'https://other.example/action'}, {'agent_api_key': 'model-private'}),
                                  ({'base_url': 'https://router.example/v1'}, {'agent_harness_api_key': 'harness-private'})):
            captured.clear()
            recipe = {'agent': {**config.model_dump(exclude={'api_key', 'harness_api_key'}), **changes}}
            with patch.object(application, 'agent', SimpleNamespace(config=config)), \
                    patch.object(application, 'composition', composition), \
                    patch.object(application, 'verify_method', return_value={'recipe': recipe}):
                application.resolve_research_method('fixture')
            self.assertEqual(captured, expected)
