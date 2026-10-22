"""Metadata-only HTTP fixtures; never send inference or inspect login files."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from server.model_capabilities import (cached_model_capabilities, discover_model_capabilities,
                                       parse_model_capabilities)


class MetadataFixture(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        self.server.requests.append((self.command, self.path, dict(self.headers)))
        status, body = self.server.responses.get(self.path, (404, {'error': 'not found'}))
        encoded = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        self.server.requests.append((self.command, self.path, dict(self.headers)))
        self.send_response(500); self.end_headers()


class ModelCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), MetadataFixture)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(2)

    def setUp(self):
        self.server.requests = []; self.server.responses = {}
        self.cache_patch = patch('server.model_capabilities._DISCOVERED', {}); self.cache_patch.start()

    def tearDown(self): self.cache_patch.stop()

    def config(self, provider='compatible', model='fixture-model', **changes):
        return {'provider': provider, 'model': model, 'base_url': self.base+'/v1',
                'api_key': SecretStr('fixture-key'), **changes}

    def discover(self, config): return asyncio.run(discover_model_capabilities(config))

    def redirect_official_to_local(self):
        native_client = httpx.AsyncClient
        local = self.base
        class RedirectFixtureClient(native_client):
            async def get(self, url, **kwargs):
                return await super().get(local+urlsplit(url).path, **kwargs)
        return patch('server.model_capabilities.httpx.AsyncClient', RedirectFixtureClient)

    def test_codex_exact_slug_uses_only_models_cache_and_keeps_model_specific_levels(self):
        with tempfile.TemporaryDirectory(prefix='codex-model-metadata-fixture-') as folder:
            path = Path(folder)/'models_cache.json'
            path.write_text(json.dumps({'models': [
                {'slug': 'fixture-astra', 'supported_reasoning_levels': [{'effort': level} for level in ('low', 'max', 'ultra')], 'default_reasoning_level': 'max'},
                {'slug': 'fixture-luna', 'supported_reasoning_levels': [{'effort': 'low'}], 'default_reasoning_level': 'low'},
            ]}), encoding='utf-8')
            (Path(folder)/'auth.json').write_text('login-must-not-be-read', encoding='utf-8')
            read_text = Path.read_text
            def guarded_read(path, *args, **kwargs):
                self.assertEqual(path.name, 'models_cache.json')
                return read_text(path, *args, **kwargs)
            with patch.dict(os.environ, {'CODEX_HOME': folder}), patch.object(Path, 'read_text', guarded_read):
                config = self.config(provider='codex', model='fixture-astra')
                result = self.discover(config)
                self.assertEqual(result['levels'], ['low', 'max', 'ultra'])
                self.assertEqual(result['default_level'], 'max'); self.assertTrue(result['verified'])
                self.assertEqual(cached_model_capabilities(config), result)
                result = self.discover(self.config(provider='codex', model='fixture-astra-suffix'))
                self.assertIsNone(result['levels']); self.assertFalse(result['verified'])
                self.assertEqual(self.discover(self.config(provider='codex', model='fixture-luna'))['levels'], ['low'])
        self.assertEqual(self.server.requests, [])

    def test_codex_missing_or_invalid_cache_preserves_real_unavailable_cause(self):
        with tempfile.TemporaryDirectory(prefix='missing-codex-models-') as folder, patch.dict(os.environ, {'CODEX_HOME': folder}):
            self.assertIn('不存在', self.discover(self.config(provider='codex'))['message'])
            (Path(folder)/'models_cache.json').write_text('{invalid', encoding='utf-8')
            result = self.discover(self.config(provider='codex'))
            self.assertIn('有效 JSON', result['message']); self.assertIsNone(result['levels'])

    def test_official_deepseek_reads_list_effort_and_default_without_global_enum(self):
        self.server.responses['/v1/models'] = (200, {'data': [
            {'id': 'deepseek-other', 'effort': {'supported_levels': ['medium'], 'default_level': 'medium'}},
            {'id': 'deepseek-flash', 'effort': {'supported_levels': ['low', 'high', 'max'], 'default_level': 'high'}},
        ]})
        with self.redirect_official_to_local():
            result = self.discover(self.config(model='deepseek-flash', base_url='https://api.deepseek.com/v1'))
        self.assertEqual(result['levels'], ['low', 'high', 'max']); self.assertNotIn('medium', result['levels'])
        self.assertEqual(result['default_level'], 'high'); self.assertEqual(result['thinking_modes'], ['enabled', 'disabled'])
        self.assertEqual(result['source'], 'deepseek_models_api'); self.assertTrue(result['verified'])
        self.assertEqual([item[:2] for item in self.server.requests], [('GET', '/v1/models')])

    def test_claude_effort_and_thinking_flags_come_from_retrieve_metadata(self):
        self.server.responses['/v1/models/claude-fixture'] = (200, {'id': 'claude-resolved-version', 'capabilities': {
            'effort': {'supported': True, 'low': {'supported': True}, 'medium': {'supported': True},
                       'high': {'supported': False}, 'xhigh': {'supported': True}, 'max': {'supported': True}},
            'thinking': {'supported': True, 'types': {'adaptive': {'supported': True}, 'disabled': {'supported': True}, 'enabled': {'supported': False}}}}})
        result = self.discover(SimpleNamespace(**self.config(provider='anthropic', model='claude-fixture', base_url=self.base)))
        self.assertEqual(result['model'], 'claude-fixture')
        self.assertEqual(result['levels'], ['low', 'medium', 'xhigh', 'max'])
        self.assertEqual(result['thinking_modes'], ['adaptive', 'disabled']); self.assertIsNone(result['default_level'])
        self.assertTrue(result['reasoning_supported']); self.assertTrue(result['verified'])
        self.assertEqual(self.server.requests[0][2]['x-api-key'], 'fixture-key')
        self.assertEqual(self.server.requests[0][2]['anthropic-version'], '2023-06-01')

    def test_compatible_service_can_declare_new_levels_and_sync_cache_is_isolated(self):
        self.server.responses['/v1/models/fixture-model'] = (200, {'id': 'fixture-model', 'effort': {
            'supported_levels': ['vendor_fast', 'vendor_deliberate'], 'default_level': 'vendor_fast'}})
        config = self.config()
        result = self.discover(config)
        self.assertEqual(result['levels'], ['vendor_fast', 'vendor_deliberate'])
        result['levels'].append('mutated-client')
        cached = cached_model_capabilities(config)
        self.assertEqual(cached['levels'], ['vendor_fast', 'vendor_deliberate'])
        self.assertIsNone(cached_model_capabilities({**config, 'api_key': 'different-account'})['levels'])
        self.assertEqual(len(self.server.requests), 1)

    def test_unknown_proxy_model_name_does_not_inherit_openai_or_deepseek_capabilities(self):
        for provider, model in (('openai', 'gpt-6-astra'), ('compatible', 'deepseek-flash'), ('custom', 'gpt-5.6-sol')):
            with self.subTest(provider=provider, model=model):
                self.server.responses['/v1/models/'+model] = (200, {'id': model, 'owned_by': 'openai', 'display_name': 'Reasoning'} )
                result = self.discover(self.config(provider=provider, model=model))
                self.assertIsNone(result['levels']); self.assertIsNone(result['reasoning_supported'])
                self.assertFalse(result['verified']); self.assertIn('手动', result['message'])

    def test_models_list_fallback_selects_exact_id_instead_of_other_model(self):
        self.server.responses['/v1/models'] = (200, {'data': [
            {'id': 'fixture-model-extra', 'effort': {'supported_levels': ['max']}},
            {'id': 'fixture-model', 'supported_reasoning_levels': ['minimal', 'high'], 'default_reasoning_level': 'minimal'},
        ]})
        result = self.discover(self.config())
        self.assertEqual(result['levels'], ['minimal', 'high'])
        self.assertEqual([item[1] for item in self.server.requests], ['/v1/models/fixture-model', '/v1/models'])
        self.server.responses['/v1/models'] = (200, {'data': [{'id': 'fixture-model-extra', 'reasoning_levels': ['high']}]})
        self.assertIsNone(self.discover(self.config())['levels'])

    def test_openai_exact_official_profile_does_not_copy_codex_ultra_or_invent_default(self):
        config = self.config(provider='openai', model='gpt-6-astra', base_url='https://api.openai.com/v1', api_key='')
        result = self.discover(config)
        self.assertEqual(result['levels'], ['low', 'medium', 'high', 'xhigh', 'max'])
        self.assertIsNone(result['default_level']); self.assertNotIn('ultra', result['levels'])
        self.assertEqual(result['source'], 'official_docs_2026-10-22'); self.assertIn('未提供 API Key', result['message'])
        self.assertEqual(cached_model_capabilities({**config, 'model': 'gpt-5.6-sol'})['default_level'], 'medium')
        self.assertIsNone(cached_model_capabilities({**config, 'model': 'gpt-6-astra-unknown-suffix'})['levels'])
        self.assertEqual(self.server.requests, [])

    def test_openai_queries_metadata_before_labelled_documentation_fallback(self):
        self.server.responses['/v1/models/gpt-6-astra'] = (200, {'id': 'gpt-6-astra', 'object': 'model'})
        config = self.config(provider='openai', model='gpt-6-astra', base_url='https://api.openai.com/v1')
        with self.redirect_official_to_local(): result = self.discover(config)
        self.assertEqual(result['source'], 'official_docs_2026-10-22')
        self.assertIn('没有返回推理强度字段', result['message'])
        self.server.responses['/v1/models/gpt-6-astra'] = (200, {'effort': {'supported_levels': ['high', 'new_level'], 'default_level': 'new_level'}})
        with self.redirect_official_to_local(): result = self.discover(config)
        self.assertEqual(result['levels'], ['high', 'new_level']); self.assertEqual(result['source'], 'service_metadata')

    def test_http_failures_bad_json_and_timeout_are_truthful_and_never_echo_secrets(self):
        for status in (401, 403, 500):
            self.server.responses['/v1/models/fixture-model'] = (status, {'error': 'raw-body fixture-key https://secret.example/?token=fixture-key'})
            result = self.discover(self.config())
            self.assertIn('HTTP '+str(status), result['message']); self.assertFalse(result['verified'])
            self.assertNotIn('fixture-key', json.dumps(result)); self.assertNotIn('secret.example', json.dumps(result))
        self.server.responses['/v1/models/fixture-model'] = (200, 'not-json fixture-key')
        self.assertIn('有效 JSON', self.discover(self.config())['message'])
        native = httpx.AsyncClient
        def transport(request): raise httpx.ReadTimeout('sensitive fixture-key https://private/?key=fixture-key', request=request)
        with patch('server.model_capabilities.httpx.AsyncClient', side_effect=lambda **kwargs: native(**kwargs, transport=httpx.MockTransport(transport))):
            result = self.discover(self.config())
        self.assertIn('超过', result['message']); self.assertNotIn('fixture-key', json.dumps(result))

    def test_invalid_url_or_metadata_cannot_expose_credentials_or_create_fake_options(self):
        for value in ('https://user:fixture-key@example.org/v1', 'https://example.org/v1?key=fixture-key',
                      'http://remote.example/v1', 'https://example.org:bad/v1'):
            result = self.discover(self.config(base_url=value))
            self.assertFalse(result['verified']); self.assertNotIn('fixture-key', json.dumps(result))
        self.assertEqual(self.server.requests, [])
        self.server.responses['/v1/models/fixture-model'] = (200, {'effort': {'supported_levels': ['fixture-key']}})
        result = self.discover(self.config())
        self.assertFalse(result['verified']); self.assertNotIn('fixture-key', json.dumps(result))
        parsed = parse_model_capabilities({'effort': {'supported_levels': ['low', {'invalid': 'value'}], 'default_level': 'high'}}, provider='custom', model='literal-label')
        self.assertIsNone(parsed['levels']); self.assertIsNone(parsed['default_level']); self.assertFalse(parsed['verified'])

    def test_explicit_unsupported_is_distinct_from_unknown_and_no_default_is_fabricated(self):
        for metadata in ({'reasoning_supported': False}, {'reasoning': {'supported': False}}):
            parsed = parse_model_capabilities(metadata, provider='custom', model='no-reasoning')
            self.assertEqual(parsed['levels'], []); self.assertTrue(parsed['verified'])
        parsed = parse_model_capabilities({'thinking': {'supported': False}}, provider='custom', model='no-thinking')
        self.assertIsNone(parsed['levels'], 'Thinking being disabled does not itself declare effort unsupported.')
        parsed = parse_model_capabilities({'capabilities': {'effort': {'supported': False}, 'thinking': {'supported': False}}}, provider='anthropic', model='no-thinking')
        self.assertEqual(parsed['levels'], []); self.assertFalse(parsed['reasoning_supported']); self.assertTrue(parsed['verified'])
        parsed = parse_model_capabilities({'effort': {'supported_levels': ['low'], 'default_level': 'high'}}, provider='custom', model='literal-label')
        self.assertEqual(parsed['levels'], ['low']); self.assertIsNone(parsed['default_level'])
        parsed = parse_model_capabilities({'capabilities': {'reasoning': {'effort': {'supported_levels': ['vendor_max'], 'default_level': 'vendor_max'}}}}, provider='custom', model='literal-label')
        self.assertEqual(parsed['levels'], ['vendor_max']); self.assertEqual(parsed['default_level'], 'vendor_max')
        parsed = parse_model_capabilities({'levels': ['light', 'deep'], 'default_level': 'deep'}, provider='custom', model='literal-label')
        self.assertEqual(parsed['levels'], ['light', 'deep']); self.assertEqual(parsed['default_level'], 'deep')

    def test_capabilities_api_reuses_key_only_for_same_provider_endpoint_and_checks_origin(self):
        import server.app as application
        from server.agent import AgentConfig
        self.server.responses['/v1/models/fixture-model'] = (200, {'effort': {'supported_levels': ['vendor_low', 'vendor_max']}})
        existing = AgentConfig(**self.config())
        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url='http://127.0.0.1:8000') as client:
                body = {'provider': 'compatible', 'model': 'fixture-model', 'base_url': self.base+'/v1', 'api_key': ''}
                with patch.object(application.agent, 'config', existing):
                    response = await client.post('/api/agent/model-capabilities', json=body)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()['levels'], ['vendor_low', 'vendor_max'])
                    self.assertEqual(self.server.requests[-1][2]['Authorization'], 'Bearer fixture-key')
                    self.assertNotIn('fixture-key', response.text)
                    await client.post('/api/agent/model-capabilities', json={**body, 'base_url': self.base+'/different'})
                    self.assertTrue(all('Authorization' not in headers for _, path, headers in self.server.requests if path.startswith('/different')))
                    await client.post('/api/agent/model-capabilities', json={**body, 'provider': 'custom'})
                    self.assertNotIn('Authorization', self.server.requests[-1][2])
                    count = len(self.server.requests)
                    blocked = await client.post('/api/agent/model-capabilities', json=body, headers={'Origin': 'https://remote.example'})
                    self.assertEqual(blocked.status_code, 403); self.assertEqual(len(self.server.requests), count)
                    self.assertEqual(application.agent.config.api_key.get_secret_value(), 'fixture-key')
        asyncio.run(check())

    def test_configure_inherits_key_before_native_guard_and_preserves_legacy_opt_out(self):
        from server.agent import AgentConfig, AgentController
        self.server.responses['/v1/models/fixture-model'] = (200, {'effort': {'supported_levels': ['low'], 'default_level': 'low'}})
        existing = AgentConfig(**self.config())
        self.assertEqual(self.discover(existing)['levels'], ['low'])
        controller = AgentController(None); controller.config = existing
        with patch('server.agent.codex_executable', side_effect=ValueError('not needed for fixture')):
            with self.assertRaisesRegex(ValueError, 'low'):
                controller.configure(AgentConfig(**{**self.config(api_key=''), 'reasoning': 'max', 'reasoning_native': True}))
            self.assertIs(controller.config, existing)
            selected = controller.configure(AgentConfig(**{**self.config(api_key=''), 'reasoning': 'low', 'reasoning_native': True}))
            self.assertEqual(selected['reasoning'], 'low'); self.assertTrue(selected['has_api_key'])
            legacy = controller.configure(AgentConfig(**{**self.config(api_key=''), 'reasoning': 'max', 'reasoning_native': False}))
            self.assertEqual(legacy['reasoning'], 'max'); self.assertFalse(legacy['reasoning_native'])
            defaults = controller.configure(AgentConfig(**{**self.config(api_key=''), 'reasoning': None, 'reasoning_native': True}))
            self.assertIsNone(defaults['reasoning'])


if __name__ == '__main__': unittest.main()
