"""Harness/model separation at configuration and transport boundaries; no paid calls."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from server.agent import AgentConfig, AgentController, invoke_agent_action, codex_action, validate_agent_access
from tests.test_provider_contracts import ACTION


class HarnessConfigurationTests(unittest.TestCase):
    def test_legacy_configs_migrate_without_changing_transport(self):
        self.assertEqual(AgentConfig().harness, 'codex_cli')
        for provider in ('openai', 'compatible', 'anthropic'):
            self.assertEqual(AgentConfig(provider=provider).harness, 'workbench')
        custom = AgentConfig(provider='custom', base_url='https://agent.example/action', api_key='old-key')
        self.assertEqual(custom.harness, 'http')
        self.assertEqual(custom.harness_url, 'https://agent.example/action')
        self.assertEqual(custom.harness_api_key.get_secret_value(), 'old-key')
        self.assertFalse(custom.api_key.get_secret_value())
        self.assertEqual(AgentConfig.model_validate(custom.model_dump()), custom)

    def test_supported_pairs_are_explicit_and_no_silent_fallback(self):
        supported = {'workbench': ('openai', 'compatible', 'anthropic'),
                     'codex_cli': ('codex', 'openai'), 'claude_code': ('anthropic',),
                     'dsh': ('openai', 'compatible', 'anthropic'),
                     'http': ('openai', 'compatible', 'anthropic', 'custom')}
        for harness, providers in supported.items():
            for provider in ('codex', 'openai', 'compatible', 'anthropic', 'custom'):
                with self.subTest(harness=harness, provider=provider):
                    data = dict(harness=harness, provider=provider, harness_url='http://localhost/action')
                    if provider in providers: self.assertEqual(AgentConfig(**data).harness, harness)
                    else:
                        with self.assertRaises(ValidationError): AgentConfig(**data)
        for data in ({'harness':'http', 'provider':'openai'},
                     {'harness':'workbench', 'provider':'openai', 'forward_model_key':True},
                     {'harness':'http', 'provider':'custom', 'harness_url':'http://localhost/action', 'forward_model_key':True}):
            with self.assertRaises(ValidationError): AgentConfig(**data)

    def test_credential_reuse_follows_independent_endpoints_and_public_config_omits_secrets(self):
        controller = AgentController(None)
        values = dict(harness='http', provider='openai', model='fixture',
                      base_url='https://models.example/v1', harness_url='https://agent.example/action')
        controller.configure(AgentConfig(**values, api_key='model-secret', harness_api_key='harness-secret'))
        same = controller.configure(AgentConfig(**{**values, 'model':'changed'}))
        self.assertTrue(same['has_api_key']); self.assertTrue(same['has_harness_api_key'])
        self.assertNotIn('model-secret', json.dumps(same)); self.assertNotIn('harness-secret', json.dumps(same))
        self.assertNotIn('api_key', same); self.assertNotIn('harness_api_key', same)
        changed = controller.configure(AgentConfig(**{**values, 'harness_url':'https://other.example/action'}))
        self.assertTrue(changed['has_api_key']); self.assertFalse(changed['has_harness_api_key'])
        changed = controller.configure(AgentConfig(**{**values, 'base_url':'https://other-model.example/v1'}))
        self.assertFalse(changed['has_api_key']); self.assertFalse(changed['has_harness_api_key'])

    def test_switching_cli_harnesses_preserves_the_independent_model_configuration(self):
        for provider, harnesses in [('openai', ('workbench','codex_cli','dsh','http')),
                                   ('anthropic', ('workbench','claude_code','dsh','http'))]:
            with self.subTest(provider=provider):
                controller = AgentController(None)
                model = dict(provider=provider, model='router/selected-model', base_url='https://router.example/v1',
                             reasoning='high', max_output_tokens=32768, timeout_s=80, max_rounds=None)
                controller.configure(AgentConfig(**model, harness='workbench', api_key='model-secret'))
                for harness in harnesses:
                    config = AgentConfig(**model, harness=harness, harness_url='https://agent.example/action')
                    saved = controller.configure(config)
                    self.assertEqual(saved['harness'], harness)
                    for key, value in model.items():
                        self.assertEqual(saved[key], value)
                    self.assertTrue(saved['has_api_key'])
                    self.assertEqual(controller.config.api_key.get_secret_value(), 'model-secret')
                    self.assertNotIn('model-secret', json.dumps(saved))
                changed = controller.configure(AgentConfig(**{**model,'base_url':'https://other.example/v1'},harness='dsh'))
                self.assertFalse(changed['has_api_key'])

    def test_cli_reasoning_options_are_checked_without_relabeling_model_efforts(self):
        for harness, provider, levels in [('claude_code','anthropic',('low','medium','high','xhigh','max')),
                                          ('dsh','openai',('off','minimal','low','medium','high','xhigh','max'))]:
            for level in (None, *levels):
                with self.subTest(harness=harness, level=level):
                    self.assertEqual(AgentConfig(harness=harness,provider=provider,reasoning=level).reasoning,level)
            for level in ('ultra','vendor_future_effort') + (('minimal','off') if harness=='claude_code' else ()):
                with self.subTest(harness=harness, invalid=level), self.assertRaises(ValidationError):
                    AgentConfig(harness=harness,provider=provider,reasoning=level)
        # Other harnesses must remain extensible for routed model-native effort.
        self.assertEqual(AgentConfig(harness='workbench',provider='openai',reasoning='vendor_future_effort').reasoning,
                         'vendor_future_effort')

    def test_disabled_deepseek_thinking_ignores_stored_effort_for_configuration(self):
        values = dict(harness='dsh',provider='compatible',base_url='https://api.deepseek.com',
                      model='deepseek-fixture',reasoning='vendor_future_effort',reasoning_native=True)
        controller = AgentController(None)
        with patch('server.model_capabilities.cached_model_capabilities',return_value={'levels':['high','max']}):
            saved = controller.configure(AgentConfig(**values,deepseek_thinking='disabled'))
        self.assertEqual(saved['deepseek_thinking'], 'disabled')
        with self.assertRaises(ValidationError):
            AgentConfig(**values,deepseek_thinking='enabled')

    def test_cli_preflight_requires_its_runtime_and_a_separate_model_key(self):
        for harness, provider, executable in [('claude_code','anthropic','server.claude_code_harness.claude_code_executable'),
                                               ('dsh','openai','server.dsh_harness.dsh_command')]:
            with self.subTest(harness=harness), patch(executable,return_value='installed-runtime') as runtime:
                with self.assertRaisesRegex(ValueError, 'API Key'):
                    validate_agent_access(AgentConfig(harness=harness,provider=provider))
                validate_agent_access(AgentConfig(harness=harness,provider=provider,api_key='model-key'))
                self.assertEqual(runtime.call_count,2)
            with self.subTest(harness=harness, missing=True), patch(executable,side_effect=ValueError('Runtime missing')):
                with self.assertRaisesRegex(ValueError, 'Runtime missing'):
                    validate_agent_access(AgentConfig(harness=harness,provider=provider,api_key='model-key'))


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.received.append({'path':self.path, 'body':body, 'headers':dict(self.headers)})
        data = json.dumps({'protocol':'embodied-agent-v1', 'output':ACTION,
                           'usage':{'input_tokens':5, 'output_tokens':2}}).encode()
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)


class HarnessDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_managed_model_does_not_probe_an_action_endpoint_for_metadata(self):
        from server.model_capabilities import discover_model_capabilities
        config = AgentConfig(provider='custom', base_url='http://localhost/agent', api_key='harness-secret')
        with patch('server.model_capabilities.httpx.AsyncClient', side_effect=AssertionError('No model endpoint configured')):
            result = await discover_model_capabilities(config)
        self.assertFalse(result['verified'])
        self.assertIn('harness', result['message'])

    async def test_harness_selects_executor_independently_of_model_provider(self):
        for harness, provider, expected in [('workbench','openai','api'), ('codex_cli','openai','codex'),
                                             ('codex_cli','codex','codex'), ('http','openai','api'),
                                             ('claude_code','anthropic','claude'), ('dsh','openai','dsh'),
                                             ('dsh','compatible','dsh'), ('dsh','anthropic','dsh')]:
            with self.subTest(harness=harness, provider=provider), \
                    patch('server.agent.codex_action', AsyncMock(return_value=('codex', {}))) as cli, \
                    patch('server.agent.api_action', AsyncMock(return_value=('api', {}))) as api, \
                    patch('server.claude_code_harness.claude_code_action', AsyncMock(return_value=('claude', {}))) as claude, \
                    patch('server.dsh_harness.dsh_action', AsyncMock(return_value=('dsh', {}))) as dsh:
                config = AgentConfig(harness=harness, provider=provider,model='router/selected-model',
                    base_url='https://router.example/v1',api_key='model-key',harness_url='http://localhost/action')
                cameras = [{'name':'front','image_base64':'fixture-only'}]
                result = await invoke_agent_action(config, 'observation', cameras, schema_name='fixture')
                self.assertEqual(result[0], expected)
                for name, adapter in [('codex',cli),('api',api),('claude',claude),('dsh',dsh)]:
                    self.assertEqual(adapter.await_count,int(name==expected))
                    if name == expected:
                        adapter.assert_awaited_once_with(config,'observation',cameras,schema_name='fixture')

    async def test_cli_failure_does_not_silently_fallback_to_another_harness(self):
        for harness,provider,module in [('claude_code','anthropic','server.claude_code_harness.claude_code_action'),
                                        ('dsh','openai','server.dsh_harness.dsh_action')]:
            with self.subTest(harness=harness), patch(module,AsyncMock(side_effect=RuntimeError('CLI protocol failed'))), \
                    patch('server.agent.api_action',AsyncMock()) as api, patch('server.agent.codex_action',AsyncMock()) as codex:
                with self.assertRaisesRegex(RuntimeError,'CLI protocol failed'):
                    await invoke_agent_action(AgentConfig(harness=harness,provider=provider),'observation',[])
                api.assert_not_called();codex.assert_not_called()

    async def test_http_harness_credentials_and_explicit_model_forwarding(self):
        server = ThreadingHTTPServer(('127.0.0.1',0),FixtureHandler); server.received=[]
        thread = Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            values = dict(harness='http', provider='openai', model='model-fixture',
                base_url='https://models.example/v1', api_key='model-secret',
                harness_url=f'http://127.0.0.1:{server.server_port}/action', harness_api_key='harness-secret')
            for forward in (False, True):
                output, usage = await invoke_agent_action(AgentConfig(**values, forward_model_key=forward), '{"task":"fixture"}', [])
                received = server.received[-1]; body=received['body']
                self.assertEqual(received['path'], '/action')
                self.assertEqual(received['headers']['Authorization'], 'Bearer harness-secret')
                self.assertEqual(body['model_config']['base_url'], values['base_url'])
                self.assertEqual(body['model_config']['provider'], 'openai')
                self.assertEqual(body['model_config'].get('api_key'), 'model-secret' if forward else None)
                self.assertNotIn('harness-secret',json.dumps(body))
                self.assertEqual(body['protocol'],'embodied-agent-v1'); self.assertEqual(body['task'],'fixture')
                self.assertEqual(output.status, 'move'); self.assertEqual(usage['output_tokens'],2)
            # Existing custom-service clients retain the original body and auth semantics.
            await invoke_agent_action(AgentConfig(provider='custom', base_url=values['harness_url'],api_key='legacy-secret'), 'task', [])
            self.assertEqual(server.received[-1]['headers']['Authorization'],'Bearer legacy-secret')
            self.assertNotIn('api_key', server.received[-1]['body']['model_config'])
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    async def test_codex_api_transport_passes_key_only_in_child_environment(self):
        captured=[]
        async def spawn(*arguments, **options):
            captured.append((arguments,options))
            output=Path(arguments[arguments.index('--output-last-message')+1])
            output.write_text(json.dumps(ACTION),encoding='utf-8')
            self.assertNotIn('model-private', ' '.join(arguments))
            self.assertNotIn('model-private', ''.join(p.read_text() for p in output.parent.iterdir()))
            process=AsyncMock(); process.returncode=0
            process.communicate.return_value=(b'{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":2}}\n',b'')
            return process
        with patch('server.agent.codex_executable',return_value='fixture-codex'), \
                patch('server.agent.asyncio.create_subprocess_exec',side_effect=spawn):
            output,usage=await codex_action(AgentConfig(harness='codex_cli',provider='openai',
                model='route-model',base_url='https://router.example/v1',api_key='model-private'),'observation',[])
            arguments,options=captured[-1]
            self.assertIn('base_url="https://router.example/v1"',' '.join(arguments))
            self.assertIn('requires_openai_auth=false',' '.join(arguments))
            self.assertEqual(options['env']['EMBODIWORKS_CODEX_MODEL_KEY'],'model-private')
            self.assertEqual(output.status,'move'); self.assertEqual(usage['output_tokens'],2)
            await codex_action(AgentConfig(),'observation',[])
            self.assertNotIn('env',captured[-1][1])
            self.assertIn('requires_openai_auth=true',' '.join(captured[-1][0]))


if __name__ == '__main__': unittest.main()
