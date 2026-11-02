"""Subprocess protocol and isolation checks; no real model or credentials used."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from server.agent import AgentConfig
from server.claude_code_harness import claude_code_action, claude_code_executable, _native_package_binary
from server.runlog import CURRENT_LOG, RunLog


ACTION = {'status': 'move', 'frame': 'world', 'position_m': [.4, 0., .5],
          'quaternion_wxyz': [0., 1., 0., 0.], 'gripper_open': 1., 'duration_s': 1.,
          'explanation': '观察后移动。', 'hand_joint_positions_rad': None, 'robot_id': None}
USAGE = {'input_tokens': 13, 'output_tokens': 7, 'cache_read_input_tokens': 3,
         'cache_creation_input_tokens': 0, 'output_tokens_details': {'thinking_tokens': 2}}
FIXTURE = r'''import json, os, sys, time
from pathlib import Path
mode, capture, payload_path = sys.argv[1:4]
arguments = sys.argv[4:]
payload = json.loads(Path(payload_path).read_text(encoding='utf-8'))
request = json.loads(sys.stdin.buffer.read().decode('utf-8'))
Path(capture).write_text(json.dumps({'arguments': arguments, 'request': request,
    'cwd': os.getcwd(), 'key_received': os.getenv('ANTHROPIC_API_KEY') == 'local-fixture-only',
    'base_url': os.getenv('ANTHROPIC_BASE_URL'), 'cap': os.getenv('CLAUDE_CODE_MAX_OUTPUT_TOKENS'),
    'inherited_token': os.getenv('ANTHROPIC_AUTH_TOKEN'), 'inherited_vertex': os.getenv('CLAUDE_CODE_USE_VERTEX'),
    'inherited_thinking': os.getenv('MAX_THINKING_TOKENS')}), encoding='utf-8')
def emit(event): print(json.dumps(event), flush=True)
emit({'type': 'system', 'subtype': 'init'})
if mode in ('timeout', 'partial_timeout'):
    if mode == 'partial_timeout':
        emit({'type': 'assistant', 'message': {'id': 'one', 'usage': payload['usage'],
              'content': [{'type': 'thinking', 'thinking': '公开返回的说明。'}]}})
    time.sleep(60)
elif mode == 'model_error':
    emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True,
          'errors': ['404 model not found: fixture-unknown-model'], 'usage': payload['usage']})
    sys.exit(1)
elif mode == 'invalid_json':
    print('{not JSON', flush=True)
else:
    emit({'type': 'assistant', 'message': {'id': 'one', 'usage': payload['usage'],
          'content': [{'type': 'thinking', 'thinking': '公开返回的说明。'},
                      {'type': 'redacted_thinking', 'data': 'never-expose-private-token'},
                      {'type': 'text', 'text': '公开动作摘要。'}]}})
    output = {'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 1,
              'usage': payload['usage'], 'modelUsage': {'fixture-model': {'costUSD': .0001}},
              'total_cost_usd': .0001}
    if mode != 'no_structured': output['structured_output'] = payload['action']
    if mode == 'bad_schema': output['structured_output'] = {'status': 'move'}
    emit(output)
'''


class ClaudeCodeHarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='claude-harness-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = self.root / 'fixture.py'
        self.fixture.write_text(FIXTURE, encoding='utf-8')
        self.payload = self.root / 'payload.json'
        self.payload.write_text(json.dumps({'action': ACTION, 'usage': USAGE}), encoding='utf-8')
        self.capture = self.root / 'capture.json'
        self.processes = []
        self.invocations = []
        self.log = RunLog('agent', {'task': 'CLI fixture'}, directory=self.root / 'runs')

    def config(self, **changes):
        config = AgentConfig(provider='anthropic', model='fixture-model', api_key='local-fixture-only',
            base_url='http://127.0.0.1:23456', reasoning='high')
        return config.model_copy(update={'harness': 'claude_code', 'timeout_s': 5, **changes})

    async def call(self, mode='success', *, timeout_s=5, cancel=False):
        create = asyncio.create_subprocess_exec
        async def launch(*arguments, **kwargs):
            self.invocations.append({'arguments': list(arguments), 'options': kwargs})
            process = await create(sys.executable, '-u', str(self.fixture), mode,
                str(self.capture), str(self.payload), *arguments[1:], **kwargs)
            self.processes.append(process)
            return process
        context = CURRENT_LOG.set(self.log)
        try:
            # Pricing is a separate tested layer. Never fetch public prices in
            # this local protocol test, including failure/cancellation cases.
            with patch('server.claude_code_harness.claude_code_executable', return_value=sys.executable), \
                    patch('server.claude_code_harness.asyncio.create_subprocess_exec', side_effect=launch), \
                    patch('server.pricing.quote_for_call', new=AsyncMock(return_value={})), \
                    patch.dict(os.environ, {'ANTHROPIC_AUTH_TOKEN': 'do-not-inherit',
                        'CLAUDE_CODE_USE_VERTEX': '1', 'MAX_THINKING_TOKENS': '1234'}):
                task = asyncio.create_task(claude_code_action(self.config(timeout_s=timeout_s), '观察并移动',
                    [{'name': name, 'image_base64': 'aW1hZ2U='} for name in ('front', 'side', 'wrist')],
                    system_prompt='严格的动作协议'))
                if cancel:
                    for _ in range(200):
                        if self.capture.exists(): break
                        await asyncio.sleep(.01)
                    self.assertTrue(self.capture.exists(), 'fixture did not start')
                    task.cancel()
                return await task
        finally:
            CURRENT_LOG.reset(context)
            for process in self.processes:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            self.log.finish({'status': 'fixture finished'})

    def events(self, event):
        return [record['data'] for record in map(json.loads, self.log.path.read_text(encoding='utf-8').splitlines())
                if record['event'] == event]

    def test_three_images_config_isolation_structured_action_usage_and_public_reasoning(self):
        action, usage = asyncio.run(self.call())
        self.assertEqual(action.model_dump(), ACTION)
        self.assertEqual(usage, {**USAGE, 'reasoning_tokens': 2, 'total_tokens': 23})
        captured = json.loads(self.capture.read_text(encoding='utf-8'))
        args = captured['arguments']
        option = lambda name: args[args.index(name) + 1]
        self.assertEqual(option('--model'), 'fixture-model')
        self.assertEqual(option('--effort'), 'high')
        self.assertEqual(option('--system-prompt'), '严格的动作协议')
        self.assertEqual(option('--tools'), '')
        self.assertEqual(option('--setting-sources'), '')
        self.assertEqual(option('--mcp-config'), '{"mcpServers":{}}')
        for flag in ('--bare', '--restricted', '--disable-slash-commands', '--strict-mcp-config', '--no-session-persistence'):
            self.assertIn(flag, args)
        self.assertEqual(set(json.loads(option('--json-schema'))['required']), set(ACTION))
        self.assertEqual(json.loads(option('--settings'))['availableModels'], ['fixture-model'])
        blocks = captured['request']['message']['content']
        self.assertEqual(len([block for block in blocks if block['type'] == 'image']), 3)
        self.assertEqual([block['text'] for block in blocks if block['type'] == 'text'],
                         ['观察并移动', 'Camera: front', 'Camera: side', 'Camera: wrist'])
        self.assertTrue(captured['key_received'])
        self.assertNotIn('local-fixture-only', json.dumps(args))
        self.assertEqual(captured['base_url'], 'http://127.0.0.1:23456')
        self.assertEqual(captured['cap'], '16384')
        for name in ('inherited_token', 'inherited_vertex', 'inherited_thinking'):
            self.assertIsNone(captured[name])
        self.assertFalse(Path(captured['cwd']).exists())
        inference, = self.events('inference')
        self.assertEqual(inference['harness'], 'claude_code')
        self.assertEqual(inference['status'], 'complete')
        self.assertEqual(inference['reasoning']['text'], '公开返回的说明。')
        self.assertEqual(inference['reasoning_tokens'], 2)
        self.assertEqual(inference['total_tokens'], 23)
        output, = self.events('model_output')
        self.assertEqual(output['cli_total_cost_usd'], .0001)
        self.assertNotIn('never-expose-private-token', json.dumps(output))

    def test_unknown_model_error_preserves_returned_usage_and_reason(self):
        with self.assertRaisesRegex(RuntimeError, '404 model not found: fixture-unknown-model'):
            asyncio.run(self.call('model_error'))
        inference, = self.events('inference')
        self.assertEqual(inference['status'], 'error')
        self.assertEqual(inference['output_tokens'], 7)

    def test_invalid_json_no_final_result_is_not_executed(self):
        with self.assertRaisesRegex(RuntimeError, 'CLI 未返回最终结果'):
            asyncio.run(self.call('invalid_json'))

    def test_unstructured_text_does_not_bypass_json_contract(self):
        with self.assertRaisesRegex(ValueError, '未返回完整 JSON 动作'):
            asyncio.run(self.call('no_structured'))

    def test_invalid_action_schema_is_not_executed(self):
        with self.assertRaises(ValidationError):
            asyncio.run(self.call('bad_schema'))

    def test_timeout_reaps_child_and_marks_no_returned_usage(self):
        with self.assertRaisesRegex(TimeoutError, 'Claude Code 请求超过 1 秒'):
            asyncio.run(self.call('timeout', timeout_s=1))
        self.assertIsNotNone(self.processes[0].returncode)
        inference, = self.events('inference')
        self.assertFalse(inference['response_returned'])
        self.assertIsNone(inference['usage'])

    def test_cancellation_reaps_child_and_preserves_partial_public_usage(self):
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(self.call('partial_timeout', cancel=True))
        self.assertIsNotNone(self.processes[0].returncode)
        inference, = self.events('inference')
        self.assertEqual(inference['status'], 'cancelled')
        self.assertEqual(inference['output_tokens'], 7)

    def test_wrong_provider_or_missing_key_or_effort_fails_before_launch(self):
        for config, reason in ((self.config(provider='openai'), 'Messages'),
                               (self.config(api_key=AgentConfig().api_key), 'API Key'),
                               (self.config(reasoning='ultra'), '--effort')):
            with self.subTest(reason=reason), patch('server.claude_code_harness.asyncio.create_subprocess_exec') as launch:
                with self.assertRaisesRegex(ValueError, reason):
                    asyncio.run(claude_code_action(config, 'task', []))
                launch.assert_not_called()

    def test_explicit_executable_and_missing_override(self):
        with patch.dict(os.environ, {'CLAUDE_CLI_PATH': sys.executable}):
            self.assertEqual(claude_code_executable(), sys.executable)
        with patch.dict(os.environ, {'CLAUDE_CLI_PATH': str(self.root / 'missing.exe')}):
            with self.assertRaisesRegex(ValueError, 'CLAUDE_CLI_PATH'):
                claude_code_executable()

    def test_native_package_precedes_placeholder_and_windows_shell_shim(self):
        package = self.root / 'node_modules' / '@anthropic-ai' / 'claude-code'
        native = package.parent / 'claude-code-win32-x64' / 'claude.exe'
        native.parent.mkdir(parents=True)
        native.write_bytes(b'MZfixture')
        wrapper = package / 'bin' / 'claude.exe'
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text('echo native binary not installed')
        with patch('server.claude_code_harness.platform.machine', return_value='AMD64'), \
                patch('server.claude_code_harness.sys.platform', 'win32'):
            self.assertEqual(_native_package_binary(package), native)
            native.unlink()
            self.assertIsNone(_native_package_binary(package))


if __name__ == '__main__':
    unittest.main()
