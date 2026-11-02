"""Opt-in pinned CLI -> loopback Messages SSE test, never a paid model request.

Run with EMBODIWORKS_CLAUDE_NATIVE_TESTS=1 after npm install. CLAUDE_CLI_PATH
can point at another official native install. This verifies the real CLI's
vision/StructuredOutput wire behavior; it does not test robot task competence.
"""
import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from threading import Thread
import unittest
from unittest.mock import patch

from PIL import Image

from server.agent import AgentConfig
from server.claude_code_harness import claude_code_action
from tests.test_claude_code_harness import ACTION


@unittest.skipUnless(os.getenv('EMBODIWORKS_CLAUDE_NATIVE_TESTS') == '1', 'opt-in real CLI loopback protocol test')
class ClaudeCodeNativeTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.outputs = []
        self.failure = False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                fixture.requests.append({'path': self.path, 'body': body,
                    'key_matches': self.headers.get('x-api-key') == 'loopback-fixture-only'})
                if 'count_tokens' in self.path:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"input_tokens":13}')
                    return
                if fixture.failure:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"type":"error","error":{"type":"not_found_error","message":"fixture unknown model"}}')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()

                def emit(kind, value):
                    self.wfile.write(f'event: {kind}\ndata: {json.dumps(value)}\n\n'.encode())
                    self.wfile.flush()

                emit('message_start', {'type': 'message_start', 'message': {
                    'id': 'fixture-message', 'type': 'message', 'role': 'assistant', 'model': body['model'],
                    'content': [], 'stop_reason': None, 'stop_sequence': None,
                    'usage': {'input_tokens': 13, 'output_tokens': 0,
                              'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}}})
                emit('content_block_start', {'type': 'content_block_start', 'index': 0,
                    'content_block': {'type': 'tool_use', 'id': 'fixture-tool', 'name': 'StructuredOutput', 'input': {}}})
                emit('content_block_delta', {'type': 'content_block_delta', 'index': 0,
                    'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(ACTION)}})
                emit('content_block_stop', {'type': 'content_block_stop', 'index': 0})
                emit('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use', 'stop_sequence': None},
                                      'usage': {'output_tokens': 7}})
                emit('message_stop', {'type': 'message_stop'})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server_thread = Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        buffer = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(buffer, format='JPEG')
        self.cameras = [{'name': name, 'image_base64': base64.b64encode(buffer.getvalue()).decode()}
                        for name in ('front', 'side', 'wrist')]
        self.config = AgentConfig(provider='anthropic', model='claude-sonnet-4-6', reasoning='high',
            api_key='loopback-fixture-only', base_url=f'http://127.0.0.1:{self.server.server_port}',
            timeout_s=30, max_output_tokens=2048).model_copy(update={'harness': 'claude_code'})

    def run_cli(self):
        with patch('server.claude_code_harness.model_output', side_effect=self.outputs.append):
            return asyncio.run(claude_code_action(self.config, 'Observe the three images and return one action.', self.cameras))

    def test_real_cli_forwards_images_model_effort_limit_and_returns_schema(self):
        action, usage = self.run_cli()
        self.assertEqual(action.model_dump(), ACTION)
        messages = [item for item in self.requests if 'count_tokens' not in item['path']]
        self.assertEqual(len(messages), 1)
        self.assertTrue(messages[0]['key_matches'])
        body = messages[0]['body']
        self.assertEqual(body['model'], 'claude-sonnet-4-6')
        self.assertEqual(body['max_tokens'], 2048)
        self.assertEqual(body['output_config']['effort'], 'high')
        self.assertEqual([tool['name'] for tool in body['tools']], ['StructuredOutput'])
        images = [block for message in body['messages'] if isinstance(message.get('content'), list)
                  for block in message['content'] if block.get('type') == 'image']
        self.assertEqual(len(images), 3)
        self.assertEqual((usage['input_tokens'], usage['output_tokens']), (13, 7))
        self.assertIsInstance(self.outputs[0]['cli_total_cost_usd'], (int, float))
        self.assertIn('claude-sonnet-4-6', self.outputs[0]['cli_model_usage'])

    def test_real_cli_reports_unavailable_model_without_fallback(self):
        self.failure = True
        # The official CLI rewrites an Anthropic 404 into its model/access
        # diagnostic; preserve that real message rather than inventing a raw
        # API body the CLI has not exposed to the adapter.
        with self.assertRaisesRegex(RuntimeError, r'selected model \(claude-sonnet-4-6\)'):
            self.run_cli()
        messages = [item for item in self.requests if 'count_tokens' not in item['path']]
        # The official CLI retries the failed stream as a non-stream request
        # even with API retry count zero; it must not silently change models.
        self.assertEqual([item['body'].get('stream') for item in messages], [True, False])
        self.assertTrue(all(item['body']['model'] == 'claude-sonnet-4-6' for item in messages))


if __name__ == '__main__':
    unittest.main()
