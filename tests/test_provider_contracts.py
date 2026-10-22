"""Local HTTP fixtures verify official SDK serialization, not model intelligence."""
import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from threading import Thread
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

from server.agent import AgentConfig, ModelResponseError, api_action


ACTION = {'status': 'move', 'frame': 'world', 'position_m': [.4, 0., .5],
          'quaternion_wxyz': [0., 1., 0., 0.], 'gripper_open': 1., 'duration_s': 1., 'explanation': '测试动作。', 'hand_joint_positions_rad': None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.received.append((self.path, body))
        if self.path.endswith('/agent'):
            response = {'protocol': 'embodied-agent-v1', 'output': ACTION,
                        'model': 'local-agent-contract-fixture', 'usage': {'input_tokens': 1, 'output_tokens': 1}}
        elif self.path.endswith('/responses'):
            response = {'id': 'fixture', 'object': 'response', 'created_at': 0, 'status': 'completed',
                        'model': body['model'], 'output': [{'type': 'message', 'id': 'message', 'status': 'completed',
                        'role': 'assistant', 'content': [{'type': 'output_text', 'text': json.dumps(ACTION), 'annotations': []}]}]}
        elif self.path.endswith('/chat/completions'):
            response = {'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                        'choices': [{'index': 0, 'finish_reason': getattr(self.server, 'finish_reason', 'stop'),
                                     'message': {'role': 'assistant', 'content': getattr(self.server, 'content', json.dumps(ACTION)),
                                                 'reasoning_content': 'private fixture reasoning'}}],
                        'usage': getattr(self.server, 'usage', None)}
            if getattr(self.server, 'empty_choices', False): response['choices'] = []
        else:
            response = {'id': 'fixture', 'type': 'message', 'role': 'assistant', 'model': body['model'],
                        'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 1, 'output_tokens': 1},
                        'content': [{'type': 'tool_use', 'id': 'tool', 'name': 'next_robot_action', 'input': ACTION}]}
        data = json.dumps(response).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)


class ProviderContracts(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler); self.server.received = []
        self.thread = Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}/v1'
        image = io.BytesIO(); Image.new('RGB', (32, 32), '#668877').save(image, 'JPEG')
        self.cameras = [{'name': name, 'image_base64': base64.b64encode(image.getvalue()).decode()} for name in ('front', 'side', 'wrist')]

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(2)

    def config(self, **changes):
        return AgentConfig(**{'provider': 'compatible', 'model': 'deepseek-flash', 'api_key': 'fixture-key',
                              'base_url': self.base_url, **changes})

    def call(self, config):
        return asyncio.run(api_action(config, 'fixture observation', self.cameras))

    def test_all_api_adapters_send_three_images_and_parse_same_absolute_action(self):
        for provider in ('openai', 'compatible', 'anthropic'):
            with self.subTest(provider=provider):
                action, _ = self.call(self.config(provider=provider, max_output_tokens=24576))
                self.assertEqual(action.model_dump(exclude={'robot_id'}), ACTION)
                path, body = self.server.received[-1]
                content = body['input'][0]['content'] if provider == 'openai' else body['messages'][-1]['content']
                images = [c for c in content if c['type'] in ('input_image', 'image_url', 'image')]
                self.assertEqual(len(images), 3)
                self.assertEqual(body['max_output_tokens' if provider == 'openai' else 'max_tokens'], 24576)
                self.assertNotIn('fixture-key', json.dumps(body))
                if provider == 'anthropic':
                    self.assertIn('Submit exactly one action using next_robot_action', body['system'])
                if provider == 'openai':
                    self.assertFalse(body['store']); self.assertTrue(body['text']['format']['strict'])
                if provider == 'compatible':
                    self.assertNotIn('thinking', body); self.assertNotIn('reasoning_effort', body)

    def test_deepseek_thinking_and_low_effort_are_sent_with_three_images(self):
        from openai import AsyncOpenAI
        # Only redirect the HTTP destination to a local fixture; serialize with the real SDK.
        with patch('openai.AsyncOpenAI', side_effect=lambda **kwargs: AsyncOpenAI(**{**kwargs, 'base_url': self.base_url})):
            for thinking in ('enabled', 'disabled'):
                with self.subTest(thinking=thinking):
                    action, _ = self.call(self.config(base_url='https://api.deepseek.com/v1', deepseek_thinking=thinking, reasoning='low'))
                    self.assertEqual(action.model_dump(exclude={'robot_id'}), ACTION)
                    body = self.server.received[-1][1]
                    self.assertEqual(body['max_tokens'], 16384)
                    self.assertEqual(body['thinking'], {'type': thinking})
                    self.assertEqual(body['model'], 'deepseek-flash')
                    self.assertEqual(body['response_format'], {'type': 'json_object'})
                    self.assertEqual(len([p for p in body['messages'][-1]['content'] if p['type'] == 'image_url']), 3)
                    if thinking == 'enabled': self.assertEqual(body['reasoning_effort'], 'low')
                    else: self.assertNotIn('reasoning_effort', body)

    def test_native_effort_is_transmitted_without_global_level_mapping(self):
        for provider in ('openai', 'compatible', 'anthropic', 'custom'):
            with self.subTest(provider=provider):
                config = self.config(provider=provider, reasoning='max', reasoning_native=True,
                    base_url=self.base_url+'/agent' if provider=='custom' else self.base_url)
                self.call(config)
                body = self.server.received[-1][1]
                actual = body['reasoning']['effort'] if provider=='openai' else body['output_config']['effort'] if provider=='anthropic' else body['model_settings']['reasoning'] if provider=='custom' else body['reasoning_effort']
                self.assertEqual(actual, 'max')
        self.call(self.config(reasoning='vendor-future-level', reasoning_native=True))
        self.assertEqual(self.server.received[-1][1]['reasoning_effort'], 'vendor-future-level')

    def test_service_default_omits_native_effort_and_legacy_recipes_keep_old_meaning(self):
        for provider in ('openai', 'compatible', 'anthropic'):
            for native in (False, True):
                with self.subTest(provider=provider, native=native):
                    self.call(self.config(provider=provider, reasoning=None, reasoning_native=native))
                    body = self.server.received[-1][1]
                    for field in ('reasoning', 'reasoning_effort', 'output_config'): self.assertNotIn(field, body)
        for provider in ('compatible', 'anthropic'):
            self.call(self.config(provider=provider, reasoning='high'))
            body = self.server.received[-1][1]
            self.assertNotIn('reasoning_effort', body); self.assertNotIn('output_config', body)

    def test_large_output_budget_is_transmitted_to_each_api_provider(self):
        budget = 262144
        for provider in ('openai', 'compatible', 'anthropic', 'custom'):
            with self.subTest(provider=provider):
                config = self.config(provider=provider, max_output_tokens=budget,
                                     base_url=self.base_url+'/agent' if provider == 'custom' else self.base_url)
                action, _ = self.call(config)
                self.assertEqual(action.model_dump(exclude={'robot_id'}), ACTION)
                body = self.server.received[-1][1]
                actual = (body['model_settings']['max_output_tokens'] if provider == 'custom' else
                          body['max_output_tokens'] if provider == 'openai' else body['max_tokens'])
                self.assertEqual(actual, budget)

    def test_length_is_rejected_even_if_content_is_valid_and_usage_is_retained(self):
        self.server.finish_reason = 'length'
        self.server.usage = {'prompt_tokens': 1100, 'completion_tokens': 4096, 'total_tokens': 5196,
                             'completion_tokens_details': {'reasoning_tokens': 4096}}
        for content in ('', '{"status":', json.dumps(ACTION)):
            with self.subTest(content=content):
                self.server.content = content
                count = len(self.server.received)
                with self.assertRaises(ModelResponseError) as caught:
                    self.call(self.config(max_output_tokens=4096))
                message = str(caught.exception)
                self.assertIn('长度限制截断', message)
                self.assertIn('本轮输出上限 4096', message)
                self.assertIn('推理 4096 tokens', message)
                self.assertIn('单轮输出上限', message)
                self.assertIn('本轮未执行动作', message)
                self.assertNotIn('视觉', message)
                self.assertNotIn('private fixture reasoning', message)
                self.assertEqual(caught.exception.usage['completion_tokens'], 4096)
                self.assertEqual(len(self.server.received), count + 1)  # No implicit paid retry.

    def test_empty_body_and_other_finish_reasons_are_not_misreported_as_length_or_vision(self):
        self.server.content = ''
        for reason, expected in [('stop', '正文为空'), ('content_filter', 'finish_reason=content_filter')]:
            with self.subTest(reason=reason):
                self.server.finish_reason = reason
                with self.assertRaisesRegex(ModelResponseError, expected) as caught:
                    self.call(self.config())
                self.assertNotIn('视觉', str(caught.exception))
                self.assertNotIn('长度限制截断', str(caught.exception))
        self.server.empty_choices = True
        with self.assertRaisesRegex(ModelResponseError, 'choices 为空'): self.call(self.config())

    def test_stop_does_not_bypass_action_schema_validation(self):
        for content in ('{"status":', json.dumps({**ACTION, 'position_m': [0, 0]})):
            with self.subTest(content=content):
                self.server.content = content
                with self.assertRaises(ValidationError): self.call(self.config())


if __name__ == '__main__': unittest.main()
