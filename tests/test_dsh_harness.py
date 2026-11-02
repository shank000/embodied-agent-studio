"""SDK protocol/lifecycle regression; optional real dsh against loopback SSE only."""
import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from server.agent import AgentConfig
from server.costing import calculate_call_cost
from server.dsh_harness import dsh_action, _profile_patch, _route, _usage, _SdkRun


ACTION = {'status': 'move', 'frame': 'world', 'position_m': [.4, 0., .5],
          'quaternion_wxyz': [0., 1., 0., 0.], 'gripper_open': 1., 'duration_s': 1.,
          'explanation': 'Local protocol fixture; not a model evaluation.',
          'hand_joint_positions_rad': None, 'robot_id': None}
KEY = 'local-dsh-protocol-fixture-key'

FIXTURE = '''import json, os, sys, time
from pathlib import Path
mode, capture, payload = sys.argv[1:4]
args = sys.argv[4:]
patch = Path(args[args.index('--patch')+1]).read_text(encoding='utf-8')
saved = {'argv': args, 'patch': json.loads(patch), 'cwd': os.getcwd(),
         'home': os.environ['DSH_HOME'], 'key': os.environ.get('EMBODIWORKS_DSH_API_KEY'),
         'ambient_key': os.environ.get('UNRELATED_API_KEY'), 'requests': []}
def emit(message): print(json.dumps(message), flush=True)
def response(request, result): emit({'jsonrpc':'2.0','id':request['id'],'result':result})
def event(session, kind, data, seq):
 emit({'jsonrpc':'2.0','method':'session.event','params':{'sessionId':session,
       'event':{'type':kind,'data':data,'seq':seq}}})
for line in sys.stdin:
 request = json.loads(line); saved['requests'].append(request)
 Path(capture).write_text(json.dumps(saved), encoding='utf-8')
 method=request['method']; params=request.get('params',{})
 if method=='initialize': response(request,{'serverInfo':{'name':'deepseek-harness-sdk-runtime','version':'0.0.1'}})
 elif method=='session/prompt':
  session=params['sessionId']; response(request,{'messageId':'fixture-message'})
  if mode=='hang': time.sleep(60)
  elif mode=='error':
   event(session,'turn/end',{'reason':{'kind':'error','message':'fixture failure'}},1)
  else:
   content=[{'type':'reasoning','text':'Public fixture reasoning.'},
            {'type':'text','text':json.dumps(json.loads(payload)) if mode!='malformed' else '{oops'}]
   if mode=='tool': content.append({'type':'tool-call','name':'bash','arguments':'{}'})
   event(session,'assistant/message',{'message':{'content':content,'source':{'model':'fixture'}},
         'usage':{'inputTokens':7,'cacheReadTokens':3,'outputTokens':11,'totalTokens':21}},1)
   event(session,'turn/end',{'reason':{'kind':'max-tokens' if mode=='length' else 'completed'}},2)
 elif method=='session/wait': response(request,{})
 elif method=='shutdown': response(request,{}); break
'''


def cameras():
    output = io.BytesIO()
    Image.new('RGB', (64, 64), 'red').save(output, format='JPEG')
    return [{'name': name, 'image_base64': base64.b64encode(output.getvalue()).decode('ascii')}
            for name in ('front', 'side', 'wrist')]


class DshProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dsh-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / 'fixture.py'
        self.script.write_text(FIXTURE, encoding='utf-8')
        self.capture = self.root / 'captured.json'
        self.processes = []
        self.outputs = []

    async def call(self, mode='ok', timeout=5):
        real_launch = asyncio.create_subprocess_exec
        async def launch(*args, **kwargs):
            process = await real_launch(sys.executable, '-u', str(self.script), mode,
                                       str(self.capture), json.dumps(ACTION), *args[1:], **kwargs)
            self.processes.append(process)
            return process
        config = AgentConfig(provider='openai', model='fixture', api_key=KEY,
                             base_url='http://127.0.0.1:1/v1', reasoning='high')
        config = config.model_copy(update={'timeout_s': timeout})
        with patch('server.dsh_harness.dsh_command', return_value=[sys.executable]), \
             patch('server.dsh_harness.asyncio.create_subprocess_exec', side_effect=launch), \
             patch('server.dsh_harness.model_output', side_effect=self.outputs.append), \
             patch.dict(os.environ, {'UNRELATED_API_KEY': 'must-not-reach-runtime'}):
            return await dsh_action(config, 'observe and move', cameras(), system_prompt='robot system')

    def test_real_subprocess_sdk_multimodal_contract_and_isolation(self):
        action, usage = asyncio.run(self.call())
        self.assertEqual(action.model_dump(), ACTION)
        self.assertEqual(usage, {'input_tokens': 10, 'cached_input_tokens': 3,
                                 'cache_write_input_tokens': 0, 'output_tokens': 11, 'total_tokens': 21})
        captured = json.loads(self.capture.read_text())
        self.assertNotIn(KEY, json.dumps(captured['patch']))
        self.assertNotIn(KEY, json.dumps(captured['argv']))
        self.assertEqual(captured['key'], KEY)
        self.assertIsNone(captured['ambient_key'])
        self.assertFalse(Path(captured['home']).exists())
        self.assertNotEqual(Path(captured['cwd']), Path.cwd())
        blocks = captured['requests'][1]['params']['contentBlocks']
        self.assertEqual(sum(b['type'] == 'image' for b in blocks), 3)
        self.assertEqual(captured['requests'][0]['params']['reasoningEffort'], 'high')
        self.assertEqual(captured['requests'][0]['params']['maxTokens'], 16384)
        self.assertEqual([r['method'] for r in captured['requests']],
                         ['initialize', 'session/prompt', 'session/wait', 'shutdown'])
        disabled = {row.get('id') for row in captured['patch'] if row.get('disabled')}
        self.assertTrue({'persistent-bash','persistent-pwsh','tool-fs','tool-working-directory',
                         'session-log-deepseek','plugin-package-inventory-deepseek','compaction-basic'} <= disabled)
        self.assertEqual(len(self.outputs), 1)
        self.assertEqual(self.outputs[0]['returned_reasoning'], 'Public fixture reasoning.')

    def test_incomplete_or_invalid_output_never_executes_an_action(self):
        for mode, error in [('error','未完整结束'), ('length','max-tokens'),
                            ('malformed','JSON 动作未通过'), ('tool','工具调用')]:
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, error):
                asyncio.run(self.call(mode))
        self.assertTrue(all(process.returncode is not None for process in self.processes))

    def test_timeout_terminates_owned_runtime_without_fabricating_output(self):
        with self.assertRaisesRegex(TimeoutError, 'dsh 请求超过'):
            asyncio.run(self.call('hang', timeout=.3))
        self.assertIsNotNone(self.processes[0].returncode)
        self.assertEqual(self.outputs, [])

    def test_user_cancel_terminates_owned_runtime(self):
        async def cancel():
            task = asyncio.create_task(self.call('hang'))
            for _ in range(100):
                if self.capture.exists():
                    break
                await asyncio.sleep(.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(cancel())
        self.assertIsNotNone(self.processes[0].returncode)

    def test_protocol_and_reasoning_boundaries_fail_before_launch(self):
        for provider, base, expected in [('openai','https://api.openai.com/v1','openai-responses'),
                                          ('compatible','https://route.example/v1','openai-completions'),
                                          ('anthropic','https://api.anthropic.com','anthropic-messages')]:
            config = AgentConfig(provider=provider, base_url=base, api_key=KEY)
            route, _, _ = _route(config, True)
            self.assertEqual(route['api'], expected)
            self.assertEqual(route['models'][0]['input'], ['text','image'])
        with self.assertRaisesRegex(ValueError, 'Codex 登录'):
            _route(AgentConfig(), False)
        with self.assertRaisesRegex(ValueError, '不支持'):
            _route(AgentConfig(provider='openai', api_key=KEY, reasoning='ultra'), False)
        config = AgentConfig(provider='compatible', base_url='https://api.deepseek.com', api_key=KEY,
                             reasoning='high', deepseek_thinking='disabled')
        route, effort, _ = _route(config, True)
        self.assertEqual(effort, 'off')
        self.assertEqual(route['compat']['thinkingFormat'], 'deepseek')

    def test_missing_usage_stays_unknown_and_disjoint_cache_is_counted_once(self):
        self.assertIsNone(_usage(None))
        self.assertIsNone(_usage({'inputTokens':float('nan'), 'outputTokens': True}))
        self.assertEqual(_usage({'inputTokens':7, 'cacheReadTokens':3, 'cacheWriteTokens':2,
                                 'outputTokens':11})['input_tokens'], 12)
        native = {'inputTokens':7, 'cacheReadTokens':3, 'cacheWriteTokens':2, 'outputTokens':11}
        self.assertEqual(_usage(native, 'anthropic'), {'input_tokens':7, 'cache_read_input_tokens':3,
                         'cache_creation_input_tokens':2, 'output_tokens':11})
        self.assertEqual(_usage({'inputTokens':7,'cacheReadTokens':3,'outputTokens':11}, 'deepseek'),
                         {'input_tokens':10,'prompt_cache_miss_tokens':7,'prompt_cache_hit_tokens':3,'output_tokens':11})

    def test_provider_native_usage_prices_cache_once(self):
        # Synthetic rates exercise the existing cost ledger, not live pricing.
        for provider in ('openai', 'anthropic', 'deepseek'):
            with self.subTest(provider=provider):
                normalized = _usage({'inputTokens':7,'cacheReadTokens':3,'outputTokens':11}, provider)
                quote = {'billing_kind':'api','status':'ready','currency':'USD','provider':provider,
                         'input_per_million':2,'cached_input_per_million':.5,'output_per_million':10}
                cost = calculate_call_cost({'usage':normalized},quote)
                self.assertEqual(cost['status'], 'complete')
                self.assertAlmostEqual(cost['amount_usd'], (7*2 + 3*.5 + 11*10)/1_000_000)

    def test_multiple_messages_preserve_unknown_usage_and_do_not_reuse_stale_action(self):
        def notify(run, seq, usage, content):
            run.notification({'method': 'session.event', 'params': {'sessionId': 'test', 'event': {
                'seq': seq, 'type': 'assistant/message', 'data': {'usage': usage,
                    'message': {'content': content}}}}})
        for missing_usage in (None, {}, {'outputTokens': 5}):
            with self.subTest(usage=missing_usage):
                run = _SdkRun('test')
                notify(run, 1, {'inputTokens': 7, 'outputTokens': 11}, [{'type': 'text', 'text': json.dumps(ACTION)}])
                notify(run, 2, missing_usage, [{'type': 'reasoning', 'text': 'Final response has no action.'}])
                self.assertIsNone(run.content)
                self.assertEqual(run.assistant_messages, 2)
                self.assertNotIn('input_tokens', run.usage or {})
                self.assertEqual(run.usage, {'output_tokens': 16} if missing_usage else None)
                notify(run, 3, {'inputTokens': 7, 'outputTokens': 11}, [{'type': 'text', 'text': json.dumps(ACTION)}])
                self.assertNotIn('input_tokens', run.usage or {})


class _ModelFixture(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.records.append({'path': self.path, 'body': body,
                                    'key': self.headers.get('x-api-key') or self.headers.get('Authorization')})
        if body['model'] == 'fixture-http-error':
            encoded = json.dumps({'error': {'message': 'controlled fixture 503; no external call',
                                            'type': 'fixture_unavailable'}}).encode()
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        text = json.dumps(ACTION)
        model = body['model']
        if '/chat/completions' in self.path:
            chunks = [{'id':'fixture','object':'chat.completion.chunk','model':model,'created':0,
                       'choices':[{'index':0,'delta':{'role':'assistant','content':text},'finish_reason':None}]},
                      {'id':'fixture','object':'chat.completion.chunk','model':model,'created':0,
                       'choices':[{'index':0,'delta':{},'finish_reason':'stop'}],
                       'usage':{'prompt_tokens':100,'completion_tokens':80,'total_tokens':180}}]
            data = ''.join('data: '+json.dumps(c)+'\n\n' for c in chunks) + 'data: [DONE]\n\n'
        elif '/messages' in self.path:
            chunks = [('message_start',{'message':{'id':'msg_fixture','type':'message','role':'assistant','model':model,
                                                   'content':[],'stop_reason':None,'stop_sequence':None,
                                                   'usage':{'input_tokens':100,'output_tokens':0}}}),
                      ('content_block_start',{'index':0,'content_block':{'type':'text','text':''}}),
                      ('content_block_delta',{'index':0,'delta':{'type':'text_delta','text':text}}),
                      ('content_block_stop',{'index':0}),
                      ('message_delta',{'delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':80}}),
                      ('message_stop',{})]
            data = ''.join('event: '+name+'\ndata: '+json.dumps({'type':name,**payload})+'\n\n' for name,payload in chunks)
        else:
            item = {'id':'msg_fixture','type':'message','status':'completed','role':'assistant',
                    'content':[{'type':'output_text','text':text,'annotations':[]}]}
            response = {'id':'resp_fixture','object':'response','status':'completed','model':model,'created_at':0,
                        'output':[item],'usage':{'input_tokens':100,'output_tokens':80,'total_tokens':180},'error':None}
            chunks = [('response.created',{'response':{**response,'status':'in_progress','output':[]}}),
                      ('response.output_item.added',{'output_index':0,'item':{**item,'status':'in_progress','content':[]}}),
                      ('response.content_part.added',{'item_id':item['id'],'output_index':0,'content_index':0,
                                                      'part':{'type':'output_text','text':'','annotations':[]}}),
                      ('response.output_text.delta',{'item_id':item['id'],'output_index':0,'content_index':0,'delta':text}),
                      ('response.output_text.done',{'item_id':item['id'],'output_index':0,'content_index':0,'text':text}),
                      ('response.output_item.done',{'output_index':0,'item':item}),
                      ('response.completed',{'response':response})]
            data = ''.join('event: '+name+'\ndata: '+json.dumps({'type':name,'sequence_number':i,**payload})+'\n\n'
                           for i,(name,payload) in enumerate(chunks))
        encoded = data.encode()
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.send_header('Content-Length',str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@unittest.skipUnless(os.getenv('RUN_DSH_RUNTIME_TESTS') == '1', 'Opt in to installed official dsh runtime; loopback models only')
class DshOfficialRuntimeTests(unittest.TestCase):
    def test_three_native_protocols_multimodal_no_tools_and_usage(self):
        server = ThreadingHTTPServer(('127.0.0.1',0), _ModelFixture)
        server.records = []
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            for provider in ('openai','compatible','anthropic'):
                with self.subTest(provider=provider):
                    config = AgentConfig(provider=provider, model='fixture', api_key=KEY,
                        base_url=f'http://127.0.0.1:{server.server_port}' + ('' if provider == 'anthropic' else '/v1'),
                        timeout_s=60, reasoning='high')
                    action, usage = asyncio.run(dsh_action(config,'Observe the three cameras and return one JSON action.',cameras()))
                    self.assertEqual(action.model_dump(), ACTION)
                    self.assertEqual(usage['input_tokens'], 100)
                    self.assertEqual(usage['output_tokens'], 80)
                    record = server.records[-1]
                    self.assertIn(KEY, record['key'])
                    self.assertFalse(record['body'].get('tools'))
                    if provider == 'openai':
                        self.assertEqual(record['body']['reasoning']['effort'], 'high')
                        self.assertEqual(record['body']['max_output_tokens'], 16384)
                    elif provider == 'compatible':
                        self.assertEqual(record['body']['reasoning_effort'], 'high')
                        self.assertEqual(record['body']['max_tokens'], 16384)
                    else:
                        self.assertEqual(record['body']['thinking']['type'], 'enabled')
                        self.assertLess(record['body']['thinking']['budget_tokens'], record['body']['max_tokens'])
                    def image_count(value):
                        if isinstance(value,dict):
                            return int(value.get('type') in ('image','image_url','input_image')) + sum(image_count(v) for v in value.values())
                        if isinstance(value,list):
                            return sum(image_count(v) for v in value)
                        return 0
                    self.assertEqual(image_count(record['body']),3)
            self.assertEqual(len(server.records),3, 'One model call per round; no title/compaction/retry calls')
            config = AgentConfig(provider='openai', model='fixture-http-error', api_key=KEY,
                                 base_url=f'http://127.0.0.1:{server.server_port}/v1', timeout_s=30)
            with self.assertRaisesRegex(RuntimeError, 'controlled fixture 503'):
                asyncio.run(dsh_action(config, 'A controlled failure.', []))
            self.assertEqual(len(server.records),4, 'No invisible transport retries')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
