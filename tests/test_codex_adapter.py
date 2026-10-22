"""Real local subprocess fixtures verify the CLI protocol, without model calls."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from server.agent import AgentConfig, codex_action
from server.runlog import CURRENT_LOG, RunLog


ACTION = {'status': 'move', 'frame': 'world', 'position_m': [.4, 0., .5],
          'quaternion_wxyz': [0., 1., 0., 0.], 'gripper_open': 1., 'duration_s': 1.,
          'explanation': '协议测试动作。', 'hand_joint_positions_rad': None, 'robot_id': None}
USAGE = {'input_tokens': 7, 'cached_input_tokens': 2, 'output_tokens': 11}
REASONING = ['公开返回的第一段说明。', '公开返回的第二段说明。']
WARNING = 'Exceeded skills context budget. All skill descriptions were removed.'

# The fixture reads the real adapter input/schema and writes the same files and
# JSONL events as a CLI process. It does not execute any Codex binary or tools.
FIXTURE = '''import json
from pathlib import Path
import sys
import time

mode, capture_path, payload_path = sys.argv[1:4]
arguments = sys.argv[4:]
def option(name):
    return arguments[arguments.index(name) + 1]
payload = json.loads(Path(payload_path).read_text(encoding='utf-8'))
prompt = sys.stdin.buffer.read().decode('utf-8')
schema = json.loads(Path(option('--output-schema')).read_text(encoding='utf-8'))
Path(capture_path).write_text(json.dumps({'arguments': arguments, 'prompt': prompt,
    'schema': schema}), encoding='utf-8')
def emit(event):
    print(json.dumps(event), flush=True)
emit({'type': 'thread.started', 'thread_id': 'local-protocol-fixture'})
emit({'type': 'turn.started'})
if mode == 'timeout':
    emit({'type': 'item.completed', 'item': {'type': 'error', 'message': payload['warning']}})
    time.sleep(60)
else:
    for text in payload['reasoning']:
        emit({'type': 'item.completed', 'item': {'type': 'reasoning', 'text': text}})
    Path(option('--output-last-message')).write_text(json.dumps(payload['action']), encoding='utf-8')
    emit({'type': 'turn.completed', 'usage': payload['usage']})
'''


class CodexAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='codex-adapter-fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = self.root / 'cli_fixture.py'
        self.fixture.write_text(FIXTURE, encoding='utf-8')
        self.payload = self.root / 'payload.json'
        self.payload.write_text(json.dumps({'action': ACTION, 'usage': USAGE,
            'reasoning': REASONING, 'warning': WARNING}), encoding='utf-8')
        self.capture = self.root / 'received.json'
        self.invocations = []
        self.processes = []
        self.log = RunLog('agent', {'task': 'local CLI protocol fixture'}, directory=self.root / 'runs')

    async def call(self, mode, timeout_s):
        # Preserve the real process implementation before replacing this single
        # boundary; record the original CLI argv for isolation-contract checks.
        create_process = asyncio.create_subprocess_exec
        async def launch(*arguments, **kwargs):
            self.invocations.append({'arguments': list(arguments), 'options': dict(kwargs)})
            process = await create_process(sys.executable, '-u', str(self.fixture), mode,
                str(self.capture), str(self.payload), *arguments[1:], **kwargs)
            self.processes.append(process)
            return process

        config = AgentConfig(provider='codex', model='codex-adapter-fixture', reasoning='high')
        # Production validation retains its normal minimum wait; only this
        # protocol test uses a short timeout so no external call is needed.
        config = config.model_copy(update={'timeout_s': timeout_s})
        token = CURRENT_LOG.set(self.log)
        status = 'error'
        try:
            with patch.dict(os.environ, {'CODEX_CLI_PATH': sys.executable}), \
                    patch('server.agent.asyncio.create_subprocess_exec', side_effect=launch):
                result = await codex_action(config, '测试观察', [], system_prompt='协议测试系统提示')
            status = 'complete'
            return result
        finally:
            CURRENT_LOG.reset(token)
            for process in self.processes:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            self.log.finish({'status': status})

    def events(self, event):
        return [record['data'] for record in
                map(json.loads, self.log.path.read_text(encoding='utf-8').splitlines())
                if record['event'] == event]

    def test_start_events_and_warning_before_timeout_are_not_a_model_response(self):
        with self.assertRaisesRegex(TimeoutError, 'Codex 请求超过 1 秒，已终止本轮') as raised:
            asyncio.run(self.call('timeout', 1))
        self.assertIn(WARNING, str(raised.exception))
        self.assertEqual(len(self.invocations), 1)
        self.assertIsNotNone(self.processes[0].returncode)
        self.assertEqual(self.events('model_output'), [])
        calls = self.events('inference')
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call['status'], 'error')
        self.assertFalse(call['response_returned'])
        self.assertEqual(call['error'], str(raised.exception))
        self.assertIsNone(call['public_text'])
        self.assertIsNone(call['usage'])
        self.assertIsNone(call['input_tokens'])
        self.assertIsNone(call['output_tokens'])
        self.assertEqual(call['reasoning']['availability'], 'unavailable')
        self.assertIsNone(call['reasoning']['text'])
        self.assertEqual(self.log.timing_snapshot()['timing']['active_calls'], 0)

    def test_success_parses_action_usage_and_public_reasoning_from_real_subprocess(self):
        action, usage = asyncio.run(self.call('success', 5))
        self.assertEqual(action.model_dump(), ACTION)
        self.assertEqual(usage, USAGE)
        self.assertEqual(self.processes[0].returncode, 0)
        received = json.loads(self.capture.read_text(encoding='utf-8'))
        self.assertEqual(received['prompt'], '协议测试系统提示\n测试观察')
        self.assertEqual(set(received['schema']['required']), set(ACTION))
        self.assertEqual(len(self.invocations), 1)
        arguments = self.invocations[0]['arguments']
        self.assertEqual(received['arguments'], arguments[1:])
        self.assertEqual(arguments[arguments.index('--model') + 1], 'codex-adapter-fixture')
        self.assertIn('model_reasoning_effort="high"', arguments)
        settings = [arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == '-c']
        disabled = [arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == '--disable']
        self.assertIn('skills.include_instructions=false', settings)
        self.assertIn('skills.bundled.enabled=false', settings)
        self.assertIn('plugins', disabled)
        self.assertNotIn('skills.max_context_tokens=1', settings)
        self.assertEqual(arguments[-1], '-')
        calls = self.events('inference')
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call['status'], 'complete')
        self.assertTrue(call['response_returned'])
        self.assertIsNone(call['error'])
        self.assertEqual(json.loads(call['public_text']), ACTION)
        self.assertEqual(call['usage'], USAGE)
        self.assertEqual((call['input_tokens'], call['output_tokens'], call['total_tokens']), (7, 11, 18))
        self.assertEqual(call['reasoning']['availability'], 'returned')
        self.assertEqual(call['reasoning']['text'], '\n\n'.join(REASONING))
        self.assertEqual(len(self.events('model_output')), 1)
        self.assertEqual(self.log.timing_snapshot()['timing']['active_calls'], 0)


if __name__ == '__main__':
    unittest.main()
