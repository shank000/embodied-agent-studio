"""Drive the official DeepSeek Harness SDK profile, with robot actions as output.

The SDK owns its Agent, model adapters, multimodal admission and session loop.
This module is only its stdio client. Local coding tools are disabled: validated
actions return to the workbench's existing simulator/controller boundary.
"""
import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit
from uuid import uuid4

from .actions import PoseAction
from .runlog import model_output
from .telemetry import track_inference


DSH_TESTED_VERSION = '0.2.1-rc.5'
DSH_PROTOCOLS = {'openai': 'openai-responses', 'compatible': 'openai-completions',
                 'anthropic': 'anthropic-messages'}
DSH_REASONING = frozenset({'off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'})


def dsh_command():
    """Resolve the official CLI without invoking npm's Windows shell wrappers."""
    override = os.environ.get('DSH_CLI_PATH', '').strip()
    root = Path(__file__).resolve().parents[1]
    candidates = [Path(override)] if override else [
        root / 'node_modules' / '@deepseek-ai' / 'dsh' / 'lib' / 'bin.js']
    if not override:
        system = shutil.which('dsh') or shutil.which('dsh.cmd')
        if system:
            candidates.append(Path(system))
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() in ('.cmd', '.ps1'):
            modules = path.parent.parent if path.parent.name == '.bin' else path.parent / 'node_modules'
            path = modules / '@deepseek-ai' / 'dsh' / 'lib' / 'bin.js'
        resolved = path.resolve()
        if resolved.suffix.lower() in ('.js', '.mjs'):
            manifest = resolved.parent.parent / 'package.json'
            if manifest.is_file():
                package = json.loads(manifest.read_text(encoding='utf-8'))
                if package.get('name') == '@deepseek-ai/dsh' and package.get('version') != DSH_TESTED_VERSION:
                    raise ValueError(f'dsh {package.get("version")} 尚未通过工作台 SDK 兼容验收；'
                        f'请在项目中运行 npm install（固定版本 {DSH_TESTED_VERSION}），或调整 DSH_CLI_PATH。')
            node = shutil.which('node')
            if not node:
                raise ValueError('dsh npm 运行时需要 Node.js；安装 Node.js 后重新启动工作台。')
            if resolved.is_file():
                return [node, str(resolved)]
        elif resolved.suffix.lower() in ('.exe', ''):
            # A native SDK runtime or Unix launcher is supported; the SDK
            # handshake and strict profile patches check its capabilities.
            return [str(resolved)]
    raise ValueError('没有找到 dsh；请在项目中运行 npm install，或设置 DSH_CLI_PATH 指向官方 dsh 程序。')


def _route(config, has_images):
    if config.provider not in DSH_PROTOCOLS:
        raise ValueError('dsh 支持 OpenAI Responses、兼容 Chat Completions 和 Anthropic Messages；'
                         '请选择其中一种模型接口并填写 API Key，不能复用 Codex 登录。')
    parsed = urlsplit(config.base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('dsh 模型地址须为不含用户名、密码、查询参数或片段的 HTTP(S) API 基础地址。')
    key = config.api_key.get_secret_value()
    if not key:
        raise ValueError('dsh 需要独立模型 API Key；请在模型配置中填写。')
    effort = config.reasoning
    deepseek = config.provider == 'compatible' and parsed.hostname == 'api.deepseek.com'
    if deepseek and config.deepseek_thinking == 'disabled':
        effort = 'off'
    if effort is not None and effort not in DSH_REASONING:
        raise ValueError('当前 dsh 官方适配器支持的推理强度为 off / minimal / low / medium / high / xhigh / max；'
                         f'不支持 {effort!r}，请改用模型默认或工作台视觉闭环 Harness。')
    model = {'id': config.model, 'input': ['text', 'image'] if has_images else ['text'],
             'maxTokens': config.max_output_tokens}
    if effort is not None:
        model['reasoningEfforts'] = {effort: None if effort == 'off' else effort}
    route = {'api': DSH_PROTOCOLS[config.provider], 'baseURL': config.base_url.rstrip('/'),
             'apiKeyEnv': 'EMBODIWORKS_DSH_API_KEY', 'models': [model],
             'timeoutMs': config.timeout_s * 1000, 'streamIdleTimeoutMs': config.timeout_s * 1000,
             'transport': 'sse', 'retryPolicy': {'mode': 'normal', 'maxRetries': 0}}
    if config.provider == 'compatible':
        route['compat'] = {'supportsDeveloperRole': False, 'maxTokensField': 'max_tokens'}
        if deepseek:
            route['compat']['thinkingFormat'] = 'deepseek'
    return route, effort, key


def _profile_patch(route):
    # sdk-minimal owns an explicit, small tree: no project instructions, user
    # credentials, MCP servers, web tools or implicit tool plugins from dsh-base.
    # Strict row IDs deliberately fail on incompatible upstream profile changes.
    disabled = ('deepseek-llm-api-extensions', 'session-log-deepseek',
                'plugin-package-inventory-deepseek', 'llm-deepseek', 'tool-fs',
                'tool-working-directory', 'persistent-bash', 'persistent-pwsh',
                'terminal-bash', 'terminal-pwsh', 'pty', 'subprocess',
                'compaction-basic', 'image-offload', 'llm-retry')
    return ([{'id': name, 'disabled': True} for name in disabled]
            + [{'id': 'sandbox-policy', 'config': {'mode': 'read-only'}},
               {'insert': [{'id': 'workbench-llm', 'name': '@deepseek-ai/dsh-llm-pi-ai',
                            'config': {'providers': {'workbench': route}}}]}])


def _environment(directory, key, system_prompt):
    allowed = {'PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP', 'TMPDIR',
               'LANG', 'LC_ALL', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
               'SSL_CERT_FILE', 'SSL_CERT_DIR', 'NODE_EXTRA_CA_CERTS', 'NUMBER_OF_PROCESSORS',
               'PROCESSOR_ARCHITECTURE'}
    env = {name: value for name, value in os.environ.items() if name.upper() in allowed}
    home = str(directory / 'home')
    Path(home).mkdir()
    env.update(DSH_HOME=home, HOME=home, USERPROFILE=home, APPDATA=home, LOCALAPPDATA=home,
               DSH_TELEMETRY_DISABLED='1', EMBODIWORKS_DSH_API_KEY=key,
               DSH_SYSTEM_PROMPT=system_prompt)
    return env


def _usage(value, provider='openai'):
    """Map dsh's disjoint counts to each provider's existing billing contract.

    The pinned llm-pi-ai mapUsage omits cacheRead/cacheWrite exactly when zero.
    Missing the entire usage object remains unknown; it never becomes zero.
    """
    if not isinstance(value, dict):
        return None
    def count(name):
        number = value.get(name)
        return number if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number) and number >= 0 else None
    result = {}
    input_count = count('inputTokens')
    cached = count('cacheReadTokens')
    written = count('cacheWriteTokens')
    output = count('outputTokens')
    if input_count is not None:
        if provider == 'anthropic':
            result.update(input_tokens=input_count, cache_read_input_tokens=cached or 0,
                          cache_creation_input_tokens=written or 0)
        elif provider == 'deepseek':
            result.update(input_tokens=input_count + (cached or 0) + (written or 0),
                          prompt_cache_miss_tokens=input_count, prompt_cache_hit_tokens=cached or 0)
        else:
            result.update(input_tokens=input_count + (cached or 0) + (written or 0),
                          cached_input_tokens=cached or 0, cache_write_input_tokens=written or 0)
    if output is not None:
        result['output_tokens'] = output
    for original, normalized in [('totalTokens', 'total_tokens'), ('reasoningTokens', 'reasoning_tokens')]:
        if count(original) is not None:
            result[normalized] = count(original)
    return result or None


class _SdkRun:
    def __init__(self, session_id, usage_provider='openai'):
        self.session_id = session_id
        self.usage_provider = usage_provider
        self.content = None
        self.reasoning = []
        self.usage = None
        self.finish = None
        self.model = None
        self.seen = set()
        self.assistant_messages = 0

    def notification(self, message):
        params = message.get('params') or {}
        if message.get('method') != 'session.event' or params.get('sessionId') != self.session_id:
            return
        event = params.get('event') or {}
        seq = event.get('seq')
        if seq is not None:
            if seq in self.seen:
                return
            self.seen.add(seq)
        data = event.get('data') or {}
        if event.get('type') == 'assistant/message':
            response = data.get('message') or {}
            blocks = response.get('content') or []
            texts = [b['text'] for b in blocks if b.get('type') == 'text' and isinstance(b.get('text'), str)]
            # Only the last completed assistant message can propose an action.
            # Do not execute an earlier JSON if the final response is empty.
            self.content = ''.join(texts) or None
            self.reasoning.extend(b['text'] for b in blocks if b.get('type') == 'reasoning' and isinstance(b.get('text'), str))
            self.model = (response.get('source') or {}).get('model') or self.model
            usage = _usage(data.get('usage'), self.usage_provider)
            if self.assistant_messages == 0:
                self.usage = usage
            elif usage is not None and self.usage is not None:
                # A total is known only if every message returned that count.
                # Missing usage must not turn an observed subtotal into zero-
                # filled, apparently complete billing data.
                self.usage = {name: self.usage[name] + usage[name]
                              for name in self.usage.keys() & usage.keys()} or None
            else:
                self.usage = None
            self.assistant_messages += 1
            if any(b.get('type') == 'tool-call' for b in blocks):
                raise RuntimeError('dsh 返回了本轮不允许的工具调用；工具未执行，请让模型直接返回 JSON 动作。')
        elif event.get('type') == 'turn/end':
            self.finish = data.get('reason')
        elif event.get('type') == 'tool/call':
            raise RuntimeError('dsh 请求了未开放的工具；本轮已停止，动作必须通过工作台控制器执行。')


@track_inference('agent_inference')
async def dsh_action(config, prompt, cameras, *, response_type=PoseAction,
                     system_prompt=None, schema_name='robot_action'):
    if system_prompt is None:
        from .agent import SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT
    route, effort, key = _route(config, bool(cameras))
    command = dsh_command()
    schema = response_type.action_schema()
    instruction = (system_prompt + '\nReturn exactly one JSON object, no Markdown and no tool calls. '
                   'Use this JSON Schema (' + schema_name + '):\n' + json.dumps(schema, ensure_ascii=False))
    usage_provider = ('deepseek' if config.provider == 'compatible' and
                      urlsplit(config.base_url).hostname == 'api.deepseek.com' else config.provider)
    result = _SdkRun('workbench-' + uuid4().hex, usage_provider)
    with tempfile.TemporaryDirectory(prefix='workbench-dsh-') as folder:
        directory = Path(folder)
        workspace = directory / 'workspace'
        workspace.mkdir()
        patch = directory / 'workbench.patch.json'
        patch.write_text(json.dumps(_profile_patch(route), ensure_ascii=False), encoding='utf-8')
        env = _environment(directory, key, instruction)
        process = await asyncio.create_subprocess_exec(*command, '--profile', 'sdk-minimal', '--patch', str(patch),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace), env=env, limit=16 * 1024 * 1024,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        diagnostics = bytearray()

        async def read_stderr():
            while chunk := await process.stderr.read(8192):
                diagnostics.extend(chunk)
                if len(diagnostics) > 16000:
                    del diagnostics[:-16000]

        errors = asyncio.create_task(read_stderr())
        request_id = 0

        async def request(method, params=None):
            nonlocal request_id
            request_id += 1
            ident = request_id
            frame = {'jsonrpc': '2.0', 'id': ident, 'method': method}
            if params is not None:
                frame['params'] = params
            process.stdin.write((json.dumps(frame, ensure_ascii=False) + '\n').encode('utf-8'))
            await process.stdin.drain()
            while True:
                line = await process.stdout.readline()
                if not line:
                    raise RuntimeError('dsh SDK 连接提前关闭；请检查运行时版本与模型接口。')
                try:
                    message = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get('method') and 'id' in message:
                    # The profile exposes no interactive tools. Never grant an
                    # unexpected SDK request or let it hold the robot round open.
                    answer = {'jsonrpc': '2.0', 'id': message['id'], 'error':
                              {'code': -32601, 'message': 'Workbench does not expose interactive tools'}}
                    process.stdin.write((json.dumps(answer) + '\n').encode('utf-8'))
                    await process.stdin.drain()
                    continue
                result.notification(message)
                if message.get('id') == ident:
                    if message.get('error'):
                        raise RuntimeError('dsh SDK: ' + str(message['error'].get('message', message['error'])))
                    return message.get('result')

        async def run():
            initialized = {'cwd': str(workspace), 'provider': 'workbench', 'model': config.model,
                           'maxTokens': config.max_output_tokens}
            if effort is not None:
                initialized['reasoningEffort'] = effort
            handshake = await request('initialize', initialized)
            if not isinstance(handshake, dict) or (handshake.get('serverInfo') or {}).get('name') != 'deepseek-harness-sdk-runtime':
                raise RuntimeError('dsh 未返回官方 SDK 握手；请安装工作台固定版本的官方 dsh。')
            blocks = [{'type': 'text', 'text': prompt}]
            for camera in cameras:
                blocks.extend([{'type': 'text', 'text': 'Camera: ' + str(camera['name'])},
                               {'type': 'image', 'data': camera['image_base64'],
                                'mimeType': camera.get('mime_type') or 'image/jpeg'}])
            await request('session/prompt', {'sessionId': result.session_id, 'contentBlocks': blocks})
            await request('session/wait', {'sessionId': result.session_id})

        failure = None
        try:
            await asyncio.wait_for(run(), timeout=config.timeout_s)
        except BaseException as exc:
            failure = exc
        finally:
            if process.returncode is None:
                # Official SDK has no cancel method. Closing its process is its
                # documented cancellation boundary; no tools/child shells exist.
                if failure is None:
                    try:
                        await asyncio.wait_for(request('shutdown'), timeout=3)
                        await asyncio.wait_for(process.wait(), timeout=3)
                    except asyncio.CancelledError as exc:
                        failure = exc
                        if process.returncode is None:
                            process.kill()
                    except Exception:
                        if process.returncode is None:
                            process.kill()
                else:
                    process.kill()
            await process.wait()
            await errors
            if result.content is not None or result.usage is not None or result.reasoning:
                model_output({'provider': config.provider, 'harness': 'dsh',
                              'model': result.model or config.model, 'content': result.content,
                              'usage': result.usage, 'returned_reasoning': '\n\n'.join(result.reasoning) or None,
                              'dsh_assistant_messages': result.assistant_messages})
        detail = diagnostics.decode('utf-8', 'replace').replace(key, '[REDACTED]')[-1800:]
        if isinstance(failure, asyncio.CancelledError):
            raise failure
        if isinstance(failure, TimeoutError):
            raise TimeoutError(f'dsh 请求超过 {config.timeout_s} 秒，已终止本轮。{detail}') from failure
        if failure is not None:
            raise RuntimeError(str(failure).replace(key, '[REDACTED]') + (('\n' + detail) if detail else '')) from failure
    if not isinstance(result.finish, dict) or result.finish.get('kind') != 'completed':
        raise RuntimeError('dsh 未完整结束本轮：' + str(result.finish or '缺少 turn/end 事件').replace(key, '[REDACTED]'))
    if not result.content:
        raise RuntimeError('dsh 未返回 JSON 动作。请确认模型支持所选接口及相机图像输入。')
    try:
        action = response_type.model_validate_json(result.content)
    except ValueError as exc:
        raise RuntimeError('dsh 返回的 JSON 动作未通过工作台校验：' + str(exc)) from exc
    return action, result.usage
