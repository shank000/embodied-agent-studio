"""Claude Code CLI in isolated API mode, proposing one workbench action per run."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from .actions import PoseAction
from .runlog import model_output
from .telemetry import track_inference

CLAUDE_CODE_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')


def _is_native_executable(path):
    if not path.is_file():
        return False
    with path.open('rb') as file:
        header = file.read(4)
    return header[:2] == b'MZ' or header == b'\x7fELF' or header in (
        b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe',
        b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca')


def _native_package_binary(package):
    arch = {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64', 'aarch64': 'arm64'}.get(platform.machine().lower())
    if not arch or sys.platform not in ('win32', 'linux', 'darwin'):
        return None
    suffix = '-musl' if sys.platform == 'linux' and platform.libc_ver()[0] != 'glibc' else ''
    filename = 'claude.exe' if sys.platform == 'win32' else 'claude'
    native_name = f'claude-code-{sys.platform}-{arch}{suffix}'
    # The pinned official package ships native optional dependencies; prefer
    # them to the wrapper's placeholder when npm install --ignore-scripts ran.
    candidates = [package.parent / native_name / filename,
                  package / 'node_modules' / '@anthropic-ai' / native_name / filename,
                  package / 'bin' / filename]
    for candidate in candidates:
        if _is_native_executable(candidate):
            return candidate
    return None


def claude_code_executable():
    """Return a native executable, never an npm shell/Node child wrapper."""
    override = os.getenv('CLAUDE_CLI_PATH')
    if override:
        candidate = Path(override).expanduser()
        if _is_native_executable(candidate):
            return str(candidate)
        raise ValueError('CLAUDE_CLI_PATH 须指向已安装的 Claude Code 原生可执行文件。')
    package = Path(__file__).resolve().parents[1] / 'node_modules' / '@anthropic-ai' / 'claude-code'
    native = _native_package_binary(package)
    if native:
        return str(native)
    for name in ('claude', 'claude.exe') if os.name == 'nt' else ('claude',):
        located = shutil.which(name)
        if not located:
            continue
        path = Path(located)
        if path.suffix.lower() in ('.cmd', '.bat', '.ps1'):
            native = _native_package_binary(path.parent / 'node_modules' / '@anthropic-ai' / 'claude-code')
            if native:
                return str(native)
            continue
        # npm may expose an extensionless shell shim on both Windows and Unix.
        with path.open('rb') as file:
            header = file.read(2048)
        if header.startswith(b'#!'):
            resolved = path.resolve()
            package = resolved.parent.parent if resolved.parent.name == 'bin' else path.parent / 'node_modules' / '@anthropic-ai' / 'claude-code'
            native = _native_package_binary(package)
            if native:
                return str(native)
            continue
        if _is_native_executable(path):
            return str(path)
    raise ValueError('未找到 Claude Code 原生 CLI。请在项目中运行 npm install，或安装 Claude Code 2.1.286+ 后用 CLAUDE_CLI_PATH 指定原生程序；此 harness 不会改用普通 API 调用。')


def _returned_output(stdout):
    result = None
    public = []
    thinking = []
    messages = {}
    errors = []
    for line in stdout.decode('utf-8', 'replace').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'result':
            result = event
        elif event.get('type') == 'assistant':
            message = event.get('message')
            if not isinstance(message, dict):
                continue
            for block in message.get('content') or []:
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'text' and isinstance(block.get('text'), str):
                    public.append(block['text'])
                if block.get('type') == 'thinking' and isinstance(block.get('thinking'), str):
                    thinking.append(block['thinking'])
            if isinstance(message.get('usage'), dict):
                messages[message.get('id') or len(messages)] = message['usage']
        elif event.get('type') == 'error':
            error = event.get('error', event.get('message'))
            errors.append(error.get('message', str(error)) if isinstance(error, dict) else str(error))
    usage = result.get('usage') if result and isinstance(result.get('usage'), dict) else None
    if usage is None and messages:
        # Keep partial returned usage when interrupted. Missing fields stay
        # missing so the cost ledger cannot silently claim full accounting.
        usage = {}
        for key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
            values = [item.get(key) for item in messages.values()]
            if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
                usage[key] = sum(values)
    if usage is not None:
        usage = dict(usage)
        # Anthropic input/cache buckets are disjoint; total usage includes
        # cached reads and writes as well as uncached input and output.
        counts = [usage.get(name) for name in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')]
        if 'total_tokens' not in usage and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts):
            usage['total_tokens'] = sum(counts)
        details = usage.get('output_tokens_details')
        thinking_tokens = details.get('thinking_tokens') if isinstance(details, dict) else None
        if isinstance(thinking_tokens, int) and not isinstance(thinking_tokens, bool) and thinking_tokens >= 0:
            usage['reasoning_tokens'] = thinking_tokens
    structured = result.get('structured_output') if result else None
    content = json.dumps(structured, ensure_ascii=False) if structured is not None else '\n\n'.join(public) or None
    if result and result.get('errors'):
        returned_errors = result['errors']
        if isinstance(returned_errors, list):
            errors.extend(str(error) for error in returned_errors)
        else:
            errors.append(str(returned_errors))
    if result and result.get('is_error') and result.get('result'):
        errors.append(str(result['result']))
    return result, structured, usage, content, '\n\n'.join(thinking) or None, errors


@track_inference('agent_inference')
async def claude_code_action(config, prompt, cameras, *, response_type=PoseAction,
                             system_prompt=None, schema_name='robot_action'):
    if config.provider != 'anthropic':
        raise ValueError('Claude Code harness 需要 Anthropic Messages 模型接口；请选择 Messages 协议并配置地址、密钥与模型名。')
    key = config.api_key.get_secret_value()
    if not key:
        raise ValueError('Claude Code 的隔离 API 模式需要模型 API Key；不会读取或复用 Claude 订阅登录。')
    if config.reasoning is not None and config.reasoning not in CLAUDE_CODE_EFFORTS:
        raise ValueError('Claude Code 2.1.316 的 --effort 仅支持 low / medium / high / xhigh / max；该等级还需所选模型支持。')
    if system_prompt is None:
        from .agent import SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT
    executable = claude_code_executable()
    content = [{'type': 'text', 'text': prompt}]
    for camera in cameras:
        content += [{'type': 'text', 'text': f'Camera: {camera["name"]}'},
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': camera['image_base64']}}]
    request = {'type': 'user', 'message': {'role': 'user', 'content': content}, 'parent_tool_use_id': None}
    with tempfile.TemporaryDirectory(prefix='robot-claude-') as folder:
        # Bare does not load OAuth/keychain credentials. Isolate SDK settings
        # too, so selected endpoints/models are not replaced by user defaults.
        environment = {name: value for name, value in os.environ.items()
                       if not name.upper().startswith(('ANTHROPIC_', 'CLAUDE_')) and name.upper() != 'MAX_THINKING_TOKENS'}
        environment.update(ANTHROPIC_API_KEY=key, ANTHROPIC_BASE_URL=config.base_url.rstrip('/'),
                           CLAUDE_CONFIG_DIR=folder, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
                           CLAUDE_CODE_MAX_RETRIES='0', MAX_STRUCTURED_OUTPUT_RETRIES='1',
                           CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(config.max_output_tokens))
        arguments = [executable, '--print', '--bare', '--restricted', '--tools', '',
                     '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
                     '--setting-sources', '', '--disable-slash-commands', '--no-session-persistence',
                     '--settings', json.dumps({'availableModels': [config.model], 'fallbackModel': []}),
                     '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose',
                     '--model', config.model, '--system-prompt', system_prompt,
                     '--json-schema', json.dumps(response_type.action_schema(), ensure_ascii=False),
                     '--max-turns', '3']
        # StructuredOutput may use an internal tool/result turn. This is one
        # workbench decision; result.usage accounts for all CLI model requests.
        if config.reasoning is not None:
            arguments += ['--effort', config.reasoning]
        try:
            process = await asyncio.create_subprocess_exec(*arguments, cwd=folder, env=environment,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except OSError as exc:
            raise RuntimeError(f'无法启动 Claude Code 原生 CLI：{exc}。请检查安装和 CLAUDE_CLI_PATH。') from exc
        communication = asyncio.create_task(process.communicate((json.dumps(request, ensure_ascii=False) + '\n').encode('utf-8')))

        def capture(stdout):
            parsed = _returned_output(stdout)
            result, structured, usage, content, reasoning, errors = parsed
            if content or usage is not None or reasoning:
                model_output({'provider': config.provider, 'harness': 'claude_code', 'model': config.model,
                              'content': content, 'usage': usage, 'returned_reasoning': reasoning,
                              'cli_num_turns': result.get('num_turns') if result else None,
                              'cli_model_usage': result.get('modelUsage') if result else None,
                              'cli_total_cost_usd': result.get('total_cost_usd') if result else None})
            return parsed

        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), config.timeout_s)
        except (asyncio.CancelledError, TimeoutError) as exc:
            if process.returncode is None:
                process.kill()
            stdout, stderr = await communication
            capture(stdout)
            if isinstance(exc, TimeoutError):
                detail = stderr.decode('utf-8', 'replace').replace(key, '[REDACTED]')[-1000:]
                raise TimeoutError(f'Claude Code 请求超过 {config.timeout_s} 秒，已终止本轮。{detail}') from exc
            raise
        result, structured, usage, content, reasoning, errors = capture(stdout)
        if process.returncode or not result or result.get('is_error') or result.get('subtype') != 'success':
            detail = '\n'.join(errors) or stderr.decode('utf-8', 'replace') or (str(result.get('subtype')) if result else 'CLI 未返回最终结果')
            raise RuntimeError(f'Claude Code 调用失败（exit {process.returncode}）：{detail.replace(key, "[REDACTED]")[-1600:]}')
        if not isinstance(structured, dict):
            raise ValueError('Claude Code 未返回完整 JSON 动作（structured_output）。请确认 CLI 版本和模型支持结构化输出；未执行任何动作。')
        return response_type.model_validate(structured), usage
