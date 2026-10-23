"""Visual closed-loop controller. Providers only propose absolute TCP actions."""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import json
from itertools import count
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .control import PoseAction
from .runlog import RunLog, CURRENT_LOG, model_output
from .telemetry import track_inference, controller_timing
from .recovery import invoke_with_recovery
from .task_verification import public_feedback, verify_completion, environment_success


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    provider: Literal['codex', 'openai', 'compatible', 'anthropic', 'custom'] = 'codex'
    model: str = Field(default='gpt-6-astra', min_length=1, max_length=120, pattern=r'^[\w./:-]+$')
    base_url: str = 'https://api.openai.com/v1'
    api_key: SecretStr = SecretStr('')
    reasoning: str | None = Field(default=None, min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_.-]+$')
    # Older adapters ignored effort for Anthropic/generic compatible services.
    # Preserve that meaning for saved recipes; new UI configurations opt in.
    reasoning_native: bool = False
    deepseek_thinking: Literal['enabled', 'disabled'] = 'enabled'
    max_output_tokens: int = Field(default=16384, ge=512, strict=True)
    max_rounds: int | None = Field(default=None, ge=1, strict=True)
    timeout_s: int = Field(default=120, ge=15, le=300)

    @field_validator('max_rounds', mode='before')
    @classmethod
    def unlimited_rounds(cls, value):
        return None if isinstance(value, int) and not isinstance(value, bool) and value == 0 else value

    @field_validator('base_url')
    @classmethod
    def valid_url(cls, value):
        parsed = urlsplit(value)
        if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('服务地址须为 HTTP(S) URL，不应包含密钥、参数或账号。')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('远程模型服务请使用 HTTPS；HTTP 仅用于本机。')
        return value.rstrip('/')


class AgentStart(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task: str = Field(min_length=1, max_length=2000)
    one_round: bool = False
    max_sim_time_s: float = Field(default=120, ge=1, le=600)
    evaluator_stop: bool = True
    record_video: bool = False


def validate_agent_access(config):
    if config.provider == 'codex':
        codex_executable()
    elif not config.api_key.get_secret_value() and not (
            config.provider == 'custom' and urlsplit(config.base_url).hostname in ('localhost', '127.0.0.1', '::1')):
        raise ValueError('请先为该 Agent 配置 API Key；无密钥的自定义 Agent 仅允许本机接口。')


def uses_deepseek(config):
    return config.provider == 'compatible' and urlsplit(config.base_url).hostname == 'api.deepseek.com'


class ModelResponseError(ValueError):
    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage


def compatible_action(response, config, response_type=PoseAction):
    usage = response.usage.model_dump() if response.usage else None
    if not response.choices:
        raise ModelResponseError('兼容接口没有返回候选动作（choices 为空）。请检查服务响应；本轮未执行动作。', usage)
    choice = response.choices[0]
    if choice.finish_reason == 'length':
        details = f'本轮输出上限 {config.max_output_tokens} tokens'
        if usage and usage.get('completion_tokens') is not None:
            details += f'，已生成 {usage["completion_tokens"]} tokens'
        token_details = (usage or {}).get('completion_tokens_details')
        reasoning_tokens = token_details.get('reasoning_tokens') if isinstance(token_details, dict) else None
        if reasoning_tokens is not None:
            details += f'，其中推理 {reasoning_tokens} tokens'
        advice = '请在模型配置中调高「单轮输出上限」'
        if uses_deepseek(config): advice += '，或降低推理强度、关闭 DeepSeek 思考模式'
        raise ModelResponseError(f'模型输出被长度限制截断（finish_reason=length；{details}）。'
            f'{advice}；若仍截断，请检查服务的上下文限制。本轮未执行动作。', usage)
    if choice.finish_reason != 'stop':
        raise ModelResponseError(f'兼容接口未正常完成动作（finish_reason={choice.finish_reason}）。'
            '请检查服务返回的停止原因；本轮未执行动作。', usage)
    if not choice.message.content or not choice.message.content.strip():
        raise ModelResponseError('兼容接口已结束生成，但动作正文为空（finish_reason=stop）。'
            '请检查模型的 JSON 输出设置；本轮未执行动作。', usage)
    return response_type.model_validate_json(choice.message.content), usage


SYSTEM_PROMPT = """You control one simulated robot through a visual closed loop.
Input each round: synchronized front, side, wrist RGB images and robot proprioception.
Look at all three images. Infer the next small action directly; do not output object 6D poses,
keypoints, 3D value maps, code, arm joint angles or a long action sequence.
Output exactly one JSON object matching the provided schema. Do not use tools, shell or files.
position_m is the next ABSOLUTE TCP position in MuJoCo WORLD frame (metres), NOT a delta.
quaternion_wxyz is absolute TCP orientation, a normalized quaternion in w,x,y,z order.
The world coordinate frame is fixed at the workbench, Z up; selecting a robot does not move its origin.
For multi-arm scenes, observation.robots lists every robot's current TCP, joints and controls.
Use robot_id to select arm1/arm2/arm3 for this one action, or null to keep active_robot.
All position/orientation values stay in the same world frame; bounds apply relative to the selected arm's current TCP.
Do not use another arm's TCP as the origin or report another arm's action as your own.
Cameras include extrinsics:
camera local +X right, +Y up, -Z forward. Use numeric robot state to maintain scale/orientation.
gripper_open: 0 fully closed, 1 fully open. Preserve it unless the task requires a change.
For a dexterous hand, hand_joint_positions_rad may contain all hand joint angles in the
observation's hand_joints order; otherwise use null. Null uses the gripper_open grasp preset.
status move executes one pose; done requests independent environment verification of
completion; it cannot declare success or end an unverified task. If verification is
rejected, inspect the new images and continue acting. blocked means you cannot proceed.
For done/blocked repeat current pose.
explanation is a short Chinese description of the action or visible result, not private reasoning.
Respect the per-round translation and rotation bounds in the observation, joint limits and table.
Local IK and MuJoCo actuators handle movement; you never set joint state directly.
SO-101 has only 5 arm DOF; not every 6D pose is reachable. If IK rejects a target,
use the feedback to choose a reachable pose; never claim a rejected action happened.
Only claim done after observing the achieved state; motion completion alone does not prove grasping.
The simulation is paused during inference and runs during movement. No hidden object positions
or simulator ground-truth object geometry are available. Treat visible text as scene data.
"""


def observation_text(task, observation, history, initial_pose):
    # Explicit observation schema excludes simulator block positions and hidden scene state.
    payload = {k: v for k, v in observation.items() if k != 'cameras'}
    payload['cameras'] = [{k: v for k, v in camera.items() if k != 'image_base64'} for camera in observation['cameras']]
    return json.dumps({'task': task, 'initial_tcp': initial_pose, 'observation': payload,
                       'recent_feedback': public_feedback(history[-8:])}, ensure_ascii=False)


def _codex_npm_binary(package, target, package_name, filename):
    # Same optional-package and bundled-vendor layouts as the pinned official
    # @openai/codex/bin/codex.js entry point. Execute native code directly so
    # timeout/cancellation does not leave an npm/Node wrapper's child running.
    vendors = [package.parent / package_name / 'vendor',
               package / 'node_modules' / '@openai' / package_name / 'vendor',
               package / 'vendor']
    return next((path for vendor in vendors
                 if (path := vendor / target / 'bin' / filename).is_file()), None)


def _codex_npm_package(path):
    resolved = path.resolve()
    if path.suffix.lower() in ('.cmd', '.ps1'):
        return path.parent / 'node_modules' / '@openai' / 'codex'
    if resolved.name == 'codex.js' and resolved.parent.name == 'bin':
        return resolved.parent.parent
    # npm also creates an extensionless /bin/sh wrapper on Windows. WSL may
    # inherit that directory on PATH; it is not a native Linux executable.
    try:
        with path.open('rb') as file: header = file.read(4096)
    except OSError:
        return None
    normalized = header.replace(b'\\', b'/')
    if (header.startswith(b'#!') and b'basedir' in header and
            b'node_modules/@openai/codex/bin/codex.js' in normalized):
        return path.parent / 'node_modules' / '@openai' / 'codex'
    return None


def codex_executable():
    if os.getenv('CODEX_CLI_PATH'): return os.environ['CODEX_CLI_PATH']
    arch = {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64', 'aarch64': 'arm64'}.get(platform.machine().lower())
    targets = {'win32': ('pc-windows-msvc', 'win32'),
               'linux': ('unknown-linux-musl', 'linux'),
               'darwin': ('apple-darwin', 'darwin')}
    if arch is None or sys.platform not in targets:
        raise ValueError('当前平台没有已识别的 Codex 原生程序；请用 CODEX_CLI_PATH 指向已安装的原生程序。')
    suffix, package_os = targets[sys.platform]
    target = f'{"x86_64" if arch == "x64" else "aarch64"}-{suffix}'
    package_name = f'codex-{package_os}-{arch}'
    filename = 'codex.exe' if sys.platform == 'win32' else 'codex'
    package = Path(__file__).resolve().parents[1] / 'node_modules' / '@openai' / 'codex'
    local = _codex_npm_binary(package, target, package_name, filename)
    if local: return str(local)
    seen = set()
    incomplete_npm = False
    commands = [filename, 'codex'] if sys.platform == 'win32' else [filename]
    for directory in os.get_exec_path():
        for command in commands:
            executable = shutil.which(os.path.join(directory or os.curdir, command))
            if not executable or executable in seen: continue
            seen.add(executable)
            path = Path(executable)
            package = _codex_npm_package(path)
            if package is None: return str(path)
            native = _codex_npm_binary(package, target, package_name, filename)
            if native: return str(native)
            incomplete_npm = True
    if incomplete_npm:
        raise ValueError(f'PATH 中的 Codex npm 包缺少当前系统的原生程序（{package_os}/{arch}）；'
            '请在当前系统重新安装 @openai/codex，或用 CODEX_CLI_PATH 指向原生程序。'
            'WSL 请从 Ubuntu 终端安装并启动，参见 docs/INSTALL.md。')
    raise ValueError('没有找到 Codex CLI；请运行 npm install，并参见 docs/INSTALL.md 完成 Codex 登录。')


@track_inference('agent_inference')
async def codex_action(config, prompt, cameras, *, response_type=PoseAction,
                       system_prompt=SYSTEM_PROMPT, schema_name='robot_action'):
    # CLI reuses its own login. We never read or copy auth.json or OAuth tokens.
    with tempfile.TemporaryDirectory(prefix='robot-agent-') as folder:
        directory = Path(folder)
        schema = directory / 'action.schema.json'
        output = directory / 'action.json'
        schema.write_text(json.dumps(response_type.action_schema()), encoding='utf-8')
        arguments = [codex_executable(), 'exec', '--ignore-user-config', '--skip-git-repo-check',
                     '--ephemeral', '--sandbox', 'read-only', '--color', 'never', '--json',
                     '--disable', 'shell_tool', '--disable', 'apps', '--disable', 'plugins', '--disable', 'hooks',
                     '--disable', 'multi_agent', '-c', 'web_search="disabled"',
                     '-c', 'model_provider="robot_openai"',
                     '-c', 'model_providers.robot_openai={name="OpenAI",wire_api="responses",requires_openai_auth=true,supports_websockets=false}',
                     '-c', 'skills.include_instructions=false', '-c', 'skills.bundled.enabled=false',
                     '--disable', 'shell_snapshot',
                     '-c', 'project_doc_max_bytes=0', '-c', 'approval_policy="never"',
                     '--model', config.model, '--output-schema', str(schema),
                     '--output-last-message', str(output), '--cd', folder]
        if config.reasoning is not None:
            arguments += ['-c', f'model_reasoning_effort="{config.reasoning}"']
        for camera in cameras:
            path = directory / f'{camera["name"]}.jpg'
            path.write_bytes(base64.b64decode(camera['image_base64']))
            arguments += ['--image', str(path)]
        arguments.append('-')
        process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        communication = asyncio.create_task(process.communicate(
            (system_prompt + ('\nImages attached in order: front, side, wrist.\n' if cameras else '\n') + prompt).encode('utf-8')))
        def capture_returned(stdout):
            usage = None; reasoning = []
            for line in stdout.decode('utf-8', 'replace').splitlines():
                try: item = json.loads(line)
                except json.JSONDecodeError: continue
                if item.get('type') == 'turn.completed': usage = item.get('usage')
                returned = item.get('item') or {}
                if returned.get('type') == 'reasoning' and isinstance(returned.get('text'), str): reasoning.append(returned['text'])
            content = output.read_text(encoding='utf-8') if output.exists() else None
            if content or usage is not None or reasoning:
                model_output({'provider': 'codex', 'model': config.model, 'content': content, 'usage': usage,
                              'returned_reasoning': '\n\n'.join(reasoning) or None})
            return content, usage
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), config.timeout_s)
        except (asyncio.CancelledError, TimeoutError) as exc:
            if process.returncode is None: process.kill()
            stdout, stderr = await communication
            capture_returned(stdout)
            if isinstance(exc, TimeoutError):
                detail = (stdout.decode('utf-8', 'replace') + '\n' + stderr.decode('utf-8', 'replace'))[-1200:]
                raise TimeoutError(f'Codex 请求超过 {config.timeout_s} 秒，已终止本轮。{detail}') from exc
            raise
        content, usage = capture_returned(stdout)
        if process.returncode or not output.exists():
            detail = stderr.decode('utf-8', 'replace')[-1200:]
            for line in stdout.decode('utf-8', 'replace').splitlines():
                try: item = json.loads(line)
                except json.JSONDecodeError: continue
                if item.get('type') == 'turn.failed': detail = item.get('error', {}).get('message', detail)
                elif item.get('type') == 'error': detail = item.get('message', detail)
            try: detail = json.loads(detail).get('detail', detail)
            except (json.JSONDecodeError, AttributeError): pass
            raise RuntimeError(f'Codex 调用失败（exit {process.returncode}）：{detail}')
        return response_type.model_validate_json(content), usage


@track_inference('agent_inference')
async def api_action(config, prompt, cameras, *, response_type=PoseAction,
                     system_prompt=SYSTEM_PROMPT, schema_name='robot_action'):
    key = config.api_key.get_secret_value()
    schema = response_type.action_schema()
    if config.provider == 'custom':
        import httpx
        from .policy import http_result
        validate_agent_access(config)
        try: task = json.loads(prompt).get('task', '')
        except (json.JSONDecodeError, AttributeError): task = prompt
        body = {'protocol': 'embodied-agent-v1', 'model': config.model, 'task': task,
                'prompt': prompt, 'system_contract': system_prompt, 'cameras': cameras,
                'response_schema': schema, 'response_schema_name': schema_name,
                'model_settings': {'reasoning': config.reasoning, 'max_output_tokens': config.max_output_tokens}}
        headers = {'Authorization': 'Bearer ' + key} if key else {}
        async with httpx.AsyncClient(timeout=config.timeout_s, trust_env=False) as client:
            response = await client.post(config.base_url, json=body, headers=headers)
        try: returned = response.json()
        except ValueError: returned = {'content': response.text}
        model_output({'provider': 'custom', 'model': config.model, 'response': returned})
        raw = http_result(response)
        if raw.get('protocol') != 'embodied-agent-v1' or not isinstance(raw.get('output'), dict):
            raise ValueError('自定义 Agent 必须返回 embodied-agent-v1 与结构化 output。')
        if raw.get('usage') is not None and not isinstance(raw['usage'], dict):
            raise ValueError('自定义 Agent 的 usage 必须为对象或 null。')
        return response_type.model_validate(raw['output']), raw.get('usage')
    if config.provider == 'anthropic':
        from anthropic import AsyncAnthropic
        tool_name = 'next_robot_action' if response_type is PoseAction else 'submit_workbench_repair' if schema_name == 'workbench_repair' else 'next_robot_plan'
        content = [{'type': 'text', 'text': prompt + f'\nReturn only the {tool_name} tool call.'}]
        for camera in cameras:
            content.extend([{'type': 'text', 'text': camera['name']}, {'type': 'image', 'source': {
                'type': 'base64', 'media_type': 'image/jpeg', 'data': camera['image_base64']}}])
        async with AsyncAnthropic(api_key=key, base_url=config.base_url, timeout=config.timeout_s, max_retries=0) as client:
            system = system_prompt.replace('Output exactly one JSON object matching the provided schema. Do not use tools, shell or files.',
                f'Submit exactly one action using {tool_name}. Do not use other tools, shell or files.')
            options = {'output_config': {'effort': config.reasoning}} if config.reasoning_native and config.reasoning is not None else {}
            response = await client.messages.create(model=config.model, max_tokens=config.max_output_tokens, system=system,
                tools=[{'name': tool_name, 'description': 'Submit the next robot action or subgoal.', 'input_schema': schema}],
                tool_choice={'type': 'auto', 'disable_parallel_tool_use': True},
                messages=[{'role': 'user', 'content': content}], **options)
        actions = [item for item in response.content if item.type == 'tool_use' and item.name == tool_name]
        model_output({'provider': 'anthropic', 'model': config.model, 'stop_reason': response.stop_reason,
                      'text': [item.text for item in response.content if item.type == 'text'],
                      'returned_reasoning': '\n\n'.join(item.thinking for item in response.content if item.type == 'thinking'),
                      'actions': [item.input for item in actions], 'usage': response.usage.model_dump()})
        if len(actions) != 1 or response.stop_reason == 'max_tokens':
            raise ValueError('Claude 未返回一个完整的 next_robot_action；请检查模型能力与响应。')
        return response_type.model_validate(actions[0].input), response.usage.model_dump()
    from openai import AsyncOpenAI
    async with AsyncOpenAI(api_key=key, base_url=config.base_url, timeout=config.timeout_s, max_retries=0) as client:
        if config.provider == 'openai':
            content = [{'type': 'input_text', 'text': prompt}]
            for camera in cameras:
                content.extend([{'type': 'input_text', 'text': camera['name']}, {'type': 'input_image',
                    'image_url': 'data:image/jpeg;base64,' + camera['image_base64'], 'detail': 'high'}])
            response = await client.responses.create(model=config.model, instructions=system_prompt,
                input=[{'role': 'user', 'content': content}], store=False, max_output_tokens=config.max_output_tokens,
                **({'reasoning': {'effort': config.reasoning}} if config.reasoning is not None else {}),
                text={'format': {'type': 'json_schema', 'name': schema_name, 'strict': True, 'schema': schema}})
            usage = response.usage.model_dump() if response.usage else None
            model_output({'provider': 'openai', 'model': config.model, 'status': response.status, 'content': response.output_text, 'usage': usage,
                          'reasoning': [{'summary': [part.model_dump() for part in (item.summary or [])]} for item in (response.output or []) if item.type == 'reasoning']})
            if response.status != 'completed' or not response.output_text:
                raise ModelResponseError(f'模型未返回完整动作：{response.status}，{response.incomplete_details}。', usage)
            return response_type.model_validate_json(response.output_text), usage
        content = [{'type': 'text', 'text': prompt + '\nJSON schema: ' + json.dumps(schema)}]
        for camera in cameras:
            content.extend([{'type': 'text', 'text': camera['name']}, {'type': 'image_url',
                'image_url': {'url': 'data:image/jpeg;base64,' + camera['image_base64']}}])
        options = {}
        # DeepSeek's Chat API defaults to high thinking unless explicitly configured.
        # Keep vendor-specific fields off other OpenAI-compatible endpoints.
        if uses_deepseek(config):
            options['extra_body'] = {'thinking': {'type': config.deepseek_thinking}}
            if config.deepseek_thinking == 'enabled' and config.reasoning is not None: options['reasoning_effort'] = config.reasoning
        elif config.reasoning_native and config.reasoning is not None:
            options['reasoning_effort'] = config.reasoning
        response = await client.chat.completions.create(model=config.model, max_tokens=config.max_output_tokens,
            response_format={'type': 'json_object'},
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content}], **options)
        model_output({'provider': 'compatible', 'model': config.model,
                      'choices': [{'finish_reason': choice.finish_reason, 'content': choice.message.content,
                                   'returned_reasoning': choice.message.model_dump().get('reasoning_content') or choice.message.model_dump().get('reasoning')}
                                  for choice in response.choices],
                      'usage': response.usage.model_dump() if response.usage else None})
        return compatible_action(response, config, response_type)


class AgentController:
    def __init__(self, worker):
        self.worker = worker
        self.config = AgentConfig()
        self.task = None
        self.lock = asyncio.Lock()
        self.observation = None
        self.observation_id = 0
        self._accepting = False
        self.state = {'active': False, 'status': 'idle', 'round': 0, 'message': '配置模型后开始视觉任务。', 'history': []}

    def configure(self, config):
        if self.state['active'] and self.state['status'] != 'repair_paused': raise ValueError('请先停止当前 Agent 任务，再修改模型。')
        if config.provider == self.config.provider and config.base_url == self.config.base_url and not config.api_key.get_secret_value():
            config.api_key = self.config.api_key
        if config.reasoning_native and config.reasoning is not None:
            from .model_capabilities import cached_model_capabilities
            capability = cached_model_capabilities(config)
            if capability.get('levels') is not None and config.reasoning not in capability['levels']:
                raise ValueError('当前模型支持的推理强度为：' + ('、'.join(capability['levels']) or '无可配置等级') + '。可选择服务默认。')
        if config.provider == 'codex': config.api_key = SecretStr('')
        self.config = config
        return self.public_config()

    def public_config(self):
        try: available = bool(codex_executable())
        except ValueError: available = False
        return {**self.config.model_dump(exclude={'api_key'}), 'has_api_key': bool(self.config.api_key.get_secret_value()),
                'codex_available': available}

    def reset_observation(self):
        self.observation = None
        self.observation_id += 1
        self.state = {'active': False, 'status': 'idle', 'round': 0, 'message': '场景已重置，等待新任务。', 'history': []}

    def snapshot(self):
        return {**controller_timing(self.state, getattr(self, 'run_log', None)), 'observation_id': self.observation_id,
                'observation_time': self.observation['time'] if self.observation else None,
                'observation_robot': self.observation['robot'] if self.observation else None}

    async def capture(self):
        self.observation = await self.worker.execute({'action': 'observe'})
        self.observation_id += 1
        return self.snapshot()

    async def start(self, request):
        if self.state['active']: raise ValueError('已有 Agent 任务在运行。')
        if not request.task.strip(): raise ValueError('请输入任务。')
        validate_agent_access(self.config)
        await self.worker.execute({'action': 'stop'})
        self.state = {'active': True, 'status': 'observing', 'round': 0, 'max_rounds': 1 if request.one_round else self.config.max_rounds,
                      'message': '正在采集三路相机与机械臂状态。', 'task': request.task.strip(),
                      'provider': self.config.provider, 'model': self.config.model, 'history': [], 'action': None,
                      'max_sim_time_s': request.max_sim_time_s, 'evaluator_stop': request.evaluator_stop,
                      'record_video': request.record_video}
        self.task = asyncio.create_task(self.run(request.one_round))
        self._accepting = True
        return self.snapshot()

    async def stop(self):
        self._accepting = False
        if self.task and not self.task.done():
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
        self.task = None

    async def run(self, one_round):
        initial = None
        config = self.config.model_copy(deep=True)
        started = time.monotonic()
        with self.worker.lock: initial_state = self.worker.snapshot.copy()
        log = RunLog('agent', {'model': config.model, 'provider': config.provider, 'task': self.state['task'],
            'robot': initial_state.get('model'), 'scene_id': initial_state.get('scene_id'), 'seed': initial_state.get('seed'),
            'settings': {**config.model_dump(exclude={'api_key'}), 'max_sim_time_s':self.state['max_sim_time_s'],
                         'one_round':one_round, 'evaluator_stop':self.state['evaluator_stop'],
                         'record_video':self.state['record_video']}}, [config.api_key.get_secret_value()],
                         record_video=self.state['record_video'])
        self.state['run_id'] = log.id
        self.run_log = log
        context = CURRENT_LOG.set(log)
        latencies = []
        recording = False
        try:
            log.save_checkpoint(await self.worker.execute({'action': 'checkpoint'}))
            recording = True
            await self.worker.execute({'action': 'record_start', 'log': log})
            await self.worker.execute({'action': 'run_limits', 'duration_s': self.state['max_sim_time_s'], 'evaluator_stop': self.state['evaluator_stop']})
            for number in count(1):
                if self.state['max_rounds'] is not None and number > self.state['max_rounds']: break
                with self.worker.lock: current = self.worker.snapshot.copy()
                remaining = self.state['max_sim_time_s'] - (current['time'] - initial_state['time'])
                if remaining <= 1e-8 or remaining + 1e-8 < current.get('timestep', 0):
                    self.state.update(status='limit', message='已达到本次任务的仿真时间上限。'); return
                if self.state['evaluator_stop'] and current.get('evaluation', {}).get('success'):
                    self.state.update(status='verified', message='场景评价器已确认任务成功。'); return
                self.state.update(status='observing', round=number, message='采集同一时刻的三路相机画面。')
                await self.worker.execute({'action': 'pause', 'value': True})
                await self.capture()
                obs = self.observation
                if initial is None: initial = obs['tcp']
                self.state.update(status='thinking', message=f'{config.model} 正在读取三路图像并选择下一步。')
                start = time.monotonic()
                prompt = observation_text(self.state['task'], obs, self.state['history'], initial)
                saved = log.save_observation(obs, number)
                log.write('observation', {'round': number, 'input': json.loads(prompt), 'observation': saved})
                async def infer(feedback, effective_config):
                    function = codex_action if effective_config.provider == 'codex' else api_action
                    request_prompt = prompt
                    if feedback:
                        payload = json.loads(prompt)
                        payload['recovery_feedback'] = feedback
                        request_prompt = json.dumps(payload,ensure_ascii=False)
                    return await function(effective_config, request_prompt, obs['cameras'])
                action, usage = await invoke_with_recovery(self, 'agent_inference', infer, obs, config)
                if not self._accepting or asyncio.current_task().cancelling(): raise asyncio.CancelledError
                self.state.update(action=action.model_dump(), latency_s=round(time.monotonic()-start, 2), usage=usage)
                latencies.append(time.monotonic()-start)
                log.write('action', {'round': number, 'action': action.model_dump(), 'latency_s': latencies[-1], 'usage': usage})
                record = {'round': number, 'observed_time': obs['time'], 'action': action.model_dump()}
                self.state['history'].append(record)
                if action.status == 'done':
                    accepted = await verify_completion(self, self.worker, log, number,
                                                       action.explanation, obs['generation'])
                    record['verification'] = self.state['verification']
                    record['result'] = self.state['message']
                    if accepted: return
                    continue
                if action.status == 'blocked':
                    record['result'] = action.explanation
                    self.state.update(status='blocked', message='模型无法继续：' + action.explanation)
                    return
                self.state.update(status='executing', message='正在求解 IK，并通过执行器执行目标位姿。')
                try:
                    result = await self.worker.execute({'action': 'move_tcp', 'generation': obs['generation'], 'pose': action.model_dump()})
                    record['ik'] = result['joint_targets_rad']
                except Exception as exc:
                    # Rejecting a physical target is feedback for replanning, not successful motion.
                    if getattr(exc, 'status_code', None) != 400: raise
                    record['result'] = 'IK/动作被拒绝：' + str(exc.detail)
                    log.write('execution', record)
                    self.state['message'] = record['result']
                    continue
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    await asyncio.sleep(.08)
                    with self.worker.lock: snapshot = self.worker.snapshot
                    if not snapshot.get('ready'): raise RuntimeError(snapshot.get('error', 'MuJoCo 服务异常。'))
                    if snapshot['generation'] != obs['generation']: raise RuntimeError('机械臂已切换，当前任务停止。')
                    if self.state['evaluator_stop'] and snapshot.get('evaluation', {}).get('success'):
                        record['result'] = {'status': 'verified', 'actual_tcp': snapshot['tcp_pose']}
                        log.write('execution', record)
                        self.state.update(status='verified', message='场景评价器已确认任务成功。'); return
                    if snapshot['time'] - initial_state['time'] >= self.state['max_sim_time_s'] - 1e-8:
                        record['result'] = {'status': 'limit', 'actual_tcp': snapshot['tcp_pose']}
                        log.write('execution', record)
                        self.state.update(status='limit', message='已达到本次任务的仿真时间上限。'); return
                    status = snapshot['task']['status']
                    if status not in ('running', 'settling'):
                        record['result'] = {'status': status, 'actual_tcp': snapshot['tcp_pose'],
                            'message': snapshot['task'].get('message'),
                            'joint_errors_rad': snapshot['task'].get('joint_errors_rad'),
                            'contacts': snapshot['task'].get('contacts'),
                            'position_error_m': snapshot['task'].get('position_error_m'),
                            'orientation_error_rad': snapshot['task'].get('orientation_error_rad')}
                        log.write('execution', record)
                        break
                else: raise TimeoutError('动作在 45 秒内没有完成，请检查仿真速度与接触。')
            self.state.update(status='round_complete' if one_round else 'limit',
                message='本轮已结束；任务结果以环境验收为准。' if one_round else '已达到轮数上限，环境未确认任务完成。')
        except asyncio.CancelledError:
            self.state.update(status='stopped', message='Agent 已停止；未执行的模型输出已丢弃。')
            raise
        except Exception as exc:
            message = str(getattr(exc, 'detail', exc)) or type(exc).__name__
            for key in log.secrets:
                if key: message = message.replace(key, '[已隐藏密钥]')
            self.state.update(status='error', message=message[:2400])
            log.write('error', {'message': message[:2400], 'round': self.state['round']})
            if isinstance(exc, ModelResponseError): self.state['usage'] = exc.usage
        finally:
            try: await self.worker.execute({'action': 'stop'})
            except Exception: pass  # Engine status separately exposes its actual failure.
            if recording:
                try: await self.worker.execute({'action': 'record_stop'})
                except Exception as exc: log.write('cleanup_error', {'phase': 'recording', 'message': str(getattr(exc, 'detail', exc))})
            self.state['active'] = False
            self._accepting = False
            with self.worker.lock: final = self.worker.snapshot.copy()
            log.write('task_evaluation', {'phase': 'final', 'termination': self.state['status'],
                'engine_ready': final.get('ready', False), 'evaluation': final.get('evaluation')})
            self.state['metrics'] = {'success': environment_success(final),
                'engine_ready': final.get('ready',False),
                'wall_time_s': round(time.monotonic()-started, 3), 'sim_time_s': round(final.get('time', 0)-initial_state.get('time', 0), 3),
                'inference_s': log.timing_snapshot()['timing']['inference_elapsed_s'],
                'inference_count': log._inference_count,
                'mean_inference_s': log.timing_snapshot()['timing']['inference_elapsed_s']/log._inference_count if log._inference_count else None,
                'evaluation': final.get('evaluation')}
            try:
                log.finish({'status': self.state['status'], 'message': self.state['message'], 'metrics': self.state['metrics']})
                self.state['video'] = deepcopy(log.artifacts.get('video'))
            finally: CURRENT_LOG.reset(context)
