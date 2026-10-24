"""Configurable coding model: diagnose, stage, validate, explicitly apply/restore.

The model submits text edits scoped to model adapters. User-authorized validation
runs candidate Python code in a source copy using the local account, not a sandbox.
"""
from __future__ import annotations

import ast
import asyncio
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .agent import AgentConfig, api_action, codex_action, validate_agent_access
from .runlog import RunLog, CURRENT_LOG, LIVE_RUNS, run_folder, list_runs

ROOT = Path(__file__).resolve().parents[1]
REPAIRS = ROOT / 'artifacts' / 'repairs'
EDITABLE = ('server/agent.py', 'server/policy.py', 'server/decision.py',
            'server/model_capabilities.py', 'server/policy_service.py', 'server/composition.py')
REGRESSION = 'tests/test_repair_candidate.py'
SKILL_DIR = Path(__file__).with_name('repair_skills')


class RepairConfig(AgentConfig):
    model: str = 'gpt-6-astra'
    reasoning: str | None = 'high'
    timeout_s: int = Field(default=300, ge=15, strict=True)
    extra_instructions: str = Field(default='', max_length=3000)
    auto_analyze: bool = False
    auto_recover: bool = True
    max_recovery_attempts: int = Field(default=2, ge=1, strict=True)


class RecoverySettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_output_tokens: int | None = Field(default=None, ge=512, strict=True)
    timeout_s: int | None = Field(default=None, ge=15, strict=True)
    reasoning: str | None = Field(default=None, min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_.-]+$')
    deepseek_thinking: Literal['enabled', 'disabled'] | None = None


class RecoveryPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal['retry', 'stop']
    instruction: str = Field(max_length=3000)
    settings: RecoverySettings | None = None


class RepairEdit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str
    before: str = Field(max_length=40000)
    after: str = Field(max_length=40000)


class RepairProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    diagnosis: str = Field(min_length=1, max_length=10000)
    category: Literal['configuration', 'provider', 'code', 'physical_control', 'insufficient_evidence']
    evidence: list[str] = Field(max_length=20)
    recommendations: list[str] = Field(max_length=20)
    edits: list[RepairEdit] = Field(max_length=12)
    recovery: RecoveryPlan | None = None

    @classmethod
    def action_schema(cls):
        schema = cls.model_json_schema()
        for node in [schema, *schema.get('$defs', {}).values()]:
            if 'properties' in node:
                node['required'] = list(node['properties'])
                for prop in node['properties'].values(): prop.pop('default', None)
        return schema


class RepairStart(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_run_id: str | None = None
    description: str = Field(default='', max_length=4000)


def digest(content): return hashlib.sha256(content).hexdigest()


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.pending')
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def secret_values(value):
    if isinstance(value, dict): return [item for child in value.values() for item in secret_values(child)]
    if isinstance(value, (list, tuple)): return [item for child in value for item in secret_values(child)]
    return [value] if isinstance(value, str) and value else []


def skills():
    return [{'name': path.parent.name, 'text': path.read_text(encoding='utf-8')}
            for path in sorted(SKILL_DIR.glob('*/SKILL.md'))]


def system_contract():
    return ('You are the dedicated coding repair agent for the embodied simulation workbench. '
            'Return one JSON instance of the supplied repair proposal schema. '
            'Explain diagnosis, evidence and advice in Chinese. Do not use tools, shell or files. '
            'For a paused live experiment, provide a structured recovery plan: retry means one bounded '
            'new inference at the same observation, never replay a previous physical command. '
            'Give a short corrective instruction for invalid/empty JSON, or an evidenced settings correction. '
            'Use stop for unavailable credentials, missing evidence, unsafe recovery or code requiring restart. '
            'Do not call a source edit applied or effective before validation/restart. '
            'The following built-in skills always apply:\n\n' + '\n\n'.join(item['text'] for item in skills()))


def redact(value, secrets=()):
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items()
                if not any(token in key.lower() for token in ('api_key', 'authorization', 'password', 'secret', 'encrypted', 'image_base64'))}
    if isinstance(value, list): return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret: value = value.replace(secret, '[已隐藏密钥]')
        value = re.sub(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+', 'Bearer [已隐藏密钥]', value)
        return re.sub(r'\bsk-[A-Za-z0-9_-]{12,}', '[已隐藏密钥]', value)
    return value


def source_context(run_id, description, secrets=()):
    metadata = {}; selected = deque(maxlen=6)
    if run_id:
        folder = run_folder(run_id)
        summary = folder / 'summary.json'
        if summary.exists(): metadata = json.loads(summary.read_text(encoding='utf-8'))
        elif run_id in LIVE_RUNS: metadata = LIVE_RUNS[run_id].metadata
        with (folder / 'events.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                try: event = json.loads(line)
                except ValueError: continue
                if event.get('event') in ('start', 'inference', 'model_output', 'error', 'recoverable_error', 'cleanup_error', 'finish'):
                    # Inference records duplicate the raw output and its reasoning.
                    # Keep call/error metadata and the actual output once, without
                    # filling the diagnostic prompt with long reasoning transcripts.
                    data = redact(event.get('data'), secrets)
                    if event['event'] == 'inference' and isinstance(data, dict):
                        data = {key:value for key,value in data.items() if key in (
                            'call_id','round','phase','provider','model','reasoning_effort',
                            'configured_reasoning','started_at','ended_at','status',
                            'inference_s','usage','error','response_returned')}
                    selected.append({'event':event.get('event'), 'timestamp':event.get('timestamp'),
                        'diagnostic_excerpt':json.dumps(diagnostic_value(data), ensure_ascii=False)})
        selected = list(selected)
        metadata = {key:metadata.get(key) for key in ('run_id','kind','provider','model','status','message','settings','metrics')}
    if not selected and not description.strip(): raise ValueError('请选择包含报错的实验，或填写待诊断问题。')
    return redact({'source_run_id': run_id, 'description': description, 'summary': metadata,
                   'recent_events': list(selected)}, secrets)


def diagnostic_value(value):
    """Compact diagnostic copy; original experiment outputs remain untouched."""
    if isinstance(value, dict):
        return {key:diagnostic_value(item) for key,item in value.items()
                if key not in ('returned_reasoning','reasoning_content','reasoning','thinking',
                               'encrypted_content','pricing','cost')}
    if isinstance(value, list): return [diagnostic_value(item) for item in value]
    if isinstance(value, str) and len(value) > 6000:
        return value[:4000] + f'\n[已省略 {len(value)-6000} 字符；完整内容保存在原实验日志]\n' + value[-2000:]
    return value


def safe_path(root, relative):
    if relative not in (*EDITABLE, REGRESSION): raise ValueError(f'修复文件超出模型接入范围：{relative}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink(): raise ValueError('修复路径不能是符号链接或位于工作区之外。')
    return path


def prepare_edits(root, edits):
    before = {}; after = {}
    for edit in edits:
        path = safe_path(root, edit.path)
        if edit.path not in before:
            content = path.read_bytes() if path.exists() else b''
            if not path.exists() and edit.path != REGRESSION: raise ValueError('只能新增专用回归测试文件。')
            before[edit.path] = content
            after[edit.path] = content.decode('utf-8').replace('\r\n', '\n')
        text = after[edit.path]
        if edit.before:
            if text.count(edit.before) != 1: raise ValueError(f'{edit.path} 的原文不是唯一匹配；请重新分析当前代码。')
            text = text.replace(edit.before, edit.after, 1)
        elif not text and edit.path == REGRESSION: text = edit.after
        else: raise ValueError('空 before 仅允许创建专用回归测试。')
        ast.parse(text, filename=edit.path)
        after[edit.path] = text
    if any(path != REGRESSION for path in after) and REGRESSION not in after:
        raise ValueError('代码修复必须提供 tests/test_repair_candidate.py 回归用例。')
    return before, {path: (text.replace('\n', '\r\n') if b'\r\n' in before[path] else text).encode('utf-8') for path,text in after.items()}


class RepairController:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.config = RepairConfig()
        self.lock = asyncio.Lock()
        self.task = None
        self._clock = None
        self.state = {'active': False, 'status': 'idle', 'message': '选择一个报错实验，配置编码模型后分析。'}
        self.current = None
        pointer = self.root/'artifacts/repairs/last_applied.json'
        if pointer.exists():
            try:
                saved = json.loads(pointer.read_text(encoding='utf-8'))
                job_id = saved['job_id']
                if not re.fullmatch(r'[a-f0-9]{32}', job_id): raise ValueError('修复编号无效。')
                folder = pointer.parent/job_id
                manifest = json.loads((folder/'applied.json').read_text(encoding='utf-8'))
                self.current = {'id':job_id, 'folder':folder, 'secrets':(),
                    'before':{path:(folder/'before'/path).read_bytes() for path in manifest['files']},
                    'after':{path:(folder/'after'/path).read_bytes() for path in manifest['files']},
                    'existed':manifest.get('existed', {path:True for path in manifest['files']})}
                for path in manifest['files']: safe_path(self.root, path)
                self.state = manifest['state']
                if self.state.get('status') == 'applied':
                    self.state.update(restart_required=False, message='已重启载入修复后的源码；仍可恢复应用前版本。')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.state['message'] = f'上次修复恢复信息无法读取：{exc}'

    def public_config(self):
        return {**self.config.model_dump(exclude={'api_key'}), 'has_api_key': bool(self.config.api_key.get_secret_value()),
                'skills': [item['name'] for item in skills()], 'editable_files': list(EDITABLE),
                'application_mode': 'review_then_apply', 'validation_scope': 'fixed_local_provider_contracts_and_candidate_regression'}

    def configure(self, config):
        if self.state['active']: raise ValueError('修复 Agent 运行中，请先停止。')
        if config.provider == self.config.provider and config.base_url == self.config.base_url and not config.api_key.get_secret_value():
            config.api_key = self.config.api_key
        if config.provider == 'codex': config.api_key = SecretStr('')
        self.config = config
        return self.public_config()

    def snapshot(self):
        state = deepcopy(self.state)
        if self._clock is not None: state['elapsed_s'] = time.monotonic()-self._clock
        if getattr(self, 'recovery_snapshot', None): state['recovery'] = self.recovery_snapshot()
        return state

    def _finish_timing(self):
        if self._clock is not None:
            self.state['elapsed_s'] = time.monotonic()-self._clock
            self._clock = None
        self.state['active'] = False

    async def start(self, request, *, secrets=(), parent_log=None):
        if self.state['active']: raise ValueError('已有修复任务在运行。')
        if self.state.get('status') in ('applying','apply_failed','restoring','restore_failed') or self.state.get('restart_required'):
            raise ValueError('上一项代码变更尚待重启或恢复，请先完成后再开始新诊断。')
        validate_agent_access(self.config)
        config = self.config.model_copy(deep=True)
        sensitive = (*secrets, config.api_key.get_secret_value())
        context = source_context(request.source_run_id, request.description, sensitive)
        source_bytes = {path:safe_path(self.root,path).read_bytes() for path in EDITABLE}
        if (self.root/REGRESSION).exists(): source_bytes[REGRESSION] = safe_path(self.root,REGRESSION).read_bytes()
        sources = {path:content.decode('utf-8').replace('\r\n','\n') for path,content in source_bytes.items()}
        job_id = uuid4().hex
        folder = self.root / 'artifacts' / 'repairs' / job_id
        folder.mkdir(parents=True)
        self.current = {'id': job_id, 'folder': folder, 'source_hashes': {path:digest(content) for path,content in source_bytes.items()},
                        'source_run_id': request.source_run_id, 'secrets': sensitive}
        self.state = {'active': True, 'status': 'analyzing', 'job_id': job_id,
                      'source_run_id': request.source_run_id, 'model': config.model,
                      'started_at': datetime.now(timezone.utc).isoformat(), 'timeout_s':config.timeout_s,
                      'message': '正在按内置修复技能分析原始报错与模型接入代码。'}
        self._clock = time.monotonic()
        self.task = asyncio.create_task(self.run(config, context, sources, parent_log=parent_log), name='model-repair')
        return self.snapshot()

    async def run(self, config, context, sources, *, parent_log=None):
        log = parent_log or RunLog('repair', {'model':config.model, 'provider':config.provider,
            'task':'诊断并修复模型调用报错', 'source_run_id':context.get('source_run_id'),
            'settings':config.model_dump(exclude={'api_key'})}, secrets=self.current['secrets'])
        if parent_log is not None:
            for secret in self.current['secrets']:
                if secret and secret not in log.secrets: log.secrets.append(secret)
        token = CURRENT_LOG.set(log)
        self.state['run_id'] = log.id
        try:
            prompt = json.dumps(redact({'diagnostic_data':context, 'editable_source':sources,
                'regression_file':REGRESSION, 'additional_user_requirements':config.extra_instructions}, self.current['secrets']), ensure_ascii=False)
            function = codex_action if config.provider == 'codex' else api_action
            proposal, usage = await function(config, prompt, [], response_type=RepairProposal,
                system_prompt=system_contract(), schema_name='workbench_repair')
            proposal = RepairProposal.model_validate(redact(proposal.model_dump(), self.current['secrets']))
            before, after = prepare_edits(self.root, proposal.edits)
            for path in after:
                if path in self.current['source_hashes'] and digest(before[path]) != self.current['source_hashes'][path]:
                    raise ValueError('分析期间代码已变更，请重新分析，未覆盖本地修改。')
            self.current.update(before=before, after=after, existed={path:safe_path(self.root,path).exists() for path in before})
            diff = ''.join(''.join(difflib.unified_diff(before[path].decode('utf-8').splitlines(True),
                text.decode('utf-8').splitlines(True), fromfile=path, tofile=path)) for path,text in after.items())
            self.state.update(status='proposed' if after else 'advice', message='已生成候选修改，请查看诊断与差异，再验证候选。' if after else '已完成诊断；本次没有建议修改代码。',
                              proposal=proposal.model_dump(), diff=diff, validation=None)
            (self.current['folder']/'proposal.json').write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
            log.write('repair_proposal', self.state)
        except asyncio.CancelledError:
            self.state.update(status='stopped', message='修复分析已停止，未修改工作区。')
            raise
        except Exception as exc:
            self.state.update(status='error', message=redact(str(exc) or type(exc).__name__, self.current['secrets']))
            log.write('error', {'message':self.state['message']})
        finally:
            self._finish_timing()
            if parent_log is None: log.finish({'status':self.state['status'], 'message':self.state['message']})
            CURRENT_LOG.reset(token)

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
        return self.snapshot()

    async def validate(self, *, allow_local_code_execution=False):
        if not allow_local_code_execution: raise ValueError('请先查看候选差异，并明确允许在本机 Python 运行候选测试。')
        if self.state['active'] or not self.current or not self.current.get('after'): raise ValueError('没有可验证的候选修改。')
        if self.state['status'] not in ('proposed', 'validation_failed', 'validated'): raise ValueError('当前状态不能验证。')
        folder = self.current['folder'] / 'candidate'
        if folder.exists(): raise ValueError('此候选已验证过；请重新分析生成新候选后再验证。')
        for name in ('server', 'tests'):
            shutil.copytree(self.root/name, folder/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for path,original in self.current['before'].items():
            target = safe_path(self.root,path)
            if (target.read_bytes() if target.exists() else b'') != original: raise ValueError('候选对应的代码已变更，请重新分析。')
        # Run the supplied regression against the original implementation first.
        (folder/REGRESSION).write_bytes(self.current['after'][REGRESSION])
        self.state.update(active=True, status='validating', timeout_s=None,
            started_at=datetime.now(timezone.utc).isoformat(),
            message='正在用本机 Python 执行副本中的候选回归与固定接口契约。')
        self._clock = time.monotonic()
        # This is explicit user-authorized local code execution, not an OS sandbox.
        self.task = asyncio.create_task(self._validate(folder), name='repair-validation')
        return self.snapshot()

    async def _validate(self, folder):
        patterns = ('test_repair_candidate.py', 'test_provider_contracts.py', 'test_repair_candidate.py')
        output = []; passed = True
        environment = {key:value for key,value in os.environ.items()
                       if not any(token in key.upper() for token in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        runner = folder/'_repair_test_runner.py'
        runner.write_text('''import json,sys,unittest
class Recorder(unittest.TextTestResult):
    def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs); self.ids=[]
    def startTest(self,test): self.ids.append(test.id()); super().startTest(test)
suite=unittest.defaultTestLoader.discover("tests",pattern=sys.argv[1])
result=unittest.TextTestRunner(verbosity=2,resultclass=Recorder).run(suite)
with open(sys.argv[2],"w",encoding="utf-8") as handle:
    json.dump({"tests":result.testsRun,"ids":result.ids,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"expected_failures":len(result.expectedFailures),"unexpected_successes":len(result.unexpectedSuccesses)},handle)
sys.exit(0 if result.wasSuccessful() else 1)
''',encoding='utf-8')
        baseline_ids = None
        try:
            for index, pattern in enumerate(patterns):
                if index == 1:
                    for path,content in self.current['after'].items(): (folder/path).write_bytes(content)
                report_path = folder.parent/f'test-result-{index}.json'
                command = [sys.executable, str(runner), pattern, str(report_path)]
                process = await asyncio.create_subprocess_exec(*command, cwd=folder, env=environment,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                try: stdout, _ = await asyncio.wait_for(process.communicate(), 90)
                except (asyncio.CancelledError, TimeoutError):
                    await self._terminate(process)
                    raise
                result = stdout.decode('utf-8', 'replace')
                output.append(('原始代码回归（预期失败）' if index==0 else '候选验证')+'\n'+result)
                report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
                valid = (report.get('tests',0)>0 and not any(report.get(key,0) for key in ('errors','skipped','expected_failures','unexpected_successes')))
                if index == 0:
                    if not valid or not report.get('failures') or process.returncode == 0:
                        passed = False; output.append('回归未在原始实现上复现有效断言失败（导入错误、跳过与空测试不算复现），不能标记为已验证修复。'); break
                    baseline_ids = report['ids']
                elif process.returncode or not valid or report.get('failures') or (index==2 and report.get('ids')!=baseline_ids):
                    passed = False; break
            text = redact('\n'.join(output)[-20000:], self.current['secrets'])
            (self.current['folder']/'validation.log').write_text(text, encoding='utf-8')
            self.state.update(status='validated' if passed else 'validation_failed',
                message='候选通过固定接口与回归测试；可查看差异后应用。' if passed else '候选未通过测试，工作区未修改。',
                validation={'passed':passed, 'output':text, 'scope':'固定本地接口契约和候选回归；未验证真实供应商账号或物理任务成功率。'})
        except asyncio.CancelledError:
            self.state.update(status='validation_failed',
                message='候选验证已取消，工作区未修改；候选副本已保留，请重新分析生成新候选后再验证。')
            raise
        except Exception as exc:
            self.state.update(status='validation_failed', message=redact(str(exc), self.current['secrets']))
        finally: self._finish_timing()

    @staticmethod
    async def _terminate(process):
        if process.returncode is None:
            if os.name == 'nt':
                killer = await asyncio.create_subprocess_exec('taskkill','/PID',str(process.pid),'/T','/F',
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                try: await asyncio.wait_for(killer.wait(),5)
                except TimeoutError: killer.kill()
            else: process.kill()
        try: await asyncio.wait_for(process.communicate(),5)
        except TimeoutError:
            if process.returncode is None: process.kill()
            raise RuntimeError('验证进程清理超时，请检查本机残留子进程。')

    def _journal(self):
        manifest = {'files':list(self.current['after']), 'existed':self.current['existed'], 'state':self.state}
        atomic_bytes(self.current['folder']/'applied.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
        atomic_bytes(self.root/'artifacts/repairs/last_applied.json',json.dumps({'job_id':self.current['id']}).encode('utf-8'))

    def apply(self):
        if self.state.get('status') != 'validated' or not (self.state.get('validation') or {}).get('passed'):
            raise ValueError('只有通过验证的当前候选才能应用。')
        for path, original in self.current['before'].items():
            target = safe_path(self.root, path)
            if (target.read_bytes() if target.exists() else b'') != original: raise ValueError(f'{path} 已被其他操作修改，请重新分析。')
        backups = self.current['folder']/'before'
        for path,content in self.current['after'].items():
            atomic_bytes(backups/path,self.current['before'][path])
            atomic_bytes(self.current['folder']/'after'/path,content)
        self.state.update(status='applying',message='已保存恢复日志，正在应用已验证的代码。')
        self._journal()
        try:
            for path, content in self.current['after'].items():
                target = safe_path(self.root, path)
                atomic_bytes(target,content)
        except OSError as exc:
            self.state.update(status='apply_failed',message=f'应用失败，备份已保存，可恢复应用前代码：{exc}')
            self._journal()
            raise
        self.state.update(status='applied', message='修复代码已写入并保留备份。后端需在停止实验后重启，当前进程不会热替换代码。', restart_required=True)
        self._journal()
        return self.snapshot()

    def rollback(self):
        if self.state.get('status') not in ('applied','applying','apply_failed','restoring','restore_failed'): raise ValueError('当前会话没有可恢复的已应用修复。')
        for path, candidate in self.current['after'].items():
            target = safe_path(self.root,path)
            content = target.read_bytes() if target.exists() else b''
            if content not in (candidate,self.current['before'][path]): raise ValueError('应用后代码已更改，不能覆盖后续工作。')
        self.state.update(status='restoring',message='正在恢复应用前代码，可在中断后继续。'); self._journal()
        try:
            for path, original in self.current['before'].items():
                target = safe_path(self.root, path)
                if self.current['existed'].get(path): atomic_bytes(target,original)
                else: target.unlink(missing_ok=True)
        except OSError as exc:
            self.state.update(status='restore_failed',message=f'恢复尚未完成，可重试：{exc}'); self._journal(); raise
        self.state.update(status='restored', message='已恢复应用前的源代码。重启后端以载入恢复版本。', restart_required=True)
        self._journal()
        return self.snapshot()
