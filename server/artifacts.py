"""Immutable, content-addressed method recipes and their local version materials."""
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / 'methods'
METHOD_ID = re.compile(r'^method-[0-9a-f]{24}$')
SECRET_KEYS = {'api_key', 'api_keys', 'apikey', 'apikeys', 'authorization', 'password', 'passwd', 'secret',
               'client_secret', 'access_token', 'refresh_token', 'id_token', 'token',
               'bearer', 'bearer_token', 'credential', 'credentials', 'secrets'}


def is_secret_key(key):
    normalized = str(key).lower().replace('-', '_')
    return normalized in SECRET_KEYS or normalized.endswith(('_api_key', '_api_keys', '_password', '_secret', '_access_token'))


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or is_secret_key(key):
                raise ValueError('方法配方不能包含密钥或凭据字段；请在运行时单独配置。')
            reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value: reject_secrets(item)
    elif isinstance(value, str) and value.strip().lower().startswith(('http://', 'https://')):
        parts = urlsplit(value.strip())
        if parts.username is not None or parts.password is not None:
            raise ValueError('方法 URL 不能包含账号或密码。')
        if any(is_secret_key(key) for key, _ in parse_qsl(parts.query, keep_blank_values=True)):
            raise ValueError('方法 URL 不能在查询参数中携带密钥。')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('方法配方不能包含非有限数值。')
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError('方法配方只能包含非保密 JSON 数据。')


class DecisionRecipe(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, validate_default=True)
    kind: Literal['typesafe', 'custom'] = 'custom'
    url: str = Field(min_length=1, max_length=2000)
    model: str = Field(min_length=1, max_length=300)
    vision: bool = False
    question_name: str = Field(default='next_action', pattern=r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
    timeout_s: int = Field(default=120, ge=1, le=600)
    weight: float = Field(default=1., gt=0, le=100)

    @field_validator('url')
    @classmethod
    def valid_url(cls, value):
        reject_secrets(value)
        parts = urlsplit(value)
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.fragment:
            raise ValueError('决策服务须使用有效的 HTTP(S) URL。')
        return value.rstrip('/')


class MethodRecipe(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, validate_default=True)
    schema_version: Literal['embodied-method-v1'] = 'embodied-method-v1'
    name: str = Field(min_length=1, max_length=120)
    mode: Literal['agent', 'policy', 'hybrid']
    parent_id: str | None = None
    evolution_generation: int = Field(default=0, ge=0)
    change_summary: str = Field(default='', max_length=2000)
    agent: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    decision: DecisionRecipe | None = None
    decisions: list[DecisionRecipe] = Field(default_factory=list, max_length=8)
    decision_strategy: Literal['single', 'cascade', 'weighted_vote'] = 'single'
    controller: dict[str, Any] = Field(default_factory=dict)
    skills: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def public_recipe_only(cls, value):
        if isinstance(value, dict): reject_secrets(value)
        return value

    @field_validator('parent_id')
    @classmethod
    def valid_parent(cls, value):
        if value is not None and not METHOD_ID.fullmatch(value):
            raise ValueError('无效父版本编号。')
        return value

    @model_validator(mode='after')
    def required_components(self):
        reject_secrets(self.model_dump(mode='json'))
        if self.mode == 'agent' and self.agent is None: raise ValueError('Agent 方法需要 agent 配方。')
        if self.mode == 'policy' and self.policy is None: raise ValueError('策略方法需要 policy 配方。')
        if self.mode != 'hybrid' and (self.decision is not None or self.decisions or self.decision_strategy != 'single'):
            raise ValueError('非混合方法不能包含决策模型或决策组合策略。')
        if self.mode == 'hybrid':
            if self.agent is None: raise ValueError('混合方法需要显式 agent 配方。')
            if self.decision is not None and self.decisions:
                raise ValueError('decision 与 decisions 不能同时配置。')
            effective = self.decisions or ([self.decision] if self.decision is not None else [])
            if self.decision_strategy == 'single' and len(effective) != 1:
                raise ValueError('single 策略需要恰好一个决策模型。')
            if self.decision_strategy != 'single' and len(effective) < 2:
                raise ValueError('cascade 和 weighted_vote 策略至少需要两个决策模型。')
        return self


def _method_directory(method_id):
    if not isinstance(method_id, str) or not METHOD_ID.fullmatch(method_id):
        raise ValueError('无效方法版本编号。')
    base = METHODS.resolve()
    folder = (base / method_id).resolve()
    if not folder.is_relative_to(base): raise ValueError('方法路径超出登记目录。')
    return folder


def _identity(recipe, footprint):
    digest = hashlib.sha256(canonical_json({'recipe': recipe, 'footprint': footprint}).encode()).hexdigest()
    return 'method-' + digest[:24]


def version_materials():
    """Read project source/requirements, never login files or runtime credentials."""
    paths = sorted((ROOT / 'server').rglob('*.py'))
    paths += sorted(ROOT.glob('requirements*.txt'))
    paths += [ROOT / name for name in ('package.json', 'package-lock.json')]
    blobs = {}
    for path in paths:
        if not path.is_file(): continue
        if not path.resolve().is_relative_to(ROOT.resolve()): raise ValueError('版本材料路径超出项目目录。')
        blobs[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    source = blobs.get('server/agent.py')
    if source:
        for node in ast.parse(source.decode('utf-8-sig')).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM_PROMPT' for t in node.targets):
                blobs['prompts/agent-system.txt'] = ast.literal_eval(node.value).encode('utf-8')
                break
    packages = {}
    for name in ('mujoco', 'numpy', 'pydantic', 'fastapi', 'uvicorn', 'Pillow', 'openai', 'anthropic', 'httpx'):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    runtime = {'python': sys.version, 'platform': platform.platform(), 'packages': packages}
    blobs['environment/runtime.json'] = canonical_json(runtime).encode('utf-8')
    footprint = {'schema_version': 'embodied-version-materials-v1',
                 'scope': ['server_source', 'system_prompt', 'dependency_specs', 'registration_runtime'],
                 'not_verified': ['remote_model_revision', 'policy_weight_revision', 'policy_runtime', 'robot_asset_bytes'],
                 'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                           for name, data in sorted(blobs.items())}}
    return footprint, blobs


def get_method(method_id):
    path = _method_directory(method_id) / 'manifest.json'
    if not path.is_file(): raise FileNotFoundError('方法版本不存在。')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    recipe = MethodRecipe.model_validate(manifest['recipe']).model_dump(mode='json')
    if manifest.get('id') != method_id or _identity(recipe, manifest['footprint']) != method_id:
        raise ValueError('方法版本内容与登记编号不一致。')
    return manifest


def get_recipe(method_id):
    return get_method(method_id)['recipe']


def verify_method(method_id):
    manifest = get_method(method_id)
    current, _ = version_materials()
    if manifest['footprint'] != current:
        raise ValueError('当前代码、提示或环境与方法版本不一致；请在当前环境登记新版本后运行。历史代码快照仅用于追溯。')
    return manifest


def register_method(recipe):
    recipe = MethodRecipe.model_validate(recipe)
    if recipe.parent_id:
        parent = get_recipe(recipe.parent_id)
        generation = parent['evolution_generation'] + 1
        if 'evolution_generation' in recipe.model_fields_set and recipe.evolution_generation != generation:
            raise ValueError('子版本代次必须等于父版本代次加一。')
        recipe = recipe.model_copy(update={'evolution_generation': generation})
    elif recipe.evolution_generation != 0:
        raise ValueError('无父版本的方法必须从第 0 代开始。')
    payload = recipe.model_dump(mode='json')
    footprint, blobs = version_materials()
    method_id = _identity(payload, footprint)
    METHODS.mkdir(parents=True, exist_ok=True)
    folder = _method_directory(method_id)
    if folder.exists(): return get_method(method_id)
    staging = Path(tempfile.mkdtemp(prefix='.register-', dir=METHODS))
    manifest = {'schema_version': 'embodied-method-version-v1', 'id': method_id, 'method_id': method_id,
                'recipe': payload, 'footprint': footprint, 'materials': 'materials',
                'model_revision_status': 'declared_unverified',
                'created_at': datetime.now(timezone.utc).isoformat()}
    try:
        for name, data in blobs.items():
            target = staging / 'materials' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (staging / 'manifest.json').write_text(canonical_json(manifest), encoding='utf-8')
        try: os.rename(staging, folder)
        except OSError:
            if not folder.exists(): raise
            return get_method(method_id)
    finally:
        if staging.exists():
            if not staging.resolve().is_relative_to(METHODS.resolve()): raise ValueError('临时版本路径超出登记目录。')
            shutil.rmtree(staging)
    return manifest


def list_methods(limit=100):
    if not METHODS.is_dir(): return []
    manifests = [get_method(folder.name) for folder in METHODS.iterdir()
                 if folder.is_dir() and METHOD_ID.fullmatch(folder.name)]
    return sorted(manifests, key=lambda item: item['created_at'], reverse=True)[:limit]
