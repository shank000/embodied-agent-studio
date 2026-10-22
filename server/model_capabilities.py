"""Read-only model capability discovery, independent of inference requests.

Codex capabilities come only from its local official models_cache.json. API
capabilities come from explicit service metadata; exact OpenAI documentation
profiles are a labelled fallback, never an inference about arbitrary proxies.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit

import httpx

DISCOVERY_TIMEOUT_S = 8.
CACHE_TTL_S = 300.
_DISCOVERED = {}
_TOKEN = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')
OPENAI_DOCS = 'https://developers.openai.com/api/docs/models/'
DEEPSEEK_DOCS = 'https://api-docs.deepseek.com/api/list-models/'
ANTHROPIC_DOCS = 'https://platform.claude.com/docs/en/api/models/retrieve'

# Each exact model page was fetched on 2026-10-22. These are API capabilities,
# deliberately separate from Codex's cache (which can additionally offer ultra).
# No prefix/family matching: a future model or a proxy with this name is unknown.
_OPENAI_PROFILES = {
    'gpt-6-astra': (['low', 'medium', 'high', 'xhigh', 'max'], None),
    'gpt-6.1-sol': (['low', 'medium', 'high', 'xhigh', 'max'], 'medium'),
    'gpt-6-sol': (['none', 'low', 'medium', 'high', 'xhigh', 'max'], 'medium'),
    'gpt-6-luna': (['none', 'low', 'medium', 'high', 'xhigh', 'max'], 'medium'),
    'gpt-5.6-sol': (['none', 'low', 'medium', 'high', 'xhigh', 'max'], 'medium'),
    'gpt-5.6-terra': (['none', 'low', 'medium', 'high', 'xhigh', 'max'], 'medium'),
    'gpt-5.6-luna': (['none', 'low', 'medium', 'high', 'xhigh', 'max'], 'medium'),
}


def _settings(config):
    def get(name, default=''):
        return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)
    provider = get('provider')
    default_url = 'https://api.anthropic.com' if provider == 'anthropic' else 'https://api.openai.com/v1'
    key = get('api_key')
    if hasattr(key, 'get_secret_value'): key = key.get_secret_value()
    return {'provider': provider, 'model': get('model'), 'base_url': get('base_url', default_url),
            'api_key': key if isinstance(key, str) else ''}


def _result(provider, model, *, source='unavailable', message='未获得明确的能力元数据；可保留服务默认值或手动填写推理强度。'):
    return {'provider': provider, 'model': model, 'levels': None, 'default_level': None,
            'source': source, 'verified': False, 'reasoning_supported': None,
            'thinking_modes': [], 'message': message}


def _levels(value):
    if not isinstance(value, list): return None
    values = [item.get('effort') if isinstance(item, dict) else item for item in value]
    if any(not isinstance(item, str) or not _TOKEN.fullmatch(item) for item in values): return None
    return list(dict.fromkeys(values))


def parse_model_capabilities(metadata, *, provider, model, source='service_metadata', source_url=None):
    """Normalize explicit Codex, DeepSeek, Claude or compatible metadata only.

    Lists select an exact model ID/slug. Single-model retrieve endpoints can
    resolve aliases, so the requested label remains unchanged in the result.
    """
    result = _result(provider, model, source=source)
    if not isinstance(metadata, dict): return result
    for key in ('data', 'models'):
        if isinstance(metadata.get(key), list):
            metadata = next((item for item in metadata[key] if isinstance(item, dict) and
                             item.get('id', item.get('slug')) == model), {})
            break
    capabilities = metadata.get('capabilities') if isinstance(metadata.get('capabilities'), dict) else {}
    reasoning = metadata.get('reasoning', capabilities.get('reasoning'))
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    effort = metadata.get('effort') if isinstance(metadata.get('effort'), dict) else capabilities.get('effort', reasoning.get('effort', {}))
    effort = effort if isinstance(effort, dict) else {}
    levels = None
    for value in (metadata.get('supported_reasoning_levels'), metadata.get('reasoning_levels'), metadata.get('levels'),
                  capabilities.get('supported_reasoning_levels'),
                  effort.get('supported_levels'), reasoning.get('supported_levels')):
        parsed = _levels(value)
        if parsed is not None: levels = parsed; break
    if levels is None and effort:
        flags = {key: value['supported'] for key, value in effort.items()
                 if isinstance(value, dict) and isinstance(value.get('supported'), bool) and _TOKEN.fullmatch(key)}
        if flags: levels = [key for key, supported in flags.items() if supported]
        elif effort.get('supported') is False: levels = []
    if levels is None and (metadata.get('reasoning_supported') is False or reasoning.get('supported') is False):
        levels = []
    thinking = capabilities.get('thinking', metadata.get('thinking'))
    thinking = thinking if isinstance(thinking, dict) else {}
    modes = _levels(metadata.get('thinking_modes'))
    if modes is None:
        types = thinking.get('types') if isinstance(thinking.get('types'), dict) else {}
        modes = [key for key, value in types.items() if isinstance(value, dict) and value.get('supported') is True and _TOKEN.fullmatch(key)]
    supported = next((value for value in (metadata.get('reasoning_supported'), reasoning.get('supported'),
                                        thinking.get('supported'), effort.get('supported')) if isinstance(value, bool)), None)
    if supported is None and levels is not None: supported = any(level != 'none' for level in levels)
    if source == 'deepseek_models_api' and levels:
        modes = modes or ['enabled', 'disabled']
    default = metadata.get('default_reasoning_level', metadata.get('default_level', effort.get('default_level', reasoning.get('default_level'))))
    result.update(levels=levels, default_level=default if isinstance(default, str) and levels is not None and default in levels else None,
                  thinking_modes=modes, reasoning_supported=supported,
                  verified=levels is not None or supported is not None or bool(modes))
    if result['verified']:
        result['message'] = '能力来自服务明确返回的模型元数据；未发送推理请求。'
        if levels is None: result['message'] += ' 服务未声明具体强度列表，仍需使用默认值或手动填写。'
    if source_url and source_url in (DEEPSEEK_DOCS, ANTHROPIC_DOCS): result['source_url'] = source_url
    return result


def codex_cache_path():
    """Only the non-secret model catalog is accessed; never read login files."""
    directory = Path(os.environ['CODEX_HOME']) if os.environ.get('CODEX_HOME') else Path.home() / '.codex'
    return directory / 'models_cache.json'


def _codex(settings):
    result = _result(settings['provider'], settings['model'], source='codex_models_cache')
    try: metadata = json.loads(codex_cache_path().read_text(encoding='utf-8'))
    except FileNotFoundError:
        result['message'] = '本机 Codex 模型缓存不存在；请在 Codex 中刷新模型目录后重试，或手动填写强度。'
        return result
    except (OSError, ValueError, UnicodeError):
        result['message'] = '本机 Codex 模型缓存无法读取或不是有效 JSON；请刷新模型目录后重试。'
        return result
    result = parse_model_capabilities(metadata, provider=settings['provider'], model=settings['model'], source='codex_models_cache')
    result['message'] = ('能力来自本机 Codex 官方模型缓存的精确 slug；未验证登录状态。' if result['verified'] else
                         '本机 Codex 模型缓存未声明此精确模型的推理强度；请刷新模型目录或手动填写。')
    return result


def _base_url(settings):
    value = settings['base_url']
    if not isinstance(value, str): raise ValueError('invalid endpoint')
    parsed = urlsplit(value)
    _ = parsed.port
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('invalid endpoint')
    if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('remote endpoint requires HTTPS')
    return value.rstrip('/'), parsed.hostname


def _profile(settings):
    try: _, hostname = _base_url(settings)
    except ValueError: return None
    if settings['provider'] != 'openai' or hostname != 'api.openai.com' or settings['model'] not in _OPENAI_PROFILES: return None
    levels, default = _OPENAI_PROFILES[settings['model']]
    return {**_result(settings['provider'], settings['model']), 'levels': list(levels), 'default_level': default,
            'reasoning_supported': True, 'verified': True, 'source': 'official_docs_2026-10-22',
            'source_url': OPENAI_DOCS + settings['model'],
            'message': '能力来自 2026-10-22 核对的精确模型官方 API 文档；不代表已验证账户访问权限。'}


def _cache_key(settings):
    fingerprint = hashlib.sha256(settings['api_key'].encode()).hexdigest()
    return settings['provider'], settings['model'], settings['base_url'], fingerprint


def cached_model_capabilities(config):
    """Synchronous serialization guard: no HTTP, no inferred proxy capabilities."""
    settings = _settings(config)
    if settings['provider'] == 'codex': return _codex(settings)
    stored = _DISCOVERED.get(_cache_key(settings))
    if stored is not None and time.monotonic() - stored[0] < CACHE_TTL_S: return deepcopy(stored[1])
    return _profile(settings) or _result(settings['provider'], settings['model'])


async def discover_model_capabilities(config):
    settings = _settings(config)
    if settings['provider'] == 'codex': return _codex(settings)
    provider, model, key = settings['provider'], settings['model'], settings['api_key']
    result = _result(provider, model)
    try: base, hostname = _base_url(settings)
    except ValueError:
        result['message'] = '元数据服务地址无效；请使用不含账号、参数或密钥的 HTTP(S) 地址，远程连接须使用 HTTPS。'
        return result
    if provider not in ('openai', 'compatible', 'anthropic', 'custom'):
        result['message'] = '该提供方未定义只读模型元数据接口；可手动填写推理强度。'
        return result
    if not key and hostname not in ('localhost', '127.0.0.1', '::1'):
        result = _profile(settings) or result
        result['message'] = '未提供 API Key，未查询远程元数据。' + result['message']
        return result
    deepseek = hostname == 'api.deepseek.com' and provider in ('compatible', 'openai')
    source = 'deepseek_models_api' if deepseek else 'anthropic_models_api' if provider == 'anthropic' else 'service_metadata'
    docs = DEEPSEEK_DOCS if deepseek else ANTHROPIC_DOCS if provider == 'anthropic' else None
    if provider == 'anthropic' and not base.endswith('/v1'): base += '/v1'
    headers = ({'x-api-key': key, 'anthropic-version': '2023-06-01'} if provider == 'anthropic'
               else {'Authorization': 'Bearer ' + key}) if key else {}
    endpoint = base + '/models' + ('' if deepseek else '/' + quote(model, safe=''))
    failure = None
    try:
        async with httpx.AsyncClient(timeout=DISCOVERY_TIMEOUT_S, trust_env=False, follow_redirects=False) as client:
            response = await client.get(endpoint, headers=headers)
            # Compatible servers may expose only the standard list endpoint.
            if response.status_code in (404, 405) and not deepseek:
                response = await client.get(base + '/models', headers=headers)
            response.raise_for_status()
            try: metadata = response.json()
            except ValueError: failure = '模型元数据接口未返回有效 JSON。'
            else: result = parse_model_capabilities(metadata, provider=provider, model=model, source=source, source_url=docs)
    except httpx.HTTPStatusError as exc: failure = f'模型元数据查询返回 HTTP {exc.response.status_code}；请检查访问权限、模型 ID 和服务接口。'
    except httpx.TimeoutException: failure = f'模型元数据查询超过 {DISCOVERY_TIMEOUT_S:g} 秒；请检查网络后重试。'
    except (httpx.RequestError, httpx.InvalidURL) as exc: failure = f'模型元数据查询连接失败（{type(exc).__name__}）；请检查网络和服务地址。'
    if failure or not result['verified']:
        profile = _profile(settings)
        if profile is not None:
            result = profile
            result['message'] = (failure or '模型接口没有返回推理强度字段。') + result['message']
        elif failure: result = _result(provider, model, source=source, message=failure + ' 可使用服务默认值或手动填写强度。')
    if key and any(key in str(value) for value in (result.get('levels'), result.get('default_level'), result.get('thinking_modes'))):
        result = _result(provider, model, source=source, message='服务返回的能力字段包含凭据信息，已拒绝展示；请检查元数据接口。')
    _DISCOVERED[_cache_key(settings)] = (time.monotonic(), deepcopy(result))
    return result
