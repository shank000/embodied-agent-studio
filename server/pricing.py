"""Hourly public price discovery and immutable per-call price selection."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
import time
from urllib.parse import urlsplit

from .pricing_sources import fetch_official_prices

REFRESH_INTERVAL_S = 3600
CACHE_PATH = Path(__file__).resolve().parents[1] / 'artifacts' / 'pricing' / 'catalog.json'
PROVIDERS = ('openai', 'deepseek', 'anthropic')
SOURCE_URLS = {'openai': 'https://developers.openai.com/api/docs/pricing',
               'deepseek': 'https://api-docs.deepseek.com/quick_start/pricing/',
               'anthropic': 'https://platform.claude.com/docs/en/about-claude/pricing'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class PriceCatalog:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else CACHE_PATH
        self.providers = {}
        self._state_lock = RLock()
        self._refresh_lock = asyncio.Lock()
        self._last_attempt = None
        self.refreshing = False
        self.running = False
        self._task = None
        self.cache_error = None
        try:
            cached = json.loads(self.path.read_text(encoding='utf-8'))
            if cached.get('schema_version') != 'official-prices-v1' or not isinstance(cached.get('providers'), dict):
                raise ValueError('价格缓存格式不正确。')
            for provider, value in cached['providers'].items():
                if provider in PROVIDERS and isinstance(value, dict) and isinstance(value.get('models'), list):
                    self.providers[provider] = value
        except FileNotFoundError: pass
        except (OSError, ValueError, TypeError) as exc: self.cache_error = f'价格缓存读取失败：{exc}'

    def status(self):
        with self._state_lock:
            rows = []
            for provider in PROVIDERS:
                item = self.providers.get(provider, {})
                state = 'unavailable'
                if item.get('models') and item.get('fetched_at'):
                    try: age = (datetime.now(timezone.utc) - datetime.fromisoformat(item['fetched_at'])).total_seconds()
                    except (TypeError, ValueError): age = float('inf')
                    state = 'stale' if age >= REFRESH_INTERVAL_S or item.get('error') else 'ready'
                rows.append({'provider': provider, 'source_url': item.get('source_url', SOURCE_URLS[provider]),
                    'updated_at': item.get('fetched_at'), 'last_attempt_at': item.get('last_attempt_at'),
                    'status': state, 'error': item.get('error'), 'model_count': len(item.get('models', []))})
            return {'updated_at': max((row['updated_at'] for row in rows if row['updated_at']), default=None),
                    'refresh_interval_s': REFRESH_INTERVAL_S, 'refreshing': self.refreshing,
                    'providers': rows, 'cache_error': self.cache_error}

    async def refresh(self, *, force=False):
        async with self._refresh_lock:
            if not force and self._last_attempt is not None and time.monotonic() - self._last_attempt < REFRESH_INTERVAL_S:
                return self.status()
            self._last_attempt = time.monotonic()
            self.refreshing = True
            attempt_at = utc_now()
            try:
                results = await asyncio.gather(*(fetch_official_prices(provider) for provider in PROVIDERS), return_exceptions=True)
                with self._state_lock:
                    for provider, result in zip(PROVIDERS, results):
                        if isinstance(result, Exception):
                            previous = self.providers.get(provider, {'models': [], 'source_url': SOURCE_URLS[provider]})
                            self.providers[provider] = {**previous, 'last_attempt_at': attempt_at,
                                                       'error': f'{type(result).__name__}: {str(result)[:1000]}'}
                        else:
                            self.providers[provider] = {**result, 'last_attempt_at': attempt_at, 'error': None}
                    try:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        pending = self.path.with_suffix('.pending.json')
                        pending.write_text(json.dumps({'schema_version': 'official-prices-v1', 'providers': self.providers}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
                        pending.replace(self.path)
                        self.cache_error = None
                    except (OSError, ValueError) as exc: self.cache_error = f'价格已读取，但本机缓存保存失败：{exc}'
            finally:
                self.refreshing = False
            return self.status()

    async def start(self):
        self.running = True
        async def loop():
            while self.running:
                await self.refresh()
                await asyncio.sleep(REFRESH_INTERVAL_S)
        self._task = asyncio.create_task(loop(), name='official-token-pricing')

    async def close(self):
        self.running = False
        if self._task is not None:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
            self._task = None

    def quote(self, settings, *, basis='recorded_snapshot'):
        provider = settings.get('provider', settings.get('kind'))
        model = settings.get('model')
        endpoint = settings.get('base_url') or settings.get('endpoint') or settings.get('url') or ''
        host = urlsplit(endpoint).hostname
        billing_kind = 'api'
        if provider == 'codex': vendor, billing_kind = 'openai', 'reference'
        elif provider in ('openai', 'compatible') and host == 'api.openai.com': vendor = 'openai'
        elif provider in ('openai', 'compatible', 'anthropic') and host == 'api.deepseek.com': vendor = 'deepseek'
        elif provider == 'anthropic' and host == 'api.anthropic.com': vendor = 'anthropic'
        elif provider == 'smolvla':
            return {'status': 'ready', 'provider': 'local', 'model': model, 'billing_kind': 'local',
                    'currency': 'USD', 'basis': basis, 'updated_at': None, 'source_url': None}
        else:
            return {'status': 'unavailable', 'provider': provider, 'model': model, 'billing_kind': 'unpriced',
                    'currency': 'USD', 'basis': basis, 'message': '此接入方式没有可确认的官方 Token 单价，未推测第三方或自定义服务价格。'}
        with self._state_lock:
            entry = self.providers.get(vendor, {})
            prices = next((row for row in entry.get('models', []) if row.get('model') == model or model in row.get('aliases', [])), None)
            status = next(row for row in self.status()['providers'] if row['provider'] == vendor)
            common = {'provider': vendor, 'model': model, 'billing_kind': billing_kind, 'currency': 'USD',
                      'basis': basis, 'source_url': entry.get('source_url', SOURCE_URLS[vendor]),
                      'updated_at': entry.get('fetched_at'), 'status': status['status']}
            if prices is None:
                return {**common, 'status': 'unavailable', 'message': entry.get('error') or '官方报价尚未包含此精确模型 ID。'}
            return {**deepcopy(prices), **common, 'refresh_error': entry.get('error')}


catalog = PriceCatalog()


async def quote_for_call(settings):
    # Price discovery never delays an observation/action loop. A request records
    # the currently available quote, including its unavailable/stale state.
    return catalog.quote(settings)


def estimate_quote(settings):
    return catalog.quote(settings, basis='current_price_estimate')
