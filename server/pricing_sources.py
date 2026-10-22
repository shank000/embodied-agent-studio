"""Public official price sources. No credentials, inference probes or seeded rates.

Rates are USD per million tokens. Ambiguous modifiers remain explicitly
unsupported; callers decide whether they can supply the missing conditions.
"""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from html import unescape
from html.parser import HTMLParser
import math
import re

import httpx

SOURCES = {
    'openai': 'https://developers.openai.com/api/docs/pricing',
    'deepseek': 'https://api-docs.deepseek.com/quick_start/pricing/',
    'anthropic': 'https://platform.claude.com/docs/en/about-claude/pricing',
}
FETCH_URLS = {**SOURCES, 'openai': SOURCES['openai'] + '.md', 'anthropic': SOURCES['anthropic'] + '.md'}
_OPENAI_WORKBENCH_MODELS = {'gpt-6-astra', 'gpt-6.1-sol', 'gpt-6-sol', 'gpt-6-luna',
                          'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna'}


class PricingSourceError(ValueError):
    """The fetched official source cannot establish an unambiguous price."""


def _text(value):
    value = re.sub(r'<sup\b[^>]*>.*?</sup>', '', value, flags=re.I | re.S)
    value = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', value)
    return re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]*>', ' ', value)).replace('`', '')).strip()


def _money(value):
    text = _text(value)
    if re.search(r'-\s*\$|\$\s*-', text): raise PricingSourceError('美元金额不能为负数。')
    matches = re.findall(r'\$\s*((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.\d+)?)(?=\s|/|$)', text)
    if len(matches) != 1: raise PricingSourceError('价格单元格缺少唯一的明确美元金额。')
    try: amount = float(Decimal(matches[0].replace(',', '')))
    except (InvalidOperation, ValueError, OverflowError) as exc: raise PricingSourceError('美元金额不是有效数字。') from exc
    if not math.isfinite(amount) or amount < 0: raise PricingSourceError('美元金额必须是有限的非负数。')
    return amount


class _HTMLTables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []; self.stack = []; self.table = None; self.row = None; self.cell = None
        self.recent = ''; self.heading = ''; self.in_heading = False; self.skip_sup = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('h1', 'h2', 'h3', 'h4'):
            self.in_heading = True; self.heading = ''
        if tag == 'sup': self.skip_sup += 1
        if tag == 'table' and self.table is None:
            context = ' '.join(f'{name}={value}' for _, values in self.stack for name, value in values.items())
            context += ' ' + ' '.join(f'{name}={value}' for name, value in attrs.items())
            self.table = {'rows': [], 'context': self.heading + ' ' + context, 'lead': self.recent[-3000:]}
        elif self.table is not None and tag == 'tr': self.row = []
        elif self.table is not None and tag in ('td', 'th'):
            self.cell = {'text': '', 'rowspan': attrs.get('rowspan', '1'), 'colspan': attrs.get('colspan', '1')}
        if tag == 'br' and self.cell is not None: self.cell['text'] += ' '
        if tag not in ('br', 'hr', 'img', 'input', 'meta', 'link', 'source', 'wbr'): self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        if tag == 'sup': self.skip_sup = max(0, self.skip_sup - 1)
        if tag in ('h1', 'h2', 'h3', 'h4'): self.in_heading = False
        if tag in ('td', 'th') and self.cell is not None and self.row is not None:
            self.row.append(self.cell); self.cell = None
        if tag == 'tr' and self.row is not None and self.table is not None:
            self.table['rows'].append(self.row); self.row = None
        if tag == 'table' and self.table is not None:
            self.table['rows'] = _expand(self.table['rows']); self.tables.append(self.table); self.table = None
        for index in range(len(self.stack)-1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]; break

    def handle_data(self, value):
        if self.skip_sup: return
        self.recent += value + ' '
        if self.in_heading: self.heading += value
        if self.cell is not None: self.cell['text'] += value


def _expand(rows):
    grid = {}; width = 0
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row:
            while (row_index, column) in grid: column += 1
            try: rowspan, colspan = int(cell['rowspan']), int(cell['colspan'])
            except (TypeError, ValueError) as exc: raise PricingSourceError('官方价格表的跨行/跨列标记无效。') from exc
            if not 1 <= rowspan <= 64 or not 1 <= colspan <= 64: raise PricingSourceError('官方价格表跨度超出可识别范围。')
            for dr in range(rowspan):
                for dc in range(colspan): grid[row_index + dr, column + dc] = _text(cell['text'])
            column += colspan; width = max(width, column)
    return [[grid.get((index, column), '') for column in range(width)] for index in range(len(rows))]


def _tables(document):
    if '<table' in document.lower():
        parser = _HTMLTables(); parser.feed(document); return parser.tables
    tables = []; lines = document.splitlines(); headings = []; index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            headings = [(depth, title) for depth, title in headings if depth < level] + [(level, line.lstrip('#').strip())]
        if '|' in line and index + 1 < len(lines) and re.match(r'^\s*\|?\s*:?-{3,}', lines[index + 1]):
            start = index; rows = []
            while index < len(lines) and '|' in lines[index] and lines[index].strip():
                row = [_text(part) for part in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\s-]*', cell) for cell in row): rows.append(row)
                index += 1
            tables.append({'rows': rows, 'context': ' / '.join(title for _, title in headings),
                           'lead': '\n'.join(lines[max(0, start-14):start])})
            continue
        index += 1
    return tables


_COLUMNS = {
    'input': 'input_per_million', 'inputtokens': 'input_per_million', 'baseinputtokens': 'input_per_million',
    'output': 'output_per_million', 'outputtokens': 'output_per_million',
    'cachedinput': 'cached_input_per_million', 'cachehitsandrefreshes': 'cached_input_per_million',
    '5mcachewrites': 'cache_write_5m_per_million', '5mwrites': 'cache_write_5m_per_million',
    '1hcachewrites': 'cache_write_1h_per_million', '1hwrites': 'cache_write_1h_per_million',
    '30mwrite': 'cache_write_30m_per_million', '30mwrites': 'cache_write_30m_per_million',
    '12hwrite': 'cache_write_12h_per_million', '12hwrites': 'cache_write_12h_per_million',
}


def _header(rows):
    for index, row in enumerate(rows[:4]):
        names = [re.sub(r'[^a-z0-9]', '', cell.lower()) for cell in row]
        if not any(name in ('model', 'name') for name in names): continue
        columns = {}
        for position, name in enumerate(names):
            if name in _COLUMNS: columns.setdefault(_COLUMNS[name], position)
        if 'input_per_million' in columns and 'output_per_million' in columns:
            return index, columns, names.index('model') if 'model' in names else names.index('name')
    return None


def _rates(row, columns):
    result = {}
    for field, index in columns.items():
        if index >= len(row): raise PricingSourceError('价格表行缺少列，未沿用旧价格。')
        if row[index].strip() in ('-', '—', 'N/A', ''):
            if field in ('input_per_million', 'output_per_million'): raise PricingSourceError('模型输入/输出价格缺失。')
            continue
        result[field] = _money(row[index])
    return result


def parse_openai_prices(document):
    models = {}; issues = []; standard_selector_used = set()
    for table in _tables(document):
        header = _header(table['rows'])
        if header is None: continue
        index, columns, model_column = header
        if any('modality' in cell.lower() or 'training' in cell.lower() for cell in table['rows'][index]): continue
        context, lead = table['context'].lower(), table['lead'].lower()
        explicit = re.findall(r'(?:data-(?:service-tier|tier|value)|value|id)\s*=\s*[\'\"]?(standard|default|flex|batch|fast|priority|ultrafast)\b', context)
        heading_tiers = re.findall(r'(?:^|/|\bpricing\s+)\s*(standard|flex|batch|fast|priority|ultrafast)(?:\s+pricing)?\s*(?:/|$)', context)
        tiers = explicit or heading_tiers
        if tiers and tiers[-1] not in ('standard', 'default'): continue
        selector = re.search(r'\bstandard\s+(?:flex/batch|batch\s+flex|flex\s+batch)\s+fast(?:\s+ultrafast)?\b', lead)
        if not tiers:
            if selector and table['context'] not in standard_selector_used:
                standard_selector_used.add(table['context'])
            else:
                issues.append('未解析缺少明确 Standard 作用域的表格；未混用 Flex/Batch/Fast 费率。'); continue
        for row in table['rows'][index+1:]:
            if model_column >= len(row): continue
            match = re.fullmatch(r'(?:gpt-[a-z0-9_.-]+|o[0-9][a-z0-9_.-]*|chat-latest)', row[model_column])
            if not match: continue
            model = match.group()
            entry = {'model': model, **_rates(row, columns)}
            if model in models and models[model] != entry: raise PricingSourceError(f'OpenAI Standard 表中 {model} 有冲突价格，无法确定上下文档位。')
            models[model] = entry
    if not models: raise PricingSourceError('未找到可明确识别的 OpenAI Standard 文本模型价格表。')
    for entry in models.values():
        entry['unsupported'] = {'long_context': '该来源未给出可安全关联到此行的长上下文阈值与完整费率；长上下文请求不能直接套短档。'}
    return list(models.values()), issues


def _claude_id(label):
    match = re.match(r'^Claude\s+(Fable|Mythos|Opus|Sonnet|Haiku)\s+(\d+(?:\.\d+)*)\b', label, re.I)
    if not match: return None
    # Official model labels normalize to the corresponding documented API alias;
    # dated model revisions are not guessed or silently mapped to this alias.
    return 'claude-' + match[1].lower() + '-' + match[2].replace('.', '-')


def parse_anthropic_prices(document):
    models = {}; issues = []
    for table in _tables(document):
        if 'model pricing' not in table['context'].lower(): continue
        header = _header(table['rows'])
        if header is None: continue
        index, columns, model_column = header
        for row in table['rows'][index+1:]:
            if model_column >= len(row): continue
            label = row[model_column]; model = _claude_id(label)
            if not model: continue
            rates = _rates(row, columns)
            threshold = re.search(r'prompts\s+(up to|over)\s+([\d,]+)\s+tokens', label, re.I)
            if threshold:
                value = int(threshold[2].replace(',', ''))
                entry = models.setdefault(model, {'model': model})
                if entry.get('long_context_threshold', value) != value: raise PricingSourceError('Claude 上下文价格阈值不一致。')
                entry['long_context_threshold'] = value
                if threshold[1].lower() == 'over': entry['long_context_rates'] = rates
                else: entry.update(rates)
            elif model in models and any(models[model].get(key) != value for key, value in rates.items()):
                raise PricingSourceError(f'Claude 标准表中 {model} 有冲突价格。')
            else: models[model] = {'model': model, **rates}
            if 'retired' in label.lower(): models[model]['retired'] = True
    if not models: raise PricingSourceError('未找到 Claude Model pricing 标准费率表。')
    for entry in models.values():
        if 'input_per_million' not in entry or 'output_per_million' not in entry: raise PricingSourceError('Claude 模型缺少基础输入/输出档位。')
        if 'long_context_threshold' in entry and 'long_context_rates' not in entry:
            entry['unsupported'] = {'long_context': '价格表声明输入长度档位，但长档价格缺失。'}
    return list(models.values()), issues


def _deepseek_schedule(document):
    text = _text(document)
    match = re.search(r'Peak hours are\s+(.+?)\s+UTC[,.;]?\s*(.*?)(?:All other|\(3\)|For more|$)', text, re.I)
    if not match: return {'unsupported': {'calendar': '官方来源没有可明确解析的峰谷时段与时区规则。'}}
    pairs = re.findall(r'(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})', match[1])
    windows = []
    for h1, m1, h2, m2 in pairs:
        start, end = int(h1)*60+int(m1), int(h2)*60+int(m2)
        if not 0 <= start < end <= 1440: return {'unsupported': {'calendar': '官方峰谷时间区间无效或跨日，未猜测边界。'}}
        windows.append([start, end])
    if not windows: return {'unsupported': {'calendar': '无法解析官方峰谷时间区间。'}}
    restriction = match[2].lower()
    result = {'peak_utc': windows}
    if 'monday through friday' in restriction:
        result['peak_weekdays_utc'] = [0, 1, 2, 3, 4]
    if 'excluding chinese public holidays' in restriction:
        result['holiday_region'] = 'CN'
        result['unsupported'] = {'calendar': '峰价仅适用于周一至五且排除中国公众假期；自动选择需权威假日日历，不能仅按每日 UTC 小时判断。'}
    elif restriction.strip(' .,;'):
        result['unsupported'] = {'calendar': '官方峰谷时段存在未识别的附加条件，未当作每日通用时间表。'}
    else:
        cursor = 0; off_peak = []
        for start, end in sorted(windows):
            if start > cursor: off_peak.append([cursor, start])
            cursor = max(cursor, end)
        if cursor < 1440: off_peak.append([cursor, 1440])
        result['off_peak_utc'] = off_peak
    return result


def parse_deepseek_prices(document):
    models = {}; issues = []; schedule = _deepseek_schedule(document)
    for table in _tables(document):
        rows = table['rows']
        if not rows: continue
        model_columns = {}
        for index, value in enumerate(rows[0]):
            match = re.fullmatch(r'deepseek-[a-z0-9_.-]+', value)
            if match: model_columns[index] = match.group()
        if not model_columns: continue
        for model in model_columns.values(): models[model] = {'model': model, 'peak': {}, 'off_peak': {}, **schedule}
        for row in rows[1:]:
            label = ' '.join(row[:min(model_columns)]).upper()
            if '1M' not in label or 'TOKENS' not in label: continue
            field = ('cached_input_per_million' if 'CACHE HIT' in label else 'input_per_million' if 'CACHE MISS' in label else 'output_per_million' if 'OUTPUT' in label else None)
            if field is None: continue
            tier = 'off_peak' if 'OFF-PEAK' in label else 'peak' if 'PEAK' in label else None
            if tier is None: raise PricingSourceError('DeepSeek 价格行未明确标记 PEAK/OFF-PEAK。')
            for column, model in model_columns.items():
                if column >= len(row): raise PricingSourceError('DeepSeek 价格行缺少模型列。')
                models[model][tier][field] = _money(row[column])
    if not models: raise PricingSourceError('未找到 DeepSeek 模型价格表。')
    text = _text(document)
    for model, entry in models.items():
        required = {'input_per_million', 'output_per_million', 'cached_input_per_million'}
        if any(set(entry[tier]) != required for tier in ('peak', 'off_peak')): raise PricingSourceError(f'DeepSeek {model} 的峰谷输入/缓存/输出价格不完整。')
        # Keep both tiers. The top-level fields are the explicitly labelled peak
        # rates, not an assertion that peak applies to any particular call.
        entry.update(entry['peak']); entry['default_rate_tier'] = 'peak_reference_only'
        if model == 'deepseek-flash':
            alias_sentence = re.search(r'The legacy names (.+?) are still accepted.*?billed at the Flash price', text, re.I)
            if alias_sentence:
                aliases = re.findall(r'\bdeepseek-[a-z0-9_.-]+\b', alias_sentence[1])
                if aliases: entry['aliases'] = list(dict.fromkeys(aliases))
    return list(models.values()), issues


def parse_official_prices(provider, document):
    if provider not in SOURCES: raise PricingSourceError('仅支持 OpenAI、DeepSeek 和 Anthropic 官方价格来源。')
    if not isinstance(document, str) or not document.strip(): raise PricingSourceError('官方价格文档为空。')
    parser = {'openai': parse_openai_prices, 'deepseek': parse_deepseek_prices, 'anthropic': parse_anthropic_prices}[provider]
    models, issues = parser(document)
    return {'provider': provider, 'currency': 'USD', 'source_url': SOURCES[provider],
            'fetched_at': datetime.now(timezone.utc).isoformat(), 'models': models, 'issues': issues,
            'service_tier': 'standard', 'scope': 'first_party_public_token_rates'}


def parse_openai_long_context_policy(document):
    """Read the exact model page's explicit threshold and price multipliers."""
    text = _text(document)
    pattern = (r'Prompts with\s+(?:more than\s+|>\s*)([\d,.]+)\s*([KM]?)\s+input tokens '
               r'are priced at\s+([\d.]+)\s*[x×]\s+input(\s+and cache rates)?\s+and '
               r'([\d.]+)\s*[x×]\s+output\s+for the full request')
    policies = []
    for match in re.finditer(pattern, text, re.I):
        try:
            threshold = Decimal(match[1].replace(',', '')) * {'': 1, 'K': 1000, 'M': 1000000}[match[2].upper()]
            input_multiplier, output_multiplier = float(Decimal(match[3])), float(Decimal(match[5]))
        except (InvalidOperation, ValueError, KeyError) as exc: raise PricingSourceError('模型页的长上下文阈值或倍率无效。') from exc
        if threshold <= 0 or threshold != threshold.to_integral_value() or any(not math.isfinite(value) or value <= 0 for value in (input_multiplier, output_multiplier)):
            raise PricingSourceError('模型页的长上下文阈值或倍率无效。')
        policies.append({'threshold': int(threshold), 'input_multiplier': input_multiplier,
                         'output_multiplier': output_multiplier,
                         'cache_multiplier': input_multiplier if match[4] else None})
    if policies and any(policy != policies[0] for policy in policies): raise PricingSourceError('模型页包含互相冲突的长上下文定价规则。')
    return policies[0] if policies else None


async def _augment_openai(catalog, client):
    async def enrich(entry):
        source = 'https://developers.openai.com/api/docs/models/' + entry['model']
        reason = None
        try:
            response = await client.get(source + '.md', headers={'Accept': 'text/markdown'})
            response.raise_for_status()
            policy = parse_openai_long_context_policy(response.text)
            if policy is None: reason = '精确模型官方页面未声明可识别的长上下文倍率规则。'
        except httpx.HTTPStatusError as exc: reason = f'精确模型长上下文来源返回 HTTP {exc.response.status_code}。'
        except httpx.TimeoutException: reason = '精确模型长上下文来源请求超过 20 秒。'
        except httpx.RequestError as exc: reason = f'精确模型长上下文来源请求失败（{type(exc).__name__}）。'
        except PricingSourceError as exc: reason = str(exc)
        if reason:
            entry.setdefault('unsupported', {})['long_context'] = reason
            return
        entry['long_context_threshold'] = policy['threshold']
        entry['long_context_source_url'] = source
        rates = {}
        for field, value in entry.items():
            if not field.endswith('_per_million'): continue
            multiplier = (policy['input_multiplier'] if field == 'input_per_million' else
                          policy['output_multiplier'] if field == 'output_per_million' else policy['cache_multiplier'])
            rates[field] = float(Decimal(str(value))*Decimal(str(multiplier))) if multiplier is not None else None
        entry['long_context_rates'] = rates
        entry.setdefault('unsupported', {}).pop('long_context', None)
        if policy['cache_multiplier'] is None and any('cache' in field for field in rates):
            entry['unsupported']['long_context_cache'] = '模型页未明确长上下文缓存倍率；长档缓存价格保留未知，未套用短档价格。'
    await asyncio.gather(*(enrich(entry) for entry in catalog['models'] if entry['model'] in _OPENAI_WORKBENCH_MODELS))
    return catalog


async def fetch_official_prices(provider):
    """Public GETs only; OpenAI enriches at most seven exact model pages in parallel."""
    if provider not in FETCH_URLS: raise PricingSourceError('未知官方价格提供方。')
    try:
        # Public documentation has no model credentials. Respect the machine's
        # normal HTTPS proxy/certificate configuration; TLS verification stays on.
        async with httpx.AsyncClient(timeout=20., follow_redirects=True) as client:
            response = await client.get(FETCH_URLS[provider], headers={'Accept': 'text/markdown' if FETCH_URLS[provider].endswith('.md') else 'text/html'})
            response.raise_for_status()
            catalog = parse_official_prices(provider, response.text)
            if provider == 'openai': await _augment_openai(catalog, client)
    except httpx.HTTPStatusError as exc: raise PricingSourceError(f'{provider} 官方价格请求返回 HTTP {exc.response.status_code}。') from exc
    except httpx.TimeoutException as exc: raise PricingSourceError(f'{provider} 官方价格请求超过 20 秒。') from exc
    except httpx.RequestError as exc: raise PricingSourceError(f'{provider} 官方价格请求失败（{type(exc).__name__}）。') from exc
    return catalog
