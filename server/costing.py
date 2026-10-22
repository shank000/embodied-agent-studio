"""USD list-price estimates from returned token counters and explicit price quotes.

No provider calls or guessed tokens. Codex API-equivalent reference amounts never
contribute to the API amount. Decimal arithmetic is used until JSON conversion.
"""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from .holiday_calendar import resolve_public_holiday


MILLION = Decimal(1_000_000)


def _counter(data, *names):
    values = [data[name] for name in names if data.get(name) is not None]
    if not values: return None
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError('Token 计数必须为非负整数：' + '/'.join(names))
    if any(value != values[0] for value in values[1:]):
        raise ValueError('同义 Token 计数不一致：' + '/'.join(names))
    return values[0]


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _split_writes(total, details, first, second, default_ttl=None):
    a, b = _counter(details, first), _counter(details, second)
    if total == 0:
        if (a or b): raise ValueError('缓存写明细超过缓存写总量。')
        return 0, 0, 0
    if total is None and a is not None and b is not None: total = a+b
    if total is not None:
        if a is not None and b is None: b = total-a
        elif b is not None and a is None: a = total-b
        elif a is None and b is None and default_ttl == first: a, b = total, 0
        elif a is None and b is None and default_ttl == second: a, b = 0, total
    if (a is not None and a < 0) or (b is not None and b < 0):
        raise ValueError('缓存写明细超过缓存写总量。')
    if total is not None and a is not None and b is not None and a+b != total:
        raise ValueError('缓存写明细之和与缓存写总量不一致。')
    return total, a, b


def _components(call, quote):
    usage = _mapping(call.get('usage'))
    incoming = _counter(usage, 'input_tokens', 'prompt_tokens')
    outgoing = _counter(usage, 'output_tokens', 'completion_tokens')
    output_details = _mapping(usage.get('output_tokens_details') or usage.get('completion_tokens_details'))
    reasoning = _counter(output_details, 'reasoning_tokens')
    top_reasoning = _counter(usage, 'reasoning_tokens')
    if reasoning is None: reasoning = top_reasoning
    elif top_reasoning is not None and reasoning != top_reasoning: raise ValueError('推理 Token 明细不一致。')
    if reasoning is not None and outgoing is not None and reasoning > outgoing:
        raise ValueError('推理 Token 超过包含它的输出 Token 总量。')
    components = [('output', outgoing, 'output_per_million')]
    provider = quote.get('provider')
    if provider == 'anthropic':
        read = _counter(usage, 'cache_read_input_tokens')
        written = _counter(usage, 'cache_creation_input_tokens')
        detail = _mapping(usage.get('cache_creation'))
        default_ttl = {'5m': 'ephemeral_5m_input_tokens', '1h': 'ephemeral_1h_input_tokens'}.get(quote.get('cache_write_default_ttl'))
        written, short, long = _split_writes(written, detail, 'ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens', default_ttl)
        components += [('input', incoming, 'input_per_million'), ('cached_input', read, 'cached_input_per_million'),
                       ('cache_write_5m', short, 'cache_write_5m_per_million'),
                       ('cache_write_1h', long, 'cache_write_1h_per_million')]
        total = incoming+read+written if all(value is not None for value in (incoming, read, written)) else None
        return components, total
    if provider == 'deepseek' or 'prompt_cache_hit_tokens' in usage or 'prompt_cache_miss_tokens' in usage:
        read = _counter(usage, 'prompt_cache_hit_tokens')
        ordinary = _counter(usage, 'prompt_cache_miss_tokens')
        if incoming is None and read is not None and ordinary is not None: incoming = read+ordinary
        if incoming is not None:
            if read is not None and ordinary is None: ordinary = incoming-read
            elif ordinary is not None and read is None: read = incoming-ordinary
            elif incoming == 0 and read is None and ordinary is None: read = ordinary = 0
        if (read is not None and read < 0) or (ordinary is not None and ordinary < 0):
            raise ValueError('缓存命中或未命中 Token 超过输入总量。')
        if incoming is not None and read is not None and ordinary is not None and read+ordinary != incoming:
            raise ValueError('缓存命中与未命中 Token 之和不等于输入总量。')
        return components+[('input', ordinary, 'input_per_million'), ('cached_input', read, 'cached_input_per_million')], incoming
    # OpenAI input includes cache reads and writes. Codex reports reads at top level.
    details = _mapping(usage.get('input_tokens_details') or usage.get('prompt_tokens_details'))
    read = _counter(details, 'cached_tokens')
    top_read = _counter(usage, 'cached_input_tokens')
    if read is not None and top_read is not None and read != top_read: raise ValueError('缓存读取 Token 明细不一致。')
    if read is None: read = top_read
    written = _counter(details, 'cache_write_tokens')
    write_details = _mapping(details.get('cache_write_tokens_details'))
    has_write_rates = any(rates.get(key) is not None
                          for rates in (quote, _mapping(quote.get('long_context_rates')), _mapping(quote.get('peak')), _mapping(quote.get('off_peak')))
                          for key in ('cache_write_30m_per_million', 'cache_write_12h_per_million'))
    if written is None and not write_details and not has_write_rates: written = 0
    if incoming == 0:
        if read is None: read = 0
        if written is None: written = 0
    default_ttl = {'30m': 'ttl_30m_tokens', '12h': 'ttl_12h_tokens'}.get(quote.get('cache_write_default_ttl'))
    written, short, long = _split_writes(written, write_details, 'ttl_30m_tokens', 'ttl_12h_tokens', default_ttl)
    minimum_written = written if written is not None else (short or 0)+(long or 0)
    if incoming is not None and (read or 0)+minimum_written > incoming:
        raise ValueError('缓存读取和写入 Token 重叠或超过输入总量。')
    ordinary = incoming-read-written if all(value is not None for value in (incoming, read, written)) else None
    return components+[('input', ordinary, 'input_per_million'), ('cached_input', read, 'cached_input_per_million'),
                       ('cache_write_30m', short, 'cache_write_30m_per_million'),
                       ('cache_write_12h', long, 'cache_write_12h_per_million')], incoming


def _in_windows(minute, windows):
    return any(start <= minute < end if start <= end else minute >= start or minute < end for start, end in windows)


def _rate_choices(call, quote, warnings, resolution=None):
    peak, off = _mapping(quote.get('peak')), _mapping(quote.get('off_peak'))
    if not peak and not off: return [quote]
    if _mapping(quote.get('unsupported')).get('calendar') and not quote.get('holiday_region'):
        warnings.append('官方峰谷附加条件未能完整解析，不能确定适用价格，仅显示金额范围。')
        return [{**quote, **off}, {**quote, **peak}]
    try:
        instant = datetime.fromisoformat(call.get('started_at', '').replace('Z', '+00:00'))
        if instant.tzinfo is None: raise ValueError('timestamp lacks timezone')
        instant = instant.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError):
        warnings.append('调用开始时间缺失或没有时区，不能确定峰谷价格。')
        return [{**quote, **off}, {**quote, **peak}]
    minute = instant.hour*60+instant.minute+instant.second/60
    if quote.get('peak_utc') is not None:
        possible_peak = _in_windows(minute, quote['peak_utc']) and instant.weekday() in quote.get('peak_weekdays_utc', range(7))
        if not possible_peak: return [{**quote, **off}]
        if quote.get('holiday_region'):
            calendar = resolve_public_holiday(instant, quote['holiday_region'])
            if resolution is not None: resolution.update(calendar)
            if calendar.get('known'):
                tier = 'off_peak' if calendar['is_holiday'] else 'peak'
                if resolution is not None: resolution['tier'] = tier
                warnings.append(f"按 {calendar['year']} 年国务院全国放假安排核对，此次调用采用{'谷' if calendar['is_holiday'] else '峰'}价估算；实际扣费仍以服务商账单为准。")
                return [{**quote, **(off if calendar['is_holiday'] else peak)}]
            warnings.append(calendar['message'])
            return [{**quote, **off}, {**quote, **peak}]
        return [{**quote, **peak}]
    if quote.get('off_peak_utc') is not None:
        return [{**quote, **(off if _in_windows(minute, quote['off_peak_utc']) else peak)}]
    warnings.append('峰谷价格未包含明确的 UTC 时间规则，不能确定适用价格。')
    return [{**quote, **off}, {**quote, **peak}]


def _price_components(components, rates, input_total, warnings):
    threshold = rates.get('long_context_threshold')
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
            raise ValueError('长上下文价格阈值无效。')
        if input_total is None:
            warnings.append('缺少完整输入计数，无法确定长上下文价格档位。')
            return [], False
        if input_total > threshold:
            tier = _mapping(rates.get('long_context_rates'))
            if not tier:
                warnings.append('调用超过长上下文阈值，但缺少对应价格。')
                return [], False
            rates = {**rates, **tier}
    breakdown, complete = [], True
    for name, tokens, rate_name in components:
        if tokens is None:
            warnings.append(f'{name} 的实际 Token 计数不完整，此部分未估算。'); complete = False; continue
        if tokens == 0: continue
        value = rates.get(rate_name)
        try:
            rate = Decimal(str(value)) if value is not None and not isinstance(value, bool) else None
            if rate is not None and (not rate.is_finite() or rate < 0): rate = None
        except InvalidOperation: rate = None
        if rate is None:
            warnings.append(f'{name} 缺少有效的官方单价，此部分未估算。'); complete = False; continue
        breakdown.append({'component': name, 'tokens': tokens, 'rate_per_million': rate,
                          'amount_usd': Decimal(tokens)*rate/MILLION})
    return breakdown, complete


def _sum(values):
    return sum((Decimal(str(value)) for value in values), Decimal(0))


def calculate_call_cost(call, quote):
    quote = deepcopy(quote) if isinstance(quote, dict) else {}
    kind = quote.get('billing_kind', 'unpriced')
    result = {'status': 'unavailable', 'currency': 'USD', 'amount_usd': None, 'known_amount_usd': 0.,
              'reference_amount_usd': None, 'known_reference_amount_usd': 0., 'priced_calls': 0,
              'total_calls': 1, 'missing_calls': 1, 'billable_calls': int(kind == 'api'),
              'reference_calls': int(kind == 'reference'), 'local_calls': int(kind == 'local'),
              'basis': quote.get('basis', 'recorded_snapshot'), 'warnings': [], 'breakdown': [],
              'price_sources': [], 'has_reference': kind == 'reference', 'pricing': quote}
    warnings = result['warnings']
    for key in ('message', 'refresh_error'):
        if quote.get(key): warnings.append(str(quote[key]))
    warnings.extend(str(value) for key, value in _mapping(quote.get('unsupported')).items()
                    if value and not (key == 'calendar' and quote.get('holiday_region')))
    if kind == 'local':
        result.update(status='complete', amount_usd=0., priced_calls=1, missing_calls=0)
        warnings.append('本地策略没有按 Token 收取的模型 API 费用；此处不包含本机算力、电费或运维成本。')
        return result
    if kind not in ('api', 'reference') or quote.get('status') not in ('ready', 'stale') or quote.get('currency') != 'USD':
        warnings.append('没有可用的 USD 官方价格快照，费用未知。'); return result
    result['price_sources'] = [{key: quote.get(key) for key in ('provider', 'model', 'source_url', 'updated_at')}]
    if _mapping(quote.get('unsupported')).get('long_context') and quote.get('long_context_threshold') is None:
        warnings.append('来源未明确当前费率的上下文适用范围，无法确认该调用应采用的价格档位。')
        return result
    warnings.append('按官方公开 Token 单价估算，实际账单还取决于服务档位、优惠、税费及其他收费项目。')
    if quote.get('status') == 'stale':
        warnings.append('使用过期的当前价格缓存，可刷新后重新估算；历史价格快照保持不变。' if result['basis'] == 'current_price_estimate' else
                        '此调用保存的是过期价格快照；刷新当前价格不会重算这条历史记录。')
    if kind == 'reference': warnings.append('Codex 登录调用仅提供同等 API 用量参考金额，不代表订阅扣款或实际 API 费用。')
    try:
        components, input_total = _components(call, quote)
        resolution = {}
        choices = _rate_choices(call, quote, warnings, resolution)
        if resolution:
            result['calendar_resolution'] = resolution
            if resolution.get('known'): result['calendar_sources'] = [{key: resolution[key] for key in
                ('calendar_id', 'year', 'coverage', 'source_url', 'document_date', 'checked_at', 'interpretation')}]
        priced = [_price_components(components, rates, input_total, warnings) for rates in choices]
    except (ValueError, TypeError) as exc:
        warnings.append(str(exc)); result['warnings'] = list(dict.fromkeys(warnings)); return result
    if len(priced) > 1:
        if all(complete for _, complete in priced):
            totals = [_sum(row['amount_usd'] for row in rows) for rows, _ in priced]
            prefix = 'reference_' if kind == 'reference' else ''
            result[prefix+'amount_min_usd'] = float(min(totals))
            result[prefix+'amount_max_usd'] = float(max(totals))
        result['status'] = 'partial' if any(rows or complete for rows, complete in priced) else 'unavailable'
    else:
        breakdown, complete = priced[0]
        amount = float(_sum(row['amount_usd'] for row in breakdown))
        result['breakdown'] = [{**row, 'rate_per_million': float(row['rate_per_million']),
                                'amount_usd': float(row['amount_usd'])} for row in breakdown]
        result['priced_calls'], result['missing_calls'] = int(complete), int(not complete)
        if kind == 'reference':
            result['known_reference_amount_usd'] = amount
            result['reference_amount_usd'] = amount if complete else None
            result['status'] = 'reference' if complete else 'partial' if breakdown else 'unavailable'
        else:
            result['known_amount_usd'] = amount
            result['amount_usd'] = amount if complete else None
            result['status'] = 'complete' if complete else 'partial' if breakdown else 'unavailable'
    result['warnings'] = list(dict.fromkeys(warnings))
    return result


def aggregate_costs(costs, *, basis=None):
    costs = list(costs)
    bases = {cost.get('basis') for cost in costs if cost.get('basis')}
    result = {'status': 'unavailable', 'currency': 'USD', 'amount_usd': None, 'known_amount_usd': 0.,
              'reference_amount_usd': None, 'known_reference_amount_usd': 0.,
              'basis': basis or ('mixed' if len(bases) > 1 else next(iter(bases), 'recorded_snapshot')),
              'warnings': [], 'breakdown': [], 'price_sources': [], 'has_reference': False, 'pricing': None}
    for key in ('total_calls', 'priced_calls', 'missing_calls', 'billable_calls', 'reference_calls', 'local_calls'):
        result[key] = sum(cost.get(key, 0) for cost in costs)
    result['call_count_incomplete'] = any(cost.get('call_count_incomplete') for cost in costs)
    result['known_amount_usd'] = float(_sum(cost.get('known_amount_usd', 0) for cost in costs))
    result['known_reference_amount_usd'] = float(_sum(cost.get('known_reference_amount_usd', 0) for cost in costs))
    result['has_reference'] = any(cost.get('has_reference') for cost in costs)
    api = [cost for cost in costs if cost.get('billable_calls', 0) or cost.get('local_calls', 0)]
    reference = [cost for cost in costs if cost.get('has_reference')]
    unknown = any(cost.get('missing_calls', 0) and not (cost.get('billable_calls', 0) or cost.get('reference_calls', 0) or cost.get('local_calls', 0)) for cost in costs)
    if api and not unknown and all(cost.get('amount_usd') is not None for cost in api): result['amount_usd'] = float(_sum(cost['amount_usd'] for cost in api))
    if reference and all(cost.get('reference_amount_usd') is not None for cost in reference):
        result['reference_amount_usd'] = float(_sum(cost['reference_amount_usd'] for cost in reference))
    for rows, prefix in ((api, ''), (reference, 'reference_')):
        if rows and not unknown:
            bounds = [(cost.get(prefix+'amount_min_usd', cost.get(prefix+'amount_usd')),
                       cost.get(prefix+'amount_max_usd', cost.get(prefix+'amount_usd'))) for cost in rows]
            if all(low is not None and high is not None for low, high in bounds) and any(low != high for low, high in bounds):
                result[prefix+'amount_min_usd'] = float(_sum(low for low, _ in bounds))
                result[prefix+'amount_max_usd'] = float(_sum(high for _, high in bounds))
    result['warnings'] = list(dict.fromkeys(warning for cost in costs for warning in cost.get('warnings', [])))
    for cost in costs:
        for source in cost.get('price_sources', []):
            if source not in result['price_sources']: result['price_sources'].append(deepcopy(source))
        for source in cost.get('calendar_sources', []):
            if source not in result.setdefault('calendar_sources', []): result['calendar_sources'].append(deepcopy(source))
    if result['total_calls'] == 0 and not result['missing_calls']:
        result.update(status='complete', amount_usd=0.)
    elif not result['missing_calls']:
        result['status'] = 'reference' if result['reference_calls'] and not api else 'complete'
    elif result['priced_calls'] or result['known_amount_usd'] or result['known_reference_amount_usd'] or any(cost.get('status') == 'partial' for cost in costs):
        result['status'] = 'partial'
    return result
