"""Read-only analysis of actual returned calls; missing historical data stay missing."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math

from .runlog import LIVE_RUNS, list_runs, run_folder, live_snapshot
from .run_management import record_labels
from .pricing import estimate_quote
from .costing import calculate_call_cost, aggregate_costs
from .telemetry import reasoning_text, response_text, usage_tokens, effective_reasoning

TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')
BATCH_KINDS = {'comparison', 'research_experiment'}
ENVIRONMENT_FIELDS = ('engine_id', 'engine_version', 'evaluator_source', 'evaluator_version',
                      'robot', 'scene_id', 'task_id')


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _elapsed(start, end):
    try: return max(0., (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError): return None


def _events(folder):
    result = []; issues = []
    path = folder / 'events.jsonl'
    if not path.is_file(): return result, ['事件日志不存在。']
    with path.open(encoding='utf-8') as file:
        for index, line in enumerate(file, 1):
            try: result.append(json.loads(line))
            except (ValueError, TypeError): issues.append(f'事件第 {index} 行不完整或无效，未计入。')
    return result, issues


def _totals(calls, *, known_empty=False):
    result = {'calls': len(calls)}
    for field in (*TOKEN_FIELDS, 'inference_s'):
        values = [_number(call.get(field)) for call in calls]
        result['known_' + field] = sum(value for value in values if value is not None)
        result[field] = sum(values) if values and all(value is not None for value in values) else 0 if not values and known_empty else None
    result['usage_complete'] = bool(calls) and all(call.get('input_tokens') is not None and call.get('output_tokens') is not None for call in calls)
    return result


def _identity(metadata, calls):
    recipe = metadata.get('recipe') or {}
    settings = metadata.get('settings') or {}
    models = sorted({str(call['model']) for call in calls if call.get('model')})
    efforts = sorted({str(call['reasoning_effort']) for call in calls if call.get('reasoning_effort') is not None})
    config = recipe.get('agent') or recipe.get('policy') or settings
    model = ' + '.join(models) if models else metadata.get('model') or config.get('model')
    effort = ' + '.join(efforts) if efforts else effective_reasoning(config)
    return model, effort or 'unspecified'


def _environment_identity(result, events, folder, issues):
    """Use recorded evidence; an installed engine is not evidence about an old run."""
    def value(item):
        return str(item).strip() if isinstance(item, (str, int, float)) and not isinstance(item, bool) and str(item).strip() else 'unknown'

    protocol = result.get('protocol') if isinstance(result.get('protocol'), dict) else {}
    spec = protocol.get('spec') if isinstance(protocol.get('spec'), dict) else {}
    contract = protocol.get('contract') if isinstance(protocol.get('contract'), dict) else {}
    backend = result.get('backend') if isinstance(result.get('backend'), dict) else {}
    checkpoint = next((event.get('data') for event in events if event.get('event') == 'initial_state'
                       and isinstance(event.get('data'), dict)), {})
    if not backend and not result.get('engine_id'):
        path = folder / 'checkpoints' / 'initial.json'
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(saved, dict): checkpoint = saved
            except (OSError, ValueError) as exc:
                issues.append(f'初态快照未能提供环境身份：{exc}')
    if not backend:
        backend = checkpoint.get('backend') or contract.get('backend') or {}
        if not isinstance(backend, dict): backend = {}
    metrics = result.get('metrics') if isinstance(result.get('metrics'), dict) else {}
    evaluation = metrics.get('evaluation') or result.get('evaluation')
    if not isinstance(evaluation, dict):
        evaluation = next((event['data'].get('evaluation') for event in reversed(events)
                           if event.get('event') == 'task_evaluation' and isinstance(event.get('data'), dict)
                           and isinstance(event['data'].get('evaluation'), dict)), {})
    source = value(backend.get('evaluator_source') or result.get('evaluator_source') or evaluation.get('source'))
    engine = value(backend.get('engine_id') or result.get('engine_id') or spec.get('engine_id'))
    if engine == 'unknown' and source == 'mujoco_privileged_state': engine = 'mujoco'
    if engine == 'unknown' and (result.get('mujoco_version') or contract.get('mujoco')): engine = 'mujoco'
    legacy_version = (result.get('mujoco_version') or contract.get('mujoco')) if engine == 'mujoco' else None
    version = value(backend.get('engine_version') or result.get('engine_version') or legacy_version)
    evaluator = value(backend.get('evaluator_version') or result.get('evaluator_version') or evaluation.get('version'))
    identity = {**backend, 'engine_id': engine, 'engine_version': version,
                'evaluator_source': source, 'evaluator_version': evaluator}
    return {'backend': identity, 'engine_id': engine, 'engine_version': version,
            'evaluator_source': source, 'evaluator_version': evaluator,
            'robot': value(result.get('robot') or checkpoint.get('model') or spec.get('robot')),
            'scene_id': value(result.get('scene_id') or checkpoint.get('scene_id') or spec.get('scene_id')),
            'task_id': value(result.get('task_id') or evaluation.get('task_id') or checkpoint.get('task_id') or spec.get('task_id'))}


def analyze_run(run_id, *, _ancestors=()):
    if run_id in _ancestors: raise ValueError('实验包含循环子运行引用。')
    folder = run_folder(run_id)
    events, issues = _events(folder)
    metadata = next((event.get('data') or {} for event in events if event.get('event') == 'start'), {})
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    live = LIVE_RUNS.get(run_id)
    if live is not None and not live._finished: metadata = {**metadata, **live_snapshot(live)}
    result = {**metadata, **summary}
    if not summary and not (live is not None and not live._finished): result['status'] = 'interrupted'
    started_at = result.get('started_at') or next((event.get('timestamp') for event in events if event.get('event') == 'start'), None)
    calls = {}; starts = {}; round_number = 0
    for event in events:
        data = event.get('data') or {}; kind = event.get('event')
        if kind == 'observation': round_number = data.get('round', data.get('step', round_number))
        if kind == 'inference_start': starts[data['call_id']] = dict(data)
        if kind == 'inference': calls[data['call_id']] = dict(data)
    now = datetime.now(timezone.utc).isoformat()
    for call_id, call in starts.items():
        if call_id not in calls:
            call.update(status='running' if result.get('status') == 'running' else 'interrupted', inference_s=None,
                        current_wait_s=_elapsed(call.get('started_at'), now) if result.get('status') == 'running' else None)
            calls[call_id] = call
    modern = bool(starts or calls)
    if not modern:
        # Old policy records may have a second copy with round+latency; prefer that copy.
        outputs = []; context_round = 0
        actions = {event['data'].get('round'): event['data'] for event in events if event.get('event') == 'action'}
        for event in events:
            if event.get('event') == 'observation': context_round = event['data'].get('round', event['data'].get('step', context_round))
            if event.get('event') == 'model_output': outputs.append((event, context_round))
        detailed_rounds = {item[0]['data'].get('round', item[1]) for item in outputs if isinstance(item[0].get('data'), dict) and 'latency_s' in item[0]['data']}
        outputs = [item for item in outputs if 'latency_s' in item[0].get('data', {}) or item[0].get('data', {}).get('round', item[1]) not in detailed_rounds]
        for index, (event, observed_round) in enumerate(outputs, 1):
            data = event.get('data') or {}
            raw = data.get('response') if isinstance(data.get('response'), dict) else data
            number = data.get('round', observed_round or index)
            action = actions.get(number, {})
            usage = raw.get('usage') if raw.get('usage') is not None else action.get('usage')
            calls[str(index)] = {'call_id': 'legacy-' + str(index), 'round': number, 'started_at': None,
                'response_at': event.get('timestamp'),
                'phase': 'legacy_provider_output', 'provider': data.get('provider', result.get('provider')),
                'model': data.get('model', result.get('model')), 'reasoning_effort': None,
                'status': 'returned', 'public_text': response_text(data), 'reasoning': reasoning_text(data),
                'usage': usage, **usage_tokens(usage), 'inference_s': _number(data.get('latency_s', action.get('latency_s'))),
                'error': None, 'response_returned': True, 'legacy': True}
    ordered = sorted(calls.values(), key=lambda call: (call.get('round') or 0, call.get('started_at') or '', call['call_id']))
    recipe = result.get('recipe') or {}
    fallback_config = recipe.get('agent') or recipe.get('policy') or result.get('settings') or {}
    for call in ordered:
        for field in (*TOKEN_FIELDS, 'inference_s'): call[field] = _number(call.get(field))
        returned_tokens = usage_tokens(call.get('usage'))
        for field in TOKEN_FIELDS:
            if call[field] is None: call[field] = returned_tokens[field]
        call.setdefault('reasoning', {'availability': 'unavailable', 'text': None, 'source': None})
        saved_price = call.get('pricing')
        # Recalculate incomplete derived amounts from the original usage and
        # quote. Calculation fixes must also reach history without rewriting it.
        recorded_cost = call.get('cost')
        if not isinstance(recorded_cost, dict) or recorded_cost.get('status') in ('partial', 'unavailable'):
            quote = saved_price if isinstance(saved_price, dict) and saved_price.get('status') in ('ready', 'stale') else estimate_quote(
                {**fallback_config, 'provider': call.get('provider') or result.get('provider') or fallback_config.get('provider', fallback_config.get('kind')),
                 'model': call.get('model') or result.get('model') or fallback_config.get('model')})
            resolved = calculate_call_cost(call, quote)
            call['cost'] = resolved
            if isinstance(recorded_cost, dict) and resolved != recorded_cost:
                call['recorded_cost'] = recorded_cost
                if quote.get('basis') == 'current_price_estimate':
                    resolved['warnings'].append('调用时未获得可用报价；此处按当前官方价格补估，原费用记录仍保留在 recorded_cost 和原始日志中。')
                elif (resolved.get('calendar_resolution') or {}).get('known'):
                    call['calendar_resolution'] = {**resolved['calendar_resolution'], 'recalculated_from_recorded_price': True,
                        'original_log_unchanged': True}
                    resolved['warnings'].append('按原调用价格与原始用量补齐官方节假日条件重新估算；旧费用保留在 recorded_cost，未改写原日志。')
                else:
                    resolved['warnings'].append('按原调用价格快照与原始用量重新计算；旧费用保留在 recorded_cost，未改写原日志。')
    rows = result.get('rows') or [event['data'] for event in events if event.get('event') == 'trial_result']
    children = []; child_analyses = []; seen = set()
    for row in rows:
        child = {**row, 'task': result.get('task')}
        child_id = row.get('run_id')
        if child_id and child_id not in seen:
            seen.add(child_id)
            try:
                analysis = analyze_run(child_id, _ancestors=(*_ancestors, run_id))
                child_analyses.append(analysis)
                child['analysis_status'] = analysis['status']
                child.update({key: analysis[key] for key in ('display_name', 'deleted_at')})
            except (ValueError, FileNotFoundError, KeyError) as exc: child['analysis_error'] = str(exc)
        children.append(child)
    known_empty = result.get('telemetry_schema_version') == 'embodied-calls-v1' and not result.get('external_telemetry')
    totals = _totals(ordered, known_empty=known_empty)
    if result.get('external_telemetry') and not ordered: totals['calls'] = None
    if result.get('kind') in BATCH_KINDS:
        child_totals = [child['total'] for child in child_analyses]
        totals = {'calls': sum(child['calls'] for child in child_totals), 'usage_complete': bool(child_totals) and all(child['usage_complete'] for child in child_totals)}
        for field in (*TOKEN_FIELDS, 'inference_s'):
            values = [child[field] for child in child_totals]
            totals[field] = sum(values) if values and all(value is not None for value in values) and not any(c.get('analysis_error') for c in children) else None
            totals['known_' + field] = sum(child.get('known_' + field, 0) for child in child_totals)
        # A recorded batch stopped before launching any child has a measured
        # zero-call clock. Historical/absent child evidence remains unknown.
        if (known_empty and not ordered and not child_totals and not issues
                and (result.get('timing') or {}).get('inference_calls') == 0
                and not any(child.get('run_id') or child.get('metrics') or child.get('analysis_error') for child in children)):
            totals = _totals([], known_empty=True)
    model, effort = _identity(result, ordered)
    if child_analyses:
        model = ' + '.join(sorted({child['model'] for child in child_analyses if child.get('model')})) or model
        effort = ' + '.join(sorted({child['reasoning'] for child in child_analyses}))
    elapsed = _number(result.get('elapsed_s', result.get('wall_time_s', (result.get('metrics') or {}).get('wall_time_s'))))
    ended_at = result.get('ended_at')
    if elapsed is None and ended_at: elapsed = _elapsed(started_at, ended_at)
    costs = [child['cost'] for child in child_analyses] if result.get('kind') in BATCH_KINDS else [call['cost'] for call in ordered]
    if result.get('external_telemetry') and not ordered:
        costs.append({'status':'unavailable','total_calls':0,'missing_calls':1,'call_count_incomplete':True,
                      'warnings':['外部策略未上报模型用量、推理耗时与费用；缺失值不按零计算。']})
    for child in children if result.get('kind') in BATCH_KINDS else []:
        if child.get('analysis_error'):
            costs.append({'status': 'unavailable', 'total_calls': 0, 'missing_calls': 1, 'call_count_incomplete': True,
                          'warnings': ['部分子实验无法读取，费用仅包含已读取的子实验；无法给出完整总额或上界。']})
    expected_calls = (result.get('timing') or {}).get('inference_calls', (result.get('metrics') or {}).get('inference_count', 0))
    incomplete_events = bool(issues) or (result.get('kind') not in BATCH_KINDS and isinstance(expected_calls, int) and expected_calls > len(ordered))
    if incomplete_events:
        saved_cost = result.get('cost')
        if isinstance(saved_cost, dict) and saved_cost.get('total_calls', 0) > 0:
            cost = deepcopy(saved_cost)
            cost.setdefault('warnings', []).append('部分调用事件缺失或损坏；总费用保留已归档的费用快照，逐轮明细可能不完整。')
        else:
            costs.append({'status': 'unavailable', 'total_calls': 0, 'missing_calls': 1, 'call_count_incomplete': True,
                          'warnings': ['调用事件缺失或损坏，无法确认完整费用；未将缺失调用按零计算。']})
            cost = aggregate_costs(costs)
    else:
        cost = aggregate_costs(costs)
    environment = _environment_identity(result, events, folder, issues)
    from .experiment_costs import attribution
    cost_attribution = attribution(run_id, ordered, cost, child_analyses if result.get('kind') in BATCH_KINDS else ())
    return {'run_id': run_id, **record_labels({**result, 'run_id': run_id, 'started_at': started_at}),
        **environment,
        'kind': result.get('kind'), 'status': result.get('status', 'interrupted'),
        'message': result.get('message'), 'task': result.get('task'), 'model': model, 'reasoning': effort,
        'started_at': started_at, 'ended_at': ended_at, 'elapsed_s': elapsed,
        'metrics': result.get('metrics'), 'video': (result.get('artifacts') or {}).get('video'),
        'benchmark': result.get('benchmark'), 'record_origin': result.get('record_origin'),
        'native_videos': (result.get('artifacts') or {}).get('native_videos', []),
        'completion_verifications': [{'timestamp': event.get('timestamp'), **event.get('data', {})}
            for event in events if event.get('event') == 'completion_verification'],
        'recovery_assisted':bool(result.get('recovery_assisted') or any(event.get('event')=='recoverable_error' for event in events)),
        'recovery_attempts':sum(event.get('event')=='recoverable_error' for event in events),
        'total': totals, 'rounds': ordered, 'children': children, 'trials': rows, 'cost': cost,
        'cost_attribution': cost_attribution,
        'issues': issues, 'aggregation_scope': 'children_once' if result.get('kind') in BATCH_KINDS else 'actual_provider_calls',
        'inference_semantics': '端到端服务调用耗时，含网络与排队；并行调用逐请求累计，可能超过墙钟时间。',
        'usage_complete_semantics': '每次调用均返回有效 input_tokens 与 output_tokens；reasoning_tokens 可独立缺失。'}


def analyze_experiments(*, task=None, model=None, reasoning=None, kind=None, limit=100):
    records = list_runs(limit=None)
    runs = []; issues = []
    for item in records:
        try: record = analyze_run(item['run_id'])
        except (ValueError, FileNotFoundError, KeyError) as exc:
            issues.append({'run_id': item.get('run_id'), 'error': str(exc)}); continue
        if any(value is not None and record.get(field) != value for field, value in (('task', task), ('model', model), ('reasoning', reasoning), ('kind', kind))): continue
        record.pop('rounds', None); runs.append(record)
    groups = defaultdict(list); excluded_active = 0
    for run in runs:
        if run['status'] == 'running': excluded_active += 1; continue
        if run['kind'] not in BATCH_KINDS and run['kind'] != 'benchmark':
            context = tuple(run[field] for field in ENVIRONMENT_FIELDS)
            groups[(*context, run['model'], run['reasoning'], run['kind'], run['task'])].append(run)
    grouped = []
    for identity, members in groups.items():
        model_name, effort, run_kind, run_task = identity[-4:]
        group = {**dict(zip(ENVIRONMENT_FIELDS, identity[:-4])),
                 'backend': {field: members[0][field] for field in ENVIRONMENT_FIELDS[:4]},
                 'model': model_name, 'reasoning': effort, 'kind': run_kind, 'task': run_task, 'runs': len(members),
                 'succeeded': sum(bool((run.get('metrics') or {}).get('success')) and run['status'] not in ('error', 'stopped', 'cancelled', 'interrupted', 'running') for run in members),
                 'cancelled': sum(run['status'] in ('stopped', 'cancelled') for run in members),
                 'failed': sum(run['status'] not in ('running', 'stopped', 'cancelled') and (run['status'] in ('error', 'interrupted') or not bool((run.get('metrics') or {}).get('success'))) for run in members)}
        for field, sample, missing in (('total_tokens', 'token_samples', 'missing_token_runs'), ('output_tokens', 'output_samples', 'missing_output_runs'), ('inference_s', 'inference_samples', 'missing_inference_runs')):
            values = [run['total'][field] for run in members if run['total'][field] is not None]
            group[field] = sum(values) if values else None
            group['mean_' + field] = sum(values) / len(values) if values else None
            group[sample] = len(values); group[missing] = len(members) - len(values)
        group['success_rate'] = group['succeeded']/len(members)
        group['recovery_runs'] = sum(run.get('recovery_assisted', False) for run in members)
        group['known_cost_usd'] = sum((run.get('cost_attribution') or {}).get('known_total_usd', 0) for run in members)
        group['success_rate_semantics'] = '成功次数 / 已结束尝试数；取消、错误和中断均保留在分母。'
        grouped.append(group)
    return {'runs': runs[:limit], 'groups': grouped, 'matched_runs': len(runs), 'returned_runs': min(limit, len(runs)),
            'aggregation_scope': 'leaf_runs_only', 'excluded_active_runs': excluded_active, 'issues': issues,
            'grouping_fields': [*ENVIRONMENT_FIELDS, 'model', 'reasoning', 'kind', 'task'],
            'message': '仅在相同后端及版本、评价来源及版本、机器人、场景、评价任务内按模型与推理强度聚合；不同种子作为重复样本。历史环境缺失保留 unknown。聚合扫描所有历史记录，列表按 limit 截取；父批次不重复计费，缺失值不按零计算。'}
