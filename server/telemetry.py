"""One trace per actual provider invocation, including failed and cancelled calls."""
import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import math
import time
from urllib.parse import urlsplit
from uuid import uuid4

CURRENT_CALL = ContextVar('robot_inference_call', default=None)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def usage_tokens(usage):
    usage = usage if isinstance(usage, dict) else {}
    def number(*names):
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0: return value
        return None
    input_tokens = number('input_tokens', 'prompt_tokens')
    output_tokens = number('output_tokens', 'completion_tokens')
    reasoning_tokens = number('reasoning_tokens')
    for name in ('output_tokens_details', 'completion_tokens_details'):
        if reasoning_tokens is None:
            details = usage.get(name)
            value = details.get('reasoning_tokens') if isinstance(details, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0: reasoning_tokens = value
    total = number('total_tokens')
    if total is None and input_tokens is not None and output_tokens is not None: total = input_tokens + output_tokens
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'reasoning_tokens': reasoning_tokens, 'total_tokens': total}


def reasoning_text(value):
    """Extract only returned, readable fields; never request encrypted/private state."""
    segments = []
    def visit(item, source='response', explicit=False):
        if isinstance(item, dict):
            if item.get('type') == 'summary_text' and isinstance(item.get('text'), str):
                segments.append((source, item['text']))
            for key, content in item.items():
                if key in ('reasoning_content', 'thinking', 'returned_reasoning') and isinstance(content, str) and content:
                    segments.append((source + '.' + key, content))
                elif key == 'reasoning' and isinstance(content, str) and content:
                    segments.append((source + '.' + key, content))
                elif explicit and key in ('summary', 'text') and isinstance(content, str) and content:
                    segments.append((source + '.' + key, content))
                elif key not in ('encrypted_content', 'signature', 'image_base64', 'cameras'):
                    if key in ('summary', 'reasoning', 'returned_reasoning') and isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get('text'), str) and (explicit or key != 'summary' or part.get('type') == 'summary_text'): segments.append((source + '.' + key, part['text']))
                            else: visit(part, source + '.' + key, explicit or key in ('reasoning', 'returned_reasoning'))
                    else: visit(content, source + '.' + key, explicit or key in ('reasoning', 'returned_reasoning'))
        elif isinstance(item, list):
            for part in item: visit(part, source, explicit)
    visit(value)
    unique = list(dict.fromkeys(segments))
    return {'availability': 'returned' if unique else 'unavailable',
            'text': '\n\n'.join(text for _, text in unique) if unique else None,
            'source': ', '.join(source for source, _ in unique) if unique else None}


def response_text(data):
    if not isinstance(data, dict): return str(data) if data is not None else None
    for key in ('content', 'text', 'summary', 'output'):
        value = data.get(key)
        if isinstance(value, str) and value: return value
        if value:
            return json.dumps(value, ensure_ascii=False)
    choices = data.get('choices') or []
    values = [choice.get('content') or (choice.get('message') or {}).get('content') for choice in choices if isinstance(choice, dict)]
    if any(values): return '\n'.join(value for value in values if isinstance(value, str))
    if isinstance(data.get('response'), dict): return response_text(data['response'])
    if data.get('actions'): return json.dumps(data['actions'], ensure_ascii=False)
    if data.get('answers'): return json.dumps(data['answers'], ensure_ascii=False)
    return None


def note_response(data):
    record = CURRENT_CALL.get()
    if record is None or not isinstance(data, dict): return
    record['response_returned'] = True
    record['public_text'] = response_text(data)
    record['reasoning'] = reasoning_text(data)
    response = data.get('response') if isinstance(data.get('response'), dict) else data
    usage = data.get('usage') if data.get('usage') is not None else response.get('usage')
    if isinstance(usage, dict): record['usage'] = usage
    if response.get('model'): record['model'] = response['model']


def effective_reasoning(settings):
    provider = settings.get('provider')
    if provider == 'anthropic' and not settings.get('reasoning_native'): return 'not_requested'
    if provider == 'compatible':
        deepseek = urlsplit(settings.get('base_url', '')).hostname == 'api.deepseek.com'
        if not deepseek and not settings.get('reasoning_native'): return 'provider_default'
        if deepseek and settings.get('deepseek_thinking') == 'disabled': return 'off'
    return settings.get('reasoning') or 'provider_default'


def track_inference(phase):
    def decorate(function):
        @wraps(function)
        async def traced(config, *args, **kwargs):
            from .runlog import CURRENT_LOG
            from .pricing import quote_for_call
            from .costing import calculate_call_cost
            log = CURRENT_LOG.get()
            if log is None: return await function(config, *args, **kwargs)
            settings = config.model_dump(exclude={'api_key'}) if hasattr(config, 'model_dump') else dict(config)
            pricing = await quote_for_call(settings)
            effort = effective_reasoning(settings)
            current_phase = 'repair_diagnosis' if kwargs.get('schema_name') == 'workbench_repair' else 'planning' if getattr(kwargs.get('response_type'), '__name__', '') == 'PlannerOutput' else phase
            record = {'call_id': uuid4().hex, 'round': log.current_round, 'phase': current_phase,
                      'provider': settings.get('provider', settings.get('kind')), 'model': settings.get('model'),
                      'reasoning_effort': effort, 'configured_reasoning': settings.get('reasoning'), 'started_at': utc_now(),
                      'status': 'running', 'public_text': None, 'response_returned': False, 'usage': None,
                      'pricing': pricing,
                      'reasoning': {'availability': 'unavailable', 'text': None, 'source': None}, 'error': None}
            started = time.monotonic()
            token = CURRENT_CALL.set(record)
            log.begin_inference(record, started)
            try:
                result = await function(config, *args, **kwargs)
                record['status'] = 'complete'
                return result
            except BaseException as exc:
                record['status'] = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'error'
                record['error'] = str(getattr(exc, 'detail', exc)) or type(exc).__name__
                if isinstance(getattr(exc, 'usage', None), dict): record['usage'] = exc.usage
                body = getattr(exc, 'body', None)
                if isinstance(body, dict) and not record['response_returned']: note_response(body)
                raise
            finally:
                record.update(inference_s=time.monotonic() - started, ended_at=utc_now())
                try:
                    record.update(usage_tokens(record['usage']))
                    record['cost'] = calculate_call_cost(record, pricing)
                    log.end_inference(record)
                finally: CURRENT_CALL.reset(token)
        return traced
    return decorate


def controller_timing(state, log=None):
    if log is not None and state.get('run_id') == log.id:
        return {**state, **log.timing_snapshot()}
    return {**state, 'started_at': state.get('started_at'), 'elapsed_s': 0.,
            'timing': {'elapsed_wall_s': 0., 'inference_elapsed_s': 0., 'current_inference_s': 0., 'active_calls': 0}}


def batch_timing(state, log, children):
    result = controller_timing(state, log)
    # A reusable controller may still expose the previous trial during start().
    # Callers bind only a newly created task/run to its row; never infer ownership
    # merely from the presence of a running row or an active controller.
    bound = {row['run_id']: row for row in state.get('rows', []) if row.get('run_id')}
    pending = {child['run_id']: child for child in children if child.get('run_id') in bound}
    timing = result['timing']
    for run_id, row in bound.items():
        child_timing = pending.get(run_id, {}).get('timing', {})
        metrics = row.get('metrics') or {}
        inference_s = metrics.get('inference_s')
        inference_calls = metrics.get('inference_count')
        # Keep the finished child's clock during the short interval before its
        # parent copies the final metrics into the row.
        timing['inference_elapsed_s'] += inference_s if inference_s is not None else child_timing.get('inference_elapsed_s', 0.)
        timing['inference_calls'] = timing.get('inference_calls', 0) + (inference_calls
            if inference_calls is not None else child_timing.get('inference_calls', 0))
        timing['current_inference_s'] = max(timing['current_inference_s'], child_timing.get('current_inference_s', 0.))
        timing['active_calls'] += child_timing.get('active_calls', 0)
    return result
