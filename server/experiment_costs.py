"""Explicit non-token costs and evidence-based attribution, separate from API estimates."""
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from .runlog import run_folder
from .costing import aggregate_costs

LOCK = RLock()


class Expense(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    category: Literal['compute', 'storage', 'tool', 'network', 'human', 'other']
    name: str = Field(min_length=1, max_length=200)
    quantity: float = Field(gt=0, le=1e12)
    unit: str = Field(min_length=1, max_length=40)
    unit_price_usd: float = Field(ge=0, le=1e9)
    source: str = Field(min_length=1, max_length=1000)
    note: str = Field(default='', max_length=2000)


def expenses(run_id):
    path = run_folder(run_id)/'expenses.jsonl'
    entries = {}
    with LOCK:
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                if record['event'] == 'add': entries[record['id']] = record
                elif record['event'] == 'void' and record['id'] in entries: entries[record['id']]['voided'] = True
    return list(entries.values())


def append_expense(run_id, expense):
    value = expense.model_dump()
    value['amount_usd'] = float(Decimal(str(expense.quantity))*Decimal(str(expense.unit_price_usd)))
    record = {**value, 'id': uuid4().hex, 'event': 'add', 'created_at': datetime.now(timezone.utc).isoformat()}
    with LOCK, (run_folder(run_id)/'expenses.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    return record


def void_expense(run_id, expense_id):
    with LOCK:
        row = next((item for item in expenses(run_id) if item['id'] == expense_id), None)
        if row is None: raise ValueError('费用条目不存在。')
        if not row.get('voided'):
            with (run_folder(run_id)/'expenses.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps({'event': 'void', 'id': expense_id, 'created_at': datetime.now(timezone.utc).isoformat()})+'\n')
    return {'ok': True}


def attribution(run_id, calls, api_cost, children=()):
    groups = defaultdict(list)
    for call in calls:
        role = 'recovery' if call.get('attempt', 0) > 0 or call.get('phase') == 'repair_diagnosis' else call.get('phase') or 'unknown'
        groups[(role, call.get('provider'), call.get('model'))].append(call)
    rows = []
    for (phase, provider, model), members in groups.items():
        rows.append({'phase': phase, 'provider': provider, 'model': model, 'calls': len(members),
                     'failed_calls': sum(c.get('status') in ('error', 'cancelled', 'interrupted') for c in members),
                     'known_inference_s': sum(c.get('inference_s') or 0 for c in members),
                     'known_input_tokens': sum(c.get('input_tokens') or 0 for c in members),
                     'known_output_tokens': sum(c.get('output_tokens') or 0 for c in members),
                     'cost': aggregate_costs([c['cost'] for c in members if isinstance(c.get('cost'), dict)])})
    ledger = expenses(run_id)
    totals = defaultdict(Decimal)
    for entry in ledger:
        if not entry.get('voided'): totals[entry['category']] += Decimal(str(entry['amount_usd']))
    for child in children:
        attribution = child.get('cost_attribution') or {}
        rows.extend({**row, 'child_run_id': child['run_id']} for row in attribution.get('sources', []))
        for category, amount in attribution.get('non_model_by_category', {}).items():
            totals[category] += Decimal(str(amount))
    extra = float(sum(totals.values(), Decimal(0)))
    known_api = api_cost.get('known_amount_usd') or api_cost.get('amount_usd') or 0.
    return {'currency': 'USD', 'sources': rows, 'expenses': ledger,
            'non_model_by_category': {k: float(v) for k, v in totals.items()}, 'non_model_amount_usd': extra,
            'known_total_usd': float(Decimal(str(known_api))+Decimal(str(extra))),
            'coverage': 'recorded_items_only',
            'note': '已记录金额包含 API 已知估算与人工登记的非模型成本；未登记的计算、存储、工具及人工开销不视为零。订阅等价参考不计入。批量实验的明细请打开子实验；同一账单勿重复登记。'}
