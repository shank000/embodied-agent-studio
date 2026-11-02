"""Explicit non-token costs and evidence-based attribution, separate from API estimates."""
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from .runlog import run_folder
from .costing import aggregate_costs, account_billing_scopes

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
    evidence_type: Literal['price_estimate', 'actual_invoice'] = 'price_estimate'
    source_invoice_id: str | None = Field(default=None, min_length=1, max_length=200)
    allocation_key: str | None = Field(default=None, min_length=1, max_length=200)
    allocation_rule: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode='after')
    def invoice_evidence(self):
        if self.evidence_type == 'actual_invoice' and not self.source_invoice_id:
            raise ValueError('实际账单费用须提供账单编号，便于对账和避免重复登记。')
        if self.allocation_key and not (self.source_invoice_id and self.allocation_rule):
            raise ValueError('资源分摊须提供账单编号和分摊依据。')
        return self


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
    with LOCK:
        if expense.source_invoice_id:
            existing = [row for row in expenses(run_id) if not row.get('voided') and
                        row.get('source') == expense.source and row.get('source_invoice_id') == expense.source_invoice_id]
            if any(bool(row.get('allocation_key')) != bool(expense.allocation_key) for row in existing):
                raise ValueError('同一账单不能同时登记全额与分摊项；请先作废原条目，再选择一种记账范围。')
            if any(row.get('allocation_key') == expense.allocation_key for row in existing):
                raise ValueError('该账单及分摊条目已登记；如需更正，请先作废原条目。')
        with (run_folder(run_id)/'expenses.jsonl').open('a', encoding='utf-8') as file:
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
    accounted = account_billing_scopes([call['cost'] for call in calls if isinstance(call.get('cost'), dict)])
    accounting = {cost['call_id']: cost for cost in accounted if cost.get('call_id')}
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
                     'cost': aggregate_costs([{**accounting.get(c.get('call_id'), c['cost']),
                          'billing_scope': 'self', 'parent_call_id': None}
                          for c in members if isinstance(c.get('cost'), dict)])})
    ledger = expenses(run_id)
    totals = defaultdict(Decimal)
    evidence = defaultdict(Decimal)
    seen_expenses = {}
    ledger_warnings = []
    all_entries = list(ledger)
    for child in children:
        attribution = child.get('cost_attribution') or {}
        rows.extend({**row, 'child_run_id': child['run_id']} for row in attribution.get('sources', []))
        child_entries = attribution.get('all_expenses')
        if child_entries is not None:
            all_entries.extend(child_entries)
        else:
            for category, amount in attribution.get('non_model_by_category', {}).items():
                totals[category] += Decimal(str(amount))
            for kind, amount in attribution.get('non_model_evidence_totals', {}).items(): evidence[kind] += Decimal(str(amount))
    invoices = defaultdict(list)
    for entry in all_entries:
        if not entry.get('voided') and entry.get('source_invoice_id'):
            invoices[(entry.get('source'), entry['source_invoice_id'])].append(entry)
    conflicts = {key: entries for key, entries in invoices.items()
                 if {bool(entry.get('allocation_key')) for entry in entries} == {True, False}}
    invoice_conflicts = [{'source': source, 'source_invoice_id': invoice,
                         'expense_ids': list(dict.fromkeys(entry['id'] for entry in entries)),
                         'reason': 'invoice_total_and_allocations_overlap'} for (source, invoice), entries in conflicts.items()]
    if conflicts:
        ledger_warnings.append('同一账单同时出现全额和分摊记录，范围冲突；该账单所有项目暂不计入已知合计，金额保持未知。请核对后作废重复范围。')
    if any(entry.get('allocation_key') for entries in invoices.values() for entry in entries):
        ledger_warnings.append('分摊项按声明金额记录；尚未提供账单总额上限，未验证各分摊之和是否超出原账单。')
    def include_entry(entry):
        if entry.get('voided'): return
        if (entry.get('source'), entry.get('source_invoice_id')) in conflicts: return
        identity = ('invoice', entry.get('source'), entry['source_invoice_id'], entry.get('allocation_key')) if entry.get('source_invoice_id') else ('entry', entry['id'])
        if identity in seen_expenses:
            ledger_warnings.append('相同账单分摊条目出现多次，已按来源、账单编号和分摊标识合计一次；实际账单优先于估算。')
            previous = seen_expenses[identity]
            if entry.get('evidence_type') != 'actual_invoice' or previous.get('evidence_type') == 'actual_invoice': return
            previous_amount = Decimal(str(previous['amount_usd']))
            totals[previous['category']] -= previous_amount
            evidence[previous.get('evidence_type', 'price_estimate')] -= previous_amount
        seen_expenses[identity] = entry
        amount = Decimal(str(entry['amount_usd']))
        totals[entry['category']] += amount
        evidence[entry.get('evidence_type', 'price_estimate')] += amount
    for entry in all_entries:
        include_entry(entry)
    extra = float(sum(totals.values(), Decimal(0)))
    known_api = api_cost.get('known_amount_usd') or api_cost.get('amount_usd') or 0.
    routing = []
    for call in calls:
        route = call.get('routing') if isinstance(call.get('routing'), dict) else {}
        routing.append({key: call.get(key) for key in ('call_id', 'trace_id', 'decision_id', 'parent_call_id',
            'round', 'phase', 'provider', 'requested_model', 'model', 'status', 'inference_s', 'billing_scope', 'input_tokens', 'output_tokens')} |
            {'routing': route, 'recovery_id': call.get('recovery_id'), 'attempt': call.get('attempt'),
             'excluded_from_total': bool(accounting.get(call.get('call_id'), {}).get('excluded_from_total'))})
    for child in children:
        routing.extend({**row, 'child_run_id': child['run_id']} for row in (child.get('cost_attribution') or {}).get('routing', []))
    coverage = {'calls': len(routing), 'routing_reported_calls': sum(c['routing'].get('availability') == 'returned' for c in routing),
                'known_usage_calls': sum(c.get('input_tokens') is not None and c.get('output_tokens') is not None for c in routing),
                'priced_calls': api_cost.get('priced_calls', 0), 'missing_price_or_usage_calls': api_cost.get('missing_calls', 0),
                'excluded_inclusive_calls': api_cost.get('excluded_calls', 0),
                'conflicting_invoices': len(invoice_conflicts),
                'unmetered_categories': [kind for kind in ('compute', 'storage', 'tool', 'network', 'human') if kind not in totals]}
    return {'currency': 'USD', 'sources': rows, 'expenses': ledger, 'all_expenses': all_entries,
            'invoice_conflicts': invoice_conflicts,
            'expense_accounting_status': 'conflict' if invoice_conflicts else 'recorded_items_only',
            'routing': routing, 'coverage_details': coverage, 'warnings': list(dict.fromkeys(ledger_warnings)),
            'evidence_totals': api_cost.get('evidence_totals', {}),
            'non_model_evidence_totals': {key: float(amount) for key, amount in evidence.items()},
            'non_model_by_category': {k: float(v) for k, v in totals.items()}, 'non_model_amount_usd': extra,
            'known_total_usd': float(Decimal(str(known_api))+Decimal(str(extra))),
            'coverage': 'recorded_items_only',
            'note': '已记录金额包含 API 已知估算与人工登记的非模型成本；未登记的计算、存储、工具及人工开销不视为零。订阅等价参考不计入。批量实验的明细请打开子实验；同一账单勿重复登记。'}
