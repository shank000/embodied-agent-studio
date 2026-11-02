import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.analysis import analyze_run
from server.costing import aggregate_costs, calculate_call_cost
from server.experiment_costs import Expense, append_expense, attribution, void_expense
from server.runlog import CURRENT_LOG, RunLog, model_output
from server.telemetry import normalize_routing, routing_request_context, track_inference


QUOTE = {'provider': 'openai', 'billing_kind': 'api', 'status': 'ready', 'currency': 'USD',
         'input_per_million': 2, 'cached_input_per_million': 1, 'output_per_million': 8}


def price(identifier, parent=None, scope='self', usage=True):
    return calculate_call_cost({'call_id': identifier, 'parent_call_id': parent,
        'trace_id': 'experiment', 'billing_scope': scope,
        'usage': {'input_tokens': 1000, 'cached_input_tokens': 0, 'output_tokens': 100} if usage else None}, QUOTE)


class BillingScopes(unittest.TestCase):
    def test_inclusive_parent_counts_each_call_but_bills_descendants_once(self):
        parent, child, grandchild = price('gateway', scope='inclusive'), price('model', 'gateway'), price('tool', 'model')
        total = aggregate_costs([child, grandchild, parent])
        self.assertEqual(total['amount_usd'], .0028)
        self.assertEqual(total['total_calls'], 3)
        self.assertEqual(total['excluded_calls'], 2)
        self.assertEqual(total['evidence_totals']['price_estimate']['known_amount_usd'], .0028)
        self.assertEqual(aggregate_costs([aggregate_costs([child, grandchild]), aggregate_costs([parent])]), total)

    def test_self_bills_add_and_missing_inclusive_parent_is_unknown(self):
        total = aggregate_costs([price('a'), price('b', 'a')])
        self.assertEqual(total['amount_usd'], .0056)
        orphan = aggregate_costs([price('b', 'absent', 'included')])
        self.assertIsNone(orphan['amount_usd'])
        self.assertEqual(orphan['missing_calls'], 1)
        self.assertEqual(orphan['known_amount_usd'], 0)
        missing = aggregate_costs([price('a', scope='inclusive', usage=False), price('b', 'a', 'included')])
        self.assertIsNone(missing['amount_usd'])
        self.assertEqual(missing['missing_calls'], 1)

    def test_same_call_replay_and_cycles_cannot_inflate_cost(self):
        a = price('a')
        self.assertEqual(aggregate_costs([a, deepcopy(a)])['amount_usd'], .0028)
        cycle = aggregate_costs([price('a', 'b', 'inclusive'), price('b', 'a', 'included')])
        self.assertIsNone(cycle['amount_usd'])
        self.assertEqual(cycle['missing_calls'], 2)
        self.assertTrue(any('循环' in warning for warning in cycle['warnings']))

    def test_references_are_not_actual_spend(self):
        reference = calculate_call_cost({'usage': {'input_tokens': 1000, 'cached_input_tokens': 0, 'output_tokens': 100}},
                                        {**QUOTE, 'billing_kind': 'reference'})
        self.assertEqual(reference['evidence_type'], 'subscription_reference')
        self.assertEqual(reference['known_amount_usd'], 0)
        self.assertEqual(reference['known_reference_amount_usd'], .0028)


class TraceAndLedger(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.patch = patch('server.runlog.RUNS', Path(self.tmp.name)); self.patch.start(); self.addCleanup(self.patch.stop)

    async def test_nested_parallel_calls_keep_parent_and_gateway_metadata(self):
        contexts = []
        @track_inference('tool')
        async def tool(settings):
            contexts.append(routing_request_context())
            await asyncio.sleep(0)
            model_output({'model': 'worker', 'usage': {'input_tokens': 1000, 'cached_input_tokens': 0, 'output_tokens': 100}})
        @track_inference('agent')
        async def gateway(settings):
            contexts.append(routing_request_context())
            await asyncio.gather(tool(settings), tool(settings))
            model_output({'model': 'router', 'content': 'next action',
                'routing': {'resolved_model': 'router', 'selected_capability': 'vision_then_plan',
                            'router_policy_version': 'v1', 'selection_probability': .4, 'reason_code': 'occlusion',
                            'api_key': 'must-not-copy'}, 'billing': {'scope': 'inclusive'},
                'usage': {'input_tokens': 1000, 'cached_input_tokens': 0, 'output_tokens': 100}})
        log = RunLog('agent', {'task': 'contract-test'}); log.current_round = 3
        token = CURRENT_LOG.set(log)
        try:
            with patch('server.pricing.quote_for_call', return_value=QUOTE):
                await gateway({'provider': 'compatible', 'model': 'router', 'base_url': 'http://fixture.invalid/v1'})
        finally: CURRENT_LOG.reset(token)
        log.finish({'status': 'limit'})
        before = log.path.read_bytes()
        analysis = analyze_run(log.id)
        self.assertEqual(log.path.read_bytes(), before)
        self.assertEqual(analysis['cost']['amount_usd'], .0028)
        self.assertEqual(analysis['cost']['total_calls'], 3)
        self.assertEqual({c['decision_id'] for c in contexts}, {f'{log.id}:round:3'})
        self.assertIsNone(contexts[0]['parent_call_id'])
        self.assertEqual({c['parent_call_id'] for c in contexts[1:]}, {contexts[0]['request_id']})
        self.assertEqual(len({c['request_id'] for c in contexts}), 3)
        self.assertEqual(analysis['cost_attribution']['coverage_details']['routing_reported_calls'], 1)
        gateway_call = next(call for call in analysis['rounds'] if call['phase'] == 'agent')
        self.assertEqual(gateway_call['routing']['selection_probability'], .4)
        self.assertNotIn('api_key', gateway_call['routing'])
        sources = analysis['cost_attribution']['sources']
        self.assertEqual(sum(row['cost']['known_amount_usd'] for row in sources), .0028)
        self.assertEqual(routing_request_context(), {})

    def test_no_probability_or_model_is_invented(self):
        for invalid in (True, -1, 1.1, float('nan'), '0.9'):
            self.assertIsNone(normalize_routing({'selection_probability': invalid})['selection_probability'])
        absent = normalize_routing(None)
        self.assertEqual(absent['availability'], 'unavailable')
        self.assertEqual(normalize_routing(absent)['availability'], 'unavailable')
        self.assertIsNone(absent['resolved_model'])

    def test_invoice_allocation_deduplicates_and_keeps_evidence_separate(self):
        log = RunLog('agent', {}); log.finish({'status': 'limit'})
        entry = Expense(category='compute', name='GPU reservation', quantity=2, unit='hour', unit_price_usd=.3,
                        source='invoice fixture', evidence_type='actual_invoice', source_invoice_id='INV-1',
                        allocation_key='slice-1', allocation_rule='reserved two hours')
        record = append_expense(log.id, entry)
        with self.assertRaisesRegex(ValueError, '已登记'): append_expense(log.id, entry)
        result = attribution(log.id, [], {'known_amount_usd': .1})
        self.assertAlmostEqual(result['known_total_usd'], .7)
        self.assertEqual(result['non_model_evidence_totals'], {'actual_invoice': .6})
        second = RunLog('agent', {}); second.finish({'status': 'limit'})
        append_expense(second.id, entry)
        combined = attribution(second.id, [], {'known_amount_usd': .1}, [{'run_id': log.id, 'cost_attribution': result}])
        self.assertAlmostEqual(combined['known_total_usd'], .7)
        self.assertTrue(combined['warnings'])
        # The same invoice number from another billing source is not a duplicate.
        append_expense(second.id, entry.model_copy(update={'source': 'another vendor'}))
        distinct = attribution(second.id, [], {'known_amount_usd': .1})
        self.assertAlmostEqual(distinct['known_total_usd'], 1.3)
        estimate_run = RunLog('agent', {}); estimate_run.finish({'status': 'limit'})
        append_expense(estimate_run.id, entry.model_copy(update={'evidence_type': 'price_estimate', 'unit_price_usd': .5}))
        preferred = attribution(estimate_run.id, [], {'known_amount_usd': .1}, [{'run_id':log.id, 'cost_attribution':result}])
        self.assertAlmostEqual(preferred['known_total_usd'], .7)
        self.assertEqual(preferred['non_model_evidence_totals']['price_estimate'], 0)
        void_expense(log.id, record['id']); append_expense(log.id, entry)
        with self.assertRaises(ValueError): Expense(category='compute', name='invoice', quantity=1,
            unit='hour', unit_price_usd=1, source='x', evidence_type='actual_invoice')

    def test_invoice_total_and_allocations_cannot_be_registered_together(self):
        log = RunLog('agent', {}); log.finish({'status': 'limit'})
        total = Expense(category='compute', name='GPU', quantity=10, unit='hour', unit_price_usd=1,
                        source='vendor', evidence_type='actual_invoice', source_invoice_id='INV')
        partial = total.model_copy(update={'quantity': 2, 'allocation_key': 'episode-1', 'allocation_rule': 'reserved hours'})
        row = append_expense(log.id, total)
        with self.assertRaisesRegex(ValueError, '全额与分摊'): append_expense(log.id, partial)
        void_expense(log.id, row['id'])
        append_expense(log.id, partial)
        with self.assertRaisesRegex(ValueError, '全额与分摊'): append_expense(log.id, total)
        append_expense(log.id, partial.model_copy(update={'allocation_key': 'episode-2'}))
        result = attribution(log.id, [], {})
        self.assertEqual(result['non_model_amount_usd'], 4)
        self.assertTrue(any('上限' in warning for warning in result['warnings']))

    def test_historical_or_cross_run_invoice_scope_conflict_is_unknown_not_doubled(self):
        parent = RunLog('comparison', {}); parent.finish({'status': 'complete'})
        child = RunLog('agent', {}); child.finish({'status': 'complete'})
        total = Expense(category='compute', name='GPU', quantity=10, unit='hour', unit_price_usd=1,
                        source='vendor', evidence_type='actual_invoice', source_invoice_id='INV')
        append_expense(parent.id, total)
        append_expense(child.id, total.model_copy(update={'quantity': 2, 'allocation_key': 'episode-1', 'allocation_rule': 'reserved hours'}))
        append_expense(parent.id, total.model_copy(update={'source_invoice_id': 'INV-2', 'quantity': 3}))
        before = (parent.folder/'expenses.jsonl').read_bytes()
        result = attribution(parent.id, [], {'known_amount_usd': .1}, [
            {'run_id': child.id, 'cost_attribution': attribution(child.id, [], {})}])
        self.assertEqual(result['expense_accounting_status'], 'conflict')
        self.assertEqual(result['coverage_details']['conflicting_invoices'], 1)
        self.assertEqual(result['invoice_conflicts'][0]['source_invoice_id'], 'INV')
        self.assertAlmostEqual(result['known_total_usd'], 3.1)
        self.assertEqual(len(result['all_expenses']), 3)
        self.assertEqual((parent.folder/'expenses.jsonl').read_bytes(), before)


if __name__ == '__main__': unittest.main()
