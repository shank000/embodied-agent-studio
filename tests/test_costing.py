"""Deterministic official-list-price math; no model, HTTP or native simulation."""
from copy import deepcopy
import unittest

from server.costing import aggregate_costs, calculate_call_cost


def quote(**changes):
    return {'status': 'ready', 'provider': 'openai', 'model': 'fixture-model', 'billing_kind': 'api',
            'currency': 'USD', 'source_url': 'https://example.org/official-fixture',
            'updated_at': '2026-10-22T00:00:00Z', 'basis': 'recorded_snapshot',
            'input_per_million': 1., 'output_per_million': 4., 'cached_input_per_million': .1, **changes}


def call(incoming=1000, outgoing=200, cached=400, **changes):
    return {'usage': {'input_tokens': incoming, 'output_tokens': outgoing,
                      'input_tokens_details': {'cached_tokens': cached}},
            'started_at': '2026-10-22T11:00:00Z', **changes}


class CostingTests(unittest.TestCase):
    def test_openai_cached_input_and_output_reasoning_are_counted_once(self):
        request = call()
        request['usage']['output_tokens_details'] = {'reasoning_tokens': 150}
        result = calculate_call_cost(request, quote())
        self.assertEqual(result['status'], 'complete')
        self.assertAlmostEqual(result['amount_usd'], .00144)
        self.assertEqual({row['component']: row['tokens'] for row in result['breakdown']},
                         {'input': 600, 'cached_input': 400, 'output': 200})
        self.assertEqual(result['billable_calls'], 1)
        self.assertEqual(result['priced_calls'], 1)

    def test_openai_two_write_lifetimes_follow_official_example(self):
        request = call(incoming=12000, outgoing=0, cached=4000)
        request['usage']['input_tokens_details'].update(cache_write_tokens=7000,
            cache_write_tokens_details={'ttl_30m_tokens': 3000, 'ttl_12h_tokens': 4000})
        result = calculate_call_cost(request, quote(cache_write_30m_per_million=1.25, cache_write_12h_per_million=2.5))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['amount_usd'], .01515)
        self.assertEqual(sum(row['tokens'] for row in result['breakdown']), 12000)

    def test_anthropic_input_excludes_reads_and_writes(self):
        request = {'usage': {'input_tokens': 1000, 'output_tokens': 200, 'cache_read_input_tokens': 4000,
                            'cache_creation_input_tokens': 7000,
                            'cache_creation': {'ephemeral_5m_input_tokens': 3000, 'ephemeral_1h_input_tokens': 4000}}}
        prices = quote(provider='anthropic', cache_write_5m_per_million=1.25, cache_write_1h_per_million=2.)
        result = calculate_call_cost(request, prices)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['amount_usd'], .01395)
        self.assertEqual(sum(row['tokens'] for row in result['breakdown']), 12200)

    def test_anthropic_write_ttl_missing_is_partial_without_assumed_five_minutes(self):
        request = {'usage': {'input_tokens': 1000, 'output_tokens': 200, 'cache_read_input_tokens': 4000,
                            'cache_creation_input_tokens': 7000}}
        prices = quote(provider='anthropic', cache_write_5m_per_million=1.25, cache_write_1h_per_million=2.)
        result = calculate_call_cost(request, prices)
        self.assertEqual(result['status'], 'partial')
        self.assertIsNone(result['amount_usd'])
        self.assertEqual(result['known_amount_usd'], .0022)
        explicit = calculate_call_cost(request, {**prices, 'cache_write_default_ttl': '5m'})
        self.assertEqual(explicit['status'], 'complete')
        self.assertEqual(explicit['amount_usd'], .01095)

    def test_deepseek_hit_and_miss_and_utc_time_windows(self):
        prices = quote(provider='deepseek', peak={'input_per_million': 2., 'output_per_million': 8., 'cached_input_per_million': .2},
                       off_peak={'input_per_million': 1., 'output_per_million': 4., 'cached_input_per_million': .1},
                       off_peak_utc=[[0, 360]])
        usage = {'prompt_tokens': 1000, 'completion_tokens': 200, 'prompt_cache_hit_tokens': 400, 'prompt_cache_miss_tokens': 600}
        off = calculate_call_cost({'usage': usage, 'started_at': '2026-10-22T13:59:59+08:00'}, prices)
        peak = calculate_call_cost({'usage': usage, 'started_at': '2026-10-22T14:00:00+08:00'}, prices)
        self.assertEqual(off['amount_usd'], .00144)
        self.assertEqual(peak['amount_usd'], .00288)

    def test_possible_peak_holiday_is_range_but_weekend_and_off_window_are_exact(self):
        prices = quote(provider='deepseek', peak={'input_per_million': 2., 'output_per_million': 8., 'cached_input_per_million': .2},
                       off_peak={'input_per_million': 1., 'output_per_million': 4., 'cached_input_per_million': .1},
                       peak_utc=[[60, 240], [360, 600]], peak_weekdays_utc=[0, 1, 2, 3, 4], holiday_region='CN')
        usage = {'prompt_tokens': 1000, 'completion_tokens': 200, 'prompt_cache_hit_tokens': 400, 'prompt_cache_miss_tokens': 600}
        uncertain = calculate_call_cost({'usage': usage, 'started_at': '2026-10-22T01:00:00Z'}, prices)
        self.assertEqual(uncertain['status'], 'partial')
        self.assertIsNone(uncertain['amount_usd'])
        self.assertEqual(uncertain['known_amount_usd'], 0.)
        self.assertEqual((uncertain['amount_min_usd'], uncertain['amount_max_usd']), (.00144, .00288))
        self.assertTrue(any('节假日' in warning for warning in uncertain['warnings']))
        for stamp in ('2026-10-24T01:00:00Z', '2026-10-22T04:00:00Z'):
            exact = calculate_call_cost({'usage': usage, 'started_at': stamp}, prices)
            self.assertEqual(exact['amount_usd'], .00144)
        no_time = calculate_call_cost({'usage': usage, 'response_at': '2026-10-22T11:00:00Z'}, prices)
        self.assertIsNone(no_time['amount_usd'])
        self.assertEqual(no_time['amount_max_usd'], .00288)

    def test_long_context_applies_to_whole_call_and_counts_anthropic_cache(self):
        prices = quote(long_context_threshold=1000, long_context_rates={
            'input_per_million': 2., 'output_per_million': 8., 'cached_input_per_million': .2})
        self.assertEqual(calculate_call_cost(call(incoming=1000), prices)['amount_usd'], .00144)
        self.assertEqual(calculate_call_cost(call(incoming=1001), prices)['amount_usd'], .002882)
        usage = {'input_tokens': 100, 'output_tokens': 200, 'cache_read_input_tokens': 1000, 'cache_creation_input_tokens': 0}
        self.assertEqual(calculate_call_cost({'usage': usage}, {**prices, 'provider': 'anthropic'})['amount_usd'], .002)

    def test_missing_cache_or_write_usage_is_not_billed_as_uncached_input(self):
        missing_cache = calculate_call_cost({'usage': {'input_tokens': 1000, 'output_tokens': 200}}, quote())
        self.assertEqual(missing_cache['status'], 'partial')
        self.assertEqual(missing_cache['known_amount_usd'], .0008)
        self.assertIsNone(missing_cache['amount_usd'])
        missing_writes = calculate_call_cost(call(), quote(cache_write_30m_per_million=1.25, cache_write_12h_per_million=2.5))
        self.assertEqual(missing_writes['known_amount_usd'], .00084)
        self.assertIsNone(missing_writes['amount_usd'])
        self.assertEqual(calculate_call_cost({'usage': None}, quote())['status'], 'unavailable')

    def test_invalid_or_overlapping_usage_is_unavailable(self):
        for request in (call(incoming=-1), call(outgoing=True), call(cached=1001), call(incoming=1.5),
                        {'usage': {'prompt_tokens': 1000, 'completion_tokens': 200,
                                   'prompt_cache_hit_tokens': 800, 'prompt_cache_miss_tokens': 500}}):
            with self.subTest(request=request):
                result = calculate_call_cost(request, quote())
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['known_amount_usd'], 0.)
                self.assertIsNone(result['amount_usd'])
        request = call()
        request['usage']['input_tokens_details'].update(cache_write_tokens=700,
            cache_write_tokens_details={'ttl_30m_tokens': 700, 'ttl_12h_tokens': 0})
        self.assertEqual(calculate_call_cost(request, quote())['status'], 'unavailable')
        request['usage']['input_tokens_details'] = {'cached_tokens': 0,
            'cache_write_tokens_details': {'ttl_30m_tokens': 1001}}
        self.assertEqual(calculate_call_cost(request, quote())['status'], 'unavailable')

    def test_cancelled_calls_use_returned_usage_and_unavailable_prices_remain_unknown(self):
        result = calculate_call_cost(call(status='cancelled'), quote(status='stale', basis='current_price_estimate'))
        self.assertEqual(result['amount_usd'], .00144)
        self.assertEqual(result['basis'], 'current_price_estimate')
        self.assertTrue(any('过期' in warning for warning in result['warnings']))
        unknown = calculate_call_cost(call(), quote(status='unavailable', billing_kind='unpriced'))
        self.assertEqual(unknown['status'], 'unavailable')
        self.assertIsNone(unknown['amount_usd'])

    def test_codex_reference_and_local_api_zero_are_separate_from_api_cash(self):
        api = calculate_call_cost(call(), quote())
        reference = calculate_call_cost({'usage': {'input_tokens': 1000, 'cached_input_tokens': 400, 'output_tokens': 200}}, quote(billing_kind='reference'))
        local = calculate_call_cost({'usage': None}, quote(provider='local', billing_kind='local'))
        self.assertEqual(reference['status'], 'reference')
        self.assertIsNone(reference['amount_usd'])
        self.assertEqual(reference['known_amount_usd'], 0.)
        self.assertEqual(reference['reference_amount_usd'], .00144)
        self.assertEqual(local['amount_usd'], 0.)
        total = aggregate_costs([api, reference, local])
        self.assertEqual(total['status'], 'complete')
        self.assertEqual(total['amount_usd'], .00144)
        self.assertEqual(total['reference_amount_usd'], .00144)
        self.assertEqual((total['billable_calls'], total['reference_calls'], total['local_calls']), (1, 1, 1))
        self.assertEqual(total['priced_calls'], 3)
        self.assertEqual(len(total['price_sources']), 1)

    def test_parent_aggregation_decimal_sums_partial_values_and_references(self):
        api = calculate_call_cost(call(), quote())
        reference = calculate_call_cost(call(), quote(billing_kind='reference'))
        partial = calculate_call_cost(call(outgoing=None), quote())
        nested = aggregate_costs([aggregate_costs([api, reference]), partial])
        direct = aggregate_costs([api, reference, partial])
        self.assertEqual(nested, direct)
        self.assertEqual(direct['status'], 'partial')
        self.assertIsNone(direct['amount_usd'])
        self.assertEqual(direct['known_amount_usd'], .00208)
        self.assertEqual(direct['reference_amount_usd'], .00144)
        self.assertEqual((direct['total_calls'], direct['priced_calls'], direct['missing_calls']), (3, 2, 1))
        self.assertEqual(aggregate_costs([api]*100)['amount_usd'], .144)
        estimated = calculate_call_cost(call(), quote(basis='current_price_estimate'))
        self.assertEqual(aggregate_costs([api, estimated])['basis'], 'mixed')
        self.assertEqual(aggregate_costs([estimated])['basis'], 'current_price_estimate')

    def test_bounds_are_not_presented_as_complete_when_another_call_is_unpriced(self):
        prices = quote(peak={'input_per_million': 2.}, off_peak={}, off_peak_utc=[[0, 360]])
        uncertain = calculate_call_cost(call(started_at=None), prices)
        full = calculate_call_cost(call(), quote())
        bounded = aggregate_costs([uncertain, full])
        self.assertEqual(bounded['amount_min_usd'], .00288)
        self.assertEqual(bounded['amount_max_usd'], .00348)
        missing = calculate_call_cost({'usage': None}, quote())
        incomplete = aggregate_costs([bounded, missing])
        self.assertNotIn('amount_max_usd', incomplete)
        self.assertIsNone(incomplete['amount_usd'])
        unknown_rule = calculate_call_cost(call(), quote(peak={'input_per_million': 2.},
            off_peak={}, peak_utc=[[0,1440]], unsupported={'calendar':'Unrecognized new eligibility condition.'}))
        self.assertEqual(unknown_rule['status'], 'partial')
        self.assertIsNone(unknown_rule['amount_usd'])
        self.assertIn('amount_max_usd', unknown_rule)
        unreadable_child = aggregate_costs([{'total_calls':0, 'missing_calls':1,
            'status':'unavailable', 'call_count_incomplete':True}])
        self.assertEqual(unreadable_child['status'], 'unavailable')
        self.assertIsNone(unreadable_child['amount_usd'])
        self.assertTrue(unreadable_child['call_count_incomplete'])

    def test_zero_usage_and_inputs_are_preserved(self):
        request, prices = call(incoming=0, outgoing=0, cached=0), quote()
        before = deepcopy((request, prices))
        result = calculate_call_cost(request, prices)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['amount_usd'], 0.)
        self.assertEqual((request, prices), before)
        self.assertEqual(aggregate_costs([])['amount_usd'], 0.)

    def test_unknown_context_scope_or_missing_long_tier_does_not_apply_short_rates(self):
        unknown = calculate_call_cost(call(), quote(unsupported={'long_context': 'No verified context scope.'}))
        self.assertEqual(unknown['status'], 'unavailable')
        self.assertIsNone(unknown['amount_usd'])
        self.assertEqual(unknown['known_amount_usd'], 0.)
        self.assertIn('No verified context scope.', unknown['warnings'])
        prices = quote(long_context_threshold=1000, unsupported={'long_context': 'Long tier unavailable.'})
        self.assertEqual(calculate_call_cost(call(), prices)['status'], 'complete')
        exceeded = calculate_call_cost(call(incoming=1001), prices)
        self.assertEqual(exceeded['status'], 'unavailable')
        self.assertIsNone(exceeded['amount_usd'])


if __name__ == '__main__': unittest.main()
