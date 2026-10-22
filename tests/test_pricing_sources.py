"""Official-source excerpts verify scope, updated prices and truthful failures."""
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx

from server.pricing_sources import (PricingSourceError, fetch_official_prices, parse_official_prices,
                                    parse_openai_long_context_policy)

FIXTURES = Path(__file__).parent/'fixtures'/'pricing'


class PricingSourceTests(unittest.TestCase):
    def fixture(self, provider):
        return (FIXTURES/(provider+('.html' if provider == 'deepseek' else '.md'))).read_text(encoding='utf-8')

    def models(self, provider, document=None):
        result = parse_official_prices(provider, document if document is not None else self.fixture(provider))
        self.assertEqual(result['currency'], 'USD')
        self.assertTrue(result['source_url'].startswith('https://'))
        self.assertIn('+00:00', result['fetched_at'])
        return {model['model']: model for model in result['models']}

    def test_openai_standard_prices_never_mix_flex_fast_and_write_lifetimes(self):
        models = self.models('openai')
        self.assertEqual(len(models), 4)
        astra = models['gpt-6-astra']
        self.assertEqual(astra['input_per_million'], 10.)
        self.assertEqual(astra['output_per_million'], 50.)
        self.assertEqual(astra['cached_input_per_million'], 1.)
        self.assertEqual(astra['cache_write_30m_per_million'], 12.5)
        self.assertEqual(astra['cache_write_12h_per_million'], 25.)
        self.assertEqual(models['gpt-6.1-sol']['cached_input_per_million'], .1)
        self.assertEqual(models['gpt-6-luna']['cache_write_30m_per_million'], .125)
        self.assertIn('long_context', astra['unsupported'])
        self.assertNotIn('long_context_rates', astra)

    def test_price_update_is_parsed_instead_of_using_embedded_model_constants(self):
        document = self.fixture('openai').replace('| gpt-6-astra | $10.00 | $50.00 |', '| gpt-6-astra | $11.25 | $55.50 |')
        astra = self.models('openai', document)['gpt-6-astra']
        self.assertEqual(astra['input_per_million'], 11.25); self.assertEqual(astra['output_per_million'], 55.5)
        deepseek = self.fixture('deepseek').replace('$0.15', '$0.19')
        self.assertEqual(self.models('deepseek', deepseek)['deepseek-flash']['off_peak']['input_per_million'], .19)

    def test_openai_html_explicit_scope_and_reordered_columns_are_supported(self):
        document = '''<h2>Token prices</h2><div data-tier="standard"><table>
        <tr><th>Model</th><th>Output</th><th>Cached input</th><th>Input</th></tr>
        <tr><td>gpt-fixture-future</td><td>$9</td><td>$0.10</td><td>$2</td></tr></table></div>
        <div data-tier="flex"><table><tr><th>Model</th><th>Input</th><th>Output</th></tr>
        <tr><td>gpt-fixture-future</td><td>$1</td><td>$4.50</td></tr></table></div>'''
        model = self.models('openai', document)['gpt-fixture-future']
        self.assertEqual(model['input_per_million'], 2); self.assertEqual(model['output_per_million'], 9)
        self.assertNotIn('cache_write_30m_per_million', model)

    def test_ambiguous_conflicting_or_missing_standard_prices_fail_instead_of_guessing(self):
        unscoped = '| Model | Input | Output |\n| --- | --- | --- |\n| gpt-fixture | $2 | $10 |\n'
        for document in (unscoped, '## Fast\n'+unscoped,
                         '## Standard\n'+unscoped+'\n## Standard\n'+unscoped.replace('$2', '$3')):
            with self.subTest(document=document[:40]), self.assertRaises(PricingSourceError):
                parse_official_prices('openai', document)
        for malformed in ('-', '$?', '$NaN', '-$1', '$1,00'):
            document = self.fixture('openai').replace('| gpt-6-astra | $10.00 |', '| gpt-6-astra | '+malformed+' |')
            with self.subTest(malformed=malformed), self.assertRaises(PricingSourceError):
                parse_official_prices('openai', document)

    def test_claude_base_model_scope_cache_ttls_and_long_prompt_tier(self):
        models = self.models('anthropic')
        opus = models['claude-opus-5-5']
        self.assertEqual(opus['input_per_million'], 4)
        self.assertEqual(opus['output_per_million'], 20)
        self.assertEqual(opus['cache_write_5m_per_million'], 5)
        self.assertEqual(opus['cache_write_1h_per_million'], 8)
        self.assertEqual(opus['cached_input_per_million'], .2)
        self.assertEqual(models['claude-fable-5-1']['cached_input_per_million'], .25)
        haiku = models['claude-haiku-5-5']
        self.assertEqual(haiku['long_context_threshold'], 100000)
        self.assertEqual(haiku['input_per_million'], .1)
        self.assertEqual(haiku['long_context_rates']['input_per_million'], .5)
        self.assertEqual(haiku['long_context_rates']['cached_input_per_million'], .05)
        self.assertTrue(models['claude-opus-4-1']['retired'])
        self.assertNotIn('claude-opus-5-5-20260921', models, 'No dated alias may be invented from the display name.')

    def test_deepseek_real_rowspans_peak_weekdays_holidays_and_explicit_legacy_aliases(self):
        models = self.models('deepseek')
        flash = models['deepseek-flash']
        self.assertEqual(flash['peak']['input_per_million'], .3)
        self.assertEqual(flash['off_peak']['input_per_million'], .15)
        self.assertEqual(flash['peak']['cached_input_per_million'], .006)
        self.assertEqual(flash['off_peak']['cached_input_per_million'], .003)
        self.assertEqual(flash['peak']['output_per_million'], 1.2)
        self.assertEqual(flash['peak_utc'], [[60, 240], [360, 600]])
        self.assertEqual(flash['peak_weekdays_utc'], [0, 1, 2, 3, 4])
        self.assertEqual(flash['holiday_region'], 'CN')
        self.assertIn('calendar', flash['unsupported'])
        self.assertNotIn('off_peak_utc', flash, 'Weekday/holiday conditions cannot be reduced to a daily time window.')
        self.assertEqual(flash['aliases'], ['deepseek-v4-flash', 'deepseek-v4-flash-vision-exp'])
        self.assertEqual(models['deepseek-v4-pro']['off_peak']['output_per_million'], 1.98)

    def test_unknown_schedule_or_incomplete_rates_remain_unsupported_or_fail(self):
        source = self.fixture('deepseek')
        no_schedule = source.replace('Peak hours are', 'The schedule is')
        flash = self.models('deepseek', no_schedule)['deepseek-flash']
        self.assertIn('calendar', flash['unsupported']); self.assertNotIn('peak_utc', flash)
        with self.assertRaises(PricingSourceError): self.models('deepseek', source.replace('$1.2', '-'))
        with self.assertRaises(PricingSourceError): parse_official_prices('anthropic', '## Model pricing\nNo table.')
        with self.assertRaises(PricingSourceError): parse_official_prices('codex', source)

    def test_official_fetch_is_unauthenticated_get_and_returns_new_source_values(self):
        native_client = httpx.AsyncClient; requests = []
        def serve(request):
            requests.append(request)
            document = self.fixture('anthropic').replace('$4 / MTok', '$4.20 / MTok')
            return httpx.Response(200, text=document)
        with patch('server.pricing_sources.httpx.AsyncClient', side_effect=lambda **kwargs: native_client(**kwargs, transport=httpx.MockTransport(serve))):
            result = asyncio.run(fetch_official_prices('anthropic'))
        opus = next(model for model in result['models'] if model['model'] == 'claude-opus-5-5')
        self.assertEqual(opus['input_per_million'], 4.2)
        self.assertEqual(requests[0].method, 'GET')
        self.assertEqual(str(requests[0].url), 'https://platform.claude.com/docs/en/about-claude/pricing.md')
        self.assertNotIn('authorization', requests[0].headers); self.assertNotIn('x-api-key', requests[0].headers)

    def test_fetch_failures_and_nonprice_responses_never_return_successful_zero_catalog(self):
        native_client = httpx.AsyncClient
        for status, document in ((403, 'access denied'), (200, '<h1>Not a pricing page</h1>')):
            def serve(request): return httpx.Response(status, text=document)
            with self.subTest(status=status), patch('server.pricing_sources.httpx.AsyncClient', side_effect=lambda **kwargs: native_client(**kwargs, transport=httpx.MockTransport(serve))), self.assertRaises(PricingSourceError):
                asyncio.run(fetch_official_prices('openai'))
        def timeout(request): raise httpx.ReadTimeout('actual timeout', request=request)
        with patch('server.pricing_sources.httpx.AsyncClient', side_effect=lambda **kwargs: native_client(**kwargs, transport=httpx.MockTransport(timeout))), self.assertRaisesRegex(PricingSourceError, '20'):
            asyncio.run(fetch_official_prices('openai'))

    def test_long_context_policy_reads_current_threshold_and_distinguishes_unstated_cache(self):
        # Exact sentence fetched from the official gpt-6-astra model page on
        # 2026-10-22; rates still come from each newly fetched pricing table.
        rule = 'Prompts with more than 272K input tokens are priced at 2x input and cache rates and 1.5x output for the full request.'
        policy = parse_openai_long_context_policy(rule)
        self.assertEqual(policy, {'threshold': 272000, 'input_multiplier': 2., 'cache_multiplier': 2., 'output_multiplier': 1.5})
        updated = parse_openai_long_context_policy(rule.replace('272K', '300K').replace('2x', '3x'))
        self.assertEqual(updated['threshold'], 300000); self.assertEqual(updated['input_multiplier'], 3.)
        sol = parse_openai_long_context_policy('Prompts with >272K input tokens are priced at 2x input and 1.5x output for the full request.')
        self.assertIsNone(sol['cache_multiplier'])
        self.assertIsNone(parse_openai_long_context_policy('Supports a 1M token context window.'))
        with self.assertRaises(PricingSourceError): parse_openai_long_context_policy(rule+' '+rule.replace('272K', '300K'))

    def test_openai_fetch_enriches_only_known_exact_models_and_preserves_failed_unknown_fields(self):
        native_client = httpx.AsyncClient; requests = []
        def serve(request):
            requests.append(request)
            self.assertEqual(request.method, 'GET'); self.assertNotIn('authorization', request.headers)
            self.assertEqual(request.headers['accept'], 'text/markdown')
            if request.url.path.endswith('/pricing.md'): return httpx.Response(200, text=self.fixture('openai'))
            model = request.url.path.rsplit('/', 1)[-1].removesuffix('.md')
            if model == 'gpt-6-luna': return httpx.Response(403, text='blocked')
            sentence = 'Prompts with more than 272K input tokens are priced at 2x input and cache rates and 1.5x output for the full request.'
            if model == 'gpt-5.6-sol': sentence = sentence.replace(' and cache rates', '')
            return httpx.Response(200, text=sentence)
        with patch('server.pricing_sources.httpx.AsyncClient', side_effect=lambda **kwargs: native_client(**kwargs, transport=httpx.MockTransport(serve))):
            catalog = asyncio.run(fetch_official_prices('openai'))
        self.assertEqual(len(requests), 5)
        models = {model['model']: model for model in catalog['models']}
        self.assertEqual(models['gpt-6-astra']['long_context_threshold'], 272000)
        self.assertEqual(models['gpt-6-astra']['long_context_rates']['input_per_million'], 20.)
        self.assertEqual(models['gpt-6-astra']['long_context_rates']['output_per_million'], 75.)
        self.assertEqual(models['gpt-6.1-sol']['long_context_rates']['cache_write_12h_per_million'], 10.)
        self.assertIn('HTTP 403', models['gpt-6-luna']['unsupported']['long_context'])
        sol = models['gpt-5.6-sol']
        self.assertEqual(sol['long_context_rates']['input_per_million'], 8.)
        self.assertIsNone(sol['long_context_rates']['cached_input_per_million'])
        self.assertIsNone(sol['long_context_rates']['cache_write_30m_per_million'])
        self.assertIn('long_context_cache', sol['unsupported'])


if __name__ == '__main__': unittest.main()
