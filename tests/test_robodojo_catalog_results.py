"""No-SDK metadata and local-result parsing contracts for RoboDojo.

All result fixtures below are constructed from the official f57e1633
src/eval_client/eval_env.py writer schema. They are not measured policy runs,
benchmark scores or leaderboard submissions.
"""
from copy import deepcopy
import sys
import unittest
from unittest.mock import patch

from server.robodojo_catalog import catalog, REVISION
from server.robodojo_results import parse_result


def result_fixture(rows):
    """Construct the official terminal detail and summary shape."""
    details = {str(index): {'layout_id': index, 'success': success, 'score': score}
               for index, (success, score) in enumerate(rows)}
    return {'success_rate': sum(success for success, _ in rows)/len(rows) if rows else 0.,
            'eval_time': len(rows), 'score': sum(score for _, score in rows)/len(rows)*100 if rows else 0.,
            'details': details}


def parse(payload, task='general_pickup', seed=0, policy='fixture-policy', embodiment='arx_x5'):
    return parse_result(payload, task, seed, policy, embodiment)


class RoboDojoCatalogTests(unittest.TestCase):
    def test_five_dimensions_42_base_tasks_54_configs_and_one_sim_embodiment(self):
        value = catalog()
        self.assertEqual(value['benchmark_id'], 'robodojo')
        self.assertEqual(value['revision'], 'f57e1633ffed59761bc8e568cfe7530c4b33381c')
        self.assertEqual(value['version'], '0.2.0')
        self.assertEqual({row['id'] for row in value['dimensions']},
                         {'generalization', 'memory', 'precision', 'long-horizon', 'open'})
        self.assertEqual(len(value['tasks']), 54)
        self.assertEqual(len({task['id'] for task in value['tasks']}), 54)
        self.assertEqual(len({task['base_task'] for task in value['tasks']}), 42)
        self.assertEqual(sum(task['variant'] == 'random' for task in value['tasks']), 12)
        expected = {'generalization': 24, 'memory': 6, 'precision': 8, 'long-horizon': 8, 'open': 8}
        for dimension, count in expected.items():
            self.assertEqual(sum(task['dimension'] == dimension for task in value['tasks']), count)
        self.assertEqual([row['id'] for row in value['embodiments']], ['arx_x5'])

    def test_native_counts_and_random_pair_identity_follow_official_configs(self):
        tasks = {row['id']: row for row in catalog()['tasks']}
        for task in tasks.values():
            self.assertEqual(task['native_episodes'], 25 if task['dimension'] == 'generalization' else 50)
            if task['variant'] == 'random':
                self.assertEqual(task['id'], task['base_task']+'_random')
                self.assertIn(task['base_task'], tasks)
                self.assertEqual(tasks[task['base_task']]['dimension'], 'generalization')

    def test_metadata_and_parser_do_not_read_files_or_import_optional_sdks(self):
        packages = ('isaacsim', 'isaaclab', 'torch', 'omni', 'newton')
        before = {name for name in sys.modules if name.split('.')[0] in packages}
        with patch('builtins.open', side_effect=AssertionError('Pure metadata and parsing must not read files')):
            value = catalog()
            parse(result_fixture([(False, .5)]), task=value['tasks'][0]['id'])
        self.assertEqual(before, {name for name in sys.modules if name.split('.')[0] in packages})
        value['tasks'][0]['name'] = 'caller edit'
        value['dimensions'][0]['name'] = 'caller edit'
        value['embodiments'][0]['name'] = 'caller edit'
        self.assertNotEqual(catalog()['tasks'][0]['name'], 'caller edit')
        self.assertNotEqual(catalog()['dimensions'][0]['name'], 'caller edit')
        self.assertNotEqual(catalog()['embodiments'][0]['name'], 'caller edit')


class RoboDojoResultTests(unittest.TestCase):
    def test_terminal_success_and_graded_score_are_independent_and_units_explicit(self):
        value = parse(result_fixture([(True, 1.), (False, .4)]))
        self.assertEqual(value['summary'], {'eval_time': 2, 'successes': 1, 'failures': 1,
                                           'success_rate': .5, 'score': 70.})
        self.assertEqual([row['native_success'] for row in value['episodes']], [True, False])
        self.assertEqual([row['score'] for row in value['episodes']], [1., .4])
        self.assertEqual(value['summary_differences'], [])
        self.assertEqual(value['coverage']['scope'], 'partial')
        # A full partial-credit value still cannot substitute for native success.
        self.assertEqual(parse(result_fixture([(False, 1.)]))['summary']['success_rate'], 0.)

    def test_top_level_initial_success_and_empty_details_are_not_terminal_results(self):
        for payload in ({'success': True}, {'success': [True], 'details': {}}, result_fixture([]),
                        {'details': []}, {'details': None}):
            with self.subTest(payload=payload), self.assertRaises(ValueError): parse(payload)

    def test_terminal_detail_types_ranges_duplicates_and_explicit_nonterminal_are_rejected(self):
        valid = {'layout_id': 0, 'success': False, 'score': .2}
        invalid = [None, {'layout_id': 0, 'score': .2}, valid|{'success': 1}, valid|{'success': 'false'},
                   valid|{'layout_id': True}, valid|{'layout_id': -1}, valid|{'layout_id': 1.5},
                   valid|{'score': True}, valid|{'score': '0.5'}, valid|{'score': -.1}, valid|{'score': 1.1},
                   valid|{'score': 10**400},
                   valid|{'score': float('nan')}, valid|{'score': float('inf')},
                   valid|{'terminal': False}, valid|{'terminated': False}, valid|{'end_flag': False}]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError): parse({'details': {'0': row}})
        for details in ({'0': valid, '1': valid.copy()}, {'1': valid, '01': valid|{'layout_id': 1}},
                        {'../0': valid}, {False: valid}):
            with self.subTest(details=details), self.assertRaises(ValueError): parse({'details': details})

    def test_source_summary_discrepancies_are_reported_without_overriding_details(self):
        payload = result_fixture([(True, 1.), (False, .4)])
        payload.update(success_rate=1., eval_time=999, score=99.)
        value = parse(payload)
        self.assertEqual(value['summary']['success_rate'], .5)
        self.assertEqual(value['summary']['score'], 70.)
        self.assertEqual(value['reported_summary'], {'success_rate': 1., 'eval_time': 999, 'score': 99.})
        self.assertEqual({row['field'] for row in value['summary_differences']}, {'success_rate', 'eval_time', 'score'})
        self.assertEqual(value['raw'], payload)
        large_summary = parse(payload|{'eval_time': 10**400, 'success_rate': 10**400})
        self.assertEqual(len(large_summary['summary_differences']), 3)

    def test_smoke_partial_task_complete_and_overfull_are_never_official_overall(self):
        for task, count, scope, expected in [('push_T', 1, 'smoke', 25), ('push_T', 2, 'partial', 25),
            ('push_T', 25, 'task_seed_complete', 25), ('push_T_random', 25, 'task_seed_complete', 25),
            ('general_pickup', 50, 'task_seed_complete', 50), ('push_T', 26, 'overfull', 25)]:
            with self.subTest(task=task, count=count):
                value = parse(result_fixture([(False, 0.)]*count), task=task)
                self.assertEqual(value['coverage']['scope'], scope)
                self.assertEqual(value['coverage']['expected_episodes'], expected)
                self.assertEqual(value['coverage']['task_seed_complete'], count == expected)
                self.assertFalse(value['coverage']['official_overall_complete'])
                self.assertFalse(value['provenance']['official_verified'])
                self.assertFalse(value['provenance']['official_overall_complete'])
                self.assertEqual(len(value['episodes']), count)

    def test_exclusions_remain_separate_and_missing_assignment_denominator_is_unknown(self):
        payload = result_fixture([(True, 1.), (False, .5)])
        payload.update(unstable_nums=3, abandoned_layout_ids=[12, 13], excluded_layout_ids=[12, 14])
        value = parse(payload)
        self.assertEqual(value['summary']['success_rate'], .5)
        self.assertEqual(value['denominator']['native_valid_completed'], 2)
        self.assertEqual(value['denominator']['unstable_nums'], 3)
        self.assertEqual(value['denominator']['abandoned_layout_ids'], [12, 13])
        self.assertIsNone(value['denominator']['assigned_episodes'])
        self.assertIsNone(value['denominator']['all_assigned_success_rate'])
        payload['assigned_episodes'] = 10
        value = parse(payload)
        self.assertEqual(value['denominator']['all_assigned_success_rate'], .1)
        absent = parse(result_fixture([(False, 0.)]))['denominator']
        self.assertIsNone(absent['unstable_nums'])
        self.assertIsNone(absent['abandoned_layout_ids'])
        self.assertIsNone(absent['excluded_layout_ids'])

    def test_invalid_exclusion_extensions_do_not_create_plausible_denominators(self):
        payload = result_fixture([(False, 0.)]*2)
        for extra in ({'assigned_episodes': 1}, {'assigned_episodes': True}, {'unstable_nums': -1},
                      {'abandoned_layout_ids': [2, 2]}, {'excluded_layout_ids': [True]},
                      {'abandoned_layout_ids': '2,3'}):
            with self.subTest(extra=extra), self.assertRaises(ValueError): parse(payload|extra)

    def test_unknown_fields_are_retained_without_mutating_the_input(self):
        payload = result_fixture([(True, 1.)])
        payload['details']['0']['future_field'] = {'nested': [1, 'original']}
        payload['vendor_extension'] = {'opaque': {'keep': True}}
        before = deepcopy(payload)
        value = parse(payload)
        self.assertEqual(value['raw'], before)
        value['raw']['details']['0']['future_field']['nested'].append('caller')
        self.assertEqual(payload, before)

    def test_video_names_are_suggestions_only_and_local_provenance_is_preserved(self):
        value = parse(result_fixture([(True, 1.), (False, .2)]), seed=9, embodiment='custom-arm')
        self.assertEqual(value['provenance']['source'], 'local_import')
        self.assertEqual(value['provenance']['upstream_revision'], REVISION)
        self.assertEqual(value['provenance']['revision_role'], 'parser_and_catalog_reference')
        self.assertFalse(value['provenance']['source_revision_verified'])
        self.assertFalse(value['provenance']['seed_in_official_protocol'])
        self.assertFalse(value['provenance']['known_embodiment'])
        self.assertEqual(len(value['native_videos']), 6)
        self.assertEqual(value['native_videos'][0]['suggested_name'], 'episode_0000000_cam_head_success.mp4')
        self.assertEqual(value['native_videos'][3]['suggested_name'], 'episode_0000001_cam_head_fail.mp4')
        self.assertTrue(all('exists' not in row and '/' not in row['suggested_name'] for row in value['native_videos']))

    def test_context_inputs_and_non_json_data_are_rejected(self):
        valid = result_fixture([(False, 0.)])
        for kwargs in ({'task': 'missing'}, {'seed': True}, {'seed': -1}, {'seed': 2**32},
                       {'policy': '  '}, {'embodiment': None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): parse(valid, **kwargs)
        for payload in ([], None, valid|{'extra': object()}, valid|{'extra': float('nan')}):
            with self.subTest(payload=payload), self.assertRaises(ValueError): parse(payload)
