"""History groups preserve the physical experiment identity across native engines."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.analysis import analyze_experiments, analyze_run
from server.runlog import RunLog


def backend(engine='mujoco', version='3.13.0', evaluator='task-evaluation-v2'):
    return {'engine_id': engine, 'engine_version': version,
            'evaluator_source': f'{engine}_privileged_state', 'evaluator_version': evaluator}


class SimulatorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='simulator-analysis-')
        self.root = Path(self.tmp.name)
        self.patch = patch('server.runlog.RUNS', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def record(self, *, identity=None, robot='panda', scene='tabletop', task='lift_red', seed=42, output=10, evaluation=None):
        metadata = {'task': 'Lift the red cube', 'model': 'same-model', 'robot': robot,
                    'scene_id': scene, 'task_id': task, 'seed': seed}
        if identity is not None: metadata['backend'] = identity
        log = RunLog('agent', metadata)
        log.write('inference', {'call_id': 'call-1', 'round': 1, 'status': 'complete', 'model': 'same-model',
                  'reasoning_effort': 'high', 'input_tokens': 2, 'output_tokens': output,
                  'total_tokens': output + 2, 'inference_s': .5})
        metrics = {'success': True}
        if evaluation is not None: metrics['evaluation'] = evaluation
        log.finish({'status': 'complete', 'metrics': metrics})
        return log

    def test_engines_never_share_a_group_but_seeds_remain_repeated_samples(self):
        self.record(identity=backend(), seed=42, output=10)
        self.record(identity=backend(), seed=43, output=20)
        self.record(identity=backend('genesis', '1.4.5', 'genesis-task-v1'), output=100)
        result = analyze_experiments(model='same-model', kind='agent')
        groups = {group['engine_id']: group for group in result['groups']}
        self.assertEqual(set(groups), {'mujoco', 'genesis'})
        self.assertEqual(groups['mujoco']['runs'], 2)
        self.assertEqual(groups['mujoco']['output_tokens'], 30)
        self.assertEqual(groups['mujoco']['mean_output_tokens'], 15)
        self.assertEqual(groups['genesis']['output_tokens'], 100)
        self.assertNotIn('seed', result['grouping_fields'])

    def test_engine_version_evaluator_robot_scene_and_task_are_separate_contexts(self):
        self.record(identity=backend())
        self.record(identity=backend(version='3.14.0'))
        self.record(identity=backend(evaluator='task-evaluation-v3'))
        self.record(identity=backend(), robot='so101')
        self.record(identity=backend(), scene='drawer')
        self.record(identity=backend(), task='place_red')
        groups = analyze_experiments(model='same-model')['groups']
        self.assertEqual(len(groups), 6)
        self.assertTrue(all(group['runs'] == 1 for group in groups))
        self.assertTrue(all(group['backend']['engine_id'] == 'mujoco' for group in groups))

    def test_legacy_mujoco_requires_recorded_evaluator_evidence_and_never_uses_current_version(self):
        recorded = self.record(evaluation={'source': 'mujoco_privileged_state', 'version': 'historical-v1', 'task_id': 'lift_red'})
        unknown = self.record(evaluation={'source': 'model_claim', 'success': True})
        before = {log.id: ((log.folder / 'events.jsonl').read_bytes(), (log.folder / 'summary.json').read_bytes()) for log in (recorded, unknown)}
        known_result, missing_result = analyze_run(recorded.id), analyze_run(unknown.id)
        self.assertEqual(known_result['engine_id'], 'mujoco')
        self.assertEqual(known_result['engine_version'], 'unknown')
        self.assertEqual(known_result['evaluator_version'], 'historical-v1')
        self.assertEqual(missing_result['engine_id'], 'unknown')
        self.assertEqual(len(analyze_experiments()['groups']), 2)
        for log in (recorded, unknown):
            self.assertEqual(before[log.id], ((log.folder / 'events.jsonl').read_bytes(), (log.folder / 'summary.json').read_bytes()))

    def test_initial_checkpoint_recovers_explicit_identity_missing_from_old_summary(self):
        log = self.record()
        initial = {'backend': backend('genesis', '1.4.5', 'genesis-task-v1'),
                   'model': 'panda', 'scene_id': 'tabletop', 'task_id': 'lift_red', 'state': [0.]}
        folder = log.folder / 'checkpoints'; folder.mkdir()
        (folder / 'initial.json').write_text(json.dumps(initial), encoding='utf-8')
        result = analyze_run(log.id)
        self.assertEqual(result['engine_id'], 'genesis')
        self.assertEqual(result['engine_version'], '1.4.5')
        self.assertEqual(result['evaluator_source'], 'genesis_privileged_state')

    def test_records_without_environment_fields_still_analyze_and_keep_missing_tokens_missing(self):
        log = RunLog('agent', {'task': 'legacy task', 'model': 'legacy-model'})
        log.metadata.pop('telemetry_schema_version')
        log.write('model_output', {'content': 'A historical response without usage'})
        log.finish({'status': 'error'})
        result = analyze_run(log.id)
        self.assertEqual(result['engine_id'], 'unknown')
        self.assertEqual(result['robot'], 'unknown')
        self.assertIsNone(result['total']['output_tokens'])
        groups = analyze_experiments(model='legacy-model')['groups']
        self.assertEqual(groups[0]['engine_id'], 'unknown')
        self.assertEqual(groups[0]['missing_output_runs'], 1)

    def test_legacy_frozen_protocol_provides_its_recorded_mujoco_version(self):
        log = RunLog('research_experiment', {'task': 'legacy protocol', 'protocol': {
            'spec': {'robot': 'so101', 'scene_id': 'drawer', 'task_id': 'open_drawer'},
            'contract': {'mujoco': '3.4.0'}}})
        log.finish({'status': 'error', 'rows': []})
        result = analyze_run(log.id)
        self.assertEqual((result['engine_id'], result['engine_version']), ('mujoco', '3.4.0'))
        self.assertEqual((result['robot'], result['scene_id'], result['task_id']), ('so101', 'drawer', 'open_drawer'))


if __name__ == '__main__': unittest.main()
