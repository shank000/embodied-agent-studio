"""Native termination evidence remains consistent through records and training export."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.rounds import evaluation_snapshot, transition_evidence
from server.runlog import RunLog
from server.task_verification import environment_failure, environment_success
from server.training_data import export_transitions


def snapshot(**evaluation):
    return {'ready': True, 'time': 1, 'backend': {'engine_id': 'robodojo', 'action_mode': 'tcp_pose'},
            'evaluation': {'source': 'robodojo_official_state', 'success': False,
                           'terminated': False, **evaluation}}


class TerminalContractTests(unittest.TestCase):
    def test_controller_and_round_agree_on_native_failure_success_and_budget(self):
        before = evaluation_snapshot(snapshot())
        scenarios = [
            (snapshot(success=True, terminated=True), True, False, 'task_success'),
            (snapshot(terminated=True), True, False, 'task_failure'),
            (snapshot(task_graph={'terminal_failure': True, 'failure_reason': 'unsafe_contact'}),
             True, False, 'unsafe_contact'),
            (snapshot(truncated=True, termination_reason='native_time_limit'),
             False, True, 'native_time_limit'),
            (snapshot(), False, False, None),
        ]
        for state, terminal, truncated, reason in scenarios:
            with self.subTest(reason=reason):
                row = transition_evidence(before, evaluation_snapshot(state), 2)
                self.assertEqual(row['terminated'], terminal)
                self.assertEqual(row['truncated'], truncated)
                self.assertEqual(row['termination_reason'], reason)
                self.assertEqual(environment_success(state) or environment_failure(state), terminal)
                self.assertEqual(row['reward'], float(environment_success(state)))

    def test_untrusted_or_unavailable_evaluation_does_not_invent_labels(self):
        before = evaluation_snapshot(snapshot())
        untrusted = snapshot(terminated=True, success=True, source='model_claim')
        unavailable = snapshot(terminated=True)
        unavailable['ready'] = False
        for state in (untrusted, unavailable):
            with self.subTest(state=state):
                row = transition_evidence(before, evaluation_snapshot(state), 1)
                self.assertIsNone(row['terminated'])
                self.assertIsNone(row['truncated'])
                self.assertIsNone(row['reward'])

    def record(self, before, after, status, *, row_status='complete', old_bug=False):
        log = RunLog('agent', {'task': 'native ending fixture'})
        log.write('round_start', {'round': 1, 'before': before})
        for event in ('observation', 'observation_after'):
            log.write(event, {'round': 1, 'observation': {'cameras': []}})
        log.write('action', {'round': 1, 'action': {'status': 'move'}})
        evidence = transition_evidence(before, after, 1)
        if old_bug: evidence.update(terminated=False, termination_reason=None)
        log.write('round_end', {'round': 1, 'status': row_status, 'after': after, **evidence})
        log.finish({'status': status})
        return log

    def test_export_preserves_failure_budget_cancel_and_unknown_without_rewriting(self):
        before = evaluation_snapshot(snapshot())
        cases = [
            (snapshot(terminated=True), 'failed', 'complete', True, False, True),
            (snapshot(success=True), 'verified', 'verified', True, False, True),
            (snapshot(), 'limit', 'complete', False, True, True),
            (snapshot(truncated=True), 'limit', 'complete', False, True, True),
            (snapshot(), 'stopped', 'cancelled', False, True, False),
            (snapshot(), 'error', 'error', False, True, False),
            (snapshot(source='untrusted'), 'error', 'error', None, None, False),
        ]
        with tempfile.TemporaryDirectory() as directory, patch('server.runlog.RUNS', Path(directory)):
            for state, status, row_status, terminal, truncated, eligible in cases:
                with self.subTest(status=status, state=state):
                    log = self.record(before, evaluation_snapshot(state), status, row_status=row_status)
                    original = log.path.read_bytes()
                    row = export_transitions(log.id)['transitions'][0]
                    self.assertEqual(row['terminated'], terminal)
                    self.assertEqual(row['truncated'], truncated)
                    self.assertEqual(row['training_eligible'], eligible)
                    if status in ('stopped', 'error'): self.assertEqual(row['termination_reason'], status)
                    self.assertEqual(log.path.read_bytes(), original)

    def test_existing_native_failure_record_is_corrected_from_trusted_evidence_on_export(self):
        with tempfile.TemporaryDirectory() as directory, patch('server.runlog.RUNS', Path(directory)):
            log = self.record(evaluation_snapshot(snapshot()), evaluation_snapshot(snapshot(terminated=True)),
                              'failed', old_bug=True)
            original = log.path.read_bytes()
            row = export_transitions(log.id)['transitions'][0]
            self.assertTrue(row['terminated'])
            self.assertFalse(row['truncated'])
            self.assertEqual(row['termination_reason'], 'task_failure')
            self.assertEqual(log.path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
