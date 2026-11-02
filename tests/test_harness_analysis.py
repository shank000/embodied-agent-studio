"""Same-model trials with different harnesses must not be pooled as one method."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.analysis import analyze_experiments, analyze_run
from server.runlog import RunLog


class HarnessAnalysisTests(unittest.TestCase):
    def test_harness_grouping_and_legacy_codex_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch('server.runlog.RUNS', Path(directory)):
            for harness in ('workbench', 'codex_cli', None):
                metadata = {'model': 'fixture', 'provider': 'codex' if harness is None else 'openai',
                            'task': 'fixture', 'settings': {'reasoning': 'high'}}
                if harness: metadata['harness'] = harness
                log = RunLog('agent', metadata)
                log.finish({'status': 'stopped', 'metrics': {'success': False}})
                self.assertEqual(analyze_run(log.id)['harness'], harness or 'codex_cli')
            result = analyze_experiments(model='fixture', task='fixture')
            self.assertEqual({row['harness']: row['runs'] for row in result['groups']},
                             {'workbench': 1, 'codex_cli': 2})
