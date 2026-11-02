"""Readiness and launch contracts; no SDK or licensed assets are loaded."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from server import omnigibson, simulators
from tools import omnigibson_setup


class OmniGibsonRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='og-runtime-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = self.root/'environment'/'python.exe'
        self.python.parent.mkdir()
        self.python.write_bytes(b'fixture only, never executed')
        self.source, self.data = self.root/'BEHAVIOR-1K', self.root/'data'
        (self.source/'OmniGibson/omnigibson').mkdir(parents=True)
        (self.source/'bddl3').mkdir()
        for folder, version in [('behavior-1k-assets', '3.9.0'), ('omnigibson-robot-assets', '3.8.2')]:
            path = self.data/folder
            path.mkdir(parents=True)
            (path/'VERSION').write_text(version)
        (self.data/'omnigibson.key').write_bytes(b'test fixture; not a real key')
        scene = self.data/'2026-challenge-task-instances/scenes/house_double_floor_lower/json'
        scene.mkdir(parents=True)
        (scene/'house_double_floor_lower_task_picking_up_trash_0_0_template.json').write_text('{}')
        self.config = {'schema_version': omnigibson.SCHEMA, 'kind': 'native', 'python': str(self.python),
                       'source_root': str(self.source), 'data_path': str(self.data)}
        self.probe = {'signature': omnigibson._signature(self.config), 'packages_ready': True,
                      'revision': omnigibson.REVISION, 'gpu': {'name': 'NVIDIA RTX test', 'memory_mib': 24576}}

    def readiness(self, config=None, memory=64, platform='linux'):
        # This complete native-runtime fixture represents a supported Linux
        # host, regardless of the operating system running the contract tests.
        with patch.object(omnigibson.sys, 'platform', platform), \
             patch.object(omnigibson, 'total_memory_gib', return_value=memory):
            return omnigibson.readiness(config or {**self.config, 'probe': self.probe})

    def test_readiness_never_launches_sdk_and_distinguishes_unverified_native_run(self):
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('No subprocess for catalog')):
            result = self.readiness()
        self.assertTrue(result['ready'])
        self.assertEqual(result['native_validation'], 'not_recorded')
        self.assertIn('仍待验收', result['message'])

    def test_macos_remains_unavailable_even_with_complete_runtime_fixture(self):
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('No subprocess for catalog')):
            result = self.readiness(platform='darwin')
        self.assertFalse(result['ready'])
        missing = [item for item in result['checks'] if item['status'] != 'passed']
        self.assertEqual(missing, [{'id': 'os', 'label': 'Windows / Linux', 'status': 'missing', 'detail': 'darwin'}])
        self.assertEqual(result['native_validation'], 'not_recorded')
        self.assertIn('不会改用其他引擎', result['message'])

    def test_low_resources_remain_unavailable_with_specific_reasons(self):
        self.probe['gpu']['memory_mib'] = 8192
        result = self.readiness(memory=15.8)
        self.assertFalse(result['ready'])
        missing = {item['id']: item['detail'] for item in result['checks'] if item['status'] != 'passed'}
        self.assertIn('15.8', missing['ram'])
        self.assertIn('8192', missing['gpu'])

    def test_changed_interpreter_invalidates_stored_dependency_probe(self):
        self.python.write_bytes(b'changed')
        result = self.readiness()
        self.assertFalse(result['ready'])
        self.assertEqual(next(x for x in result['checks'] if x['id'] == 'dependencies')['status'], 'missing')

    def test_missing_asset_version_does_not_claim_readiness(self):
        (self.data/'behavior-1k-assets/VERSION').unlink()
        self.assertFalse(self.readiness()['ready'])

    def test_unknown_config_and_shell_interpreters_are_rejected(self):
        for config in [[], self.config|{'command': 'powershell'}, self.config|{'python': str(self.root/'cmd.exe')},
                       self.config|{'python': 'relative/python'}, self.config|{'kind': 'wsl'}, self.config|{'probe': []}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                omnigibson.validate_config(config)

    def test_probe_checks_pinned_source_and_package_origin_without_importing_sdk(self):
        packages = {'python': [3, 11], 'omnigibson': '3.9.3', 'isaacsim': '5.1.0.0', 'bddl': str(self.source/'bddl3/bddl/__init__.py'),
                    'sdk_origin': str(self.source/'OmniGibson/omnigibson/__init__.py'), 'pillow': '11.0.0', 'torch': '2.7.0+cu128'}
        responses = [subprocess.CompletedProcess([], 0, json.dumps(packages), ''),
                     subprocess.CompletedProcess([], 0, omnigibson.REVISION+'\n', ''),
                     subprocess.CompletedProcess([], 0, 'NVIDIA RTX fixture, 24576\n', '')]
        with patch.object(subprocess, 'run', side_effect=responses) as run, patch.dict('os.environ', {'OMNIGIBSON_GPU_ID': '0'}):
            probe = omnigibson.probe_runtime(self.config)
        self.assertTrue(probe['packages_ready'])
        self.assertEqual(probe['gpu']['memory_mib'], 24576)
        self.assertEqual(run.call_args_list[0].args[0][:3], [str(self.python), '-I', '-c'])
        self.assertNotIn('import omnigibson', run.call_args_list[0].args[0][3])
        self.assertFalse(any(call.kwargs.get('shell') for call in run.call_args_list))

    def test_wrong_dependency_or_source_revision_fails_before_registration(self):
        packages = {'python': [3, 12], 'omnigibson': '3.9.3', 'isaacsim': '6.0.1.0'}
        with patch.object(subprocess, 'run', side_effect=[subprocess.CompletedProcess([], 0, json.dumps(packages), ''),
                                                        subprocess.CompletedProcess([], 0, 'other\n', '')]):
            with self.assertRaisesRegex(ValueError, '版本不匹配'):
                omnigibson.probe_runtime(self.config)

    def test_register_preserves_existing_unknown_configuration(self):
        marker = self.root/'runtime.json'
        original = b'{"future-schema": true}'
        marker.write_bytes(original)
        with patch.object(omnigibson, 'RUNTIME_FILE', marker), patch.object(omnigibson, 'probe_runtime', return_value=self.probe):
            with self.assertRaisesRegex(ValueError, '未知字段'):
                omnigibson.configure(self.config, replace=True)
        self.assertEqual(marker.read_bytes(), original)

    def test_registered_paths_are_preserved_and_read_only_check_does_not_write(self):
        marker = self.root/'config/runtime.json'
        with patch.object(omnigibson, 'RUNTIME_FILE', marker), patch.object(omnigibson, 'probe_runtime', return_value=self.probe), \
             patch.object(omnigibson.sys, 'platform', 'linux'), \
             patch.object(omnigibson, 'total_memory_gib', return_value=64):
            self.assertTrue(omnigibson.configure(self.config)['ready'])
            before = marker.read_bytes()
            with patch.object(omnigibson_setup, 'probe_runtime', return_value=self.probe), patch('builtins.print'):
                self.assertEqual(omnigibson_setup.main(['--check']), 0)
            self.assertEqual(marker.read_bytes(), before)

    def test_launch_uses_exact_configured_interpreter_and_rechecks_health(self):
        info = self.config|{'installed': True, 'readiness': {'ready': True, 'checks': []}}
        with patch.object(simulators, 'runtime_info', return_value=info), \
             patch.object(omnigibson, 'runtime_config', return_value=self.config), \
             patch.object(omnigibson, 'probe_runtime', return_value=self.probe), \
             patch.object(omnigibson, 'readiness', return_value={'ready': True, 'checks': []}):
            command = simulators.runtime_command('omnigibson', generation_base=9)
        self.assertEqual(command[0], str(self.python))
        self.assertEqual(command[-4:], ['--engine', 'omnigibson', '--generation-base', '9'])


if __name__ == '__main__':
    unittest.main()
