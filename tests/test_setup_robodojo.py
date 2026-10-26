"""Source/runtime setup contracts; no network, assets, policy weights or SDKs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1]/'tools/setup-robodojo.py'
spec = importlib.util.spec_from_file_location('setup_robodojo', TOOL)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class RoboDojoSetupTests(unittest.TestCase):
    def test_new_source_dry_run_has_fixed_source_and_one_gitlink_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'source with spaces'
            with patch.object(setup, 'git', side_effect=AssertionError('Dry-run must not fetch')):
                report = setup.prepare_source(target, dry_run=True)
            self.assertFalse(target.exists())
        self.assertEqual(report['revision'], 'f57e1633ffed59761bc8e568cfe7530c4b33381c')
        self.assertEqual(report['xpolicy_revision'], '5d0be1499634d3176258c8f2e31f5fe816497b0a')
        self.assertFalse(report['source_prepared'])
        commands = report['commands']
        self.assertEqual(commands[0][0:4], ['git', 'clone', '--no-checkout', '--filter=blob:none'])
        self.assertIn(setup.SOURCE_URL, commands[0])
        self.assertIn(setup.REVISION, commands[1])
        self.assertIn('--no-overwrite-ignore', commands[2])
        self.assertIn(f'submodule.XPolicyLab.url={setup.XPOLICY_URL}', commands[3])
        self.assertEqual(commands[3][-2:], ['--', 'XPolicyLab'])
        self.assertFalse(any('--remote' in row or '--recursive' in row for row in commands))
        self.assertFalse(any('IsaacLab' in arg or 'curobo' in arg or 'pip' == arg
                             for row in commands for arg in row))

    def test_existing_pinned_source_and_xpolicy_are_not_fetched_or_rechecked_out(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for relative in ('scripts/internal/task_inventory.py', 'src/eval_client/main.py',
                             'XPolicyLab/XPolicyLab.py', 'env_cfg/arx_x5.yml'):
                target = source/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('fixture', encoding='utf-8')
            (source/'XPolicyLab/.git').write_text('gitdir: fixture', encoding='utf-8')
            states = lambda path: setup.XPOLICY_REVISION if path.name == 'XPolicyLab' else setup.REVISION
            gitlink = f'160000 commit {setup.XPOLICY_REVISION}\tXPolicyLab'
            with patch.object(setup, '_checkout_state', side_effect=states), patch.object(setup, 'git', return_value=gitlink) as git, \
                    patch.object(setup, 'normalize_shell_scripts', return_value=[]):
                result = setup.prepare_source(source)
            self.assertTrue(result['source_prepared'])
            self.assertEqual(result['commands'], [])
            # macOS temporary paths may use /var -> /private/var aliases.
            # The installer intentionally validates and uses the real root.
            self.assertEqual(result['source_root'], str(source.resolve()))
            git.assert_called_once_with('-C', source.resolve(), 'ls-tree', setup.REVISION, '--', 'XPolicyLab')

    def test_windows_checkout_scripts_get_lf_without_overwriting_user_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            def git(*args):
                subprocess.run(['git', '-C', str(source), *args], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git('init')
            git('config', 'core.autocrlf', 'true')
            script = source/'entry.sh'
            original = b'#!/usr/bin/env bash\nset -euo pipefail\n'
            script.write_bytes(original)
            git('add', 'entry.sh')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
            script.write_bytes(original.replace(b'\n', b'\r\n'))
            self.assertEqual(setup.normalize_shell_scripts(source), ['entry.sh'])
            self.assertEqual(script.read_bytes(), original)
            self.assertEqual(setup.normalize_shell_scripts(source), [])
            self.assertEqual(setup._checkout_state(source), setup.git('-C', source, 'rev-parse', 'HEAD'))
            edited = original.replace(b'\n', b'\r\n') + b'echo user change\r\n'
            script.write_bytes(edited)
            with self.assertRaisesRegex(ValueError, '用户修改'):
                setup.normalize_shell_scripts(source)
            self.assertEqual(script.read_bytes(), edited)
            with self.assertRaisesRegex(ValueError, '未提交修改'):
                setup._checkout_state(source)

    def test_tracked_source_edits_stop_without_reset_or_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source/'.git').mkdir()
            edited = source/'user-edit.py'
            edited.write_text('preserve me', encoding='utf-8')
            with patch.object(setup, 'git', side_effect=[str(source), ' M user-edit.py']) as git:
                with self.assertRaisesRegex(ValueError, '未提交修改'):
                    setup.prepare_source(source)
            self.assertEqual(edited.read_text(encoding='utf-8'), 'preserve me')
            self.assertEqual(git.call_count, 2)

    def test_existing_non_git_content_and_unexpected_gitlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with patch.object(setup, 'git') as git:
                with self.assertRaisesRegex(ValueError, '不是独立 Git'):
                    setup.prepare_source(source)
                git.assert_not_called()
            with patch.object(setup, '_checkout_state', return_value=setup.REVISION), \
                    patch.object(setup, 'git', return_value='160000 commit unexpected XPolicyLab'):
                with self.assertRaisesRegex(ValueError, 'gitlink'):
                    setup.prepare_source(source, dry_run=True)

    def test_runtime_paths_are_host_absolute_and_interpreter_probe_is_sdk_free(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            payload = setup.runtime_payload(source, Path(sys.executable).absolute())
            self.assertEqual(set(payload), {'kind', 'source_root', 'python'})
            self.assertEqual(payload['kind'], 'native')
            self.assertEqual(payload['source_root'], str(source.resolve()))
            self.assertEqual(payload['python'], str(Path(sys.executable).absolute()))
            with self.assertRaisesRegex(ValueError, '绝对路径'):
                setup.runtime_payload(source, 'relative/python')
            with self.assertRaisesRegex(ValueError, '不存在或不可执行'):
                setup.runtime_payload(source, source/'absent-python')
            with self.assertRaisesRegex(ValueError, '源码目录不存在'):
                setup.runtime_payload(source/'absent-source', sys.executable)

    def test_runtime_write_round_trips_exact_contract_and_does_not_accept_license(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            payload = setup.runtime_payload(source, sys.executable)
            marker = source/'.venv-robodojo/runtime.json'
            setup.write_runtime(payload, marker)
            self.assertEqual(json.loads(marker.read_text(encoding='utf-8')), payload)
            self.assertEqual(list(marker.parent.iterdir()), [marker])
            with self.assertRaisesRegex(ValueError, '目标不是文件'):
                setup.write_runtime(payload, marker.parent)
            self.assertNotIn('ACCEPT_EULA', marker.read_text(encoding='utf-8'))

    def test_cli_dry_run_preserves_existing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            marker = source/'runtime.json'
            marker.write_text('existing configuration', encoding='utf-8')
            prepared = {'source_root': str(source), 'revision': setup.REVISION,
                        'xpolicy_revision': setup.XPOLICY_REVISION, 'commands': [], 'source_prepared': False}
            stream = io.StringIO()
            with patch.object(setup, 'prepare_source', return_value=prepared), contextlib.redirect_stdout(stream):
                result = setup.main(['--source-root', str(source), '--python', sys.executable,
                                     '--runtime-file', str(marker), '--dry-run'])
            self.assertEqual(result, 0)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'existing configuration')
            report = json.loads(stream.getvalue())
            self.assertTrue(report['dry_run'])
            self.assertFalse(report['runtime_configured'])
            self.assertFalse(report['sdk_validated'])
            self.assertEqual(report['runtime']['kind'], 'native')

    def test_cli_invalid_python_stops_before_source_changes(self):
        stream = io.StringIO()
        with patch.object(setup, 'prepare_source') as prepare, contextlib.redirect_stderr(stream):
            result = setup.main(['--python', 'relative/python'])
        self.assertEqual(result, 1)
        prepare.assert_not_called()
        self.assertIn('绝对路径', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
