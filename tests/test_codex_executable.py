"""Native Codex lookup fixtures; no CLI or model is executed."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from server.agent import codex_executable

REAL_WHICH = shutil.which
NPM_SHELL_SHIM = '''#!/bin/sh
basedir=$(dirname "$0")
exec node "$basedir/node_modules/@openai/codex/bin/codex.js" "$@"
'''


class CodexExecutableTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='codex-lookup-')
        self.addCleanup(temporary.cleanup)
        # Windows TEMP can contain an 8.3 alias (e.g. RUNNER~1), while the
        # production lookup resolves paths to their long spelling.
        self.root = Path(temporary.name).resolve()
        self.package = self.root / 'node_modules' / '@openai' / 'codex'
        self.addCleanup(patch.stopall)
        patch('server.agent.__file__', str(self.root / 'server' / 'agent.py')).start()
        patch.dict(os.environ, {'CODEX_CLI_PATH': ''}).start()
        self.which = patch('server.agent.shutil.which', return_value=None).start()

    def native(self, vendor, target, filename):
        path = vendor / target / 'bin' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture only; never executed')
        return path

    def test_project_packages_cover_official_windows_linux_and_macos_targets(self):
        targets = [('win32', 'AMD64', 'win32-x64', 'x86_64-pc-windows-msvc', 'codex.exe'),
                   ('win32', 'ARM64', 'win32-arm64', 'aarch64-pc-windows-msvc', 'codex.exe'),
                   ('linux', 'x86_64', 'linux-x64', 'x86_64-unknown-linux-musl', 'codex'),
                   ('linux', 'aarch64', 'linux-arm64', 'aarch64-unknown-linux-musl', 'codex'),
                   ('darwin', 'x86_64', 'darwin-x64', 'x86_64-apple-darwin', 'codex'),
                   ('darwin', 'arm64', 'darwin-arm64', 'aarch64-apple-darwin', 'codex')]
        for system, machine, package_name, target, filename in targets:
            with self.subTest(system=system, machine=machine), \
                    patch('server.agent.sys.platform', system), patch('server.agent.platform.machine', return_value=machine):
                native = self.native(self.package.parent / ('codex-' + package_name) / 'vendor', target, filename)
                self.assertEqual(Path(codex_executable()).resolve(), native.resolve())
        self.which.assert_not_called()

    def test_nested_optional_package_and_bundled_vendor_layouts(self):
        target = 'x86_64-unknown-linux-musl'
        for vendor in [self.package / 'node_modules/@openai/codex-linux-x64/vendor', self.package / 'vendor']:
            with self.subTest(vendor=vendor), patch('server.agent.sys.platform', 'linux'), \
                    patch('server.agent.platform.machine', return_value='x86_64'):
                native = self.native(vendor, target, 'codex')
                try:
                    self.assertEqual(Path(codex_executable()).resolve(), native.resolve())
                finally:
                    native.unlink()

    def test_global_npm_shims_resolve_native_program_without_a_shell(self):
        global_root = self.root / 'global'
        package = global_root / 'node_modules/@openai/codex'
        with patch('server.agent.sys.platform', 'win32'), patch('server.agent.platform.machine', return_value='AMD64'):
            shim = global_root / 'codex.cmd'
            self.which.side_effect = lambda name, **kwargs: str(shim) if Path(name).name == 'codex' else None
            native = self.native(package.parent / 'codex-win32-x64/vendor', 'x86_64-pc-windows-msvc', 'codex.exe')
            self.assertEqual(Path(codex_executable()).resolve(), native.resolve())
        with patch('server.agent.sys.platform', 'darwin'), patch('server.agent.platform.machine', return_value='arm64'):
            entry = package / 'bin/codex.js'
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text('// fixture', encoding='utf-8')
            self.which.side_effect = None
            self.which.return_value = str(entry)
            native = self.native(package.parent / 'codex-darwin-arm64/vendor', 'aarch64-apple-darwin', 'codex')
            self.assertEqual(Path(codex_executable()).resolve(), native.resolve())

    def test_absent_or_incomplete_install_has_an_actionable_error(self):
        with patch('server.agent.sys.platform', 'linux'), patch('server.agent.platform.machine', return_value='x86_64'):
            with self.assertRaisesRegex(ValueError, 'docs/INSTALL.md'):
                codex_executable()
            self.which.return_value = str(self.root / 'bin/codex.js')
            with self.assertRaisesRegex(ValueError, 'CODEX_CLI_PATH'):
                codex_executable()

    def test_explicit_native_override_is_preserved(self):
        with patch.dict(os.environ, {'CODEX_CLI_PATH': str(self.root / 'custom-codex')}):
            self.assertEqual(codex_executable(), str(self.root / 'custom-codex'))

    def test_real_path_skips_foreign_npm_shell_shim_for_later_linux_native(self):
        foreign = self.root / 'windows-node'
        foreign.mkdir()
        shim = foreign / 'codex'
        shim.write_text(NPM_SHELL_SHIM, encoding='utf-8')
        shim.chmod(0o755)
        self.native(foreign / 'node_modules/@openai/codex-win32-x64/vendor',
                    'x86_64-pc-windows-msvc', 'codex.exe')
        native_dir = self.root / 'linux-bin'
        native_dir.mkdir()
        native = native_dir / 'codex'
        native.write_bytes(b'\x7fELFfixture; never executed')
        native.chmod(0o755)
        with patch('server.agent.sys.platform', 'linux'), \
                patch('server.agent.platform.machine', return_value='x86_64'), \
                patch('server.agent.shutil.which', REAL_WHICH), \
                patch.dict(os.environ, {'PATH': os.pathsep.join([str(foreign), str(native_dir)])}):
            self.assertEqual(Path(codex_executable()).resolve(), native.resolve())

    def test_real_path_npm_shell_shim_resolves_current_os_or_reports_missing_package(self):
        directory = self.root / 'npm-bin'
        directory.mkdir()
        shim = directory / 'codex'
        shim.write_text(NPM_SHELL_SHIM, encoding='utf-8')
        shim.chmod(0o755)
        with patch('server.agent.sys.platform', 'linux'), \
                patch('server.agent.platform.machine', return_value='x86_64'), \
                patch('server.agent.shutil.which', REAL_WHICH), \
                patch.dict(os.environ, {'PATH': str(directory)}):
            with self.assertRaisesRegex(ValueError, r'linux/x64.*CODEX_CLI_PATH'):
                codex_executable()
            native = self.native(directory / 'node_modules/@openai/codex-linux-x64/vendor',
                                 'x86_64-unknown-linux-musl', 'codex')
            self.assertEqual(Path(codex_executable()).resolve(), native.resolve())


if __name__ == '__main__':
    unittest.main()
