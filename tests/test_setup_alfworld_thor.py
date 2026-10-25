"""ALFWorld visual asset/install boundaries; no download, SDK or WSL process."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('setup_alfworld_thor', ROOT/'tools/setup-household.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
EPISODE = 'json_2.1.1/valid_unseen/pick_and_place_simple-Mug-None-Sink-1/trial_1'
GAME = {'pddl_domain': 'domain', 'pddl_problem': 'problem', 'grammar': 'grammar', 'solvable': True}
TRAJECTORY = {'task_type': 'pick_and_place_simple', 'scene': {'scene_num': 1},
              'pddl_params': {'object_target': 'Mug'}, 'plan': {}}


def archive_at(path, records):
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in records.items():
            archive.writestr(name, value if isinstance(value, str) else json.dumps(value))
    return path


class ALFWorldVisualAssetTests(unittest.TestCase):
    def test_visual_and_text_records_share_official_episode_and_preserve_user_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); destination = root/'data'
            games = archive_at(root/'text.zip', {EPISODE+'/game.tw-pddl': GAME})
            visual = archive_at(root/'visual.zip', {
                EPISODE+'/traj_data.json': TRAJECTORY,
                EPISODE.replace('valid_unseen', 'train')+'/traj_data.json': TRAJECTORY,
                EPISODE+'/raw_images/frame.png': 'not part of task metadata',
                '../escaped.json': 'outside',
                '/json_2.1.1/valid_unseen/task/trial/traj_data.json': TRAJECTORY,
                'json_2.1.1/valid_unseen/../trial/traj_data.json': TRAJECTORY,
                'json_2.1.1/valid_unseen/task\\outside/trial/traj_data.json': TRAJECTORY,
            })
            self.assertEqual(setup.extract_alfworld_games(games, destination), 1)
            self.assertEqual(setup.extract_alfworld_visual(visual, destination), 1)
            episode = destination/EPISODE
            self.assertEqual({path.name for path in episode.iterdir()}, {'game.tw-pddl', 'traj_data.json'})
            self.assertEqual(json.loads((episode/'traj_data.json').read_text()), TRAJECTORY)
            self.assertEqual(len(list(destination.rglob('traj_data.json'))), 1)
            self.assertFalse((root/'escaped.json').exists())
            self.assertFalse((destination/'json_2.1.1/train').exists())
            (episode/'traj_data.json').write_text('user-authored scene; preserve verbatim', encoding='utf-8')
            self.assertEqual(setup.extract_alfworld_visual(visual, destination), 1)
            self.assertEqual((episode/'traj_data.json').read_text(encoding='utf-8'),
                             'user-authored scene; preserve verbatim')
            self.assertEqual(json.loads((episode/'game.tw-pddl').read_text()), GAME)

    def test_wrong_corrupt_or_incomplete_archives_never_publish_visual_records(self):
        for records in ({'notes.txt': 'no tasks'}, {EPISODE+'/game.tw-pddl': GAME},
                        {EPISODE+'/traj_data.json': 'not json'},
                        {EPISODE+'/traj_data.json': {'scene': {}}}):
            with self.subTest(records=records), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); destination = root/'data'
                archive = archive_at(root/'invalid.zip', records)
                with self.assertRaises(ValueError): setup.extract_alfworld_visual(archive, destination)
                self.assertFalse((destination/EPISODE/'traj_data.json').exists())
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'corrupt.zip'; archive.write_bytes(b'not a ZIP')
            with self.assertRaises(zipfile.BadZipFile): setup.extract_alfworld_visual(archive, Path(folder)/'data')

    def test_existing_directory_symlink_cannot_redirect_asset_writes_outside_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); destination = root/'data'; outside = root/'outside'
            destination.mkdir(); outside.mkdir()
            sentinel = outside/'preserve.txt'; sentinel.write_text('user data')
            try: (destination/'json_2.1.1').symlink_to(outside, target_is_directory=True)
            except OSError as exc: self.skipTest(f'Directory symlinks unavailable: {exc}')
            archive = archive_at(root/'visual.zip', {EPISODE+'/traj_data.json': TRAJECTORY})
            with self.assertRaisesRegex(ValueError, '越界'):
                setup.extract_alfworld_visual(archive, destination)
            self.assertEqual(list(outside.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(), 'user data')


class ALFWorldVisualRuntimeTests(unittest.TestCase):
    def installer(self, root):
        # Configure the install operation directly; all external commands are
        # intercepted below, including interpreter and native SDK verification.
        installer = object.__new__(setup.RuntimeInstaller)
        installer.distribution, installer.home = 'Ubuntu', '/home/fixture'
        installer.prefix = ['wsl.exe', '--distribution', 'Ubuntu', '--exec']
        installer.uv = '/home/fixture/.local/bin/uv'
        installer.archive = archive_at(root/'text.zip', {EPISODE+'/game.tw-pddl': GAME})
        installer.visual_archive = archive_at(root/'visual.zip', {EPISODE+'/traj_data.json': TRAJECTORY})
        return installer

    def external_commands(self, recorded, *, probe_version='0.4.2', fail_probe=False, fail_install=False):
        def run(arguments, **kwargs):
            argv = [str(arg) for arg in arguments]; recorded.append(argv)
            if argv[-1] == '--version': return SimpleNamespace(returncode=1, stdout='')
            if fail_install and 'install' in argv:
                raise subprocess.CalledProcessError(1, argv, stderr='fixture pip failure')
            output = ''
            if '-c' in argv:
                program = argv[argv.index('-c')+1]
                if program.startswith('from importlib.metadata import version;'):
                    self.assertIn("version('alfworld')", program)
                    output = '0.4.2\n'
                elif 'prepare_build' in program:
                    output = ''
                else:
                    self.assertIn('ALFWorldThorSimulation', program)
                    if fail_probe: raise subprocess.CalledProcessError(1, argv, stderr='fixture native failure')
                    output = setup.RESULT_PREFIX+json.dumps({'version': probe_version,
                                                            'image_shape': [480, 640, 3], 'cameras': 1})+'\n'
            return SimpleNamespace(returncode=0, stdout=output)
        return run

    def test_separate_python311_and_alfworld_package_mapping_stay_unverified_without_native_probe(self):
        requirements = (ROOT/'requirements-alfworld-thor.txt').read_text(encoding='utf-8').splitlines()
        self.assertIn('alfworld==0.4.2', requirements)
        self.assertIn('ai2thor==2.1.0', requirements)
        self.assertIn('numpy==1.23.5', requirements)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); installer = self.installer(root); commands = []
            with patch.object(setup, 'ROOT', root), patch.object(setup.subprocess, 'run',
                    side_effect=self.external_commands(commands)), \
                    patch.object(setup.urllib.request, 'urlopen', side_effect=AssertionError('No download allowed')), \
                    contextlib.redirect_stdout(io.StringIO()):
                installer.install('alfworld_thor', verify=False)
            environment = '/home/fixture/.local/share/embodiworks/alfworld_thor'
            self.assertTrue(any(argv[-4:] == ['venv', '--python', '3.11', environment] for argv in commands))
            installs = [argv for argv in commands if 'install' in argv]
            self.assertEqual(len(installs), 1)
            self.assertEqual(installs[0][installs[0].index('--python')+1], environment+'/bin/python')
            self.assertEqual(installs[0][installs[0].index('-r')+1], 'requirements-alfworld-thor.txt')
            marker = json.loads((root/'.venv-alfworld_thor/runtime.json').read_text())
            self.assertEqual((marker['package'], marker['version']), ('alfworld', '0.4.2'))
            self.assertEqual(marker['python'], environment+'/bin/python')
            self.assertFalse(marker['validated']); self.assertFalse(marker['assets_ready'])
            self.assertTrue((root/'vendor/alfworld_data'/EPISODE/'traj_data.json').is_file())
            self.assertTrue((root/'vendor/alfworld_data'/EPISODE/'game.tw-pddl').is_file())

    def test_successful_probe_marks_its_package_version_but_mismatch_does_not(self):
        for probe_version in ('0.4.2', 'wrong-version'):
            with self.subTest(version=probe_version), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); installer = self.installer(root); commands = []
                with patch.object(setup, 'ROOT', root), patch.object(setup.subprocess, 'run',
                        side_effect=self.external_commands(commands, probe_version=probe_version)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if probe_version == '0.4.2': installer.install('alfworld_thor')
                    else:
                        with self.assertRaisesRegex(RuntimeError, '版本'):
                            installer.install('alfworld_thor')
                marker = root/'.venv-alfworld_thor/runtime.json'
                if probe_version == '0.4.2':
                    value = json.loads(marker.read_text())
                    self.assertTrue(value['validated']); self.assertTrue(value['assets_ready'])
                elif marker.exists():
                    self.assertFalse(json.loads(marker.read_text())['validated'])

    def test_failed_reinstall_invalidates_previous_success_marker(self):
        for phase in ('install', 'probe'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                root = Path(folder); installer = self.installer(root); commands = []
                marker = root/'.venv-alfworld_thor/runtime.json'; marker.parent.mkdir()
                python = '/home/fixture/.local/share/embodiworks/alfworld_thor/bin/python'
                marker.write_text(json.dumps({'kind': 'wsl', 'distribution': 'Ubuntu',
                    'python': python, 'package': 'alfworld', 'version': '0.4.2',
                    'validated': True, 'assets_ready': True}))
                with patch.object(setup, 'ROOT', root), patch.object(setup.subprocess, 'run',
                        side_effect=self.external_commands(commands, fail_probe=phase == 'probe',
                                                          fail_install=phase == 'install')), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(subprocess.CalledProcessError): installer.install('alfworld_thor')
                value = json.loads(marker.read_text())
                self.assertFalse(value['validated'], 'A failed reinstall must not retain old native verification')
                self.assertFalse(value['assets_ready'])
                self.assertEqual((value['distribution'], value['python']), ('Ubuntu', python))


if __name__ == '__main__': unittest.main()
