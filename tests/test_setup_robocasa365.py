"""Offline installer contracts using tiny local ZIPs, never real SDK/assets."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


TOOL = Path(__file__).resolve().parents[1]/'tools/setup-robocasa365.py'
spec = importlib.util.spec_from_file_location('setup_robocasa365', TOOL)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def tiny_zip(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return stream.getvalue()


class RoboCasaSetupTests(unittest.TestCase):
    def test_fixed_source_dry_run_only_prepares_two_repositories(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = [(url, revision, Path(directory)/folder.name) for url, revision, folder in setup.SOURCES]
            output = io.StringIO()
            with patch.object(setup, 'SOURCES', sources), contextlib.redirect_stdout(output):
                setup.prepare_sources(dry_run=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
        commands = [json.loads(row)['command'] for row in output.getvalue().splitlines()]
        self.assertEqual(len(commands), 6)
        self.assertEqual(commands[2][-1], '2ce1aecdd76e29b52d32fdb160436c5a3e230b07')
        self.assertEqual(commands[5][-1], '5ce6643f3092639d08f7b0f90ed1c6a84f50552c')
        self.assertTrue(all('--recursive' not in argv for argv in commands))

    def test_runtime_dry_run_has_separate_cpu_and_normal_index_transactions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch.object(setup, 'shutil') as shutil, contextlib.redirect_stdout(output):
                shutil.which.return_value = None
                setup.install_runtime(Path(directory)/'venv', uv=Path('uv'), dry_run=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
        commands = [json.loads(row)['command'] for row in output.getvalue().splitlines()]
        self.assertEqual(commands[0][1:4], ['venv', '--python', '3.11'])
        cpu = commands[1]
        self.assertIn('torch==2.7.1' + ('' if setup.sys.platform == 'darwin' else '+cpu'), cpu)
        self.assertIn('torchvision==0.22.1' + ('' if setup.sys.platform == 'darwin' else '+cpu'), cpu)
        normal = commands[2]
        self.assertIn(str(setup.ROOT/'requirements-robocasa365.txt'), normal)
        self.assertEqual(normal[-2:], ['--index-url', 'https://pypi.org/simple'])
        self.assertIn('check', commands[3])

    def test_official_metadata_pins_asset_revisions_and_correct_destinations(self):
        sha = 'a'*40
        def metadata(url):
            if '/tree/' not in url:
                return {'sha': sha}
            paths = (['textures.zip', 'fixtures.zip', 'objaverse.zip', 'generative_textures.zip']
                     if setup.AUTHOR_ASSETS in url else
                     ['fixtures_lightwheel/cabinets.zip'] if 'fixtures_lightwheel' in url else
                     ['objects_lightwheel/apple.zip'])
            return [{'type': 'file', 'path': name, 'size': 10, 'lfs': {'oid': 'b'*64}} for name in paths]
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, '_json_url', side_effect=metadata):
            root = Path(directory)
            kitchen = setup.asset_jobs(root, 'kitchen')
            complete = setup.asset_jobs(root, 'complete')
        self.assertEqual(len(kitchen), 4)
        self.assertEqual(len(complete), 5)
        self.assertTrue(all(job['revision'] == sha and job['expected_sha256'] == 'b'*64 for job in complete))
        self.assertFalse(any('generative' in job['path'] or 'aigen' in job['path'] for job in complete))
        objaverse = next(job for job in complete if job['path'] == 'objaverse.zip')
        self.assertEqual(Path(objaverse['destination']), root/'robocasa/models/assets/objects')

    def test_zip_validation_precedes_extraction_and_preserves_existing_different_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'download.zip'
            destination = root/'assets'
            archive.write_bytes(tiny_zip([('textures/ok.txt', b'new'), ('../outside.txt', b'bad')]))
            with self.assertRaisesRegex(ValueError, '不安全路径'):
                setup.extract_assets(archive, destination)
            self.assertFalse(destination.exists())
            self.assertFalse((root/'outside.txt').exists())
            archive.write_bytes(tiny_zip([('textures/existing.txt', b'official')]))
            existing = destination/'textures/existing.txt'
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b'user edit')
            with self.assertRaisesRegex(ValueError, '不会覆盖'):
                setup.extract_assets(archive, destination)
            self.assertEqual(existing.read_bytes(), b'user edit')
            existing.write_bytes(b'official')
            self.assertEqual(setup.extract_assets(archive, destination), ['textures/existing.txt'])

    def test_zip_symlinks_and_windows_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'download.zip'
            symlink = zipfile.ZipInfo('link')
            symlink.create_system = 3
            symlink.external_attr = 0o120777 << 16
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr(symlink, '../outside')
            with self.assertRaisesRegex(ValueError, '符号链接'):
                setup.extract_assets(archive, root/'assets')
            for name in ('C:/outside.txt', '..\\outside.txt', '/outside.txt'):
                archive.write_bytes(tiny_zip([(name, b'invalid')]))
                with self.assertRaises(ValueError):
                    setup.extract_assets(archive, root/'assets')

    def test_small_download_checks_official_size_hash_and_reuses_record(self):
        data = tiny_zip([('textures/sample.txt', b'official tiny fixture')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'repo': setup.AUTHOR_ASSETS, 'revision': 'a'*40, 'path': 'textures.zip',
                   'compressed_bytes': len(data), 'expected_sha256': hashlib.sha256(data).hexdigest(),
                   'destination': str(root/'robocasa/models/assets')}
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(data)) as get:
                record = setup.download_asset(job, root)
            self.assertIn('/resolve/'+'a'*40+'/textures.zip', get.call_args.args[0])
            self.assertEqual(record['verification'], 'official_lfs_sha256')
            self.assertEqual(record['files'], ['textures/sample.txt'])
            self.assertEqual(list((root/'downloads').iterdir()), [root/'downloads/textures.json'])
            with patch.object(setup.urllib.request, 'urlopen', side_effect=AssertionError('Already downloaded')):
                self.assertTrue(setup.download_asset(job, root)['reused'])

    def test_bad_download_does_not_publish_marker_or_extract(self):
        data = tiny_zip([('textures/sample.txt', b'fixture')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {'repo': setup.AUTHOR_ASSETS, 'revision': 'a'*40, 'path': 'textures.zip',
                   'compressed_bytes': len(data), 'expected_sha256': '0'*64,
                   'destination': str(root/'assets')}
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(data)):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    setup.download_asset(job, root)
            self.assertEqual(list((root/'downloads').iterdir()), [])
            self.assertFalse((root/'assets').exists())

    def test_legacy_marker_reuse_retains_unknown_revision_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root/'assets'
            destination.mkdir()
            (destination/'existing.txt').write_text('fixture')
            job = {'repo': setup.AUTHOR_ASSETS, 'revision': 'a'*40, 'path': 'textures.zip',
                   'compressed_bytes': 10, 'expected_sha256': None, 'destination': str(destination)}
            old = {key: value for key, value in job.items() if key not in ('revision', 'expected_sha256')}
            old.update({'sha256': 'b'*64, 'uncompressed_bytes': 25})
            (root/'downloads').mkdir()
            marker = root/'downloads/textures.json'
            marker.write_text(json.dumps(old), encoding='utf-8')
            with patch.object(setup.urllib.request, 'urlopen', side_effect=AssertionError('Do not redownload')):
                record = setup.download_asset(job, root)
            self.assertTrue(record['reused'])
            self.assertNotIn('revision', record)
            self.assertIn('legacy', record['verification'])
            self.assertEqual(json.loads(marker.read_text(encoding='utf-8')), old)

    def test_asset_dry_run_is_offline_and_never_touches_actual_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with patch.object(setup, 'prepare_sources'), \
                    patch.object(setup, 'cache_root', return_value=root/'cache'), \
                    patch.object(setup.urllib.request, 'urlopen', side_effect=AssertionError('Offline dry-run')), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(setup.main(['--stage', 'assets', '--asset-set', 'complete', '--dry-run']), 0)
            self.assertEqual(list(root.iterdir()), [])
            lines = [json.loads(row) for row in output.getvalue().splitlines()]
            assets = next(row for row in lines if 'asset_bytes_downloaded' in row)
            self.assertTrue(assets['includes_objaverse'])
            self.assertEqual(assets['asset_bytes_downloaded'], 0)
            self.assertFalse(lines[-1]['native_validated'])


if __name__ == '__main__':
    unittest.main()
