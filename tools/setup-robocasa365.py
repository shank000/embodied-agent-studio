#!/usr/bin/env python3
"""Prepare official RoboCasa365 sources, an isolated runtime, and kitchen assets.

Stages are explicit; the default downloads source code, not assets or weights.
  python tools/setup-robocasa365.py --dry-run
  python tools/setup-robocasa365.py --stage source
  python tools/setup-robocasa365.py --stage runtime --python /path/to/python3.11
  python tools/setup-robocasa365.py --stage assets --asset-set complete

Set EMBODI_ROBOCASA_CACHE in both the installer and workbench environments to
override the shared cache base. Windows requires an ASCII cache path. Kitchen
assets use the authors' and NVIDIA's official Hugging Face mirrors; complete
also includes Objaverse objects used by the default native object registry.
No demonstrations, policy checkpoints, AI-generated textures or objects are
downloaded, and no SDK source or global Python settings are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '2ce1aecdd76e29b52d32fdb160436c5a3e230b07'
SUITE_COMMIT = '5ce6643f3092639d08f7b0f90ed1c6a84f50552c'
SOURCES = (
    ('https://github.com/robocasa/robocasa.git', SOURCE_COMMIT, ROOT/'vendor/robocasa365'),
    ('https://github.com/ARISE-Initiative/robosuite.git', SUITE_COMMIT, ROOT/'vendor/robosuite365'),
)
AUTHOR_ASSETS = 'robocasa/robocasa-assets'
NVIDIA_ASSETS = 'nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF'


def run(arguments, *, capture=False, dry_run=False):
    argv = list(map(str, arguments))
    if dry_run:
        print(json.dumps({'command': argv}, ensure_ascii=False))
        return ''
    result = subprocess.run(argv, cwd=ROOT, check=True, text=True, encoding='utf-8',
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None,
                            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    return result.stdout.strip() if capture else ''


def prepare_sources(*, dry_run=False, prepare=True):
    for url, revision, folder in SOURCES:
        current = None
        if folder.exists():
            if not (folder/'.git').exists():
                raise ValueError(f'已有目录不是独立 Git 工作区，已保留：{folder}')
            actual = Path(run(['git', '-C', folder, 'rev-parse', '--show-toplevel'], capture=True)).resolve()
            if actual != folder.resolve():
                raise ValueError(f'源码目录不是 Git 工作区根目录：{folder}')
            if run(['git', '-C', folder, 'status', '--porcelain', '--untracked-files=no'], capture=True):
                raise ValueError(f'源码有未提交修改；请先保存或提交，工具不会重置：{folder}')
            current = run(['git', '-C', folder, 'rev-parse', 'HEAD'], capture=True)
        if current == revision:
            continue
        if not prepare:
            raise ValueError(f'缺少固定源码 {folder.name}@{revision}；请先执行 --stage source。')
        if not folder.exists():
            run(['git', 'clone', '--no-checkout', '--filter=blob:none', '--', url, folder], dry_run=dry_run)
        run(['git', '-C', folder, 'fetch', '--depth', '1', url, revision], dry_run=dry_run)
        run(['git', '-C', folder, 'checkout', '--no-overwrite-ignore', '--detach', revision], dry_run=dry_run)


def venv_python(folder):
    return Path(folder)/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def cache_root(venv):
    # Same platform/path contract as robocasa_backend.runtime_cache(), evaluated
    # against the target venv rather than the installer's unrelated sys.prefix.
    default = Path(Path(venv).anchor)/'EmbodiWorksCache/robocasa365' if os.name == 'nt' else Path.home()/'.cache/embodiworks/robocasa365'
    folder = Path(os.getenv('EMBODI_ROBOCASA_CACHE', str(default)))/SOURCE_COMMIT[:12]
    if not folder.is_absolute():
        raise ValueError('EMBODI_ROBOCASA_CACHE 必须为绝对路径。')
    if os.name == 'nt' and not str(folder).isascii():
        raise ValueError('Windows 原生资产读取要求英文缓存路径；请设置 EMBODI_ROBOCASA_CACHE。')
    return folder


def _python311(python):
    value = json.loads(run([python, '-I', '-c', 'import sys,json;print(json.dumps(list(sys.version_info[:2])))'], capture=True))
    if value != [3, 11]:
        raise ValueError(f'RoboCasa365 独立环境要求 Python 3.11，当前为 {value}。')


def install_runtime(venv, *, python=None, uv=None, dry_run=False):
    executable = venv_python(venv)
    uv_path = str(uv) if uv else shutil.which('uv')
    if not executable.is_file():
        if Path(venv).exists() and any(Path(venv).iterdir()):
            raise ValueError(f'独立环境目录已有内容但没有 Python；已保留：{venv}')
        if python:
            base = Path(python).expanduser()
            if not base.is_absolute() or not base.is_file():
                raise ValueError('--python 需要已有 Python 3.11 的绝对文件路径。')
            if not dry_run:
                _python311(base)
            run([base, '-m', 'venv', venv], dry_run=dry_run)
        elif uv_path:
            run([uv_path, 'venv', '--python', '3.11', venv], dry_run=dry_run)
        else:
            raise ValueError('未找到 uv 或 Python 3.11；请用 --python 指定现有解释器，或 --uv 指定 uv 可执行文件。')
    if not dry_run:
        _python311(executable)
    pip = [uv_path, 'pip', 'install', '--python', executable] if uv_path else [executable, '-m', 'pip', 'install']
    cpu_suffix = '' if sys.platform == 'darwin' else '+cpu'
    torch_arguments = [f'torch==2.7.1{cpu_suffix}', f'torchvision==0.22.1{cpu_suffix}']
    if sys.platform != 'darwin':
        torch_arguments += ['--index-url', 'https://download.pytorch.org/whl/cpu']
    run(pip+torch_arguments, dry_run=dry_run)
    # Install the official editable sources and complete upstream dependencies.
    # A separate normal-index transaction avoids uv's multi-index resolution
    # choosing a NumPy candidate from the CPU wheel index.
    run(pip+['-r', ROOT/'requirements-robocasa365.txt', '--index-url', 'https://pypi.org/simple'], dry_run=dry_run)
    check = [uv_path, 'pip', 'check', '--python', executable] if uv_path else [executable, '-m', 'pip', 'check']
    run(check, dry_run=dry_run)


def prepare_cache(folder, *, dry_run=False):
    for package, source in (('robocasa', SOURCES[0][2]/'robocasa'), ('robosuite', SOURCES[1][2]/'robosuite')):
        target = folder/package
        marker = target/'.embodiworks-source-copied'
        if marker.is_file():
            continue
        if dry_run:
            print(json.dumps({'copy_official_source': str(source), 'destination': str(target)}, ensure_ascii=False))
        else:
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            marker.write_text(str(source), encoding='utf-8')


def _json_url(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'EmbodiWorks-RoboCasa-Setup/1'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def asset_jobs(folder, asset_set='kitchen'):
    """Resolve official dataset revisions and compressed sizes; no asset bytes."""
    jobs = []
    for repo in (AUTHOR_ASSETS, NVIDIA_ASSETS):
        revision = _json_url('https://huggingface.co/api/datasets/'+repo).get('sha')
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError(f'官方资产库没有返回固定 revision：{repo}')
        groups = [('', folder/'robocasa/models/assets')] if repo == AUTHOR_ASSETS else [
            ('fixtures_lightwheel', folder/'robocasa/models/assets/fixtures'),
            ('objects_lightwheel', folder/'robocasa/models/assets/objects/lightwheel')]
        for group, destination in groups:
            url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{group}?recursive=false&expand=false'
            entries = _json_url(url)
            selected = {'textures.zip', 'fixtures.zip'}
            if asset_set == 'complete':
                selected.add('objaverse.zip')
            found = set()
            for item in entries:
                path = item.get('path', '')
                if item.get('type') != 'file' or not path.endswith('.zip'):
                    continue
                if repo == AUTHOR_ASSETS and path not in selected:
                    continue
                relative = PurePosixPath(path)
                if relative.is_absolute() or '..' in relative.parts or '\\' in path or ':' in path:
                    raise ValueError(f'官方资产元数据含无效路径：{path}')
                if group and relative.parent.as_posix() != group:
                    raise ValueError(f'官方资产元数据不在请求目录中：{path}')
                size = item.get('size')
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ValueError(f'官方资产元数据缺少压缩大小：{path}')
                digest = (item.get('lfs') or {}).get('oid')
                digest = digest.removeprefix('sha256:') if isinstance(digest, str) else None
                if digest is not None and not re.fullmatch(r'[0-9a-f]{64}', digest):
                    raise ValueError(f'官方资产元数据含无效 LFS 哈希：{path}')
                actual_destination = destination/'objects' if path == 'objaverse.zip' else destination
                jobs.append({'repo': repo, 'revision': revision, 'path': path, 'compressed_bytes': size,
                             'expected_sha256': digest, 'destination': str(actual_destination)})
                found.add(path)
            if (repo == AUTHOR_ASSETS and found != selected) or (group and not found):
                raise ValueError(f'官方资产镜像缺少请求的厨房资源：{repo}/{group}')
    return jobs


def _safe_members(archive, destination):
    root = Path(destination).resolve()
    members = archive.infolist()
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if (path.is_absolute() or any(part in ('.', '..') for part in name.rstrip('/').split('/'))
                or '\\' in name or ':' in name or stat.S_ISLNK(member.external_attr >> 16)
                or not (root/Path(*path.parts)).resolve().is_relative_to(root)):
            raise ValueError(f'资产 ZIP 含不安全路径或符号链接：{name}')
    return members


def extract_assets(archive_path, destination):
    """Merge official archives while preserving any differing existing file."""
    destination = Path(destination)
    files = []
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive, destination)
        for member in members:
            target = destination/Path(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                with archive.open(member) as source, target.open('rb') as current:
                    if hashlib.file_digest(source, 'sha256').digest() != hashlib.file_digest(current, 'sha256').digest():
                        raise ValueError(f'资产目标已有不同内容，工具不会覆盖：{target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                created = False
                try:
                    with archive.open(member) as source:
                        with target.open('xb') as output:
                            created = True
                            shutil.copyfileobj(source, output, length=1024*1024)
                except Exception:
                    # Only our newly created, incomplete file is removed.
                    if created and target.is_file():
                        target.unlink()
                    raise
            files.append(member.filename)
    return files


def download_asset(job, folder):
    downloads = folder/'downloads'
    archive_path = downloads/job['path'].replace('/', '_')
    marker = archive_path.with_suffix('.json')
    destination = Path(job['destination'])
    if marker.is_file():
        record = json.loads(marker.read_text(encoding='utf-8'))
        same = all(record.get(key) == job[key] for key in ('repo', 'path', 'compressed_bytes', 'destination'))
        digest = record.get('sha256', '')
        members = record.get('files')
        present = destination.is_dir() and (all((destination/name).is_file() for name in members)
                                             if isinstance(members, list) and members else next(destination.iterdir(), None) is not None)
        revision_matches = 'revision' not in record or record['revision'] == job['revision']
        hash_matches = not job['expected_sha256'] or digest == job['expected_sha256']
        if same and present and revision_matches and hash_matches and re.fullmatch(r'[0-9a-f]{64}', digest):
            return {**record, 'reused': True,
                    'verification': 'recorded_revision_and_local_download_hash' if 'revision' in record
                                    else 'legacy_download_record_without_dataset_revision; extracted_files_not_rehashed'}
    downloads.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{job['repo']}/resolve/{job['revision']}/{urllib.parse.quote(job['path'], safe='/')}?download=true"
    temporary = None
    try:
        with urllib.request.urlopen(url, timeout=90) as response, tempfile.NamedTemporaryFile(
                dir=downloads, suffix='.partial', delete=False) as output:
            temporary = Path(output.name)
            count = 0
            while chunk := response.read(1024*1024):
                count += len(chunk)
                if count > job['compressed_bytes']:
                    raise ValueError(f"资产下载超过官方元数据大小：{job['path']}")
                output.write(chunk)
        if count != job['compressed_bytes']:
            raise ValueError(f"资产下载不完整：{job['path']} ({count}/{job['compressed_bytes']})")
        with temporary.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if job['expected_sha256'] and digest != job['expected_sha256']:
            raise ValueError(f"资产 SHA256 与官方 LFS 元数据不一致：{job['path']}")
        files = extract_assets(temporary, destination)
        record = {**job, 'sha256': digest, 'files': files, 'reused': False,
                  'verification': 'official_lfs_sha256' if job['expected_sha256'] else 'official_size_and_recorded_local_sha256'}
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        return record
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['source', 'runtime', 'assets', 'all'], default='source')
    parser.add_argument('--python', type=Path, help='创建独立环境所用的已有 Python 3.11 绝对路径')
    parser.add_argument('--uv', type=Path, help='可选 uv 可执行文件，不修改系统 uv/Python')
    parser.add_argument('--venv', type=Path, default=ROOT/'.venv-robocasa365')
    parser.add_argument('--asset-set', choices=['kitchen', 'complete'], default='kitchen',
                        help='kitchen: 厨房基础资产与 Lightwheel；complete: 额外下载默认物品任务所需 Objaverse')
    parser.add_argument('--dry-run', action='store_true', help='离线输出准备步骤，不写文件、联网或导入 SDK')
    args = parser.parse_args(argv)
    try:
        venv = args.venv.expanduser().absolute()
        folder = cache_root(venv)
        prepare_sources(dry_run=args.dry_run, prepare=args.stage in ('source', 'all') or args.dry_run)
        if args.stage in ('runtime', 'all'):
            install_runtime(venv, python=args.python, uv=args.uv, dry_run=args.dry_run)
            prepare_cache(folder, dry_run=args.dry_run)
        if args.stage in ('assets', 'all'):
            prepare_cache(folder, dry_run=args.dry_run)
            if args.dry_run:
                print(json.dumps({'asset_set': args.asset_set, 'cache': str(folder),
                                  'official_repositories': [AUTHOR_ASSETS, NVIDIA_ASSETS],
                                  'includes_objaverse': args.asset_set == 'complete',
                                  'asset_bytes_downloaded': 0}, ensure_ascii=False))
            else:
                jobs = asset_jobs(folder, args.asset_set)
                print(json.dumps({'archives': len(jobs), 'compressed_bytes': sum(job['compressed_bytes'] for job in jobs),
                                  'cache': str(folder), 'asset_set': args.asset_set}, ensure_ascii=False), flush=True)
                records = []
                for job in jobs:
                    record = download_asset(job, folder)
                    records.append(record)
                    print(json.dumps({key: record.get(key) for key in ('repo', 'path', 'compressed_bytes', 'reused', 'verification')}, ensure_ascii=False), flush=True)
                (folder/'assets-manifest.json').write_text(json.dumps({'source_commit': SOURCE_COMMIT,
                    'asset_set': args.asset_set, 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'stage': args.stage, 'dry_run': args.dry_run, 'cache': str(folder),
                          'native_validated': False,
                          'notice': '准备成功；真实原生加载、任务成功率以工作台实际实验为准。无训练数据或预训练策略下载。'}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f'RoboCasa365 准备失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
