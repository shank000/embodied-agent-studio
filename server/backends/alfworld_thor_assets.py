"""Install the official legacy Unity build without touching AI2-THOR's cache.

AI2-THOR 2.1's default downloader prunes other releases in ~/.ai2thor/releases.
Keep this build independent and use ThorEnv's public build_path parameter.
Preparation is explicit; opening an experiment never downloads a simulator.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import tempfile
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen
import zipfile


def _build_spec():
    if version('ai2thor') != '2.1.0':
        raise RuntimeError('ALFWorld 视觉构建需要独立 ai2thor==2.1.0 环境。')
    from ai2thor._builds import BUILDS
    system = platform.system()
    if system not in ('Linux', 'Darwin') or system not in BUILDS:
        raise RuntimeError('官方 AI2-THOR 2.1 构建仅支持 Linux/macOS；Windows 请使用独立 WSL 环境。')
    spec = BUILDS[system]
    parsed = urlsplit(spec['url'])
    checksum = spec.get('sha256', '')
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or not re.fullmatch(r'[0-9a-f]{64}', checksum):
        raise RuntimeError('已安装 AI2-THOR SDK 缺少有效的官方构建 URL/SHA256。')
    name = PurePosixPath(parsed.path).name
    if not re.fullmatch(r'thor-[A-Za-z0-9-]+\.zip', name):
        raise RuntimeError('已安装 AI2-THOR SDK 的构建文件名无效。')
    build = name[:-4]
    executable = Path(build) if system == 'Linux' else Path(build+'.app')/'Contents/MacOS'/build
    return build, urlunsplit(parsed._replace(scheme='https')), checksum, executable


def _root(cache_root):
    return Path(cache_root or Path.home()/'.cache/embodiworks/alfworld_thor').expanduser().resolve()


def executable_path(cache_root=None):
    """Return an already prepared executable; never invoke the SDK downloader."""
    build, _, _, relative = _build_spec()
    executable = _root(cache_root)/build/relative
    if not executable.is_file():
        raise RuntimeError('ALFWorld 官方视觉构建尚未安装，请运行环境安装工具准备独立 Unity 2.1 构建：'+str(executable))
    return executable


def _check_archive(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError('ALFWorld Unity 归档 SHA256 与官方 SDK 不一致，未安装；请检查或重新下载该归档：'+str(path))


def _extract(archive, destination):
    # Validate the complete member list before writing anything. Reject links
    # so a member cannot redirect a later write outside the staging directory.
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in member.filename or ':' in member.filename:
                raise RuntimeError('官方构建归档包含越界路径，未安装。')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError('官方构建归档包含符号链接，未安装。')
            if not (destination/member.filename).resolve().is_relative_to(destination.resolve()):
                raise RuntimeError('官方构建归档包含越界路径，未安装。')
        source.extractall(destination)


def prepare_build(archive_path=None, cache_root=None):
    """Verify/download and atomically install the SDK's pinned native build.

    archive_path can point at an existing download; it is verified and preserved.
    An existing incomplete destination is reported, never overwritten/deleted.
    """
    build, url, checksum, relative = _build_spec()
    root = _root(cache_root)
    target = root/build
    if (target/relative).is_file():
        return target/relative
    if target.exists():
        raise RuntimeError('ALFWorld 构建目录已存在但缺少可执行文件；已保留原文件，请先检查：'+str(target))
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(archive_path).expanduser().resolve() if archive_path else root/(build+'.zip')
    if not archive.is_file():
        if archive_path:
            raise FileNotFoundError('指定的 ALFWorld 构建归档不存在：'+str(archive))
        with tempfile.TemporaryDirectory(prefix='download-', dir=root) as temporary:
            downloaded = Path(temporary)/archive.name
            with urlopen(url, timeout=120) as response, downloaded.open('wb') as output:
                if urlsplit(response.geturl()).scheme != 'https':
                    raise RuntimeError('官方构建下载跳转到了非 HTTPS 地址，未接受该下载。')
                shutil.copyfileobj(response, output, length=1024*1024)
            _check_archive(downloaded, checksum)
            if not archive.exists():
                downloaded.rename(archive)
    _check_archive(archive, checksum)
    with tempfile.TemporaryDirectory(prefix='install-', dir=root) as temporary:
        staging = Path(temporary)/build
        staging.mkdir()
        _extract(archive, staging)
        executable = staging/relative
        if not executable.is_file():
            raise RuntimeError('官方 Unity 归档未包含预期可执行文件：'+str(relative))
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        # No replacement of user files or other simulator releases.
        if target.exists():
            raise RuntimeError('ALFWorld 构建目录在安装期间已出现，未覆盖：'+str(target))
        staging.rename(target)
    return target/relative
