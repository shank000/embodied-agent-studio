#!/usr/bin/env python3
"""Install the official ALFWorld/AI2-THOR runtimes without training weights.

Run with the workbench Python. Windows uses an existing WSL Ubuntu distribution;
Linux/macOS use a project-local environment. No system packages are modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ASSET_URL = 'https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip'
RESULT_PREFIX = 'EMBODI_HOUSEHOLD_RESULT='


def extract_alfworld_games(archive, destination):
    """Extract only official unseen task games, preserving existing local files."""
    destination = Path(destination).resolve()
    count = 0
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            if (len(path.parts) != 5 or path.parts[:2] != ('json_2.1.1', 'valid_unseen')
                    or path.name != 'game.tw-pddl' or any(part in ('', '.', '..') for part in path.parts)
                    or '\\' in item.filename):
                continue
            target = (destination/Path(*path.parts)).resolve()
            if not target.is_relative_to(destination): raise ValueError('官方数据归档含越界路径，已停止解压。')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                data = source.read(item)
                game = json.loads(data)
                if not all(key in game for key in ('pddl_domain', 'pddl_problem', 'grammar', 'solvable')):
                    raise ValueError(f'官方回合缺少必需字段：{item.filename}')
                # Exclusive creation keeps a concurrent user's existing file.
                try:
                    with target.open('xb') as output: output.write(data)
                except FileExistsError: pass
            count += 1
    if not count: raise ValueError('归档中没有官方 json_2.1.1/valid_unseen/*/*/game.tw-pddl。')
    return count


def alfworld_assets(archive=None):
    destination = ROOT/'vendor/alfworld_data'
    if archive is not None:
        return extract_alfworld_games(archive, destination)
    cache = ROOT/'vendor/alfworld-download/json_2.1.3_tw-pddl.zip'
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f'下载官方 ALFWorld 文字任务归档（约 36.5 MB）：{ASSET_URL}', flush=True)
        temporary = None
        try:
            with urllib.request.urlopen(ASSET_URL, timeout=60) as response, tempfile.NamedTemporaryFile(
                    dir=cache.parent, suffix='.partial', delete=False) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output, length=1024*1024)
            # Validate before publishing the downloaded cache.
            with zipfile.ZipFile(temporary) as source: source.infolist()
            temporary.replace(cache)
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()
    count = extract_alfworld_games(cache, destination)
    print(f'已准备 {count} 个官方 valid_unseen 回合；未下载训练集、视觉检测权重或预训练策略。', flush=True)
    return count


class RuntimeInstaller:
    def __init__(self, distribution=None, archive=None):
        self.distribution = distribution
        self.archive = archive
        self.prefix = []
        if os.name == 'nt':
            if not distribution: raise ValueError('Windows 需要已安装的 WSL 发行版，例如 --wsl Ubuntu。')
            self.prefix = ['wsl.exe', '--distribution', distribution, '--exec']
            self.home = self.capture(['/usr/bin/printenv', 'HOME']).strip()
            if not self.home.startswith('/'): raise ValueError('WSL 未返回有效 HOME 路径。')
            linux_root = self.capture(['wslpath', '-u', str(ROOT)]).strip()
            self.prefix = ['wsl.exe', '--distribution', distribution, '--cd', linux_root, '--exec']
            probe = subprocess.run(self.prefix+['/usr/bin/which', 'uv'], capture_output=True, text=True)
            self.uv = probe.stdout.strip() if probe.returncode == 0 else self.home+'/.local/bin/uv'
        else:
            if distribution: raise ValueError('--wsl 仅用于 Windows；Linux/macOS 直接安装独立环境。')
            self.uv = shutil.which('uv')
            if not self.uv: raise ValueError('未找到 uv。请先按 https://docs.astral.sh/uv/getting-started/installation/ 安装 uv。')
        try: self.run([self.uv, '--version'])
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('运行环境中未找到可用 uv。请先安装 uv，再重试；工具不会修改系统 Python。') from exc

    def run(self, arguments):
        return subprocess.run(self.prefix+[str(arg) for arg in arguments], cwd=ROOT, check=True,
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})

    def capture(self, arguments):
        result = subprocess.run(self.prefix+[str(arg) for arg in arguments], cwd=ROOT, check=True,
                                stdout=subprocess.PIPE, text=True, encoding='utf-8',
                                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        return result.stdout

    def install(self, engine, verify=True):
        folder = self.home+'/.local/share/embodiworks/'+engine if self.prefix else str(ROOT/f'.venv-{engine}')
        python = folder+'/bin/python'
        exists = subprocess.run(self.prefix+[python, '--version'], capture_output=True).returncode == 0 if self.prefix else Path(python).is_file()
        if not exists: self.run([self.uv, 'venv', '--python', '3.12', folder])
        requirement = 'requirements-alfworld.txt' if engine == 'alfworld' else 'requirements-thor.txt'
        self.run([self.uv, 'pip', 'install', '--python', python, '-r', requirement, '--index-url', 'https://pypi.org/simple'])
        self.run([self.uv, 'pip', 'check', '--python', python])
        version = self.capture([python, '-c', f'from importlib.metadata import version;print(version({engine!r}))']).strip()
        if engine == 'alfworld': alfworld_assets(self.archive)
        validated = False
        if verify:
            module, name = ('alfworld_backend', 'ALFWorldSimulation') if engine == 'alfworld' else ('thor_backend', 'ThorSimulation')
            probe = (
                f'from server.backends.{module} import {name};import json;'
                f's={name}();'
                '\ntry:\n'
                ' o=s.observe();image=s.render_rgb();assert len(o["environment"]["admissible_commands"])>0;'
                'assert image.ndim==3 and image.shape[2]==3;'
                f'print({RESULT_PREFIX!r}+json.dumps({{"version":s.backend["engine_version"],"image_shape":list(image.shape),"cameras":len(o["cameras"])}}))'
                '\nfinally:s.close()\n'
            )
            print(f'实际初始化 {engine}、读取观察与画面后关闭。AI2-THOR 首次会下载官方 Unity 构建（约 769 MB）。', flush=True)
            output = self.capture([python, '-u', '-c', probe])
            print(output, end='')
            lines = [line[len(RESULT_PREFIX):] for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
            if not lines: raise RuntimeError('原生 SDK 验证未返回完整结果，未标记为可用。')
            report = json.loads(lines[-1])
            validated = report.get('version') == version
            if not validated: raise RuntimeError('原生 SDK 版本与安装版本不一致，未标记为可用。')
        if self.prefix:
            marker_folder = ROOT/f'.venv-{engine}'
            marker_folder.mkdir(exist_ok=True)
            marker = {'kind': 'wsl', 'distribution': self.distribution, 'python': python,
                      'package': engine, 'version': version, 'validated': validated, 'assets_ready': validated}
            # This is local machine configuration, never committed to Git.
            (marker_folder/'runtime.json').write_text(json.dumps(marker, indent=2), encoding='utf-8')
        print(f'{engine} {version}：'+('已通过真实启动验证。' if validated else '已安装；尚未运行验证，请去掉 --skip-verify 后重试。'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['alfworld', 'ai2thor', 'all'], required=True)
    parser.add_argument('--wsl', default='Ubuntu' if os.name == 'nt' else None, help='现有 WSL 发行版名称；不会安装/修改 WSL')
    parser.add_argument('--archive', type=Path, help='已有官方 ALFWorld json_2.1.3_tw-pddl.zip，免重复下载')
    parser.add_argument('--skip-verify', action='store_true', help='只安装；WSL 后端未通过验证时不会标为可用')
    args = parser.parse_args()
    installer = RuntimeInstaller(args.wsl, args.archive)
    for engine in (['alfworld', 'ai2thor'] if args.engine == 'all' else [args.engine]):
        installer.install(engine, verify=not args.skip_verify)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    try: main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f'安装未完成：{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
