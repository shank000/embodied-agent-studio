#!/usr/bin/env python3
"""Prepare pinned RoboDojo source, without installing SDKs or downloading assets.

    python tools/setup-robodojo.py --dry-run
    python tools/setup-robodojo.py
    python tools/setup-robodojo.py --python /absolute/linux/environment/bin/python

The last command records an existing interpreter in .venv-robodojo/runtime.json.
It does not validate Isaac Sim, accept its license, or mark the simulator ready.
Run it again on the Linux host where the workbench and RoboDojo will execute.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://github.com/RoboDojo-Benchmark/RoboDojo.git'
REVISION = 'f57e1633ffed59761bc8e568cfe7530c4b33381c'
XPOLICY_URL = 'https://github.com/XPolicyLab/XPolicyLab.git'
XPOLICY_REVISION = '5d0be1499634d3176258c8f2e31f5fe816497b0a'
DEFAULT_SOURCE = ROOT/'vendor/robodojo'
DEFAULT_RUNTIME = ROOT/'.venv-robodojo/runtime.json'


def git(*arguments):
    """Use argument arrays throughout; no shell, activation script or SDK import."""
    result = subprocess.run(['git', *map(str, arguments)], check=True, text=True,
                            encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def _checkout_state(path):
    if not (path/'.git').exists():
        raise ValueError(f'已有目录不是独立 Git 工作区，已保留原内容：{path}')
    actual_root = Path(git('-C', path, 'rev-parse', '--show-toplevel')).resolve()
    if actual_root != path.resolve():
        raise ValueError(f'源码目录不是 Git 工作区根目录：{path}')
    # Untracked assets/results are user data. Do not remove them, or treat them
    # as tracked source edits. Each requested submodule is checked separately.
    if git('-C', path, 'status', '--porcelain', '--untracked-files=no', '--ignore-submodules=all'):
        raise ValueError(f'源码存在未提交修改，请先保存或提交；工具不会重置它们：{path}')
    return git('-C', path, 'rev-parse', 'HEAD')


def prepare_source(source_root, *, dry_run=False):
    """Fetch only the fixed source and XPolicyLab gitlink; preserve local edits."""
    source = Path(source_root).expanduser().resolve()
    commands = []

    def execute(arguments):
        commands.append(['git', *map(str, arguments)])
        if not dry_run:
            git(*arguments)

    exists = source.exists()
    current = _checkout_state(source) if exists else None
    xpolicy = source/'XPolicyLab'
    # Check this before changing the parent checkout, so a user's policy edits
    # are preserved even when the parent needs to move to the pinned revision.
    xpolicy_current = _checkout_state(xpolicy) if (xpolicy/'.git').exists() else None
    if xpolicy.exists() and any(xpolicy.iterdir()) and xpolicy_current is None:
        raise ValueError(f'XPolicyLab 目录已有非 Git 内容，已保留：{xpolicy}')
    if current != REVISION:
        if not exists:
            execute(['clone', '--no-checkout', '--filter=blob:none', '--', SOURCE_URL, source])
        execute(['-C', source, 'fetch', '--depth', '1', SOURCE_URL, REVISION])
        execute(['-C', source, 'checkout', '--no-overwrite-ignore', '--detach', REVISION])
    if not dry_run or current == REVISION:
        gitlink = git('-C', source, 'ls-tree', REVISION, '--', 'XPolicyLab').split()
        if gitlink != ['160000', 'commit', XPOLICY_REVISION, 'XPolicyLab']:
            raise ValueError('固定 RoboDojo 提交中的 XPolicyLab gitlink 与已审核版本不一致。')
    if xpolicy_current != XPOLICY_REVISION:
        # The official parent gitlink selects the exact commit. Do not use
        # --remote/--recursive: those would track main or prepare other SDKs.
        execute(['-c', f'submodule.XPolicyLab.url={XPOLICY_URL}', '-C', source,
                 'submodule', 'update', '--init', '--depth', '1', '--checkout', '--', 'XPolicyLab'])
    if not dry_run:
        if _checkout_state(source) != REVISION or _checkout_state(xpolicy) != XPOLICY_REVISION:
            raise ValueError('源码提交核验失败；未写入运行时配置。')
        for relative in ('scripts/internal/task_inventory.py', 'src/eval_client/main.py',
                         'XPolicyLab/XPolicyLab.py', 'env_cfg/arx_x5.yml'):
            if not (source/relative).exists():
                raise ValueError(f'官方源码缺少所需入口：{relative}')
    return {'source_root': str(source), 'revision': REVISION,
            'xpolicy_revision': XPOLICY_REVISION, 'source_prepared': not dry_run,
            'commands': commands}


def runtime_payload(source_root, python):
    """Validate paths on this host, without importing any optional packages."""
    source, executable = Path(source_root).expanduser(), Path(python).expanduser()
    if not source.is_absolute() or not executable.is_absolute():
        raise ValueError('source_root 与 Python 必须为当前运行主机上的绝对路径。')
    if not source.is_dir():
        raise ValueError(f'源码目录不存在：{source}')
    if not executable.is_file() or (os.name != 'nt' and not os.access(executable, os.X_OK)):
        raise ValueError(f'Python 解释器不存在或不可执行：{executable}')
    probe = subprocess.run([str(executable), '-I', '-c',
                            'import json,sys;print(json.dumps(list(sys.version_info[:2])))'],
                           check=True, text=True, encoding='utf-8', stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=15)
    version = json.loads(probe.stdout)
    if not isinstance(version, list) or len(version) != 2 or tuple(version) < (3, 11):
        raise ValueError('官方 RoboDojo 源码要求 Python >= 3.11；请指定匹配的独立运行环境。')
    # Preserve the interpreter's venv entry path: resolving its symlink on Linux
    # can select the base Python and lose the environment's installed packages.
    return {'kind': 'native', 'source_root': str(source.resolve()), 'python': str(executable.absolute())}


def write_runtime(payload, runtime_file):
    target = Path(runtime_file).expanduser().absolute()
    if target.exists() and not target.is_file():
        raise ValueError(f'运行时配置目标不是文件：{target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.runtime-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE,
                        help='官方源码工作区（默认 vendor/robodojo）')
    parser.add_argument('--python', type=Path,
                        help='已有独立 Python 的本机绝对路径；指定后写入 native runtime.json')
    parser.add_argument('--runtime-file', type=Path, default=DEFAULT_RUNTIME,
                        help='本机运行时配置路径（默认 .venv-robodojo/runtime.json）')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅输出固定源码准备命令和可选配置，不写文件或启动 SDK')
    args = parser.parse_args(argv)
    try:
        # Reject missing/relative interpreter paths before source mutations.
        payload = None
        if args.python is not None:
            # New source directories are created by prepare_source; validate
            # their existence after preparation, but reject relative Python now.
            if not args.python.expanduser().is_absolute():
                raise ValueError('Python 必须为当前运行主机上的绝对路径。')
            if not args.python.expanduser().is_file():
                raise ValueError(f'Python 解释器不存在：{args.python}')
        report = prepare_source(args.source_root, dry_run=args.dry_run)
        if args.python is not None:
            if args.dry_run and not Path(report['source_root']).exists():
                payload = {'kind': 'native', 'source_root': report['source_root'], 'python': str(args.python.absolute())}
            else:
                payload = runtime_payload(report['source_root'], args.python)
        report.update({'dry_run': args.dry_run, 'runtime': payload, 'runtime_configured': False,
                       'sdk_validated': False,
                       'notice': '未安装 Isaac Sim/IsaacLab/curobo、未下载仿真资产、未接受许可。原生仿真需官方 Linux GPU 运行栈；源码就绪不表示 SDK 已可运行。'})
        if payload is not None:
            report['runtime_file'] = str(args.runtime_file.expanduser().absolute())
            if not args.dry_run:
                report['runtime_file'] = write_runtime(payload, args.runtime_file)
                report['runtime_configured'] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        print(f'RoboDojo 准备失败：{detail}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
