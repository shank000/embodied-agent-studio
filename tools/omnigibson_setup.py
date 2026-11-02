"""Register an already installed BEHAVIOR v3.9.4 runtime; no installation or license acceptance."""
import argparse
import json
import subprocess
import sys

from server.omnigibson import SCHEMA, configure, probe_runtime, readiness, runtime_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', help='Absolute path to the independent Python 3.11 interpreter')
    parser.add_argument('--source-root', help='Absolute path to the official BEHAVIOR-1K v3.9.4 checkout')
    parser.add_argument('--data-path', help='Directory containing behavior-1k-assets and omnigibson-robot-assets')
    parser.add_argument('--check', action='store_true', help='Read-only checks; do not update configuration')
    parser.add_argument('--replace', action='store_true', help='Update a recognized config and keep a backup')
    args = parser.parse_args(argv)
    try:
        paths = (args.python, args.source_root, args.data_path)
        if any(paths) and not all(paths):
            raise ValueError('--python、--source-root 和 --data-path 必须一起提供。')
        if args.check and args.replace:
            raise ValueError('--check 不会写入配置，不能与 --replace 同用。')
        config = ({'schema_version': SCHEMA, 'kind': 'native', 'python': args.python,
                   'source_root': args.source_root, 'data_path': args.data_path} if all(paths) else runtime_config())
        if args.check:
            if config:
                config = {**config, 'probe': probe_runtime(config)}
            result = readiness(config)
        elif all(paths):
            result = configure(config, replace=args.replace)
        else:
            raise ValueError('请提供三个绝对路径以注册运行时，或使用 --check 检查现有准备状态。')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['ready'] else 2
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'ready': False, 'error': str(exc)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
