"""Plan, run or import pinned BEHAVIOR evaluations from the same workbench code."""
import argparse
import asyncio
import json
from pathlib import Path
import signal

from server.behavior_benchmark import BehaviorBenchmarkController, import_result, plan


async def run(spec):
    controller = BehaviorBenchmarkController()
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(lambda: asyncio.create_task(controller.stop())))
    try:
        state = await controller.start(spec)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if controller.task:
            try:
                await controller.task
            except asyncio.CancelledError:
                pass
        state = controller.snapshot()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state['status'] == 'complete' else 1
    finally:
        await controller.stop()
        signal.signal(signal.SIGINT, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'run'):
        item = sub.add_parser(name)
        item.add_argument('--spec', type=Path, required=True, help='BehaviorBenchmarkSpec JSON file')
    item = sub.add_parser('import')
    item.add_argument('--result', type=Path, required=True, help='Official per-rollout JSON object or list')
    item.add_argument('--policy-name', default='external-policy')
    item.add_argument('--mode', choices=('train', 'public_test'), default='public_test')
    args = parser.parse_args()
    try:
        if args.command == 'import':
            result = import_result({'payload': json.loads(args.result.read_text(encoding='utf-8-sig')),
                                    'policy_name': args.policy_name, 'mode': args.mode})
        else:
            spec = json.loads(args.spec.read_text(encoding='utf-8-sig'))
            if args.command == 'run':
                return asyncio.run(run(spec))
            result = plan(spec)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
