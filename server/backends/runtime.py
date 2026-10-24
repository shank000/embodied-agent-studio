"""JSON-lines entry point for allowlisted native engines in isolated environments."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import sys
import traceback


def image_png(rgb):
    from PIL import Image
    import numpy as np
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or not all(value.shape[:2]):
        raise ValueError('视频帧必须为非空 uint8 RGB 图像。')
    output = io.BytesIO()
    Image.fromarray(value).save(output, 'PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def native_state(simulation):
    return {**simulation.state(), 'render_suspended': simulation.render_suspended,
            'ready': True}


class Samples:
    """Collect physics samples only while the parent is recording an episode."""
    def __init__(self, simulation, recording):
        if recording is not None and not isinstance(recording, dict):
            raise ValueError('录制采样配置必须为对象。')
        self.simulation = simulation
        self.enabled = recording is not None
        self.values = []
        config = recording or {}
        self.video_enabled = bool(config.get('video_enabled', False))
        self.fps = config.get('video_fps', 20.)
        self.next_time = config.get('next_video_time')
        if self.video_enabled:
            if isinstance(self.fps, bool) or not isinstance(self.fps, (float, int)) or not math.isfinite(self.fps) or not 1 <= self.fps <= 60:
                raise ValueError('视频帧率须为 1–60 fps。')
            if self.next_time is not None and (isinstance(self.next_time, bool) or not isinstance(self.next_time, (float, int)) or not math.isfinite(self.next_time)):
                raise ValueError('视频采样时间须为有限数值。')

    def capture(self):
        if not self.enabled:
            return
        trajectory = self.simulation.trajectory_frame()
        sample = {'trajectory': trajectory}
        current = trajectory['time']
        if self.video_enabled and (self.next_time is None or current >= self.next_time - 1e-9):
            try:
                sample['png_base64'] = image_png(self.simulation.render_rgb())
                if self.next_time is None:
                    self.next_time = current + 1. / self.fps
                else:
                    ticks = math.floor((current - self.next_time) * self.fps + 1e-7) + 1
                    self.next_time += max(1, ticks) / self.fps
            except Exception as exc:
                # A video encoder/render failure must not discard the trajectory.
                sample['rgb_error'] = f'{type(exc).__name__}: {exc}'
                self.video_enabled = False
        self.values.append(sample)


def dispatch(simulation, operation, payload):
    samples = None
    if operation == 'command':
        result = simulation.command(payload)
    elif operation == 'step':
        result = simulation.step()
    elif operation == 'advance':
        steps = payload.get('steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 10000:
            raise ValueError('物理批次须为 0–10000 步。')
        samples = Samples(simulation, payload.get('recording'))
        advanced = 0
        for _ in range(steps):
            if simulation.paused:
                break
            simulation.step()
            advanced += 1
            samples.capture()
        result = {'advanced': advanced}
    elif operation == 'verify_task':
        samples = Samples(simulation, payload.get('recording'))
        result = simulation.verify_task(payload['generation'], on_step=samples.capture if samples.enabled else None)
    elif operation == 'render':
        result = {'jpeg_base64': base64.b64encode(simulation.render()).decode('ascii')}
    elif operation == 'render_rgb':
        result = {'png_base64': image_png(simulation.render_rgb())}
    elif operation == 'checkpoint':
        result = simulation.checkpoint()
    elif operation == 'trajectory_frame':
        result = simulation.trajectory_frame()
    elif operation == 'replay_frame':
        result = simulation.replay_frame(payload)
    elif operation == 'close':
        simulation.close()
        return {'result': None, 'state': {'ready': False}}
    else:
        raise ValueError(f'未知仿真进程操作：{operation}')
    response = {'result': result, 'state': native_state(simulation)}
    if samples is not None and samples.enabled:
        response['samples'] = samples.values
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['genesis','maniskill','libero','alfworld','ai2thor','isaac','isaacsim','robocasa365'], required=True)
    parser.add_argument('--generation-base', type=int, default=0)
    arguments = parser.parse_args()
    # Native libraries may print directly to fd 1, bypassing redirect_stdout.
    # Keep a separate copy for protocol messages before importing the engine.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(value):
        protocol.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        protocol.flush()

    simulation = None
    try:
        if arguments.generation_base < 0:
            raise ValueError('generation-base 必须为非负整数。')
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import importlib
        from server.simulators import RUNTIMES
        _, _, module_name, class_name, _ = RUNTIMES[arguments.engine]
        module = importlib.import_module('server.backends.'+module_name)
        simulation = getattr(module, class_name)()
        simulation.generation += arguments.generation_base
        send({'id': 0, 'ok': True, 'state': native_state(simulation)})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        send({'id': 0, 'ok': False, 'error': {'type': type(exc).__name__, 'message': str(exc)}})
        if simulation is not None:
            simulation.close()
        return 1
    try:
        for line in sys.stdin.buffer:
            request_id = None
            operation = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or type(request.get('id')) is not int or not isinstance(request.get('op'), str) or not isinstance(request.get('payload'), dict):
                    raise ValueError('请求须包含整数 id、字符串 op 和对象 payload。')
                request_id, operation = request['id'], request['op']
                response = dispatch(simulation, operation, request['payload'])
                send({'id': request_id, 'ok': True, **response})
                if operation == 'close':
                    simulation = None
                    break
            except ValueError as exc:
                send({'id': request_id, 'ok': False, 'error': {'type': 'ValueError', 'message': str(exc)},
                      'state': native_state(simulation)})
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                send({'id': request_id, 'ok': False, 'error': {'type': type(exc).__name__, 'message': str(exc)}})
                return 1
    finally:
        if simulation is not None:
            simulation.close()
        protocol.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
