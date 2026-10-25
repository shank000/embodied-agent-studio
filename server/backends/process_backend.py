"""Synchronous, isolated simulation bridge owned by the physics worker thread."""
from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
import io
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
from threading import Event, Lock, RLock, Thread
import time

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SCRIPT = Path(__file__).with_name('runtime.py')
from ..simulators import RUNTIMES, runtime_command
ENGINES = frozenset(RUNTIMES)


class BackendProcessError(RuntimeError):
    """The child cannot safely accept more physical commands."""


def _decode_image(value, image_format, *, decode_rgb=True):
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != image_format:
                raise ValueError(f'Expected {image_format}, received {image.format}')
            image.load()
            # The browser preview consumes the original JPEG bytes. Only video
            # samples need an owned RGB array; avoid allocating one to discard it.
            rgb = np.array(image if image.mode == 'RGB' else image.convert('RGB'), copy=True) if decode_rgb else None
        return raw, rgb
    except MemoryError as exc:
        raise BackendProcessError(f'解析仿真子进程的 {image_format} 图像时内存不足（MemoryError）。'
                                  '请关闭其他仿真进程或降低相机分辨率，再重新加载环境。') from exc
    except Exception as exc:
        raise BackendProcessError(f'仿真子进程返回无效的 {image_format} 图像：{exc}') from exc


class ProcessBackend:
    """A real engine in its own Python environment; no shell or arbitrary command.

    Only the owning worker calls this object. The child advances a batch locally.
    During recording callbacks, trajectory_frame/render_rgb expose the recorded
    sample rather than the final state of the batch.
    """
    def __init__(self, engine_id='genesis', generation_base=0, python_path=None,
                 *, startup_timeout_s=600., command_timeout_s=60.):
        if engine_id not in ENGINES:
            raise ValueError(f'不支持的隔离仿真引擎：{engine_id}')
        if isinstance(generation_base, bool) or not isinstance(generation_base, int) or generation_base < 0:
            raise ValueError('generation_base 必须为非负整数。')
        command = runtime_command(engine_id, generation_base, python_path, RUNTIME_SCRIPT)
        self.engine_id = engine_id
        self.command_timeout_s = command_timeout_s
        self._state = {}
        self._preview = None
        self._sample = None
        self._closed = False
        self._failure = None
        self._sequence = 0
        self._responses = Queue()
        self._writes = Queue(maxsize=1)
        self._closing = Event()
        self._stderr = deque(maxlen=12)
        self._stderr_lock = Lock()
        self._rpc_lock = RLock()
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
        self.process = subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._output_thread = Thread(target=self._read_stdout, name=f'{engine_id}-protocol', daemon=True)
        self._error_thread = Thread(target=self._read_stderr, name=f'{engine_id}-diagnostics', daemon=True)
        self._input_thread = Thread(target=self._write_stdin, name=f'{engine_id}-input', daemon=True)
        self._output_thread.start()
        self._error_thread.start()
        self._input_thread.start()
        try:
            self._receive(0, startup_timeout_s)
            if self._state.get('ready') is not True:
                raise BackendProcessError(self._state.get('error') or '仿真子进程初始化后未就绪。')
        except Exception:
            self._abort()
            raise

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self._responses.put(('line', line))
        except (OSError, ValueError) as exc:
            self._responses.put(('error', str(exc)))
        finally:
            self._responses.put(('eof', None))

    def _write_stdin(self):
        while not self._closing.is_set():
            packet = self._writes.get()
            if packet is None or self._closing.is_set():
                return
            raw, completed = packet
            try:
                remaining = memoryview(raw)
                while remaining:
                    count = self.process.stdin.write(remaining)
                    if not count:
                        raise BrokenPipeError('仿真输入管道已关闭。')
                    remaining = remaining[count:]
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                completed.set_exception(exc)
            else:
                completed.set_result(None)

    def _read_stderr(self):
        try:
            while chunk := self.process.stderr.read1(1024):
                with self._stderr_lock:
                    self._stderr.append(chunk.decode('utf-8', errors='replace'))
        except (OSError, ValueError):
            pass

    def _diagnostics(self):
        with self._stderr_lock:
            detail = ''.join(self._stderr)[-12000:].strip()
        return f'\n{detail}' if detail else ''

    def _abort(self):
        self._closing.set()
        try:
            self._writes.put_nowait(None)
        except Full:
            pass
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Killing the child releases a blocked pipe write before its buffered
        # stream is closed. One owned writer serves all requests and exits here.
        self._input_thread.join(timeout=1)
        self._output_thread.join(timeout=1)
        self._error_thread.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        self._closed = True

    def _broken(self, message, cause=None):
        self._failure = message
        self._abort()
        error = BackendProcessError(message + self._diagnostics())
        if cause is not None:
            raise error from cause
        raise error

    def _receive(self, request_id, timeout, *, timeout_budget=None):
        try:
            kind, value = self._responses.get(timeout=timeout)
        except Empty as exc:
            self._broken(f'{self.engine_id} 仿真操作超过 {timeout if timeout_budget is None else timeout_budget:g} 秒，子进程已终止；物理命令不会自动重试。', exc)
        if kind != 'line':
            self._broken(f'{self.engine_id} 仿真子进程中断（退出码 {self.process.poll()}）：{value or "协议输出已关闭"}')
        try:
            response = json.loads(value)
            if not isinstance(response, dict) or response.get('id') != request_id or type(response.get('ok')) is not bool:
                raise ValueError('响应编号或状态字段不符合协议')
        except (ValueError, UnicodeError) as exc:
            self._broken(f'{self.engine_id} 仿真子进程返回无效 JSON 协议：{exc}', exc)
        if isinstance(response.get('state'), dict):
            self._state = response['state']
        if not response['ok']:
            error = response.get('error') or {}
            message = f'{error.get("type", "RuntimeError")}: {error.get("message", "仿真操作失败")}'
            if error.get('type') == 'ValueError':
                raise ValueError(message)
            self._broken(f'{self.engine_id} {message}')
        if not isinstance(response.get('state'), dict):
            self._broken(f'{self.engine_id} 仿真子进程响应缺少状态。')
        return response

    def _request(self, operation, payload=None, *, timeout=None, mutation=False):
        with self._rpc_lock:
            if self._closed or self._failure:
                raise BackendProcessError(self._failure or '仿真子进程已关闭。')
            budget = self.command_timeout_s if timeout is None else timeout
            deadline = time.monotonic() + budget
            self._sequence += 1
            # Serialize before writing: invalid input must not corrupt the stream.
            line = json.dumps({'id': self._sequence, 'op': operation, 'payload': payload or {}},
                              ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
            sent = Future()
            try:
                self._writes.put((line, sent), timeout=max(0., deadline-time.monotonic()))
                sent.result(timeout=max(0., deadline-time.monotonic()))
            except (Full, FutureTimeout) as exc:
                self._broken(f'{self.engine_id} 仿真操作超过 {budget:g} 秒（发送命令），子进程已终止；物理命令不会自动重试。', exc)
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._broken(f'{self.engine_id} 仿真子进程无法接收命令：{exc}', exc)
            if mutation:
                self._preview = None
            return self._receive(self._sequence, max(0., deadline-time.monotonic()), timeout_budget=budget)

    @property
    def paused(self): return self._state['paused']

    @property
    def speed(self): return self._state['speed']

    @property
    def render_suspended(self): return self._state.get('render_suspended', False)

    @property
    def timestep(self): return float(self._state['timestep'])

    @property
    def model_id(self): return self._state['model']

    @property
    def generation(self): return self._state['generation']

    def state(self):
        state = self._state.copy()
        if self._closed or self._failure:
            state.update(ready=False, error=self._failure or f'{self.engine_id} 仿真子进程已关闭。')
        elif self.process.poll() is not None:
            state.update(ready=False, error=f'{self.engine_id} 仿真子进程已退出（退出码 {self.process.returncode}）。'+self._diagnostics())
        return state

    def command(self, payload):
        return self._request('command', payload, mutation=True).get('result')

    def step(self):
        return self._request('step', mutation=True).get('result')

    def render(self):
        if self._preview is None:
            result = self._request('render')['result']
            self._preview, _ = _decode_image(result['jpeg_base64'], 'JPEG', decode_rgb=False)
        return self._preview

    def render_rgb(self):
        if self._sample is not None:
            if self._sample.get('rgb_error'):
                raise RuntimeError(self._sample['rgb_error'])
            if 'rgb' not in self._sample:
                raise RuntimeError('该轨迹样本没有视频帧；请使用视频采样配置推进仿真。')
            return self._sample['rgb']
        result = self._request('render_rgb')['result']
        return _decode_image(result['png_base64'], 'PNG')[1]

    def checkpoint(self):
        return self._request('checkpoint')['result']

    def trajectory_frame(self):
        if self._sample is not None:
            return self._sample['trajectory'].copy()
        return self._request('trajectory_frame')['result']

    def replay_frame(self, frame):
        return self._request('replay_frame', frame, mutation=True).get('result')

    def _consume_samples(self, samples, on_step):
        try:
            for sample in samples:
                self._sample = {'trajectory': sample['trajectory']}
                if sample.get('png_base64') is not None:
                    self._sample['rgb'] = _decode_image(sample['png_base64'], 'PNG')[1]
                if sample.get('rgb_error'):
                    self._sample['rgb_error'] = sample['rgb_error']
                if on_step is not None:
                    on_step()
        finally:
            self._sample = None

    def advance(self, steps, on_step=None, recording=None):
        response = self._request('advance', {'steps': steps, 'recording': recording}, mutation=True)
        self._consume_samples(response.get('samples', []), on_step)
        return response['result']['advanced']

    def verify_task(self, generation, on_step=None, recording=None):
        response = self._request('verify_task', {'generation': generation, 'recording': recording}, mutation=True)
        self._consume_samples(response.get('samples', []), on_step)
        return response['result']

    def close(self):
        if self._closed:
            return
        last_state = self._state.copy()
        try:
            self._request('close', timeout=5.)
        except (BackendProcessError, ValueError):
            pass
        finally:
            self._state = {**last_state, **self._state}
            self._abort()
