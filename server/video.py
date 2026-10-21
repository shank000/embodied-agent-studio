"""Bounded RGB streaming to the installed FFmpeg H.264 encoder."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def find_ffmpeg():
    return shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')


class SimulationVideo:
    """Sample native RGB by simulation time, keeping at most one pending frame."""
    def __init__(self, path, enabled=False, fps=20., write_timeout_s=3., finalize_timeout_s=15.):
        if not math.isfinite(fps) or not 1 <= fps <= 60:
            raise ValueError('视频帧率须为 1–60 fps。')
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.stem + '.pending.mp4')
        self.stderr_path = self.path.with_name(self.path.stem + '-encoder.log')
        self.write_timeout_s = write_timeout_s
        self.finalize_timeout_s = finalize_timeout_s
        self.process = None
        self._writer = self._pending = self._stderr = None
        self._owns_partial = False
        self._next_time = None
        self.metadata = {'enabled': bool(enabled), 'status': 'pending' if enabled else 'disabled',
                         'path': None, 'fps': float(fps), 'frame_count': 0, 'width': None, 'height': None,
                         'start_sim_time': None, 'end_sim_time': None, 'dropped_intervals': 0, 'error': None}

    def due(self, sim_time):
        if self.metadata['status'] not in ('pending', 'recording'): return False
        if not isinstance(sim_time, (int, float)) or not math.isfinite(sim_time):
            raise ValueError('视频需要有限的仿真时间。')
        last = self.metadata['end_sim_time']
        if last is not None and sim_time < last - 1e-8:
            raise ValueError('录制期间仿真时间倒退，请在重置场景前结束当前录制。')
        if last is not None and sim_time <= last + 1e-9: return False
        return self._next_time is None or sim_time >= self._next_time - 1e-9

    def _start(self, width, height):
        executable = find_ffmpeg()
        if not executable:
            raise FileNotFoundError('未找到 FFmpeg；请安装 FFmpeg 并加入 PATH，再重新运行录制。')
        if self.path.exists() or self.partial.exists():
            raise FileExistsError('视频目标文件已存在，不能覆盖。')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_path.open('xb')
        arguments = [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                     '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                     '-framerate', f'{self.metadata["fps"]:g}', '-i', 'pipe:0', '-an',
                     '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-threads', '2',
                     '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                     '-movflags', '+faststart', str(self.partial)]
        self.process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self._stderr, bufsize=0, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._owns_partial = True
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mujoco-video')
        self.metadata.update(status='recording', width=width, height=height)

    def _write_frame(self, raw):
        remaining = memoryview(raw)
        while remaining:
            count = self.process.stdin.write(remaining)
            if not count: raise BrokenPipeError('FFmpeg 视频输入管道已关闭。')
            remaining = remaining[count:]

    def capture(self, rgb, sim_time):
        try:
            if not self.due(sim_time): return False
            frame = np.asarray(rgb)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1]:
                raise ValueError('视频帧必须为非空 uint8 RGB 图像。')
            height, width = frame.shape[:2]
            if self.process is None: self._start(width, height)
            elif (width, height) != (self.metadata['width'], self.metadata['height']):
                raise ValueError('录制期间相机图像尺寸发生变化。')
            self._pending = self._writer.submit(self._write_frame, frame.tobytes(order='C'))
            try: self._pending.result(timeout=self.write_timeout_s)
            except FutureTimeout as exc:
                raise TimeoutError(f'FFmpeg 写入超过 {self.write_timeout_s:g} 秒，已终止视频编码。') from exc
            self.metadata['frame_count'] += 1
            if self.metadata['start_sim_time'] is None: self.metadata['start_sim_time'] = float(sim_time)
            start = self.metadata['start_sim_time']
            tick = math.floor((sim_time - start) * self.metadata['fps'] + 1e-7)
            if self._next_time is not None:
                previous_tick = round((self._next_time - start) * self.metadata['fps'])
                self.metadata['dropped_intervals'] += max(0, tick - previous_tick)
            self._next_time = start + (tick + 1) / self.metadata['fps']
            self.metadata['end_sim_time'] = float(sim_time)
            return True
        except Exception as exc:
            self.fail(exc)
            return False

    def _close_process(self, abort=False):
        if self.process is not None:
            if abort and self.process.poll() is None: self.process.kill()
            if self.process.stdin is not None:
                try: self.process.stdin.close()
                except (OSError, ValueError): pass
            try: code = self.process.wait(timeout=self.finalize_timeout_s)
            except subprocess.TimeoutExpired as exc:
                self.process.kill()
                self.process.wait(timeout=5)
                raise TimeoutError('FFmpeg 未能按时封装视频，已终止编码进程。') from exc
            finally:
                if self._writer is not None:
                    self._writer.shutdown(wait=True, cancel_futures=True)
                    self._writer = None
                if self._stderr is not None:
                    self._stderr.close()
                    self._stderr = None
            return code
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        return None

    def fail(self, error):
        if not self.metadata['enabled'] or self.metadata['status'] == 'failed': return self.metadata
        message = str(error) or type(error).__name__
        try: self._close_process(abort=True)
        except Exception as exc: message += f'；编码资源清理失败：{exc}'
        if self.stderr_path.is_file():
            try:
                with self.stderr_path.open('rb') as file:
                    file.seek(max(0, self.stderr_path.stat().st_size - 8000))
                    detail = file.read().decode('utf-8', errors='replace').strip()
                if detail: message += '\n' + detail
            except OSError as exc: message += f'；无法读取编码诊断：{exc}'
        if self._owns_partial:
            try: self.partial.unlink(missing_ok=True)
            except OSError as exc: message += f'；未完成视频无法清理：{exc}'
        self.metadata.update(status='failed', path=None, error=message[:12000])
        return self.metadata

    def finalize(self):
        if self.metadata['status'] in ('disabled', 'saved', 'failed'): return self.metadata
        try:
            if not self.metadata['frame_count']:
                raise ValueError('本次录制没有收到 MuJoCo 渲染帧，未生成视频。')
            code = self._close_process()
            if code != 0: raise RuntimeError(f'FFmpeg 编码失败，退出码 {code}。')
            if not self.partial.is_file() or not self.partial.stat().st_size:
                raise RuntimeError('FFmpeg 未生成有效的视频文件。')
            if self.path.exists(): raise FileExistsError('视频目标文件已存在，不能覆盖。')
            os.rename(self.partial, self.path)
            self._owns_partial = False
            self.metadata.update(status='saved', path=self.path.name, error=None)
        except Exception as exc: self.fail(exc)
        return self.metadata
