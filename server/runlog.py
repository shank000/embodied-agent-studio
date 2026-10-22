"""Append-only local run records. Public model outputs only; no credentials or hidden reasoning."""
from contextvars import ContextVar
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from threading import RLock
from weakref import WeakValueDictionary
from uuid import uuid4

from PIL import Image

from .artifacts import is_secret_key
from .video import SimulationVideo

RUNS = Path(__file__).resolve().parents[1] / 'runs'
CURRENT_LOG = ContextVar('robot_run_log', default=None)
RUN_ID = re.compile(r'^\d{8}T\d{6}-[0-9a-f]{10}$')
LIVE_RUNS = WeakValueDictionary()


class RunLog:
    def __init__(self, kind, metadata, secrets=(), directory=None, *, record_video=False, video_fps=20.):
        self.id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:10]
        self.folder = (directory or RUNS) / self.id
        self.folder.mkdir(parents=True, exist_ok=False)
        self.path = self.folder / 'events.jsonl'
        self.secrets = [value for value in secrets if value]
        self.metadata = {'run_id': self.id, 'kind': kind, **metadata}
        self.metadata['telemetry_schema_version'] = 'embodied-calls-v1'
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._started_clock = time.monotonic()
        self._elapsed = None
        self.ended_at = None
        self.current_round = 0
        self._active_inference = {}
        self._inference_s = 0.
        self._inference_count = 0
        self._call_costs = []
        self.metadata.setdefault('started_at', self.started_at)
        self._lock = RLock()
        self._trajectory = None
        self._finished = False
        self._video = SimulationVideo(self.folder / 'trajectory.mp4', enabled=record_video, fps=video_fps)
        self.artifacts = {'schema_version': 'embodied-run-artifacts-v1', 'observation_manifest': None,
                          'observation_count': 0, 'trajectory': None, 'trajectory_frames': 0,
                          'checkpoints': {}, 'video': self._video.metadata}
        self.write('start', self.metadata)
        LIVE_RUNS[self.id] = self

    def clean(self, value):
        if isinstance(value, dict):
            return {key: self.clean(item) for key, item in value.items()
                    if not is_secret_key(key) and key.lower() not in ('reasoning_content', 'encrypted_content', 'image_base64')}
        if isinstance(value, (list, tuple)): return [self.clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        if isinstance(value, str):
            for secret in self.secrets: value = value.replace(secret, '[已隐藏密钥]')
        return value

    def write(self, event, data):
        if event == 'observation' and isinstance(data, dict): self.current_round = data.get('round', data.get('step', self.current_round))
        record = self.clean({'timestamp': datetime.now(timezone.utc).isoformat(), 'event': event, 'data': data})
        with self._lock, self.path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def begin_inference(self, record, started_clock):
        with self._lock:
            self._active_inference[record['call_id']] = started_clock
            self._inference_count += 1
            self.write('inference_start', record)

    def end_inference(self, record):
        with self._lock:
            self._active_inference.pop(record['call_id'], None)
            self._inference_s += record['inference_s']
            if isinstance(record.get('cost'), dict): self._call_costs.append(record['cost'])
            self.write('inference', record)

    def timing_snapshot(self):
        with self._lock:
            elapsed = self._elapsed if self._elapsed is not None else time.monotonic() - self._started_clock
            inference = self._inference_s + sum(time.monotonic() - started for started in self._active_inference.values())
            return {'started_at': self.started_at, 'ended_at': self.ended_at, 'elapsed_s': elapsed,
                    'timing': {'elapsed_wall_s': elapsed, 'inference_elapsed_s': inference,
                               'inference_calls': self._inference_count,
                               'current_inference_s': max((time.monotonic() - started for started in self._active_inference.values()), default=0.),
                               'active_calls': len(self._active_inference)}}

    def save_observation(self, observation, step):
        """Persist the actual three JPEGs; return observation metadata and relative references."""
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 999999999:
            raise ValueError('观察步骤必须为非负整数。')
        cameras = observation.get('cameras', [])
        if len(cameras) != 3 or {camera.get('name') for camera in cameras} != {'front', 'side', 'wrist'}:
            raise ValueError('观察必须包含 front、side、wrist 各一张图像。')
        blobs = {}
        for camera in cameras:
            try:
                raw = base64.b64decode(camera['image_base64'], validate=True)
                with Image.open(io.BytesIO(raw)) as jpeg:
                    if jpeg.format != 'JPEG': raise ValueError('相机图像必须为 JPEG。')
                    jpeg.verify()
            except Exception as exc:
                raise ValueError('相机 JPEG 数据无效。') from exc
            blobs[camera['name']] = raw
        saved = self.clean(observation)
        for camera in saved['cameras']:
            name = camera['name']
            camera['image_ref'] = f'observations/{step:06d}/{name}.jpg'
            camera['image_sha256'] = hashlib.sha256(blobs[name]).hexdigest()
        with self._lock:
            root = self.folder / 'observations'
            root.mkdir(exist_ok=True)
            destination = root / f'{step:06d}'
            if destination.exists(): raise FileExistsError('该步骤的观察已保存，不能覆盖。')
            staging = Path(tempfile.mkdtemp(prefix='.observation-', dir=root))
            try:
                for name, raw in blobs.items(): (staging / f'{name}.jpg').write_bytes(raw)
                os.rename(staging, destination)
            finally:
                if staging.exists():
                    if not staging.resolve().is_relative_to(root.resolve()): raise ValueError('临时观察路径超出运行目录。')
                    shutil.rmtree(staging)
            record = {'step': step, 'time': saved.get('time'), 'robot': saved.get('robot'),
                      'generation': saved.get('generation'), 'cameras': saved['cameras']}
            with (root / 'manifest.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            self.artifacts['observation_manifest'] = 'observations/manifest.jsonl'
            self.artifacts['observation_count'] += 1
        return saved

    def save_checkpoint(self, checkpoint, name='initial'):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
            raise ValueError('无效状态快照名称。')
        content = json.dumps(self.clean(checkpoint), ensure_ascii=False, allow_nan=False)
        relative = f'checkpoints/{name}.json'
        with self._lock:
            path = self.folder / relative
            path.parent.mkdir(exist_ok=True)
            with path.open('x', encoding='utf-8') as file: file.write(content)
            self.artifacts['checkpoints'][name] = {'path': relative, 'sha256': hashlib.sha256(content.encode()).hexdigest()}
        return relative

    def write_trajectory(self, frame):
        content = json.dumps(self.clean(frame), ensure_ascii=False, allow_nan=False)
        with self._lock:
            if self._finished: raise ValueError('已结束的运行不能追加轨迹。')
            if self._trajectory is None:
                self._trajectory = (self.folder / 'trajectory.jsonl').open('x', encoding='utf-8', buffering=65536)
                self.artifacts['trajectory'] = 'trajectory.jsonl'
            self._trajectory.write(content + '\n')
            self.artifacts['trajectory_frames'] += 1

    def flush_trajectory(self):
        with self._lock:
            if self._trajectory is not None: self._trajectory.flush()

    def video_due(self, sim_time):
        with self._lock:
            if self._finished: return False
            try: return self._video.due(sim_time)
            except Exception as exc:
                self.fail_video(exc)
                return False

    def capture_video_frame(self, rgb, sim_time):
        with self._lock:
            if self._finished: return False
            previous = self._video.metadata['status']
            captured = self._video.capture(rgb, sim_time)
            if previous != 'failed' and self._video.metadata['status'] == 'failed':
                self.write('video_failed', self._video.metadata)
            return captured

    def fail_video(self, error):
        with self._lock:
            previous = self._video.metadata['status']
            metadata = self._video.fail(error)
            if previous != 'failed' and metadata['status'] == 'failed': self.write('video_failed', metadata)
            return metadata.copy()

    def finalize_video(self):
        with self._lock:
            previous = self._video.metadata['status']
            metadata = self._video.finalize()
            if previous not in ('disabled', 'saved', 'failed'):
                self.write('video_finalized' if metadata['status'] == 'saved' else 'video_failed', metadata)
            return metadata.copy()

    def finish(self, result):
        with self._lock:
            if self._finished: return
            cleanup_errors = []
            try: self.finalize_video()
            except Exception as exc: cleanup_errors.append({'phase': 'video', 'message': str(exc)})
            if self._trajectory is not None:
                try: self._trajectory.flush()
                except Exception as exc: cleanup_errors.append({'phase': 'trajectory_flush', 'message': str(exc)})
                try: self._trajectory.close()
                except Exception as exc: cleanup_errors.append({'phase': 'trajectory_close', 'message': str(exc)})
                finally: self._trajectory = None
            self._finished = True
            self._elapsed = time.monotonic() - self._started_clock
            self.ended_at = datetime.now(timezone.utc).isoformat()
            timing = self.timing_snapshot()
            if isinstance(result.get('timing'), dict):
                timing['timing'].update({key: value for key, value in result['timing'].items() if key != 'elapsed_wall_s'})
            result = {**result, **timing, 'artifacts': self.artifacts}
            from .costing import aggregate_costs
            costs = self._call_costs
            if self.metadata.get('kind') in ('comparison', 'research_experiment'):
                costs = []; seen = set()
                for row in result.get('rows') or self.metadata.get('rows') or []:
                    child_id = row.get('run_id')
                    if not child_id or child_id in seen: continue
                    seen.add(child_id)
                    try:
                        child_summary = json.loads((run_folder(child_id) / 'summary.json').read_text(encoding='utf-8'))
                        if isinstance(child_summary.get('cost'), dict): costs.append(child_summary['cost'])
                        else: raise ValueError('子实验未记录调用时价格。')
                    except (OSError, ValueError) as exc:
                        costs.append({'status': 'unavailable', 'total_calls': 0, 'missing_calls': 1, 'call_count_incomplete': True,
                                      'warnings': [f'子实验 {child_id} 费用尚未归档：{exc}']})
            result['cost'] = aggregate_costs(costs)
            if cleanup_errors: result['cleanup_errors'] = cleanup_errors
            try: self.write('finish', result)
            except OSError as exc:
                result.setdefault('cleanup_errors', []).append({'phase': 'events_finish', 'message': str(exc)})
            summary = self.clean({**self.metadata, **result})
            pending = self.folder / 'summary.pending.json'
            pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            pending.replace(self.folder / 'summary.json')


def model_output(data):
    from .telemetry import note_response
    note_response(data)
    log = CURRENT_LOG.get()
    if log: log.write('model_output', data)


def run_folder(run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id): raise ValueError('无效日志编号。')
    folder = (RUNS / run_id).resolve()
    if not folder.is_relative_to(RUNS.resolve()): raise ValueError('日志路径超出运行目录。')
    if not folder.is_dir(): raise FileNotFoundError('日志不存在。')
    return folder


def safe_artifact_path(run_id, relative_path):
    folder = run_folder(run_id)
    if not isinstance(relative_path, str) or not relative_path or ':' in relative_path:
        raise ValueError('无效运行材料路径。')
    relative = Path(relative_path)
    if relative.is_absolute() or '..' in relative.parts or '\\' in relative_path:
        raise ValueError('无效运行材料路径。')
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder): raise ValueError('运行材料路径超出日志目录。')
    if not path.is_file(): raise FileNotFoundError('运行材料不存在。')
    return path


def live_snapshot(log):
    """Parent controllers bind newly created children before results are read."""
    provider = getattr(log, 'snapshot_provider', None)
    snapshot = provider() if callable(provider) else log.timing_snapshot()
    return {**log.metadata, **snapshot, 'status': 'running'}


def list_runs(limit=100, offset=0, *, deleted=False):
    from .run_management import read_run_record, record_labels
    if not RUNS.exists(): return []
    result = []
    for folder in sorted(RUNS.iterdir(), reverse=True):
        if not RUN_ID.fullmatch(folder.name): continue
        if not (folder / 'summary.json').is_file() and not (folder / 'events.jsonl').is_file(): continue
        record = read_run_record(folder.name)
        record.update(record_labels(record))
        if bool(record['deleted_at']) != deleted: continue
        result.append(record)
        if limit is not None and len(result) >= limit + offset: break
    return result[offset:] if limit is None else result[offset:offset + limit]
