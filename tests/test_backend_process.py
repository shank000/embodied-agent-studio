"""Real child-process transport checks without installing or invoking Genesis."""
import asyncio
import base64
import importlib
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from fastapi import HTTPException

from server.backends.process_backend import BackendProcessError, ProcessBackend, ROOT, _decode_image
from server.backends.runtime import dispatch


FAKE_RUNTIME = r'''
import io
import os
import sys
import time
import types
import numpy as np
from PIL import Image
sys.path.insert(0, __PROJECT_ROOT__)

class Simulation:
    def __init__(self):
        self.time = 0.
        self.generation = 1
        self.paused = False
        self.speed = 1.
        self.render_suspended = False
        self.timestep = .01
        self.model_id = 'panda'
        self.pause_after = None
        print('native Python output 中文', flush=True)
        os.write(1, b'native fd output\n')
    def state(self):
        return {'model': self.model_id, 'generation': self.generation,
                'paused': self.paused, 'speed': self.speed,
                'timestep': self.timestep, 'time': self.time,
                'backend': {'engine_id': 'genesis'}, 'actuators': [{'index': 0}]}
    def command(self, payload):
        action = payload['action']
        if action == 'pause': self.paused = payload['value']
        elif action == 'speed': self.speed = payload['value']
        elif action == 'bad': raise ValueError('真实参数错误 中文')
        elif action == 'crash':
            os.write(2, b'native crash diagnostic\n')
            os._exit(17)
        elif action == 'hang': time.sleep(10)
        elif action == 'pause_after': self.pause_after = payload['time']
        elif action == 'load': self.model_id = payload['model']; self.generation += 1
        return 'received 中文'
    def step(self):
        self.time = round(self.time + self.timestep, 8)
        if self.pause_after is not None and self.time >= self.pause_after:
            self.paused = True
    def render_rgb(self):
        return np.full((2, 3, 3), round(self.time * 100), dtype=np.uint8)
    def render(self):
        stream = io.BytesIO()
        Image.fromarray(self.render_rgb()).save(stream, 'JPEG')
        return stream.getvalue()
    def checkpoint(self):
        return {'engine': 'genesis', 'generation': self.generation, 'time': self.time}
    def trajectory_frame(self):
        return {'time': self.time, 'generation': self.generation}
    def replay_frame(self, frame):
        self.time = frame['time']; self.paused = True
        return {'time': self.time}
    def verify_task(self, generation, on_step=None):
        if generation != self.generation: raise ValueError('过期的generation')
        for _ in range(3):
            self.step()
            if on_step: on_step()
        self.paused = True
        return {'success': False, 'source': 'genesis_privileged_state'}
    def close(self): pass

module = types.ModuleType('server.backends.genesis_backend')
module.GenesisSimulation = Simulation
sys.modules['server.backends.genesis_backend'] = module
from server.backends.runtime import main
raise SystemExit(main())
'''


class ImageTransportTests(unittest.TestCase):
    def encoded(self, image_format):
        output = io.BytesIO()
        Image.new('RGB', (3, 2), 'green').save(output, image_format)
        return output.getvalue(), base64.b64encode(output.getvalue()).decode('ascii')

    def test_jpeg_preview_validates_image_without_allocating_discarded_rgb_array(self):
        raw, encoded = self.encoded('JPEG')
        with patch('server.backends.process_backend.np.array', side_effect=MemoryError('RGB allocation')) as array:
            decoded, rgb = _decode_image(encoded, 'JPEG', decode_rgb=False)
        self.assertEqual(decoded, raw)
        self.assertIsNone(rgb)
        array.assert_not_called()
        with self.assertRaisesRegex(BackendProcessError, '无效的 JPEG'):
            _decode_image(base64.b64encode(b'not a JPEG').decode('ascii'), 'JPEG', decode_rgb=False)

    def test_rgb_video_array_owns_its_pixels_after_image_is_closed(self):
        raw, encoded = self.encoded('PNG')
        decoded, rgb = _decode_image(encoded, 'PNG')
        self.assertEqual(decoded, raw)
        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertTrue(rgb.flags.owndata)
        self.assertTrue(rgb.flags.writeable)
        rgb[0, 0] = [1, 2, 3]
        self.assertEqual(rgb[0, 0].tolist(), [1, 2, 3])

    def test_memory_failure_preserves_real_cause_and_recovery_advice(self):
        _, encoded = self.encoded('PNG')
        failure = MemoryError('fixture frame allocation failed')
        with patch('server.backends.process_backend.np.array', side_effect=failure):
            with self.assertRaisesRegex(BackendProcessError, '内存不足.*MemoryError.*重新加载') as error:
                _decode_image(encoded, 'PNG')
        self.assertIs(error.exception.__cause__, failure)
        self.assertNotIn('无效', str(error.exception))
        _, jpeg = self.encoded('JPEG')
        with patch.object(Image.Image, 'load', side_effect=failure):
            with self.assertRaisesRegex(BackendProcessError, '内存不足'):
                _decode_image(jpeg, 'JPEG', decode_rgb=False)


class BackendProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='embodiworks-process-')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.script = self.folder / 'fake_runtime.py'
        self.script.write_text(FAKE_RUNTIME.replace('__PROJECT_ROOT__', repr(str(ROOT))), encoding='utf-8')
        patcher = patch('server.backends.process_backend.RUNTIME_SCRIPT', self.script)
        patcher.start()
        self.addCleanup(patcher.stop)

    def backend(self, **kwargs):
        backend = ProcessBackend(python_path=sys.executable, **kwargs)
        self.addCleanup(backend.close)
        return backend

    def stalled_input_runtime(self):
        output = io.BytesIO()
        Image.new('RGB', (3, 2), 'green').save(output, 'JPEG')
        preview = base64.b64encode(output.getvalue()).decode('ascii')
        # The transport accepts valid action data plus its tracing metadata.
        # A large trace makes this exceed pipe capacity on every CI platform;
        # the original Windows failure also occurs with the 8.6 kB action alone.
        payload = {'action': 'policy_chunk', 'generation': 1,
                   'joint_names': [f'joint{i}' for i in range(1, 8)] + ['finger_joint1'],
                   'actions': [[.12345678901234568]*8]*50, 'dt_s': .02,
                   'trace_context': 'x' * (512 * 1024)}
        script = '''
import json, sys, time
generation = int(sys.argv[sys.argv.index('--generation-base')+1])+1
state = {'ready': True, 'model': 'panda', 'generation': generation, 'paused': True,
         'speed': 1., 'timestep': .01, 'time': 0., 'backend': {'engine_id': 'genesis'},
         'actuators': [{'index': 0}], 'task': {'status': 'idle'}}
print(json.dumps({'id': 0, 'ok': True, 'state': state}), flush=True)
request = json.loads(sys.stdin.buffer.readline())
assert request['op'] == 'render'
print(json.dumps({'id': request['id'], 'ok': True, 'state': state,
                  'result': {'jpeg_base64': __PREVIEW__}}), flush=True)
print('native stdin stopped reading fixture', file=sys.stderr, flush=True)
time.sleep(30)
'''
        self.script.write_text(script.replace('__PREVIEW__', repr(preview)), encoding='utf-8')
        return payload, output.getvalue()

    def test_whitelist_and_missing_environment_fail_before_launch(self):
        with self.assertRaisesRegex(ValueError, '不支持'):
            self.backend(engine_id='shell')
        with self.assertRaisesRegex(ValueError, 'docs/simulators.md'):
            ProcessBackend(python_path=self.folder / 'missing-python')

    def test_native_stdout_is_separate_and_mutations_update_state(self):
        backend = self.backend(generation_base=40)
        writer = backend._input_thread
        self.assertEqual(backend.generation, 41)
        self.assertEqual(backend.timestep, .01)
        self.assertFalse(backend.paused)
        self.assertEqual(backend.command({'action': 'pause', 'value': True}), 'received 中文')
        self.assertTrue(backend.paused)
        backend.command({'action': 'load', 'model': 'other'})
        self.assertEqual(backend.model_id, 'other')
        self.assertEqual(backend.generation, 42)
        self.assertIs(backend._input_thread, writer)
        backend.close()
        self.assertFalse(writer.is_alive())
        self.assertIn('native fd output', backend._diagnostics())

    def test_validation_error_preserves_process_and_real_reason(self):
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, '真实参数错误 中文'):
            backend.command({'action': 'bad'})
        backend.command({'action': 'speed', 'value': .5})
        self.assertEqual(backend.speed, .5)
        with self.assertRaises(ValueError):
            backend.command({'action': 'speed', 'value': float('nan')})
        backend.step()
        self.assertEqual(backend.state()['time'], .01)

    def test_crash_reports_exit_and_native_diagnostic(self):
        backend = self.backend()
        with self.assertRaisesRegex(BackendProcessError, 'native crash diagnostic'):
            backend.command({'action': 'crash'})
        self.assertEqual(backend.process.poll(), 17)
        self.assertFalse(backend.state()['ready'])
        self.assertIn('中断', backend.state()['error'])
        with self.assertRaises(BackendProcessError):
            backend.step()

    def test_idle_process_exit_and_normal_close_cannot_report_cached_ready_state(self):
        backend = self.backend()
        last = backend.state()
        backend.process.kill(); backend.process.wait(timeout=5)
        state = backend.state()
        self.assertFalse(state['ready']); self.assertIn('退出', state['error'])
        self.assertEqual(state['generation'], last['generation'])
        self.assertEqual(state['model'], last['model'])
        backend.close()
        self.assertFalse(backend.state()['ready'])
        healthy = self.backend(); healthy_state = healthy.state(); healthy.close()
        self.assertFalse(healthy.state()['ready'])
        self.assertIn('已关闭', healthy.state()['error'])
        self.assertEqual(healthy.state()['generation'], healthy_state['generation'])
        self.assertEqual(healthy.state()['model'], healthy_state['model'])

    def test_worker_detects_an_idle_child_exit_without_another_user_command(self):
        from server.app import Worker
        backend = self.backend()
        backend.command({'action': 'pause', 'value': True})
        worker = Worker(); worker.initial_engine_id = 'genesis'
        with patch.object(Worker, 'create_backend', return_value=backend):
            worker.thread.start()
            try:
                self.assertTrue(worker.ready.wait(5))
                self.assertTrue(worker.snapshot['ready'])
                backend.process.kill(); backend.process.wait(timeout=5)
                worker.thread.join(timeout=5)
                self.assertFalse(worker.thread.is_alive())
                self.assertFalse(worker.snapshot['ready'])
                self.assertIn('退出', worker.snapshot['error'])
            finally:
                worker.shutdown.set(); worker.thread.join(timeout=5)

    def test_timeout_stops_child_without_replaying_command(self):
        backend = self.backend(command_timeout_s=.1)
        with self.assertRaisesRegex(BackendProcessError, '不会自动重试'):
            backend.command({'action': 'hang'})
        self.assertIsNotNone(backend.process.poll())

    def test_send_deadline_reaps_child_that_stops_reading_and_reclaims_owned_threads(self):
        payload, _ = self.stalled_input_runtime()
        backend = self.backend(command_timeout_s=.15)
        backend.render()
        started = time.monotonic()
        with self.assertRaisesRegex(BackendProcessError, '发送命令.*|native stdin stopped reading fixture') as failure:
            backend.command(payload)
        self.assertIn('发送命令', str(failure.exception))
        self.assertIn('native stdin stopped reading fixture', str(failure.exception))
        self.assertLess(time.monotonic()-started, 4.)
        self.assertIsNotNone(backend.process.poll())
        self.assertFalse(backend._input_thread.is_alive())
        self.assertFalse(backend._output_thread.is_alive())
        self.assertFalse(backend._error_thread.is_alive())

    def test_blocked_send_becomes_worker_failure_and_can_explicitly_reload(self):
        from server.simulation import Simulation
        application = importlib.import_module('server.app')
        payload, preview = self.stalled_input_runtime()
        real_backend = ProcessBackend
        def isolated(engine_id, generation_base):
            return real_backend(engine_id, generation_base, python_path=sys.executable,
                                command_timeout_s=.15)
        def native():
            simulation = Simulation(render=False)
            simulation.paused = True
            simulation.render = lambda: preview
            return simulation
        async def check():
            worker = application.Worker()
            worker.initial_engine_id = 'genesis'
            worker.thread.start()
            try:
                self.assertTrue(await asyncio.to_thread(worker.ready.wait, 10))
                self.assertIsNone(worker.failure)
                generation, session = worker.snapshot['generation'], worker.session_id
                with self.assertRaises(HTTPException) as failed:
                    await worker.execute(payload)
                self.assertEqual(failed.exception.status_code, 500)
                self.assertIn('发送命令', failed.exception.detail)
                await asyncio.to_thread(worker.thread.join, 5)
                self.assertFalse(worker.thread.is_alive())
                self.assertFalse(worker.snapshot['ready'])
                self.assertIn('发送命令', worker.failure)
                recovered = await worker.restart('mujoco')
                self.assertTrue(recovered['restarted'])
                self.assertEqual(worker.engine_id, 'mujoco')
                self.assertTrue(worker.snapshot['ready'])
                self.assertIsNone(worker.failure)
                self.assertGreater(worker.snapshot['generation'], generation)
                self.assertNotEqual(worker.session_id, session)
                before = worker.snapshot['time']
                await worker.execute({'action': 'step'})
                self.assertGreater(worker.snapshot['time'], before)
            finally:
                worker.shutdown.set()
                await asyncio.to_thread(worker.thread.join, 5)
                self.assertFalse(worker.thread.is_alive())
        with patch.object(application, 'Simulation', side_effect=native), \
                patch('server.backends.process_backend.ProcessBackend', side_effect=isolated), \
                patch('server.app.traceback.print_exc'):
            asyncio.run(check())

    def test_malformed_json_is_fatal_with_native_diagnostics(self):
        self.script.write_text('import os\nos.write(2,b"fixture broken protocol\\n")\nprint("not-json",flush=True)\n', encoding='utf-8')
        with self.assertRaisesRegex(BackendProcessError, '无效 JSON 协议.*|fixture broken protocol'):
            self.backend()

    def test_success_without_state_is_rejected(self):
        self.script.write_text('print(\'{"id":0,"ok":true}\',flush=True)\n', encoding='utf-8')
        with self.assertRaisesRegex(BackendProcessError, '缺少状态'):
            self.backend()

    def test_batch_returns_all_trajectory_samples_and_only_due_rgb(self):
        backend = self.backend()
        samples = []
        def record():
            trajectory = backend.trajectory_frame()
            image = None
            if abs(trajectory['time'] - .03) < 1e-9:
                image = backend.render_rgb().copy()
            samples.append((trajectory, image))
        advanced = backend.advance(5, on_step=record,
            recording={'video_enabled': True, 'video_fps': 20., 'next_video_time': .025})
        self.assertEqual(advanced, 5)
        self.assertEqual([sample[0]['time'] for sample in samples], [.01, .02, .03, .04, .05])
        # Only .03 is due: subsequent samples must not silently use the final RGB.
        self.assertIsNone(samples[0][1])
        self.assertEqual(int(samples[2][1][0, 0, 0]), 3)
        self.assertEqual(backend.state()['time'], .05)
        self.assertIsNone(backend._sample)

    def test_batch_without_recording_does_not_collect_samples_and_respects_pause(self):
        backend = self.backend()
        calls = []
        backend.command({'action': 'pause_after', 'time': .03})
        self.assertEqual(backend.advance(10, on_step=lambda: calls.append(True)), 3)
        self.assertTrue(backend.paused)
        self.assertEqual(calls, [])
        self.assertEqual(backend.advance(2), 0)

    def test_verification_uses_recorded_samples_and_final_state(self):
        backend = self.backend(generation_base=2)
        frames = []
        result = backend.verify_task(3, on_step=lambda: frames.append(backend.trajectory_frame()),
                                     recording={'video_enabled': False})
        self.assertFalse(result['success'])
        self.assertEqual([frame['time'] for frame in frames], [.01, .02, .03])
        self.assertTrue(backend.paused)
        with self.assertRaisesRegex(ValueError, '过期'):
            backend.verify_task(2)

    def test_images_and_replay_round_trip_real_bytes(self):
        backend = self.backend()
        backend.step()
        jpeg = backend.render()
        with Image.open(io.BytesIO(jpeg)) as image:
            self.assertEqual(image.format, 'JPEG')
            self.assertEqual(image.size, (3, 2))
        self.assertEqual(backend.render_rgb().shape, (2, 3, 3))
        self.assertEqual(int(backend.render_rgb()[0, 0, 0]), 1)
        self.assertEqual(backend.checkpoint()['time'], .01)
        self.assertEqual(backend.replay_frame({'time': .4}), {'time': .4})
        self.assertEqual(backend.state()['time'], .4)
        self.assertTrue(backend.paused)

    def test_callback_exception_clears_sample_cache(self):
        backend = self.backend()
        def reject(): raise ValueError('recorder failed')
        with self.assertRaisesRegex(ValueError, 'recorder failed'):
            backend.advance(2, on_step=reject, recording={'video_enabled': False})
        self.assertIsNone(backend._sample)
        self.assertEqual(backend.trajectory_frame()['time'], .02)


class RuntimeSamplingTests(unittest.TestCase):
    def test_idle_batch_does_not_read_privileged_trajectory_or_render(self):
        class Simulation:
            paused = False
            render_suspended = False
            steps = 0
            def state(self): return {'time': self.steps * .01}
            def step(self): self.steps += 1
            def trajectory_frame(self): raise AssertionError('not recording')
            def render_rgb(self): raise AssertionError('not recording video')
        simulation = Simulation()
        result = dispatch(simulation, 'advance', {'steps': 4})
        self.assertEqual(result['result'], {'advanced': 4})
        self.assertNotIn('samples', result)

    def test_video_failure_keeps_every_trajectory_sample(self):
        class Simulation:
            paused = False
            render_suspended = False
            steps = 0
            def state(self): return {'time': self.steps * .01}
            def step(self): self.steps += 1
            def trajectory_frame(self): return {'time': self.steps * .01}
            def render_rgb(self): raise ValueError('native camera failed')
        result = dispatch(Simulation(), 'advance', {'steps': 3, 'recording': {'video_enabled': True}})
        self.assertEqual(len(result['samples']), 3)
        self.assertIn('native camera failed', result['samples'][0]['rgb_error'])
        self.assertNotIn('png_base64', result['samples'][1])


if __name__ == '__main__':
    unittest.main()
