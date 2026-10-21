"""Native RGB sampling, actual FFmpeg decoding and bounded encoder failure cleanup."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from server.runlog import RunLog
from server.simulation import Simulation
from server.video import SimulationVideo, find_ffmpeg


@unittest.skipUnless(find_ffmpeg() and shutil.which('ffprobe'), 'Native video acceptance needs FFmpeg and ffprobe.')
class NativeVideoTests(unittest.TestCase):
    def test_native_rgb_stream_samples_sim_time_and_skips_inference_pause(self):
        with tempfile.TemporaryDirectory(prefix='mujoco-video-验证-') as temporary:
            sim = Simulation('so101', render=True)
            log = RunLog('video-fixture', {'task': 'Native movement recording.'},
                         directory=Path(temporary), record_video=True)
            try:
                sim.set_control(0, .3)
                self.assertTrue(log.video_due(sim.data.time))
                self.assertTrue(log.capture_video_frame(sim.render_rgb(), sim.data.time))
                # A paused observation/inference phase must not extend the video.
                rgb = sim.render_rgb()
                for _ in range(30):
                    self.assertFalse(log.video_due(sim.data.time))
                    self.assertFalse(log.capture_video_frame(rgb, sim.data.time))
                while sim.data.time < .2 - 1e-8:
                    sim.step()
                    if log.video_due(sim.data.time):
                        self.assertTrue(log.capture_video_frame(sim.render_rgb(), sim.data.time))
                log.finish({'status': 'complete'})
                video = log.artifacts['video']
                self.assertEqual(video['status'], 'saved', video)
                self.assertEqual(video['path'], 'trajectory.mp4')
                self.assertEqual(video['frame_count'], 5)
                self.assertEqual(video['fps'], 20.)
                self.assertEqual(video['dropped_intervals'], 0)
                self.assertAlmostEqual(video['end_sim_time'] - video['start_sim_time'], .2, places=8)
                path = log.folder / video['path']
                probe = subprocess.run([shutil.which('ffprobe'), '-v', 'error', '-count_frames',
                    '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
                    '-of', 'json', str(path)], capture_output=True, timeout=15,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.assertEqual(probe.returncode, 0, probe.stderr.decode('utf-8', errors='replace'))
                stream = json.loads(probe.stdout)['streams'][0]
                self.assertEqual(stream['codec_name'], 'h264')
                self.assertEqual((stream['width'], stream['height']), (960, 640))
                self.assertEqual(int(stream['nb_read_frames']), video['frame_count'])
                self.assertEqual(stream['r_frame_rate'], '20/1')
                self.assertAlmostEqual(float(stream['duration']), video['frame_count'] / video['fps'], places=3)
                decoded = subprocess.run([find_ffmpeg(), '-hide_banner', '-v', 'error', '-i', str(path),
                    '-f', 'framemd5', '-'], capture_output=True, timeout=15,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.assertEqual(decoded.returncode, 0, decoded.stderr.decode('utf-8', errors='replace'))
                frames = [line for line in decoded.stdout.decode().splitlines() if line and not line.startswith('#')]
                self.assertEqual(len(frames), 5)
                self.assertGreater(len({line.rsplit(',', 1)[-1] for line in frames}), 1)
                self.assertIsNotNone(log._video.process.poll())
                self.assertIsNone(log._video._writer)
                summary = json.loads((log.folder / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['artifacts']['video'], video)
                self.assertEqual(log.finalize_video(), video)
            finally:
                log.finalize_video()
                sim.close()


class EncoderFailureTests(unittest.TestCase):
    def test_disabled_video_never_resolves_or_starts_an_encoder(self):
        with tempfile.TemporaryDirectory(prefix='video-disabled-') as temporary:
            log = RunLog('test', {}, directory=Path(temporary), record_video=False)
            with patch('server.video.find_ffmpeg', side_effect=AssertionError('Disabled video must not look up FFmpeg.')):
                self.assertFalse(log.video_due(0.))
                self.assertFalse(log.capture_video_frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.))
                log.finish({'status': 'complete'})
            self.assertEqual(log.artifacts['video']['status'], 'disabled')
            self.assertIsNone(log._video.process)
            self.assertEqual(list(log.folder.glob('*.mp4')), [])

    def test_missing_ffmpeg_and_renderer_failure_are_explicit_and_nonfatal(self):
        with tempfile.TemporaryDirectory(prefix='video-failed-') as temporary:
            log = RunLog('test', {}, directory=Path(temporary), record_video=True)
            with patch('server.video.find_ffmpeg', return_value=None):
                self.assertFalse(log.capture_video_frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.))
            log.finish({'status': 'complete'})
            video = log.artifacts['video']
            self.assertEqual(video['status'], 'failed')
            self.assertIn('FFmpeg', video['error'])
            self.assertIsNone(video['path'])
            self.assertFalse(log.video_due(1.))
            self.assertEqual(list(log.folder.glob('*.mp4')), [])
            events = [json.loads(line) for line in log.path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sum(event['event'] == 'video_failed' for event in events), 1)
            rendered = RunLog('test', {}, directory=Path(temporary), record_video=True)
            rendered.fail_video(RuntimeError('Native renderer fixture failure.'))
            rendered.finish({'status': 'complete'})
            self.assertEqual(rendered.artifacts['video']['status'], 'failed')
            self.assertIn('Native renderer fixture failure', rendered.artifacts['video']['error'])
            self.assertIsNone(rendered.artifacts['video']['path'])

    def test_stalled_stdin_writer_times_out_and_reaps_real_child(self):
        original_popen = subprocess.Popen
        children = []
        def not_reading_stdin(_arguments, **options):
            child = original_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **options)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory(prefix='video-stalled-') as temporary:
            video = SimulationVideo(Path(temporary) / 'trajectory.mp4', enabled=True,
                                    write_timeout_s=.15, finalize_timeout_s=1.)
            started = time.monotonic()
            with patch('server.video.find_ffmpeg', return_value=sys.executable), \
                 patch('server.video.subprocess.Popen', side_effect=not_reading_stdin):
                self.assertFalse(video.capture(np.zeros((640, 960, 3), dtype=np.uint8), 0.))
            self.assertLess(time.monotonic() - started, 5.)
            self.assertEqual(video.metadata['status'], 'failed', video.metadata)
            self.assertIn('写入超过', video.metadata['error'])
            self.assertTrue(children)
            self.assertIsNotNone(children[0].poll())
            self.assertIsNone(video._writer)
            self.assertFalse(any(thread.name.startswith('mujoco-video') for thread in threading.enumerate()))
            self.assertEqual(list(Path(temporary).glob('*.mp4')), [])

    def test_stalled_container_finalization_is_bounded_and_reaps_real_child(self):
        original_popen = subprocess.Popen
        children = []
        def never_finalizing(_arguments, **options):
            child = original_popen([sys.executable, '-c',
                'import sys,time; sys.stdin.buffer.read(); time.sleep(60)'], **options)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory(prefix='video-finalize-stalled-') as temporary:
            video = SimulationVideo(Path(temporary) / 'trajectory.mp4', enabled=True,
                                    write_timeout_s=1., finalize_timeout_s=.15)
            with patch('server.video.find_ffmpeg', return_value=sys.executable), \
                 patch('server.video.subprocess.Popen', side_effect=never_finalizing):
                self.assertTrue(video.capture(np.zeros((40, 40, 3), dtype=np.uint8), 0.))
            started = time.monotonic()
            outcome = video.finalize()
            self.assertLess(time.monotonic() - started, 5.)
            self.assertEqual(outcome['status'], 'failed')
            self.assertIn('按时封装', outcome['error'])
            self.assertIsNone(outcome['path'])
            self.assertIsNotNone(children[0].poll())
            self.assertIsNone(video._writer)

    @unittest.skipUnless(find_ffmpeg(), 'Actual FFmpeg required for the encoder rejection check.')
    def test_actual_ffmpeg_rejection_retains_stderr_and_removes_partial_video(self):
        original_popen = subprocess.Popen
        def invalid_codec(arguments, **options):
            arguments = arguments.copy()
            arguments[arguments.index('-c:v') + 1] = 'explicit-unsupported-codec-fixture'
            return original_popen(arguments, **options)
        with tempfile.TemporaryDirectory(prefix='video-rejected-') as temporary:
            video = SimulationVideo(Path(temporary) / 'trajectory.mp4', enabled=True)
            with patch('server.video.subprocess.Popen', side_effect=invalid_codec):
                video.capture(np.zeros((40, 40, 3), dtype=np.uint8), 0.)
                outcome = video.finalize()
            self.assertEqual(outcome['status'], 'failed', outcome)
            self.assertIn('explicit-unsupported-codec-fixture', outcome['error'])
            self.assertIsNone(outcome['path'])
            self.assertEqual(list(Path(temporary).glob('*.mp4')), [])
            self.assertIsNotNone(video.process.poll())
            self.assertIsNone(video._writer)


if __name__ == '__main__':
    unittest.main()
