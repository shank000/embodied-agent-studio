"""Agent video finalization using native MuJoCo frames and an actual FFmpeg decoder."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from server.agent import AgentConfig, AgentController, AgentStart, ModelResponseError
from server.app import Worker
from server.control import PoseAction
from server.comparison import ComparisonController, ComparisonStart
from server.policy import PolicyConfig, PolicyController
from server.runlog import RunLog
from server.simulation import Simulation


class AgentVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tempfile.TemporaryDirectory(prefix='agent-video-fixture-')
        cls.log_patch = patch('server.runlog.RUNS', Path(cls.logs.name)); cls.log_patch.start()
        cls.worker = Worker(); cls.worker.thread.start()
        assert cls.worker.ready.wait(30)
        assert cls.worker.failure is None, cls.worker.failure

    @classmethod
    def tearDownClass(cls):
        cls.worker.shutdown.set(); cls.worker.thread.join(5)
        cls.log_patch.stop(); cls.logs.cleanup()

    async def prepare(self):
        await self.worker.execute({'action': 'load', 'model': 'panda', 'scene_id': 'tabletop', 'seed': 42})
        await self.worker.execute({'action': 'settle'})
        agent = AgentController(self.worker)
        agent.configure(AgentConfig(provider='compatible', api_key='local-boundary-fixture'))
        return agent

    def saved_video(self, agent):
        from server.video import find_ffmpeg
        folder = Path(self.logs.name)/agent.state['run_id']
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        video = agent.state['video']
        self.assertEqual(video, summary['artifacts']['video'])
        self.assertTrue(video['enabled'])
        self.assertEqual(video['status'], 'saved', video)
        self.assertGreater(video['frame_count'], 0)
        path = folder/video['path']
        self.assertTrue(path.is_file())
        ffmpeg = find_ffmpeg()
        self.assertIsNotNone(ffmpeg, 'Video acceptance requires an actual FFmpeg decoder.')
        # framemd5 decodes every frame while keeping captured output small.
        decoded = subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-v', 'error', '-i', str(path),
            '-map', '0:v:0', '-f', 'framemd5', '-'], capture_output=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertEqual(decoded.returncode, 0, decoded.stderr.decode('utf-8', 'replace'))
        frames = [line for line in decoded.stdout.decode('utf-8').splitlines() if line.strip() and not line.startswith('#')]
        self.assertGreater(len(frames), 0, 'The MP4 must contain decodable video frames.')
        return frames

    def test_stopping_after_motion_finalizes_playable_native_video(self):
        async def check():
            agent = await self.prepare(); thinking = asyncio.Event(); calls = 0
            async def boundary(config, prompt, cameras):
                nonlocal calls
                calls += 1
                if calls > 1:
                    thinking.set()
                    await asyncio.sleep(60)
                pose = json.loads(prompt)['observation']['tcp']; pose['position_m'][2] += .02
                return PoseAction(status='move', **pose, gripper_open=1., duration_s=.7,
                                  explanation='Native motion for video acceptance.'), None
            with patch('server.agent.api_action', boundary):
                await agent.start(AgentStart(task='Raise, then stop.', record_video=True))
                await asyncio.wait_for(thinking.wait(), 20)
                await agent.stop()
            self.assertEqual(agent.state['status'], 'stopped', agent.state)
            self.assertEqual(agent.state['history'][0]['result']['status'], 'complete')
            frames = self.saved_video(agent)
            self.assertGreater(len(frames), 1)
            self.assertGreater(len({frame.rsplit(',', 1)[-1].strip() for frame in frames}), 1,
                               'The recorded camera must show actual changing native frames.')
            self.assertIsNone(self.worker.recorder)
            with self.worker.lock: self.assertTrue(self.worker.snapshot['paused'])
        asyncio.run(check())

    def test_model_error_finalizes_initial_frame_as_playable_video(self):
        async def check():
            agent = await self.prepare()
            async def boundary(*args):
                raise ModelResponseError('Explicit local model fixture failure.', {'output_tokens': 1})
            with patch('server.agent.api_action', boundary):
                await agent.start(AgentStart(task='Error before motion.', record_video=True))
                await asyncio.wait_for(agent.task, 20)
            self.assertEqual(agent.state['status'], 'error', agent.state)
            self.assertIn('fixture failure', agent.state['message'])
            self.saved_video(agent)
            self.assertIsNone(self.worker.recorder)
        asyncio.run(check())

    def test_normal_one_round_finalizes_playable_video(self):
        async def check():
            agent = await self.prepare()
            async def boundary(config, prompt, cameras):
                pose = json.loads(prompt)['observation']['tcp']; pose['position_m'][2] += .01
                return PoseAction(status='move', **pose, gripper_open=1., duration_s=.7,
                                  explanation='Normal native video fixture.'), None
            with patch('server.agent.api_action', boundary):
                await agent.start(AgentStart(task='Finish one recorded round.', one_round=True, record_video=True))
                await asyncio.wait_for(agent.task, 20)
            self.assertEqual(agent.state['status'], 'round_complete', agent.state)
            self.assertEqual(agent.state['history'][0]['result']['status'], 'complete')
            self.assertGreater(len(self.saved_video(agent)), 1)
            self.assertIsNone(self.worker.recorder)
        asyncio.run(check())

    def test_disabled_recording_produces_no_mp4(self):
        async def check():
            agent = await self.prepare()
            async def boundary(config, prompt, cameras):
                pose = json.loads(prompt)['observation']['tcp']; pose['position_m'][2] += .01
                return PoseAction(status='move', **pose, gripper_open=1., duration_s=.7,
                                  explanation='Video disabled fixture.'), None
            with patch('server.agent.api_action', boundary):
                await agent.start(AgentStart(task='Move without video.', one_round=True, record_video=False))
                await asyncio.wait_for(agent.task, 20)
            self.assertEqual(agent.state['status'], 'round_complete', agent.state)
            self.assertFalse(agent.state['video']['enabled'])
            self.assertEqual(agent.state['video']['status'], 'disabled')
            self.assertEqual(list((Path(self.logs.name)/agent.state['run_id']).rglob('*.mp4')), [])
        asyncio.run(check())


class WorkerVideoCleanupTests(unittest.TestCase):
    """Each fault owns its Worker so it cannot damage the shared Agent fixture."""

    def check_flush_failure(self, stop_command):
        worker = Worker()
        closed = []
        thread_errors = []
        original_close = Simulation.close
        def close(simulation):
            closed.append(simulation)
            return original_close(simulation)
        with tempfile.TemporaryDirectory(prefix='worker-video-io-fixture-') as temporary, \
                patch.object(Simulation, 'close', close):
            log = RunLog('test', {}, directory=Path(temporary), record_video=True)
            worker.thread.start()
            try:
                self.assertTrue(worker.ready.wait(30))
                self.assertIsNone(worker.failure)
                asyncio.run(worker.execute({'action': 'pause', 'value': True}))
                asyncio.run(worker.execute({'action': 'record_start', 'log': log}))
                encoder = log._video.process
                self.assertIsNotNone(encoder)
                self.assertIsNone(encoder.poll())
                message = 'Explicit trajectory flush IO failure at ' + ('record_stop' if stop_command else 'worker shutdown')
                with patch.object(log, 'flush_trajectory', side_effect=OSError(message)), \
                        patch('server.app.traceback.print_exc'), \
                        patch('threading.excepthook', lambda args: thread_errors.append(args.exc_value)):
                    if stop_command:
                        with self.assertRaises(HTTPException) as error:
                            asyncio.run(worker.execute({'action': 'record_stop'}))
                        self.assertEqual(error.exception.status_code, 500)
                        self.assertIn(message, error.exception.detail)
                    else:
                        worker.shutdown.set()
                    worker.thread.join(20)
                self.assertFalse(worker.thread.is_alive())
                self.assertIsNone(worker.recorder)
                self.assertIsNotNone(encoder.poll(), 'The encoder must exit even if trajectory flush fails.')
                self.assertEqual(log.artifacts['video']['status'], 'saved', log.artifacts['video'])
                self.assertEqual(len(closed), 1, 'Native Simulation.close must run in the owning Worker thread.')
                if stop_command:
                    self.assertIn(message, worker.failure)
                else:
                    self.assertEqual(len(thread_errors), 1)
                    self.assertIsInstance(thread_errors[0], OSError)
                    self.assertEqual(str(thread_errors[0]), message)
            finally:
                worker.shutdown.set(); worker.thread.join(20)
                log.finalize_video()
                log.finish({'status': 'test_complete'})

    def test_record_stop_flush_error_still_closes_encoder_and_renderer(self):
        self.check_flush_failure(stop_command=True)

    def test_shutdown_flush_error_still_closes_encoder_and_renderer(self):
        self.check_flush_failure(stop_command=False)


class ComparisonVideoCancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tempfile.TemporaryDirectory(prefix='comparison-video-fixture-')
        cls.log_patch = patch('server.runlog.RUNS', Path(cls.logs.name)); cls.log_patch.start()
        cls.worker = Worker(); cls.worker.thread.start()
        assert cls.worker.ready.wait(30)
        assert cls.worker.failure is None, cls.worker.failure

    @classmethod
    def tearDownClass(cls):
        cls.worker.shutdown.set(); cls.worker.thread.join(5)
        cls.log_patch.stop(); cls.logs.cleanup()

    def controllers(self):
        agent = AgentController(self.worker)
        agent.configure(AgentConfig(provider='compatible', api_key='local-comparison-fixture'))
        policy = PolicyController(self.worker)
        for kind in ('vla', 'wam'):
            policy.configure(PolicyConfig(kind=kind, endpoint='http://127.0.0.1:9/local-boundary-fixture',
                                          model='local-policy-fixture', max_chunks=1))
        return agent, policy, ComparisonController(self.worker, agent, policy)

    def test_cancelled_current_trial_keeps_saved_video_in_comparison_row(self):
        async def check():
            agent, policy, comparison = self.controllers(); thinking = asyncio.Event()
            async def boundary(*args):
                thinking.set(); await asyncio.sleep(60)
            with patch('server.agent.api_action', boundary):
                await comparison.start(ComparisonStart(robot='so101', methods=['agent', 'vla'], record_video=True))
                await asyncio.wait_for(thinking.wait(), 15)
                await comparison.stop()
            self.assertEqual(comparison.state['status'], 'stopped')
            row = comparison.state['rows'][0]
            self.assertEqual(row['status'], 'stopped')
            self.assertEqual(row['run_id'], agent.state['run_id'])
            self.assertEqual(row['video'], agent.state['video'])
            self.assertEqual(row['video']['status'], 'saved', row)
            folder = Path(self.logs.name)/row['run_id']
            self.assertTrue((folder/row['video']['path']).is_file())
            self.assertTrue((folder/'final.jpg').is_file())
            summary = json.loads((Path(self.logs.name)/comparison.state['run_id']/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['rows'][0]['run_id'], row['run_id'])
            self.assertEqual(summary['rows'][0]['video']['path'], row['video']['path'])
            self.assertIsNone(comparison.state['rows'][1]['run_id'])
            from server.video import find_ffmpeg
            decoded = subprocess.run([find_ffmpeg(), '-v', 'error', '-i', str(folder/row['video']['path']),
                '-f', 'framemd5', '-'], capture_output=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.assertEqual(decoded.returncode, 0, decoded.stderr.decode('utf-8', 'replace'))
            self.assertTrue(any(line and not line.startswith('#') for line in decoded.stdout.decode().splitlines()))
        asyncio.run(check())

    def test_cancel_in_next_start_does_not_reuse_previous_policy_video(self):
        async def check():
            agent, policy, comparison = self.controllers(); next_start = asyncio.Event()
            original_start = policy.start
            original_execute = self.worker.execute
            async def policy_boundary(config, body, client=None):
                actuators = body['observation']['actuators']
                return {'protocol': 'embodied-policy-v1', 'action_type': 'actuator_targets',
                    'joint_names': [actuator['name'] for actuator in actuators],
                    'actions': [[actuator['target'] for actuator in actuators]], 'dt_s': .05}
            async def start_boundary(request):
                if request.kind != 'wam': return await original_start(request)
                async def wait_before_stop(command):
                    if command['action'] == 'stop':
                        next_start.set(); await asyncio.sleep(60)
                    return await original_execute(command)
                with patch.object(self.worker, 'execute', wait_before_stop):
                    return await original_start(request)
            with patch('server.policy.predict_policy', policy_boundary), patch.object(policy, 'start', start_boundary):
                await comparison.start(ComparisonStart(robot='so101', methods=['vla', 'wam'], record_video=True))
                await asyncio.wait_for(next_start.wait(), 15)
                await comparison.stop()
            first, second = comparison.state['rows']
            # Comparison runs continuous trials; max_chunks=1 ends at its round budget.
            self.assertEqual(first['status'], 'limit', first)
            self.assertEqual(first['video']['status'], 'saved', first)
            self.assertTrue(first['run_id'])
            self.assertEqual(second['status'], 'stopped', second)
            self.assertIsNone(second['run_id'], 'An unstarted trial must not link the previous policy run.')
            self.assertIsNone(second.get('video'))
            self.assertIsNone(second['metrics'])
        asyncio.run(check())


if __name__ == '__main__': unittest.main()
