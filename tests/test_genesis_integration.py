"""Opt-in real Genesis process + shared visual loop + central episode artifacts.

Run with EMBODI_TEST_GENESIS=1 using the main workbench Python. Only model
inference is a local fixture; physics, IK, three cameras, recording and replay
use the installed native Genesis runtime.
"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from fastapi import HTTPException

from server.actions import PoseAction
from server.agent import AgentConfig, AgentController, AgentStart
from server.app import Worker
from server.video import find_ffmpeg


@unittest.skipUnless(os.getenv('EMBODI_TEST_GENESIS') == '1', 'Opt-in native Genesis process acceptance')
class GenesisIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_visual_episode_preserves_recording_verification_and_replay(self):
        config = AgentConfig(provider='compatible', model='local-test-fixture',
            base_url='http://127.0.0.1:9/v1', api_key='fixture-only', max_rounds=2)
        worker = Worker()
        worker.thread.start()
        agent = None
        try:
            self.assertTrue(await asyncio.to_thread(worker.ready.wait, 30))
            self.assertIsNone(worker.failure)
            await worker.execute({'action': 'stop'})
            switched = await worker.execute({'action': 'backend_select', 'engine_id': 'genesis'})
            self.assertEqual(switched['backend']['engine_id'], 'genesis')
            await worker.execute({'action': 'scene', 'scene_id': 'tabletop', 'task_id': 'reach_up', 'seed': 42})
            await worker.execute({'action': 'settle'})
            agent = AgentController(worker)
            agent.config = config
            calls = []

            async def inference(config, prompt, cameras):
                observation = json.loads(prompt)['observation']
                calls.append(observation)
                self.assertEqual(observation['backend']['engine_id'], 'genesis')
                self.assertNotIn('evaluation', observation)
                self.assertNotIn('blocks', observation)
                self.assertEqual({c['name'] for c in cameras}, {'front', 'side', 'wrist'})
                self.assertEqual(len({c['image_base64'] for c in cameras}), 3)
                pose = observation['tcp']
                status = 'move' if len(calls) == 1 else 'done'
                if status == 'move': pose['position_m'][2] += .02
                return PoseAction(status=status, **pose, gripper_open=1., duration_s=1.,
                                  explanation='Explicit local native-process acceptance action.'), None

            with tempfile.TemporaryDirectory(prefix='genesis-integration-') as folder:
                with patch('server.runlog.RUNS', Path(folder)), patch('server.agent.api_action', inference):
                    await agent.start(AgentStart(task='沿世界 Z 轴抬高末端 2 cm。', max_sim_time_s=5., evaluator_stop=False,
                                                record_video=bool(find_ffmpeg())))
                    await asyncio.wait_for(agent.task, 120)
                self.assertEqual(agent.state['status'], 'verified', agent.state)
                self.assertTrue(agent.state['metrics']['success'], agent.state)
                self.assertEqual(len(calls), 2)
                saved = Path(folder) / agent.state['run_id']
                summary = json.loads((saved / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['backend']['engine_id'], 'genesis')
                self.assertGreater(summary['artifacts']['trajectory_frames'], 20)
                self.assertEqual(summary['artifacts']['observation_count'], 2)
                events = [json.loads(line) for line in (saved / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
                verification = next(e['data'] for e in events if e['event'] == 'completion_verification')
                self.assertTrue(verification['accepted'])
                self.assertEqual(verification['evaluation']['source'], 'genesis_privileged_state')
                frames = [json.loads(line) for line in (saved / 'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertTrue(all(f['backend']['engine_id'] == 'genesis' for f in frames))
                self.assertGreater(frames[-1]['time'], frames[0]['time'])
                self.assertNotEqual(frames[-1]['qpos'], frames[0]['qpos'])
                if find_ffmpeg():
                    self.assertEqual(summary['artifacts']['video']['status'], 'saved', summary['artifacts']['video'])
                    video = saved / summary['artifacts']['video']['path']
                    decoded = subprocess.run([find_ffmpeg(), '-v', 'error', '-i', str(video), '-f', 'null', '-'],
                                             capture_output=True, timeout=30)
                    self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors='replace'))
                replay = frames[len(frames)//2]
                await worker.execute({'action': 'replay_frame', 'frame': replay})
                with worker.lock: state = worker.snapshot.copy()
                self.assertAlmostEqual(state['time'], replay['time'])
                self.assertIsNotNone(state['playback'])
                await worker.execute({'action': 'backend_select', 'engine_id': 'mujoco'})
                with self.assertRaises(HTTPException) as rejected:
                    await worker.execute({'action': 'replay_frame', 'frame': replay})
                self.assertEqual(rejected.exception.status_code, 400)
                self.assertIn('其他仿真引擎', str(rejected.exception))
                self.assertIsNone(worker.failure)
        finally:
            if agent is not None: await agent.stop()
            worker.shutdown.set()
            await asyncio.to_thread(worker.thread.join, 10)
            self.assertFalse(worker.thread.is_alive(), 'Native simulator process must be released.')
