"""Opt-in real native workers and visual Agent episodes, without external models.

Run in the main environment with EMBODI_TEST_MANISKILL=1 and/or
EMBODI_TEST_ISAAC=1. Tests execute sequentially and close each native process.
"""
import asyncio
import base64
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


class RobotBackendsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(os.getenv('EMBODI_TEST_MANISKILL') == '1', 'Opt-in native ManiSkill worker acceptance')
    async def test_maniskill_visual_episode_rejects_false_actor_completion(self):
        # A 2 cm TCP movement cannot complete official PickCube. Verifying that
        # the actor's second-round done is rejected is part of the contract.
        await self._episode('maniskill', 'pick_cube', 'pick_cube', False, 'maniskill_official_evaluate')

    @unittest.skipUnless(os.getenv('EMBODI_TEST_ISAAC') == '1', 'Opt-in native Isaac worker acceptance')
    async def test_isaac_visual_episode_verifies_actual_reach(self):
        await self._episode('isaac', 'isaac_tabletop', 'isaac_reach_up', True, 'isaaclab_newton_state')

    async def _episode(self, engine, scene, task, expected_success, evaluator_source):
        worker = Worker(); worker.thread.start(); agent = None
        try:
            self.assertTrue(await asyncio.to_thread(worker.ready.wait, 30))
            self.assertIsNone(worker.failure)
            await worker.execute({'action': 'stop'})
            switched = await worker.execute({'action': 'backend_select', 'engine_id': engine})
            self.assertEqual(switched['backend']['engine_id'], engine)
            # Exercise the first-load UI path before any scene/reset/settle.
            initial_observation = await worker.execute({'action': 'observe'})
            json.dumps(initial_observation, allow_nan=False)
            self.assertEqual(initial_observation['time'], 0.)
            capture = os.getenv('EMBODI_NATIVE_CAPTURE_DIR')
            if capture:
                target = Path(capture).resolve(); target.mkdir(parents=True, exist_ok=True)
                with worker.lock: viewer = bytes(worker.jpeg)
                self.assertTrue(viewer)
                (target/'viewer-initial.jpg').write_bytes(viewer)
                for camera in initial_observation['cameras']:
                    (target/(camera['name']+'-initial.jpg')).write_bytes(base64.b64decode(camera['image_base64']))
            await worker.execute({'action': 'scene', 'scene_id': scene, 'task_id': task, 'seed': 42})
            await worker.execute({'action': 'settle'})
            agent = AgentController(worker)
            agent.config = AgentConfig(provider='compatible', model='local-native-fixture',
                base_url='http://127.0.0.1:9/v1', api_key='fixture-only', max_rounds=2)
            calls = []

            async def inference(config, prompt, cameras):
                observation = json.loads(prompt)['observation']; calls.append(observation)
                self.assertEqual(observation['backend']['engine_id'], engine)
                for key in ('evaluation', 'blocks', 'checkpoint'): self.assertNotIn(key, observation)
                self.assertEqual([c['name'] for c in cameras], ['front', 'side', 'wrist'])
                self.assertEqual(len({c['image_base64'] for c in cameras}), 3)
                pose = observation['tcp']; status = 'move' if len(calls) == 1 else 'done'
                if status == 'move': pose['position_m'][2] += .02
                return PoseAction(status=status, **pose, gripper_open=1., duration_s=1.,
                    explanation='Local two-round fixture; only the simulator may verify task success.'), None

            with tempfile.TemporaryDirectory(prefix=engine+'-workbench-') as folder:
                with patch('server.runlog.RUNS', Path(folder)), patch('server.agent.api_action', inference):
                    await agent.start(AgentStart(task='沿世界 Z 轴抬高末端 2 cm，然后申请环境验收。',
                        max_sim_time_s=5., evaluator_stop=False, record_video=bool(find_ffmpeg())))
                    await asyncio.wait_for(agent.task, 180)
                self.assertEqual(len(calls), 2, agent.state)
                self.assertEqual(agent.state['status'], 'verified' if expected_success else 'limit', agent.state)
                self.assertEqual(agent.state['metrics']['success'], expected_success, agent.state)
                folder_path = Path(folder)/agent.state['run_id']
                summary = json.loads((folder_path/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['backend']['engine_id'], engine)
                self.assertGreater(summary['artifacts']['trajectory_frames'], 10)
                self.assertEqual(summary['artifacts']['observation_count'], 2)
                events = [json.loads(line) for line in (folder_path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                verification = next(e['data'] for e in events if e['event'] == 'completion_verification')
                self.assertEqual(verification['accepted'], expected_success)
                self.assertEqual(verification['evaluation']['source'], evaluator_source)
                frames = [json.loads(line) for line in (folder_path/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertTrue(all(frame['backend']['engine_id'] == engine for frame in frames))
                self.assertGreater(frames[-1]['time'], frames[0]['time'])
                self.assertNotEqual(frames[-1]['qpos'], frames[0]['qpos'])
                if find_ffmpeg():
                    self.assertEqual(summary['artifacts']['video']['status'], 'saved', summary['artifacts']['video'])
                    video = folder_path/summary['artifacts']['video']['path']
                    decoded = subprocess.run([find_ffmpeg(), '-v', 'error', '-i', str(video), '-f', 'null', '-'], capture_output=True, timeout=30)
                    self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors='replace'))
                replay = frames[len(frames)//2]
                await worker.execute({'action': 'replay_frame', 'frame': replay})
                with worker.lock: state = worker.snapshot.copy()
                self.assertAlmostEqual(state['time'], replay['time']); self.assertIsNotNone(state['playback'])
                await worker.execute({'action': 'backend_select', 'engine_id': 'mujoco'})
                with self.assertRaises(HTTPException) as rejected:
                    await worker.execute({'action': 'replay_frame', 'frame': replay})
                self.assertEqual(rejected.exception.status_code, 400)
                self.assertIn('其他仿真引擎', str(rejected.exception)); self.assertIsNone(worker.failure)
        finally:
            if agent is not None: await agent.stop()
            worker.shutdown.set(); await asyncio.to_thread(worker.thread.join, 10)
            self.assertFalse(worker.thread.is_alive(), 'Native simulator process must be released.')
