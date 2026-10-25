"""Opt-in real Worker/WSL/household SDK acceptance, without paid inference.

EMBODI_TEST_HOUSEHOLD=1 runs ALFWorld. The separate
EMBODI_TEST_HOUSEHOLD_THOR=1 flag starts a real Unity instance and should be run
after other heavy SDKs have closed. All experiment artifacts use temporary dirs.
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

from server.actions import EnvironmentAction
from server.agent import AgentConfig, AgentController, AgentStart
from server.app import Worker
from server.runlog import RunLog
from server.video import find_ffmpeg


class HouseholdWorkerCase(unittest.IsolatedAsyncioTestCase):
    async def open_worker(self, engine):
        worker = Worker(); worker.thread.start()
        self.addAsyncCleanup(self.close_worker, worker)
        self.assertTrue(await asyncio.to_thread(worker.ready.wait, 30))
        self.assertIsNone(worker.failure)
        await worker.execute({'action': 'stop'})
        result = await worker.execute({'action': 'backend_select', 'engine_id': engine})
        self.assertEqual(result['backend']['engine_id'], engine)
        return worker

    async def close_worker(self, worker):
        worker.shutdown.set()
        await asyncio.to_thread(worker.thread.join, 10)
        self.assertFalse(worker.thread.is_alive(), 'Household process must close with its worker.')

    def assert_saved_video(self, summary, folder):
        if not find_ffmpeg(): return
        video = summary['artifacts']['video']
        self.assertEqual(video['status'], 'saved', video)
        self.assertGreaterEqual(video['frame_count'], 2)
        result = subprocess.run([find_ffmpeg(), '-v', 'error', '-i', str(folder/video['path']), '-f', 'null', '-'],
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))


@unittest.skipUnless(os.getenv('EMBODI_TEST_HOUSEHOLD') == '1', 'Opt-in native ALFWorld/WSL integration')
class ALFWorldIntegrationTests(HouseholdWorkerCase):
    async def test_inconsistent_native_restore_marks_worker_unready_instead_of_publishing_partial_state(self):
        worker = Worker(); worker.initial_engine_id = 'alfworld'; worker.thread.start()
        self.addAsyncCleanup(self.close_worker, worker)
        self.assertTrue(await asyncio.to_thread(worker.ready.wait, 30))
        self.assertIsNone(worker.failure)
        checkpoint = await worker.execute({'action':'checkpoint'})
        old_session = worker.session_id
        observation = await worker.execute({'action':'observe'})
        action = next(value for value in observation['environment']['admissible_commands'] if value.startswith('go to '))
        await worker.execute({'action':'environment_action','generation':observation['generation'],'environment_action':action})
        # A mismatched saved observation simulates a replay that the current
        # official assets cannot reproduce. Reset has already changed the world.
        checkpoint['environment']['observation_text'] += '\nExpected observation differs.'
        with self.assertRaises(HTTPException) as failure:
            await worker.execute({'action':'restore','checkpoint':checkpoint})
        self.assertEqual(failure.exception.status_code,500)
        with worker.lock:
            self.assertFalse(worker.snapshot['ready'])
            self.assertIn('检查点恢复失败',worker.snapshot['error'])
        self.assertIsNotNone(worker.failure)
        recovered = await worker.restart('alfworld')
        self.assertEqual(recovered['backend']['engine_id'],'alfworld')
        self.assertNotEqual(worker.session_id,old_session)
        self.assertIsNone(worker.failure)
        with worker.lock: self.assertTrue(worker.snapshot['ready'])
        with self.assertRaises(HTTPException) as stale:
            await worker.execute({'action':'environment_action','generation':observation['generation'],'environment_action':action})
        self.assertEqual(stale.exception.status_code,400)
        with worker.lock: self.assertTrue(worker.snapshot['ready'])

    async def test_false_done_replans_native_action_and_archives_real_text_episode(self):
        worker = await self.open_worker('alfworld')
        agent = AgentController(worker)
        self.addAsyncCleanup(agent.stop)
        agent.config = AgentConfig(provider='compatible', model='local-household-test-fixture',
            base_url='http://127.0.0.1:9/v1', api_key='fixture-only', max_rounds=3)
        observations, selected = [], []

        async def inference(config, prompt, cameras, **options):
            payload = json.loads(prompt); observation = payload['observation']
            observations.append(observation)
            self.assertIs(options.get('response_type'), EnvironmentAction)
            self.assertEqual(observation['backend']['action_mode'], 'text')
            self.assertEqual(cameras, [])
            self.assertEqual(observation['cameras'], [])
            self.assertIsNone(observation['tcp'])
            for hidden in ('evaluation', 'verification', 'pddl_problem', 'pddl_domain', 'walkthrough'):
                self.assertNotIn(hidden, observation)
                self.assertNotIn(hidden, observation['environment'])
            if len(observations) == 2:
                # A native navigation action changes the actual text world.
                action = next(value for value in observation['environment']['admissible_commands'] if value.startswith('go to '))
                selected.append(action)
                return EnvironmentAction(status='move', action=action, explanation='Fixed local native action fixture.'), None
            return EnvironmentAction(status='done', action='look', explanation='Deliberately incorrect completion claim.'), None

        with tempfile.TemporaryDirectory(prefix='alfworld-integration-') as temporary:
            with patch('server.runlog.RUNS', Path(temporary)), patch('server.agent.api_action', inference):
                await agent.start(AgentStart(task='完成当前官方 ALFWorld 任务。', max_sim_time_s=10.,
                                            evaluator_stop=False, record_video=bool(find_ffmpeg())))
                await asyncio.wait_for(agent.task, 120)
            self.assertEqual(agent.state['status'], 'limit', agent.state)
            self.assertFalse(agent.state['metrics']['success'], agent.state)
            self.assertEqual(len(observations), 3)
            self.assertEqual([item['time'] for item in observations], [0., 0., 1.])
            self.assertNotEqual(observations[0]['environment']['observation_text'], observations[-1]['environment']['observation_text'])
            folder = Path(temporary)/agent.state['run_id']
            summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['backend']['engine_id'], 'alfworld')
            self.assertFalse(summary['metrics']['success'])
            self.assertIsNone(summary['metrics']['sim_time_s'])
            self.assertEqual(summary['metrics']['environment_steps'], 1)
            self.assertEqual(summary['artifacts']['observation_count'], 3)
            self.assertGreaterEqual(summary['artifacts']['trajectory_frames'], 2)
            events = [json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            verifications = [event['data'] for event in events if event['event'] == 'completion_verification']
            self.assertEqual(len(verifications), 2)
            self.assertTrue(all(event['accepted'] is False for event in verifications))
            self.assertTrue(all(event['evaluation']['source'] == 'alfworld_native_task' for event in verifications))
            saved = [event['data']['observation'] for event in events if event['event'] == 'observation']
            self.assertEqual([item['environment']['observation_text'] for item in saved],
                             [item['environment']['observation_text'] for item in observations])
            self.assertTrue(all(item['cameras'] == [] for item in saved))
            manifests = [json.loads(line) for line in (folder/'observations/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(manifests)
            self.assertTrue(all(item['backend']['action_mode'] == 'text' for item in manifests))
            self.assertEqual(manifests[-1]['environment']['observation_text'], observations[-1]['environment']['observation_text'])
            self.assertEqual(list((folder/'observations').rglob('*.jpg')), [])
            trajectory = [json.loads(line) for line in (folder/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(trajectory[-1]['actions'], selected)
            self.assertEqual(trajectory[-1]['backend']['engine_id'], 'alfworld')
            self.assertEqual(trajectory[-1]['time'], 1.)
            self.assert_saved_video(summary, folder)
            old_generation = observations[-1]['generation']
            old_session = worker.session_id
            await worker.execute({'action': 'backend_select', 'engine_id': 'mujoco'})
            self.assertNotEqual(worker.session_id, old_session)
            with self.assertRaises(HTTPException) as rejected:
                await worker.execute({'action': 'environment_action', 'generation': old_generation, 'environment_action': selected[0]})
            self.assertEqual(rejected.exception.status_code, 400)
            with self.assertRaises(HTTPException) as foreign:
                await worker.execute({'action': 'replay_frame', 'frame': trajectory[-1]})
            self.assertEqual(foreign.exception.status_code, 400)
            self.assertIsNone(worker.failure)


@unittest.skipUnless(os.getenv('EMBODI_TEST_HOUSEHOLD_THOR') == '1', 'Opt-in native Unity/WSLg bridge integration')
class ThorIntegrationTests(HouseholdWorkerCase):
    async def test_one_native_camera_metadata_verification_and_explicit_action_replay(self):
        worker = await self.open_worker('ai2thor')
        observation = await worker.execute({'action': 'observe'})
        self.assertEqual([camera['name'] for camera in observation['cameras']], ['front'])
        self.assertIsNone(observation['tcp'])
        self.assertNotIn('evaluation', observation)
        self.assertNotIn('|', json.dumps(observation['environment']))
        self.assertNotIn('distance', json.dumps(observation['environment']))
        self.assertEqual(worker.snapshot['backend']['state_schema'], 'ai2thor-integration-v2')
        with self.assertRaises(HTTPException) as native_id:
            await worker.execute({'action': 'environment_action', 'generation': observation['generation'],
                                  'environment_action': '{"action":"OpenObject","objectId":"Cabinet|-01.55|+00.50|+00.38"}'})
        self.assertEqual(native_id.exception.status_code, 400)
        evaluation = await worker.execute({'action': 'verify_task', 'generation': observation['generation']})
        self.assertEqual(evaluation['source'], 'ai2thor_native_task')
        self.assertFalse(evaluation['success'])
        with tempfile.TemporaryDirectory(prefix='thor-integration-') as temporary:
            with patch('server.runlog.RUNS', Path(temporary)):
                log = RunLog('agent', {'model': 'native-bridge-fixture', 'backend': worker.snapshot['backend']},
                             record_video=bool(find_ffmpeg()))
                recording = False
                try:
                    saved = log.save_observation(observation, 1)
                    self.assertEqual(len(saved['cameras']), 1)
                    await worker.execute({'action': 'record_start', 'log': log}); recording = True
                    response = await worker.execute({'action': 'environment_action', 'generation': observation['generation'],
                                                     'environment_action': 'RotateRight'})
                    self.assertTrue(response['last_action_success'], response)
                    frame = await worker.execute({'action': 'checkpoint'})
                    self.assertEqual(frame['actions'], ['RotateRight'])
                    await worker.execute({'action': 'record_stop'}); recording = False
                    log.finish({'status': 'completed', 'metrics': {'success': False}})
                    summary = json.loads((log.folder/'summary.json').read_text(encoding='utf-8'))
                    self.assertGreaterEqual(summary['artifacts']['trajectory_frames'], 2)
                    self.assert_saved_video(summary, log.folder)
                    await worker.execute({'action': 'environment_action', 'generation': observation['generation'],
                                          'environment_action': 'RotateLeft'})
                    replay = await worker.execute({'action': 'replay_frame', 'frame': frame})
                    self.assertEqual(replay['restore_mode'], 'native_action_replay')
                    after = await worker.execute({'action': 'observe'})
                    self.assertEqual(after['environment'], frame['environment'])
                    self.assertEqual([camera['name'] for camera in after['cameras']], ['front'])
                    self.assertNotIn('|', json.dumps(after['environment']))
                finally:
                    if recording: await worker.execute({'action': 'record_stop'})
                    log.finish({'status': 'stopped'})
        await worker.execute({'action': 'backend_select', 'engine_id': 'mujoco'})
        self.assertIsNone(worker.failure)


@unittest.skipUnless(os.getenv('EMBODI_TEST_ALFWORLD_THOR_WORKBENCH') == '1', 'Opt-in official ALFWorld visual Worker/Agent acceptance')
class ALFWorldVisualIntegrationTests(HouseholdWorkerCase):
    async def test_visual_agent_records_native_action_rejects_false_done_and_replays(self):
        worker = await self.open_worker('alfworld_thor')
        agent = AgentController(worker)
        self.addAsyncCleanup(agent.stop)
        agent.config = AgentConfig(provider='compatible', model='local-visual-fixture',
            base_url='http://127.0.0.1:9/v1', api_key='fixture-only', max_rounds=2)
        images = []

        async def inference(config, prompt, cameras, **options):
            observation = json.loads(prompt)['observation']
            self.assertIs(options.get('response_type'), EnvironmentAction)
            self.assertEqual(observation['backend']['engine_id'], 'alfworld_thor')
            self.assertEqual([camera['name'] for camera in cameras], ['front'])
            self.assertIsNone(observation['tcp'])
            public = json.dumps(observation)
            for hidden in ('pddl_params', 'object_poses', 'low_actions', 'high_pddl', 'evaluation', 'goal_conditions_met'):
                self.assertNotIn(hidden, public)
            images.append(cameras[0]['image_base64'])
            return EnvironmentAction(status='move' if len(images) == 1 else 'done',
                action='RotateRight', explanation='Fixed local test, not model performance.'), None

        with tempfile.TemporaryDirectory(prefix='alfworld-visual-agent-') as temporary:
            with patch('server.runlog.RUNS', Path(temporary)), patch('server.agent.api_action', inference):
                await agent.start(AgentStart(task='完成当前官方 ALFWorld 视觉任务。', max_environment_steps=4,
                                            evaluator_stop=False, record_video=bool(find_ffmpeg())))
                await asyncio.wait_for(agent.task, 120)
            self.assertEqual(agent.state['status'], 'limit', agent.state)
            self.assertEqual(len(images), 2)
            self.assertNotEqual(images[0], images[1])
            folder = Path(temporary)/agent.state['run_id']
            summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['backend']['engine_id'], 'alfworld_thor')
            self.assertFalse(summary['metrics']['success'])
            self.assertEqual(summary['metrics']['environment_steps'], 1)
            self.assertIsNone(summary['metrics']['sim_time_s'])
            self.assertEqual(summary['artifacts']['observation_count'], 2)
            self.assert_saved_video(summary, folder)
            events = [json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            verification = [item['data'] for item in events if item['event'] == 'completion_verification']
            self.assertEqual(len(verification), 1)
            self.assertFalse(verification[0]['accepted'])
            self.assertEqual(verification[0]['evaluation']['source'], 'alfworld_thor_native_task')
            frame = json.loads((folder/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()[-1])
            self.assertEqual(frame['actions'], ['RotateRight'])
            await worker.execute({'action':'environment_action','generation':frame['generation'], 'environment_action':'RotateLeft'})
            replay = await worker.execute({'action':'replay_frame','frame':frame})
            self.assertEqual(replay['restore_mode'], 'native_action_replay')
            self.assertEqual((await worker.execute({'action':'observe'}))['environment'], frame['environment'])
        await worker.execute({'action':'backend_select','engine_id':'mujoco'})
        self.assertIsNone(worker.failure)


if __name__ == '__main__': unittest.main()
