import asyncio
import base64
import io
import json
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image
from pydantic import ValidationError

from server.agent import AgentConfig, AgentController, AgentStart, ModelResponseError, observation_text
from server.app import Worker
from server.control import PoseAction, tcp_pose, solve_ik
from server.simulation import Simulation


def action_at(pose, **changes):
    return PoseAction.model_validate({'status': 'move', **pose, 'gripper_open': 1., 'duration_s': .7,
                                      'explanation': '小幅移动末端。', **changes})


class TCPTests(unittest.TestCase):
    def test_absolute_pose_reaches_with_actuators_on_both_models(self):
        for model in ('panda', 'so101'):
            with self.subTest(model=model):
                sim = Simulation(model, render=False)
                try:
                    for _ in range(500): sim.step()
                    scratch = mujoco.MjData(sim.model); scratch.qpos[:] = sim.data.qpos
                    scratch.qpos[sim.actuators[0]['qpos_index']] += .06
                    goal = tcp_pose(sim.model, scratch, sim.site_id)
                    action = action_at(goal)
                    before = sim.data.qpos.copy()
                    solution = solve_ik(sim, action)
                    np.testing.assert_array_equal(sim.data.qpos, before)
                    self.assertEqual(len(solution), sim.config['arm_dof'])
                    sim.move_tcp({'generation': sim.generation, 'pose': action.model_dump()})
                    np.testing.assert_array_equal(sim.data.qpos, before)
                    for _ in range(2200): sim.step()
                    self.assertEqual(sim.task['status'], 'complete', sim.task)
                    error = np.linalg.norm(np.array(tcp_pose(sim.model, sim.data, sim.site_id)['position_m']) - goal['position_m'])
                    self.assertLess(error, .006 * sim.config['scale'])
                finally: sim.close()

    def test_reject_far_invalid_and_underactuated_pose_without_mutation(self):
        sim = Simulation('so101', render=False)
        try:
            pose = tcp_pose(sim.model, sim.data, sim.site_id)
            before_q, before_ctrl = sim.data.qpos.copy(), sim.data.ctrl.copy()
            far = action_at(pose, position_m=[3., 3., 3.])
            with self.assertRaisesRegex(ValueError, '平移超过'): solve_ik(sim, far)
            impossible = action_at(pose, position_m=(np.array(pose['position_m'])+[0, .035, 0]).tolist())
            with self.assertRaisesRegex(ValueError, 'IK|关节变化'): solve_ik(sim, impossible)
            for changes in ({'quaternion_wxyz': [0., 0., 0., 0.]}, {'position_m': [float('nan'), 0., 0.]},
                            {'frame': 'relative'}, {'gripper_open': 2.}, {'object_pose': [0, 0, 0]}):
                with self.subTest(changes=changes), self.assertRaises(ValidationError): action_at(pose, **changes)
            np.testing.assert_array_equal(before_q, sim.data.qpos)
            np.testing.assert_array_equal(before_ctrl, sim.data.ctrl)
        finally: sim.close()

    def test_configuration_never_returns_key_or_reuses_across_hosts(self):
        agent = AgentController(None)
        settings = AgentConfig(provider='openai', api_key='test-secret')
        agent.configure(settings)
        self.assertNotIn('test-secret', json.dumps(agent.public_config()))
        agent.configure(AgentConfig(provider='openai', base_url='https://example.com/v1'))
        self.assertFalse(agent.public_config()['has_api_key'])
        with self.assertRaises(ValidationError): AgentConfig(base_url='http://example.com/v1')

    def test_output_budget_and_thinking_settings_round_trip_without_losing_key(self):
        agent = AgentController(None)
        agent.configure(AgentConfig(provider='compatible', base_url='https://api.deepseek.com', api_key='test-secret'))
        for tokens in (32768, 131073, 262144, 1048576):
            with self.subTest(tokens=tokens):
                result = agent.configure(AgentConfig(provider='compatible', base_url='https://api.deepseek.com',
                                                    max_output_tokens=tokens, deepseek_thinking='disabled'))
                self.assertTrue(result['has_api_key'])
                self.assertEqual(result['max_output_tokens'], tokens)
                self.assertEqual(result['deepseek_thinking'], 'disabled')
                self.assertNotIn('test-secret', json.dumps(result))
        for tokens in (-1, 0, 511, True, 512.5, '16384'):
            with self.subTest(tokens=tokens), self.assertRaises(ValidationError): AgentConfig(max_output_tokens=tokens)


class VisualLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tempfile.TemporaryDirectory(prefix='robot-test-logs-')
        cls.log_patch = patch('server.runlog.RUNS', Path(cls.logs.name)); cls.log_patch.start()
        cls.worker = Worker(); cls.worker.thread.start()
        assert cls.worker.ready.wait(30)
        assert not cls.worker.failure, cls.worker.failure

    @classmethod
    def tearDownClass(cls):
        cls.worker.shutdown.set(); cls.worker.thread.join(5)
        cls.log_patch.stop(); cls.logs.cleanup()

    def test_three_views_are_synchronized_and_no_object_truth_in_prompt(self):
        async def check():
            for model in ('panda', 'so101'):
                await self.worker.execute({'action': 'load', 'model': model})
                observation = await self.worker.execute({'action': 'observe'})
                self.assertEqual([c['name'] for c in observation['cameras']], ['front', 'side', 'wrist'])
                fingerprints = []
                for camera in observation['cameras']:
                    self.assertEqual(camera['time'], observation['time'])
                    jpeg = base64.b64decode(camera['image_base64'])
                    pixels = np.asarray(Image.open(io.BytesIO(jpeg)))
                    self.assertEqual(pixels.shape, (426, 640, 3))
                    self.assertGreater(pixels.std(), 8)
                    fingerprints.append(jpeg)
                self.assertEqual(len(set(fingerprints)), 3)
                text = observation_text('测试', observation, [], observation['tcp'])
                self.assertNotIn('block_', text); self.assertNotIn('image_base64', text)
                self.assertNotIn('blocks', json.loads(text)['observation'])
        asyncio.run(check())

    def test_controller_waits_for_physics_and_observes_again(self):
        # Stub only the external model boundary. Images, IK and physics are real.
        async def check():
            await self.worker.execute({'action': 'load', 'model': 'panda'})
            agent = AgentController(self.worker)
            seen = []
            async def model_boundary(config, prompt, cameras):
                info = json.loads(prompt)['observation']; seen.append(info)
                pose = info['tcp']
                if len(seen) == 1:
                    pose['position_m'][2] += .02
                    return action_at(pose), None
                return action_at(pose, status='done', explanation='已看到末端抬高。'), None
            with patch('server.agent.codex_action', model_boundary):
                await agent.start(AgentStart(task='抬高两厘米'))
                await asyncio.wait_for(agent.task, 20)
            self.assertEqual(agent.state['status'], 'model_done', agent.state)
            self.assertEqual(len(seen), 2)
            self.assertGreater(seen[1]['time'], seen[0]['time'])
            self.assertEqual(agent.state['history'][0]['result']['status'], 'complete')
            self.assertFalse(agent.state['active'])
        asyncio.run(check())

    def test_stopping_inference_prevents_late_control(self):
        async def check():
            await self.worker.execute({'action': 'load', 'model': 'panda'})
            agent = AgentController(self.worker); started = asyncio.Event()
            async def slow_model(config, prompt, cameras):
                started.set(); await asyncio.sleep(10)
                pose = json.loads(prompt)['observation']['tcp']; pose['position_m'][2] += .02
                return action_at(pose), None
            with patch('server.agent.codex_action', slow_model):
                await agent.start(AgentStart(task='等待中停止'))
                await asyncio.wait_for(started.wait(), 5)
                with self.worker.lock: before = self.worker.snapshot['time']
                await agent.stop()
                await asyncio.sleep(.15)
                with self.worker.lock: after = self.worker.snapshot
                self.assertEqual(after['time'], before)
                self.assertTrue(after['paused'])
                self.assertEqual(agent.state['status'], 'stopped')
                self.assertEqual(agent.state['history'], [])
        asyncio.run(check())

    def test_recovery_retries_same_round_then_continues_real_physics_and_video(self):
        from server.repair import RepairController, RepairConfig, RepairProposal, EDITABLE
        from server.recovery import RecoveryCoordinator
        from server.telemetry import track_inference
        from server.runlog import model_output
        async def check(root):
            for name in EDITABLE:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# protocol fixture\n')
            repair=RepairController(root)
            repair.configure(RepairConfig(provider='custom',model='local-coding-fixture',
                base_url='http://127.0.0.1:1/unused',auto_analyze=True))
            await self.worker.execute({'action':'load','model':'panda'})
            agent=AgentController(self.worker)
            agent.configure(AgentConfig(provider='compatible',model='local-visual-fixture',api_key='fixture-key'))
            agent.recovery=RecoveryCoordinator(repair)
            seen=[]
            @track_inference('agent_inference')
            async def boundary(config,prompt,cameras):
                payload=json.loads(prompt);seen.append(payload)
                self.assertEqual(len(cameras),3)
                if len(seen)==1:raise ModelResponseError('模型正文为空，本轮未执行动作。',{'input_tokens':2,'output_tokens':3})
                pose=payload['observation']['tcp']
                if len(seen)==2:
                    pose['position_m'][2]+=.02
                    action=action_at(pose)
                else:action=action_at(pose,status='done',explanation='实际末端已抬高。')
                usage={'input_tokens':2,'output_tokens':3}
                model_output({'provider':'compatible','model':config.model,'content':action.model_dump_json(),'usage':usage})
                return action,usage
            @track_inference('agent_inference')
            async def diagnose(config,prompt,cameras,**kwargs):
                self.assertEqual(cameras,[])
                proposal=RepairProposal(diagnosis='空正文在物理动作前失败。',category='provider',evidence=['empty response'],
                    recommendations=['返回严格动作实例后重试。'],edits=[],
                    recovery={'action':'retry','instruction':'Return the exact action JSON instance.','settings':None})
                usage={'input_tokens':1,'output_tokens':1}
                model_output({'provider':'custom','model':config.model,'output':proposal.model_dump(),'usage':usage})
                return proposal,usage
            with patch('server.agent.api_action',boundary),patch('server.repair.api_action',diagnose), \
                    patch.object(self.worker,'execute',wraps=self.worker.execute) as execute:
                await agent.start(AgentStart(task='抬高两厘米并确认',record_video=True))
                await asyncio.wait_for(agent.task,25)
                commands=[call.args[0]['action'] for call in execute.call_args_list]
            self.assertEqual(agent.state['status'],'model_done',agent.state)
            self.assertEqual(agent.state['round'],2);self.assertEqual(len(seen),3)
            self.assertEqual(seen[0]['observation']['time'],seen[1]['observation']['time'])
            self.assertGreater(seen[2]['observation']['time'],seen[1]['observation']['time'])
            self.assertIn('recovery_feedback',seen[1])
            self.assertEqual(commands.count('move_tcp'),1)
            self.assertEqual(commands.count('record_start'),1);self.assertEqual(commands.count('record_stop'),1)
            self.assertEqual(agent.state['history'][0]['round'],1)
            self.assertEqual(agent.state['history'][0]['result']['status'],'complete')
            self.assertEqual(agent.run_log.artifacts['observation_count'],2)
            self.assertEqual(agent.run_log._inference_count,4)
            self.assertEqual(agent.state['video']['status'],'saved')
            self.assertGreater(agent.state['video']['frame_count'],1)
            events=list(map(json.loads,agent.run_log.path.read_text(encoding='utf-8').splitlines()))
            self.assertEqual(sum(row['event']=='finish' for row in events),1)
            self.assertEqual([row['data']['round'] for row in events if row['event']=='inference'],[1,1,1,2])
        with tempfile.TemporaryDirectory(prefix='native-recovery-') as folder:
            asyncio.run(check(Path(folder)))

    def test_truncated_response_stops_before_motion_and_preserves_usage(self):
        async def check():
            await self.worker.execute({'action': 'load', 'model': 'panda'})
            agent = AgentController(self.worker)
            agent.configure(AgentConfig(provider='compatible', api_key='fixture-key'))
            usage = {'completion_tokens': 4096, 'completion_tokens_details': {'reasoning_tokens': 4096}}
            async def truncated_model(*args):
                raise ModelResponseError('模型输出被长度限制截断（finish_reason=length）。本轮未执行动作。', usage)
            with patch('server.agent.api_action', truncated_model), patch.object(self.worker, 'execute', wraps=self.worker.execute) as execute:
                await agent.start(AgentStart(task='抬高两厘米'))
                await asyncio.wait_for(agent.task, 10)
                self.assertFalse(any(call.args[0]['action'] == 'move_tcp' for call in execute.call_args_list))
            self.assertEqual(agent.state['status'], 'error')
            self.assertEqual(agent.state['usage'], usage)
            self.assertIsNone(agent.state['action'])
            self.assertEqual(agent.state['history'], [])
            self.assertFalse(agent.state['active'])
            with self.worker.lock: self.assertTrue(self.worker.snapshot['paused'])
        asyncio.run(check())


if __name__ == '__main__': unittest.main()
