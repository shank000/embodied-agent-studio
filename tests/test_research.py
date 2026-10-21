"""Real MuJoCo regression checks; only remote policy inference is a fixture."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import socket
from threading import Thread
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from server.agent import AgentController, AgentStart
from server.app import Worker
from server.comparison import ComparisonController, ComparisonStart
from server.control import PoseAction
from server.policy import PolicyConfig, PolicyController, PolicyStart, ensure_local_service, release_local_service
from server.runlog import RunLog
from server.scenes import SCENES, add_scene
from server.simulation import Simulation


class SceneTests(unittest.TestCase):
    def test_seeded_layouts_have_no_initial_object_interpenetration(self):
        for scene in SCENES:
            for scale in (1., .42):
                for seed in (*range(256), 4294967295):
                    with self.subTest(scene=scene, scale=scale, seed=seed):
                        root = ET.Element('mujoco')
                        world = ET.SubElement(root, 'worldbody')
                        add_scene(world, scale, scene, seed)
                        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
                        data = mujoco.MjData(model)
                        mujoco.mj_forward(model, data)
                        free_bodies = set(model.jnt_bodyid[model.jnt_type == mujoco.mjtJoint.mjJNT_FREE])
                        overlaps = [(model.geom(c.geom1).name, model.geom(c.geom2).name, c.dist)
                                    for c in data.contact
                                    if (model.geom_bodyid[c.geom1] in free_bodies or model.geom_bodyid[c.geom2] in free_bodies)
                                    and c.dist < -1e-7 * scale]
                        self.assertFalse(overlaps, overlaps)

    def test_drawer_objects_stay_clear_of_fixtures_on_all_robots(self):
        sim = Simulation(render=False)
        try:
            for robot in ('panda', 'so101', 'rm65_gaia'):
                for seed in (1, 17, 42, 43, 255):
                    with self.subTest(robot=robot, seed=seed):
                        sim.load(robot, 'drawer', seed)
                        initial = sim.data.qpos.copy()
                        sim.command({'action': 'settle'})
                        self.assertTrue(np.isfinite(sim.data.qpos).all())
                        for contact in sim.data.contact:
                            names = [sim.model.geom(g).name or '' for g in contact.geom]
                            object_fixture = (any(n.startswith('block_') for n in names)
                                              and any(n.startswith(('drawer_', 'cabinet_')) for n in names))
                            self.assertFalse(object_fixture and contact.dist < -1e-7 * sim.config['scale'],
                                             (names, contact.dist))
                        sim.command({'action': 'reset'})
                        np.testing.assert_array_equal(sim.data.qpos, initial)
        finally:
            sim.close()

    def test_all_scenes_settle_and_checkpoint_restores_identical_physics(self):
        sim = Simulation(render=False)
        try:
            for scene in SCENES:
                with self.subTest(scene=scene):
                    sim.load('panda', scene, 17)
                    sim.command({'action':'settle'})
                    self.assertTrue(np.isfinite(sim.data.qpos).all())
                    self.assertFalse(sim.evaluator.success)
                    saved = sim.checkpoint()
                    sim.set_control(0, .15)
                    for _ in range(100): sim.step()
                    sim.restore(saved)
                    # mj_forward recomputes warm-start accelerations; qpos/qvel/ctrl/time
                    # must be restored exactly and produce the same future trajectory.
                    first = []
                    for _ in range(20): sim.step(); first.append(sim.data.qpos.copy())
                    sim.restore(saved)
                    second = []
                    for _ in range(20): sim.step(); second.append(sim.data.qpos.copy())
                    np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)
            sim.load('panda','tabletop',17); a=sim.data.qpos.copy()
            sim.load('panda','tabletop',17); np.testing.assert_array_equal(a,sim.data.qpos)
            sim.load('panda','tabletop',18); self.assertFalse(np.array_equal(a,sim.data.qpos))
        finally: sim.close()

    def test_objective_requires_sustained_state_and_drawer_displacement(self):
        sim=Simulation(render=False,scene_id='drawer',task_id='open_drawer')
        try:
            sim.command({'action':'settle'})
            self.assertFalse(sim.evaluator.success)
            sim.data.joint('drawer_slide').qpos[0]=.15
            mujoco.mj_forward(sim.model,sim.data)
            sim.evaluator.update(sim)
            self.assertFalse(sim.evaluator.success)
            for _ in range(160): sim.step()
            self.assertTrue(sim.evaluator.success,sim.evaluator.snapshot())
        finally: sim.close()

    def test_official_rm65_gaia_has_independent_actuated_fingers(self):
        sim=Simulation('rm65_gaia',render=False)
        try:
            self.assertEqual(sim.model.nu,21)
            self.assertEqual(sim.config['hand_dof'],15)
            self.assertGreater(sim.model.nmesh,20)
            before=sim.data.qpos.copy()
            finger=sim.actuators[7]
            sim.set_control(7,.4)
            np.testing.assert_array_equal(before,sim.data.qpos)
            for _ in range(700): sim.step()
            self.assertGreater(sim.data.qpos[finger['qpos_index']],.15)
            self.assertTrue(np.isfinite(sim.data.qpos).all())
        finally: sim.close()

    def test_action_chunks_validate_order_and_units_before_actuation(self):
        sim=Simulation('so101',render=False)
        try:
            initial=sim.data.qpos.copy(); controls=sim.data.ctrl.copy()
            payload={'generation':sim.generation,'joint_names':[a['name'] for a in sim.actuators],
                     'actions':[controls.tolist()],'dt_s':.05}
            for changes in ({'joint_names':list(reversed(payload['joint_names']))}, {'actions':[[0]*7]}, {'actions':[[float('nan')]*6]}):
                with self.assertRaises(ValueError): sim.policy_chunk({**payload,**changes})
                np.testing.assert_array_equal(initial,sim.data.qpos)
                np.testing.assert_array_equal(controls,sim.data.ctrl)
            payload['actions']=[[100]*6]*10
            result=sim.policy_chunk(payload)
            self.assertGreater(result['limited_targets'],0)
            np.testing.assert_array_equal(initial,sim.data.qpos)
            sim.command({'action':'run_limits','duration_s':.2,'evaluator_stop':False})
            while not sim.paused: sim.step()
            self.assertAlmostEqual(sim.data.time,.2,places=8)
            self.assertEqual(sim.task['status'],'limit')
            self.assertGreater(np.linalg.norm(sim.data.qpos-initial),.001)
        finally: sim.close()


class PolicyFixture(BaseHTTPRequestHandler):
    requests=[]
    def log_message(self,*args): pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append(body)
        controls=body['observation']['actuators']
        target=[a['target'] for a in controls]; target[0]+=.03
        response={'protocol':'embodied-policy-v1','action_type':'actuator_targets',
                  'joint_names':[a['name'] for a in controls], 'actions':[target]*10,'dt_s':.05,
                  'summary':'HTTP contract fixture; not a trained policy'}
        data=json.dumps(response).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(data)


class ResearchLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs=tempfile.TemporaryDirectory(prefix='robot-research-test-')
        cls.log_patch=patch('server.runlog.RUNS',Path(cls.logs.name)); cls.log_patch.start()
        cls.worker=Worker(); cls.worker.thread.start()
        assert cls.worker.ready.wait(30)
        assert not cls.worker.failure,cls.worker.failure
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),PolicyFixture)
        cls.http_thread=Thread(target=cls.server.serve_forever,daemon=True); cls.http_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.http_thread.join(2)
        cls.worker.shutdown.set(); cls.worker.thread.join(5)
        cls.log_patch.stop(); cls.logs.cleanup()

    def test_http_vla_wam_comparison_shares_start_and_logs_real_execution(self):
        async def check():
            policy=PolicyController(self.worker); agent=AgentController(self.worker)
            for kind in ('vla','wam'):
                policy.configure(PolicyConfig(kind=kind,model='test-fixture',endpoint=f'http://127.0.0.1:{self.server.server_port}/predict',api_key='fixture-secret'))
            comparison=ComparisonController(self.worker,agent,policy)
            await comparison.start(ComparisonStart(robot='so101',methods=['vla','wam'],max_sim_time_s=1,record_video=True))
            await asyncio.wait_for(comparison.task,30)
            self.assertEqual(comparison.state['status'],'complete',comparison.state)
            rows=comparison.state['rows']; self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['initial_state_id'],rows[1]['initial_state_id'])
            for row in rows:
                self.assertEqual(row['status'],'limit',row)
                self.assertAlmostEqual(row['metrics']['sim_time_s'],1,places=3)
                folder=Path(self.logs.name)/row['run_id']
                text=(folder/'events.jsonl').read_text(encoding='utf-8')
                events=[json.loads(line)['event'] for line in text.splitlines()]
                self.assertIn('model_output',events); self.assertIn('execution',events)
                self.assertNotIn('fixture-secret',text); self.assertNotIn('image_base64',text)
                self.assertTrue((folder/'final.jpg').exists())
                self.assertEqual(row['video']['status'], 'saved', row)
                summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(row['video'],summary['artifacts']['video'])
                self.assertEqual(row['video']['frame_count'],21)
                self.assertTrue((folder/row['video']['path']).is_file())
            for body in PolicyFixture.requests:
                self.assertEqual(len(body['observation']['cameras']),3)
                self.assertNotIn('blocks',body['observation'])
            with self.worker.lock:
                self.assertTrue(self.worker.snapshot['paused'])
        asyncio.run(check())

    def test_agent_time_budget_stops_in_progress_physics(self):
        async def check():
            await self.worker.execute({'action':'load','model':'panda','scene_id':'tabletop'})
            await self.worker.execute({'action':'settle'})
            agent=AgentController(self.worker)
            async def boundary(config,prompt,cameras):
                pose=json.loads(prompt)['observation']['tcp']; pose['position_m'][2]+=.02
                return PoseAction(status='move',**pose,gripper_open=1,duration_s=4,explanation='fixture'),None
            with patch('server.agent.codex_action',boundary):
                await agent.start(AgentStart(task='fixture',max_sim_time_s=1))
                await asyncio.wait_for(agent.task,10)
            self.assertEqual(agent.state['status'],'limit',agent.state)
            self.assertAlmostEqual(agent.state['metrics']['sim_time_s'],1,places=3)
            with self.worker.lock: self.assertTrue(self.worker.snapshot['paused'])
        asyncio.run(check())

    def test_log_keeps_public_outputs_and_redacts_credentials(self):
        log=RunLog('test',{'task':'fixture'},['known-secret'])
        log.write('model_output',{'content':'known-secret public action', 'reasoning_content':'private',
                                  'nested':{'api_key':'known-secret'},'actions':[[1,2,3]]})
        log.finish({'status':'complete'})
        text=log.path.read_text(encoding='utf-8')
        self.assertNotIn('known-secret',text); self.assertNotIn('private',text)
        self.assertIn('public action',text); self.assertIn('actions',text)

    def test_cancel_loading_releases_owned_sidecar_and_can_start_again(self):
        # Start a real sidecar/Windows venv launcher, but stop before loading weights.
        # An existing user-started service must not be disturbed by this test.
        try:
            with socket.create_connection(('127.0.0.1',8010),timeout=3):
                self.skipTest('8010 already hosts an external policy service')
        except OSError: pass
        async def check():
            await self.worker.execute({'action':'load','model':'so101','scene_id':'tabletop'})
            policy=PolicyController(self.worker); started=asyncio.Event()
            async def loading_boundary():
                await ensure_local_service()
                started.set()
                await asyncio.Future()
            try:
                with patch.object(policy,'warmup',loading_boundary):
                    await policy.start(PolicyStart(task='lifecycle fixture'))
                    await asyncio.wait_for(started.wait(),20)
                    await policy.stop()
                self.assertFalse(policy.state['active'])
                self.assertEqual(policy.state['status'],'stopped',policy.state)
                with self.assertRaises(OSError): socket.create_connection(('127.0.0.1',8010),timeout=3)
                await ensure_local_service()
                self.assertTrue(await release_local_service())
                self.assertFalse(await release_local_service())
                with self.assertRaises(OSError): socket.create_connection(('127.0.0.1',8010),timeout=3)
                with self.worker.lock: self.assertTrue(self.worker.snapshot['ready'])
            finally:
                await policy.stop(); await release_local_service()
        asyncio.run(check())


if __name__=='__main__': unittest.main()
