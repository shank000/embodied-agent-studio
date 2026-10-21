"""Regression checks use the real models and MuJoCo, without simulated API responses."""
import asyncio
import io
import math
import time
import unittest

import numpy as np
from PIL import Image
from pydantic import ValidationError

from server.simulation import Simulation
from server.app import Command, Worker


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation(render=False)

    def test_real_models_and_actuated_joint_motion(self):
        for name, actuators, meshes in [('panda', 8, 67), ('so101', 6, 18)]:
            with self.subTest(model=name):
                self.sim.load(name)
                self.assertEqual(self.sim.model.nu, actuators)
                self.assertEqual(self.sim.model.nmesh, meshes)
                before = self.sim.data.qpos[0]
                self.sim.command({'action': 'instruction', 'text': '关节1转到30度'})
                self.assertEqual(self.sim.data.qpos[0], before, 'a command must not teleport the joint')
                for _ in range(1000): self.sim.step()
                self.assertAlmostEqual(self.sim.data.qpos[0], math.pi / 6, delta=.02)
                self.assertTrue(np.isfinite(self.sim.data.qpos).all())

    def test_gravity_and_real_table_contacts(self):
        for name in ('panda', 'so101'):
            with self.subTest(model=name):
                self.sim.load(name)
                self.sim.drop()
                start_z = self.sim.data.body('block_red').xpos[2]
                for _ in range(500): self.sim.step()
                end_z = self.sim.data.body('block_red').xpos[2]
                self.assertLess(end_z, start_z * .2)
                self.assertAlmostEqual(end_z, .022*self.sim.config['scale'], delta=.002)
                self.assertGreater(self.sim.data.ncon, 0)
                self.sim.drop()
                self.sim.command({'action':'gravity','value':False})
                start_z = self.sim.data.body('block_red').xpos[2]
                for _ in range(150): self.sim.step()
                self.assertAlmostEqual(self.sim.data.body('block_red').xpos[2], start_z, delta=1e-6)

    def test_gripper_and_demo_reach_actual_targets(self):
        for name in ('panda', 'so101'):
            with self.subTest(model=name):
                self.sim.load(name)
                self.sim.instruction('闭合夹爪')
                for _ in range(1000): self.sim.step()
                gripper = self.sim.actuators[-1]
                self.assertAlmostEqual(self.sim.data.qpos[gripper['qpos_index']], 0, delta=.01)
                self.sim.demo()
                for _ in range(7600): self.sim.step()
                self.assertEqual(self.sim.task['status'], 'complete')
                self.assertAlmostEqual(self.sim.data.qpos[0], 0, delta=.025)
                expected = .04 if name=='panda' else 1.25
                self.assertAlmostEqual(self.sim.data.qpos[gripper['qpos_index']], expected, delta=.01)

    def test_rejected_commands_and_reset_preserve_contract(self):
        self.sim.load('so101')
        before = self.sim.data.ctrl.copy()
        for text in ('不要闭合夹爪', '关节1转到900度', '关节7转到10度', '把红色方块放到收纳区'):
            with self.subTest(text=text), self.assertRaises(ValueError): self.sim.instruction(text)
            np.testing.assert_equal(self.sim.data.ctrl, before)
        self.sim.demo()
        for _ in range(300): self.sim.step()
        self.sim.reset()
        self.assertIsNone(self.sim.motion)
        self.assertEqual(self.sim.data.time, 0)
        np.testing.assert_equal(self.sim.data.ctrl, self.sim.home_ctrl)

    def test_http_command_validation(self):
        for payload in ({'action':'control','index':0}, {'action':'control','index':0,'value':float('nan')},
                        {'action':'load','model':'unknown'}, {'action':'pause','value':1}, {'action':'camera','zoom':100}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError): Command.model_validate(payload)
        self.assertIs(Command.model_validate({'action':'pause','value':True}).value, True)


class NativeServiceTests(unittest.TestCase):
    def test_render_pause_single_step_model_switch_and_resume(self):
        worker = Worker()
        worker.thread.start()
        try:
            self.assertTrue(worker.ready.wait(30))
            self.assertIsNone(worker.failure)
            def execute(payload): return asyncio.run(worker.execute(payload))
            def snapshot():
                with worker.lock: return worker.snapshot.copy()
            execute({'action':'pause','value':True})
            before = snapshot()
            with worker.lock: paused_frame = worker.frame; paused_jpeg = worker.jpeg
            time.sleep(.15)
            self.assertEqual(snapshot()['time'], before['time'])
            with worker.lock:
                self.assertEqual(worker.frame, paused_frame)
                self.assertEqual(worker.jpeg, paused_jpeg)
            execute({'action':'view','view':'top'})
            with worker.lock:
                self.assertGreater(worker.frame, paused_frame)
                self.assertNotEqual(worker.jpeg, paused_jpeg)
            execute({'action':'step'})
            self.assertAlmostEqual(snapshot()['time']-before['time'], .002, places=8)
            execute({'action':'control','index':0,'value':.3})
            paused_joint = snapshot()['actuators'][0]['actual']
            time.sleep(.1)
            self.assertEqual(snapshot()['actuators'][0]['actual'], paused_joint)
            execute({'action':'pause','value':False})
            deadline = time.monotonic()+4
            while abs(snapshot()['actuators'][0]['actual']-.3)>.02 and time.monotonic()<deadline: time.sleep(.05)
            self.assertAlmostEqual(snapshot()['actuators'][0]['actual'], .3, delta=.02)
            for name, count in [('so101',6),('panda',8)]:
                execute({'action':'load','model':name})
                self.assertEqual(snapshot()['nu'],count)
                with worker.lock: pixels=Image.open(io.BytesIO(worker.jpeg)).copy()
                self.assertEqual(pixels.size,(960,640))
                self.assertGreater(np.asarray(pixels).std(),10)
        finally:
            worker.shutdown.set();worker.thread.join(5)
            self.assertFalse(worker.thread.is_alive())


if __name__=='__main__': unittest.main()
