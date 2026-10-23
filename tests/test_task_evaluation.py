"""Task decisions use real MuJoCo state and simulated time, without a model."""
import json
import unittest

import mujoco
import numpy as np

from server.scenes import SCENES, TASKS, TaskEvaluator
from server.simulation import Simulation


class TaskEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation(render=False, scene_id='drawer', task_id='open_drawer')

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def drawer(self, robot='panda'):
        self.sim.load(robot, 'drawer', 42, 'open_drawer')
        self.sim.model.opt.gravity[:] = 0
        return self.sim

    def set_drawer(self, position):
        sim = self.sim
        sim.data.joint('drawer_slide').qpos[0] = position
        sim.data.joint('drawer_slide').qvel[0] = 0
        mujoco.mj_forward(sim.model, sim.data)
        sim.evaluator.update(sim)
        return sim.evaluator.snapshot()

    def advance(self, count):
        for _ in range(count): self.sim.step()
        return self.sim.evaluator.snapshot()

    def test_closed_drawer_is_not_complete_despite_model_declaration(self):
        sim = self.drawer()
        sim.task['status'] = 'model_done'
        result = self.advance(400)
        self.assertFalse(result['success'])
        self.assertFalse(result['currently_satisfied'])
        self.assertEqual(result['source'], 'mujoco_privileged_state')
        displacement = next(item for item in result['checks'] if item['key'] == 'drawer_displacement')
        self.assertFalse(displacement['passed'])
        self.assertAlmostEqual(displacement['actual'], 0, places=8)
        self.assertEqual(displacement['unit'], 'm')
        self.assertIn('抽屉相对闭合位置的位移', result['failure_reasons'])

    def test_drawer_threshold_and_continuous_hold_scale_for_both_robots(self):
        for robot in ('panda', 'so101'):
            with self.subTest(robot=robot):
                sim = self.drawer(robot)
                scale = sim.config['scale']
                self.assertFalse(self.set_drawer(.119*scale)['currently_satisfied'])
                self.assertFalse(self.set_drawer(.12*scale)['currently_satisfied'])
                result = self.set_drawer(.15*scale)
                self.assertTrue(result['currently_satisfied'])
                self.assertFalse(result['success'])
                self.assertEqual(result['hold_s'], 0)
                self.assertAlmostEqual(result['values']['drawer_threshold_m'], .12*scale)
                self.assertFalse(self.advance(149)['success'])
                result = self.advance(1)
                self.assertTrue(result['success'], result)
                self.assertAlmostEqual(result['hold_s'], .3, places=6)

    def test_success_is_revoked_after_retraction_but_achievement_is_retained(self):
        self.drawer()
        self.set_drawer(.15)
        self.assertTrue(self.advance(155)['success'])
        result = self.set_drawer(.02)
        self.assertFalse(result['success'])
        self.assertTrue(result['ever_success'])
        self.assertEqual(result['hold_s'], 0)
        self.set_drawer(.15)
        self.assertFalse(self.advance(145)['success'], 'A new hold must begin after retraction')

    def test_paused_repeated_observations_never_supply_hold_time(self):
        sim = self.drawer()
        self.set_drawer(.15)
        state = sim.checkpoint()['state']
        for _ in range(500): sim.evaluator.update(sim)
        self.assertEqual(sim.checkpoint()['state'], state)
        self.assertFalse(sim.evaluator.success)
        self.assertEqual(sim.evaluator.snapshot()['hold_s'], 0)
        self.assertFalse(self.advance(149)['success'])
        for _ in range(500): sim.evaluator.update(sim)
        self.assertFalse(sim.evaluator.success)
        self.assertTrue(self.advance(1)['success'])

    def test_duration_uses_simulated_time_and_resets_after_interruption(self):
        sim = self.drawer()
        sim.model.opt.timestep = .001
        self.set_drawer(.15)
        self.assertFalse(self.advance(299)['success'])
        self.assertTrue(self.advance(1)['success'])
        self.set_drawer(.05)
        self.set_drawer(.15)
        self.assertFalse(self.advance(299)['success'])
        self.assertTrue(self.advance(1)['success'])

    def test_restore_starts_fresh_hold_from_closed_relative_position(self):
        sim = self.drawer()
        self.set_drawer(.15)
        self.assertTrue(self.advance(160)['success'])
        expected_open = sim.evaluator.snapshot()['values']['drawer_open_m']
        checkpoint = sim.checkpoint()
        self.set_drawer(0)
        sim.restore(checkpoint)
        result = sim.evaluator.snapshot()
        self.assertFalse(result['success'])
        self.assertFalse(result['ever_success'])
        self.assertEqual(result['hold_s'], 0)
        self.assertTrue(result['currently_satisfied'])
        self.assertAlmostEqual(result['values']['drawer_open_m'], expected_open, places=8)
        self.assertTrue(self.advance(150)['success'])

    def test_invalid_or_out_of_range_drawer_state_cannot_pass_and_remains_serializable(self):
        sim = self.drawer()
        result = self.set_drawer(.21)
        self.assertFalse(result['currently_satisfied'])
        limits = next(item for item in result['checks'] if item['key'] == 'drawer_joint_limits')
        self.assertFalse(limits['passed'])
        result = self.set_drawer(float('nan'))
        self.assertFalse(result['success'])
        self.assertFalse(result['currently_satisfied'])
        json.dumps(result, allow_nan=False)

    def test_soft_joint_stop_tolerance_allows_solver_excursions_but_rejects_off_rail_state(self):
        for robot in ('panda', 'so101'):
            sim = self.drawer(robot)
            scale = sim.config['scale']
            for unscaled_position, within_limits in [(.201, True), (.203, False), (-.001, True), (-.003, False)]:
                with self.subTest(robot=robot, position=unscaled_position):
                    result = self.set_drawer(unscaled_position*scale)
                    limits = next(item for item in result['checks'] if item['key'] == 'drawer_joint_limits')
                    self.assertEqual(limits['passed'], within_limits)
                    self.assertAlmostEqual(result['values']['limit_tolerance_m'], .002*scale)
                    self.assertEqual(result['currently_satisfied'], within_limits and unscaled_position > .12)

    def test_verify_closed_drawer_advances_real_time_and_rejects_completion(self):
        sim = self.drawer()
        sim.paused = True
        before = sim.physics_steps
        callbacks = []
        result = sim.verify_task(sim.generation, lambda: callbacks.append(float(sim.data.time)))
        self.assertFalse(result['success'])
        self.assertFalse(result['currently_satisfied'])
        self.assertAlmostEqual(result['verification_sim_time_s'], .3, places=8)
        self.assertEqual(len(callbacks), sim.physics_steps-before)
        self.assertEqual(len(callbacks), 150)
        self.assertTrue(sim.paused)

    def test_verify_open_drawer_requires_physics_after_paused_observations_and_records_each_step(self):
        sim = self.drawer()
        self.set_drawer(.15)
        sim.paused = True
        for _ in range(500): sim.evaluator.update(sim)
        self.assertEqual(sim.evaluator.snapshot()['hold_s'], 0)
        before = sim.physics_steps
        callbacks = []
        result = sim.verify_task(sim.generation, lambda: callbacks.append(sim.trajectory_frame()))
        self.assertTrue(result['success'], result)
        self.assertAlmostEqual(result['verification_sim_time_s'], .3, places=8)
        self.assertAlmostEqual(result['hold_s'], .3, places=6)
        self.assertEqual(len(callbacks), sim.physics_steps-before)
        self.assertEqual(len(callbacks), 150)
        self.assertAlmostEqual(callbacks[-1]['time']-callbacks[0]['time'], .298, places=8)
        self.assertTrue(callbacks[-1]['evaluation']['success'])
        self.assertTrue(sim.paused)

    def test_verify_obeys_remaining_episode_budget(self):
        sim = self.drawer()
        self.set_drawer(.15)
        sim.paused = True
        sim.run_deadline = sim.data.time + .1
        callbacks = []
        before = sim.physics_steps
        result = sim.verify_task(sim.generation, lambda: callbacks.append(float(sim.data.time)))
        self.assertFalse(result['success'], 'The remaining budget cannot supply the required 0.3 s hold')
        self.assertTrue(result['currently_satisfied'])
        self.assertAlmostEqual(result['verification_sim_time_s'], .1, places=8)
        self.assertLessEqual(sim.data.time, sim.run_deadline + 1e-9)
        self.assertEqual(len(callbacks), sim.physics_steps-before)
        self.assertEqual(len(callbacks), 50)
        self.assertEqual(sim.task['status'], 'limit')
        self.assertTrue(sim.paused)

    def test_verify_with_stale_generation_does_not_advance_or_record_state(self):
        sim = self.drawer()
        self.set_drawer(.15)
        sim.paused = True
        before = sim.checkpoint()['state']
        steps = sim.physics_steps
        callbacks = []
        with self.assertRaisesRegex(ValueError, '场景已切换'):
            sim.verify_task(sim.generation-1, lambda: callbacks.append(float(sim.data.time)))
        self.assertEqual(sim.checkpoint()['state'], before)
        self.assertEqual(sim.physics_steps, steps)
        self.assertFalse(callbacks)

    def test_all_scene_tasks_have_initial_auditable_evidence(self):
        sim = self.sim
        for scene, spec in SCENES.items():
            for task in spec['tasks']:
                with self.subTest(scene=scene, task=task):
                    sim.load('panda', scene, 42, task)
                    result = sim.evaluator.snapshot()
                    self.assertEqual(result['task_id'], task)
                    self.assertEqual(result['version'], TaskEvaluator.VERSION)
                    self.assertEqual(result['required_hold_s'], .3)
                    self.assertFalse(result['success'])
                    self.assertTrue(result['checks'])
                    self.assertTrue(result['values'])
                    json.dumps(result, allow_nan=False)
        with self.assertRaisesRegex(ValueError, '未知评价任务'):
            TaskEvaluator(sim, 'invented_task')

    def test_rotated_object_corners_must_fit_inside_bin(self):
        sim = self.sim
        sim.load('panda', 'sorting', 42, 'place_red')
        joint = sim.model.joint('block_red_joint')
        adr = int(joint.qposadr[0])
        sim.data.qpos[adr:adr+3] = [.61, .25, .04]
        sim.data.qpos[adr+3:adr+7] = [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)]
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        sim.evaluator.update(sim)
        result = sim.evaluator.snapshot()
        inside = next(item for item in result['checks'] if item['key'] == 'red_inside_bin_xy')
        self.assertFalse(inside['passed'], 'A rotated corner protrudes despite its center being inside the old tolerance')
        self.assertFalse(result['currently_satisfied'])
        sim.data.qpos[adr] = .5
        mujoco.mj_forward(sim.model, sim.data)
        sim.evaluator.update(sim)
        self.assertTrue(next(item for item in sim.evaluator.snapshot()['checks'] if item['key'] == 'red_inside_bin_xy')['passed'])

    def test_extra_evaluations_do_not_change_physics_trajectory(self):
        sim = self.drawer()
        sim.command({'action': 'settle'})
        saved = sim.checkpoint()
        expected = []
        for _ in range(40):
            sim.step()
            expected.append(sim.data.qpos.copy())
        sim.restore(saved)
        actual = []
        for _ in range(40):
            sim.step()
            for _ in range(3): sim.evaluator.update(sim)
            actual.append(sim.data.qpos.copy())
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


if __name__ == '__main__': unittest.main()
