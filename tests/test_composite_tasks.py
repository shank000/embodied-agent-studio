"""Task semantics fixtures plus a real actuator-driven feasibility reference.

State assignment is used only for negative/edge-case evaluator fixtures. The
reference rollout moves only via the public TCP/IK/actuator path, with simulator
positions supplied to this privileged test controller (not to a research actor).
"""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from server.control import tcp_pose
from server.scenes import TaskEvaluator
from server.simulation import Simulation
from server.task_graph import TaskGraph
from server.task_verification import environment_failure, public_feedback


def reference_mover(sim, trace, orientation=None, target_tolerance=.008):
    if orientation is None: orientation = tcp_pose(sim.model, sim.data, sim.site_id)['quaternion_wxyz']

    def move(target, opening, label):
        target = np.asarray(target, dtype=float)
        holding = np.linalg.norm(target-sim.data.site_xpos[sim.site_id]) < .008
        for attempt in range(40):
            current = sim.data.site_xpos[sim.site_id].copy()
            delta = target-current
            distance = float(np.linalg.norm(delta))
            step = .06
            for retry in range(5):
                waypoint = current + delta * min(1., step/max(distance, 1e-12))
                try:
                    sim.move_tcp({'generation': sim.generation, 'pose': {
                        'status': 'move', 'frame': 'world', 'position_m': waypoint.tolist(),
                        'quaternion_wxyz': list(orientation), 'gripper_open': opening,
                        'duration_s': .6, 'explanation': label}})
                    break
                except ValueError:
                    step /= 2
                    if retry == 4: raise
            for _ in range(2200):
                sim.step()
                if sim.motion is None and sim.task['status'] != 'settling': break
            error = float(np.linalg.norm(sim.data.site_xpos[sim.site_id]-waypoint))
            trace.append({'label': label, 'sim_s': float(sim.data.time), 'position_error_m': error,
                          'motion_status': sim.task['status'], 'evaluation': sim.evaluator.snapshot()})
            if sim.evaluator.graph.terminal_failure: raise AssertionError(sim.evaluator.graph.failure_reason)
            if error > .018: raise AssertionError(f'{label}: TCP tracking error {error}')
            if holding or np.linalg.norm(sim.data.site_xpos[sim.site_id]-target) < target_tolerance: return
        raise AssertionError(f'{label}: waypoint budget exceeded')

    return move


def run_order_reference(sim):
    trace = []
    for _ in range(500): sim.step()
    move = reference_mover(sim, trace, target_tolerance=.012)
    for name, tray in [('red', 'tray'), ('blue', 'blue_bin')]:
        start = sim.data.body('block_'+name).xpos.copy()
        destination = sim.data.geom(tray+'_base').xpos.copy()
        move([*start[:2], .16], 1., f'{name}: approach')
        start = sim.data.body('block_'+name).xpos.copy()
        move([*start[:2], .025], 1., f'{name}: descend')
        move(sim.data.site_xpos[sim.site_id].copy(), 0., f'{name}: close')
        move([*start[:2], .16], 0., f'{name}: lift')
        if sim.data.body('block_'+name).xpos[2] < .10: raise AssertionError(f'{name}: grasp failed')
        move([*destination[:2], .16], 0., f'{name}: transport')
        move(sim.data.site_xpos[sim.site_id].copy(), 1., f'{name}: release')
        move([*destination[:2], .25], 1., f'{name}: withdraw')
        for _ in range(300): sim.step()
    for _ in range(600): sim.step()
    return trace


def run_drawer_reference(sim):
    trace = []
    for _ in range(500): sim.step()
    initial_orientation = np.asarray(tcp_pose(sim.model, sim.data, sim.site_id)['quaternion_wxyz'])
    orientation = initial_orientation.copy()
    move = reference_mover(sim, trace, orientation, target_tolerance=.012)
    handle = sim.data.geom('drawer_handle').xpos.copy()
    current = sim.data.site_xpos[sim.site_id].copy()
    move([*current[:2], .42], 1., 'drawer: lift clear')
    move([handle[0], -.30, .42], 1., 'drawer: front clearance')
    for degrees in (30, 60):
        angle = np.deg2rad(degrees)/2
        mujoco.mju_mulQuat(orientation, np.array([np.cos(angle), np.sin(angle), 0., 0.]), initial_orientation)
        move(sim.data.site_xpos[sim.site_id].copy(), 1., 'drawer: horizontal wrist')
    move([handle[0], -.30, handle[2]], 1., 'drawer: front descend')
    move(handle, 1., 'drawer: approach')
    move(sim.data.site_xpos[sim.site_id].copy(), 0., 'drawer: grip')
    opened = handle + [0, -.15, 0]
    move(opened, 0., 'drawer: pull')
    for _ in range(300): sim.step()
    if not sim.evaluator.graph.stages['opened']['satisfied']:
        raise AssertionError(f'drawer did not open: {sim.data.joint("drawer_slide").qpos[0]}')
    move(handle, 0., 'drawer: close')
    for _ in range(300): sim.step()
    move(sim.data.site_xpos[sim.site_id].copy(), 1., 'drawer: release')
    move([*handle[:2], .22], 1., 'drawer: withdraw')
    for _ in range(600): sim.step()
    return trace


class TaskGraphTests(unittest.TestCase):
    def test_holds_require_sim_time_and_dependencies_cannot_complete_together(self):
        graph = TaskGraph('drawer_cycle')
        for _ in range(100): graph.update(0, {'opened': True, 'closed': True, 'released': True})
        self.assertFalse(graph.snapshot()['complete'])
        graph.update(.3, {'opened': True, 'closed': True, 'released': True})
        self.assertTrue(graph.stages['opened']['satisfied'])
        self.assertFalse(graph.stages['closed']['satisfied'])

    def test_revoke_propagates_and_first_achievement_cannot_be_farmed(self):
        graph = TaskGraph('order_fulfillment')
        predicates = {'red_delivered': True, 'blue_delivered': True, 'withdrawn': True}
        for t in (0, .3, .31, .61, .62, .92): graph.update(t, predicates)
        self.assertTrue(graph.snapshot()['complete'])
        graph.update(1., {**predicates, 'red_delivered': False})
        self.assertEqual(graph.snapshot()['potential'], 0)
        for t in (1.1, 1.4, 1.41, 1.71, 1.72, 2.02): graph.update(t, predicates)
        self.assertTrue(graph.snapshot()['complete'])
        self.assertEqual(graph.snapshot()['first_achievements'], 3)
        self.assertEqual(sum(e.get('first_achievement', False) for e in graph.events), 3)
        self.assertTrue(any(e['type'] == 'revoked' for e in graph.events))

    def test_terminal_violation_cannot_be_undone(self):
        graph = TaskGraph('drawer_cycle')
        graph.update(0, {}, 'joint out of range')
        for t in (1, 2, 3): graph.update(t, {'opened': True, 'closed': True, 'released': True})
        self.assertTrue(graph.snapshot()['terminal_failure'])
        self.assertFalse(graph.snapshot()['complete'])
        self.assertEqual(sum(e['type']=='violation' for e in graph.events), 1)

    def test_evaluation_is_removed_from_actor_feedback(self):
        clean = public_feedback({'tcp': {'position_m': [1,2,3]}, 'evaluation': {'success': True},
            'details': {'task_graph': {'stages': []}, 'reward_components': {'potential': 1}},
            'evaluator_state': {'graph': {}}})
        self.assertEqual(clean, {'tcp': {'position_m': [1,2,3]}, 'details': {}})

    def test_failure_requires_ready_and_trusted_evaluator(self):
        snapshot = {'ready': True, 'evaluation': {'source': 'mujoco_privileged_state',
                                                 'task_graph': {'terminal_failure': True}}}
        self.assertTrue(environment_failure(snapshot))
        self.assertFalse(environment_failure({**snapshot, 'ready': False}))
        self.assertFalse(environment_failure({'ready': True, 'evaluation': {
            'source': 'model', 'task_graph': {'terminal_failure': True}}}))


class NativeCompositeTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation(render=False, scene_id='sorting', task_id='order_fulfillment')

    def tearDown(self): self.sim.close()

    def fixture_place(self, name, xy):
        sim = self.sim
        joint = sim.model.joint('block_'+name+'_joint')
        adr = int(joint.qposadr[0])
        sim.data.qpos[adr:adr+7] = [*xy, .039, 1, 0, 0, 0]
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        sim.evaluator.update(sim)

    def test_final_positions_without_required_order_fail(self):
        self.fixture_place('blue', [.17, .26])
        self.fixture_place('red', [.5, .25])
        for _ in range(500): self.sim.step()
        result = self.sim.evaluator.snapshot()
        self.assertFalse(result['success'])
        self.assertTrue(result['task_graph']['terminal_failure'])
        self.assertIn('之前交付', result['task_graph']['failure_reason'])

    def test_initial_closed_drawer_does_not_pass_cycle(self):
        sim = self.sim
        sim.load('panda', 'drawer', 42, 'drawer_cycle')
        for _ in range(600): sim.step()
        self.assertFalse(sim.evaluator.success)
        self.assertFalse(sim.evaluator.graph.stages['opened']['satisfied'])

    def test_drawer_cycle_fixture_requires_open_close_release_and_revokes_on_reopen(self):
        sim = self.sim
        sim.load('panda', 'drawer', 42, 'drawer_cycle')
        for position, steps in ((.15, 200), (.005, 500)):
            sim.data.joint('drawer_slide').qpos[0] = position
            sim.data.joint('drawer_slide').qvel[0] = 0
            mujoco.mj_forward(sim.model, sim.data)
            sim.evaluator.update(sim)
            for _ in range(steps): sim.step()
        self.assertTrue(sim.evaluator.success)
        self.assertEqual(sim.evaluator.graph.snapshot()['first_achievements'], 3)
        sim.data.joint('drawer_slide').qpos[0] = .15
        mujoco.mj_forward(sim.model, sim.data)
        sim.evaluator.update(sim)
        self.assertFalse(sim.evaluator.success)
        self.assertTrue(sim.evaluator.graph.stages['opened']['satisfied'])
        self.assertFalse(sim.evaluator.graph.stages['closed']['satisfied'])
        self.assertFalse(sim.evaluator.graph.stages['released']['satisfied'])

    def test_moving_protected_object_is_irreversible_failure(self):
        self.fixture_place('yellow', [.6, -.3])
        self.fixture_place('yellow', [.32, -.37])
        self.assertTrue(self.sim.evaluator.graph.terminal_failure)
        self.assertFalse(self.sim.evaluator.success)
        self.sim.paused = True
        before = float(self.sim.data.time)
        result = self.sim.verify_task(self.sim.generation)
        self.assertEqual(result['verification_sim_time_s'], 0)
        self.assertEqual(float(self.sim.data.time), before)

    def test_composite_checkpoint_retains_history_and_failures(self):
        sim = self.sim
        self.fixture_place('red', [.5, .25])
        for _ in range(250): sim.step()
        checkpoint = sim.checkpoint()
        graph = sim.evaluator.graph.snapshot()
        frame = sim.trajectory_frame()
        sim.reset()
        sim.replay_frame(frame)
        self.assertEqual(sim.evaluator.graph.snapshot(), graph)
        sim.reset()
        sim.restore(checkpoint)
        self.assertEqual(sim.evaluator.graph.snapshot(), graph)
        self.assertEqual(sim.evaluator.initial_yellow_xy.tolist(), checkpoint['evaluator_state']['initial_yellow_xy'])
        self.fixture_place('yellow', [.6, -.3])
        failed = sim.checkpoint()
        sim.reset()
        sim.restore(failed)
        self.assertTrue(sim.evaluator.graph.terminal_failure)

    def test_real_actuator_reference_completes_order(self):
        trace = run_order_reference(self.sim)
        result = self.sim.evaluator.snapshot()
        self.assertTrue(result['success'], result)
        self.assertEqual(result['task_graph']['first_achievements'], 3)
        self.assertTrue(all(row['position_error_m'] <= .018 for row in trace))

    def test_real_actuator_reference_completes_drawer_cycle(self):
        self.sim.load('panda', 'drawer', 42, 'drawer_cycle')
        trace = run_drawer_reference(self.sim)
        self.assertTrue(self.sim.evaluator.success, self.sim.evaluator.snapshot())
        self.assertEqual(self.sim.evaluator.graph.snapshot()['first_achievements'], 3)
        self.assertTrue(all(row['position_error_m'] <= .018 for row in trace))


if __name__ == '__main__': unittest.main()
