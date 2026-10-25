"""Failure-boundary contracts with SDK construction replaced, never a native rollout."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from server.agent import AgentController
from server.policy import PolicyController
from server.runlog import RunLog
from server.backends.libero_backend import LiberoSimulation
from server.backends.libero_catalog import TASK_MAP
from server.backends.maniskill_backend import ManiSkillSimulation


class BackendLifecycleTests(unittest.TestCase):
    def libero(self):
        sim = LiberoSimulation.__new__(LiberoSimulation)
        sim.model_id, sim.scene_id, sim.task_id, sim.seed = 'panda', 'libero_spatial', 'libero_spatial:0', 42
        sim.generation, sim.env = 7, SimpleNamespace(close=Mock())
        sim.render_enabled = True
        sim.robot = sim.sim = sim.native = object()
        return sim

    def maniskill(self):
        sim = ManiSkillSimulation.__new__(ManiSkillSimulation)
        sim.model_id, sim.scene_id, sim.task_id, sim.seed = 'panda', 'pick_cube', 'pick_cube', 42
        sim.generation, sim.env = 7, SimpleNamespace(close=Mock())
        sim.render_enabled = True
        sim._sdk_directory = None
        sim.robot = sim.kinematics = object(); sim.cameras = {}
        return sim

    def test_invalid_selection_preserves_the_existing_native_environment(self):
        for sim in (self.libero(), self.maniskill()):
            with self.subTest(engine=type(sim).__name__):
                old = sim.env
                with self.assertRaises(ValueError): sim.command({'action': 'scene', 'scene_id': 'unknown'})
                self.assertIs(sim.env, old); old.close.assert_not_called()
                self.assertEqual(sim.generation, 7)

    def test_libero_native_construction_failure_is_fatal_and_retains_asset_cause(self):
        sim = self.libero(); old = sim.env
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bddl = root/'libero_goal'/(TASK_MAP['libero_goal'][0]+'.bddl')
            bddl.parent.mkdir(); bddl.write_text('construction fixture', encoding='utf-8')
            sim._libero_path = lambda kind: str(root)
            sim._environment_class = Mock(side_effect=ValueError('fixture mesh is missing'))
            with self.assertRaisesRegex(RuntimeError, '未就绪.*ValueError: fixture mesh is missing'):
                sim.command({'action': 'scene', 'scene_id': 'libero_goal', 'task_id': 'libero_goal:0'})
        old.close.assert_called_once()
        self.assertIsNone(sim.env); self.assertIsNone(sim.sim)

    def test_maniskill_native_construction_failure_is_fatal_and_retains_asset_cause(self):
        sim = self.maniskill(); old = sim.env
        # The public command exercises the actual adapter loader; only Gym's
        # external factory is replaced by the SDK error we need to preserve.
        with patch('server.backends.maniskill_backend.gym', SimpleNamespace(
                make=Mock(side_effect=ValueError('fixture Vulkan device unavailable'))), create=True):
            with self.assertRaisesRegex(RuntimeError, '未就绪.*ValueError: fixture Vulkan device unavailable'):
                sim.command({'action': 'scene', 'scene_id': 'push_cube', 'task_id': 'push_cube'})
        old.close.assert_called_once()
        self.assertIsNone(sim.env); self.assertIsNone(sim.robot)

    def test_reset_after_native_mutation_cannot_report_a_recoverable_input_error(self):
        for sim in (self.libero(), self.maniskill()):
            with self.subTest(engine=type(sim).__name__):
                old = sim.env
                sim.reset = Mock(side_effect=ValueError('native state initialization failed'))
                with self.assertRaisesRegex(RuntimeError, 'native state initialization failed'):
                    sim.command({'action': 'reset'})
                old.close.assert_called_once(); self.assertIsNone(sim.env)
                self.assertEqual(sim.generation, 7)

    def test_cleanup_error_does_not_replace_the_initial_reset_error(self):
        sim = self.libero()
        sim.reset = Mock(side_effect=ValueError('primary native reset error'))
        sim.close = Mock(side_effect=OSError('secondary cleanup error'))
        with self.assertRaises(RuntimeError) as error: sim.command({'action': 'reset'})
        self.assertIn('primary native reset error', str(error.exception))
        self.assertIn('secondary cleanup error', str(error.exception))


class ControllerSceneResetTests(unittest.TestCase):
    def test_new_scene_clears_current_controller_display_without_deleting_archived_run(self):
        with tempfile.TemporaryDirectory() as directory:
            for cls in (AgentController, PolicyController):
                with self.subTest(controller=cls.__name__):
                    log = RunLog('agent' if cls is AgentController else 'policy',
                                 {'model': 'fixture-policy', 'task': 'old scene task'}, directory=Path(directory))
                    log.finish({'status': 'verified', 'metrics': {'success': True}})
                    summary = (log.folder/'summary.json').read_bytes()
                    controller = cls(None)
                    controller.run_log = log
                    controller.state = {'active': False, 'status': 'verified', 'run_id': log.id, 'round': 4,
                        'history': [{'action': 'old action'}], 'action': {'status': 'done'},
                        'metrics': {'success': True}, 'video': {'status': 'ready'}}
                    controller.observation = {'robot': 'old robot', 'time': 5., 'cameras': []}
                    previous_id = controller.observation_id
                    controller.reset_observation()
                    state = controller.snapshot()
                    self.assertEqual(state['status'], 'idle'); self.assertFalse(state['active'])
                    self.assertEqual(state['history'], []); self.assertEqual(state['round'], 0)
                    self.assertEqual(state['observation_id'], previous_id+1)
                    self.assertIsNone(state['observation_time']); self.assertEqual(state['elapsed_s'], 0.)
                    for field in ('run_id', 'action', 'metrics', 'video'): self.assertNotIn(field, state)
                    self.assertIsNone(controller.run_log)
                    self.assertEqual((log.folder/'summary.json').read_bytes(), summary)

    def test_active_controllers_cannot_be_silently_detached_from_their_run(self):
        for cls in (AgentController, PolicyController):
            with self.subTest(controller=cls.__name__):
                controller = cls(None)
                controller.state.update(active=True, status='thinking', run_id='active-fixture')
                observation = {'robot': 'active robot'}; controller.observation = observation
                with self.assertRaises(ValueError): controller.reset_observation()
                self.assertTrue(controller.state['active'])
                self.assertEqual(controller.state['run_id'], 'active-fixture')
                self.assertIs(controller.observation, observation)
