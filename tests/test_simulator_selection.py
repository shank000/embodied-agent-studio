"""Simulator selection preserves live state and frozen evaluation boundaries."""
import asyncio
from copy import deepcopy
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
import httpx
import numpy as np
from PIL import Image

from server import research, simulators
from server.comparison import ComparisonController, ComparisonStart
from server.simulation import Simulation
from server.simulators import backend_identity, supported_catalog, validate_selection
from server.task_verification import environment_success, verify_completion


application = importlib.import_module('server.app')


def image_bytes(color='green'):
    output = io.BytesIO()
    Image.new('RGB', (16, 12), color).save(output, 'JPEG')
    return output.getvalue()


class SelectionApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.controllers = {}
        for name in ('agent', 'policy', 'comparison', 'composition', 'research', 'repair'):
            state = {'active': False}
            controller = SimpleNamespace(state=state, snapshot=lambda state=state: state,
                                         reset_observation=Mock())
            controller.stop = AsyncMock(side_effect=lambda state=state: state.update(active=False))
            if name == 'agent': controller.lock = asyncio.Lock()
            self.controllers[name] = controller
        self.worker = SimpleNamespace(lock=Lock(), switching=False, engine_id='mujoco',
            session_id='existing-session', snapshot={'task': {'status': 'idle'}},
            thread=SimpleNamespace(is_alive=lambda: True),
            execute=AsyncMock(return_value={'backend': backend_identity('genesis', 'fixture')}))
        for name, value in {**self.controllers, 'worker': self.worker}.items():
            item = patch.object(application, name, value)
            item.start()
            self.addCleanup(item.stop)

    async def request(self, engine_id='genesis'):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                     base_url='http://127.0.0.1:8000') as client:
            return await client.post('/api/simulators/select', json={'engine_id': engine_id},
                                     headers={'origin': 'http://127.0.0.1:8000'})

    async def test_all_active_controller_types_block_selection_without_cancelling_them(self):
        for name, controller in self.controllers.items():
            with self.subTest(controller=name):
                controller.state['active'] = True
                response = await self.request()
                self.assertEqual(response.status_code, 409, response.text)
                self.assertTrue(controller.state['active'])
                self.worker.execute.assert_not_awaited()
                controller.state['active'] = False

    async def test_pending_switch_and_manual_motion_block_selection(self):
        self.worker.switching = True
        self.assertEqual((await self.request()).status_code, 409)
        self.worker.switching = False
        for status in ('running', 'settling'):
            self.worker.snapshot['task']['status'] = status
            self.assertEqual((await self.request()).status_code, 409)
        self.worker.execute.assert_not_awaited()

    async def test_out_of_scope_engine_and_arbitrary_command_are_not_executable(self):
        for engine in ('omnigibson', 'habitat', 'cmd.exe'):
            self.assertEqual((await self.request(engine)).status_code, 422)
        self.worker.execute.assert_not_awaited()

    async def test_idle_selection_resets_old_observations(self):
        response = await self.request()
        self.assertEqual(response.status_code, 200, response.text)
        self.worker.execute.assert_awaited_once_with({'action': 'backend_select', 'engine_id': 'genesis'})
        self.controllers['agent'].reset_observation.assert_called_once()
        self.controllers['policy'].reset_observation.assert_called_once()

    async def test_custom_local_port_allows_its_own_origin_and_rejects_cross_site_requests(self):
        for address in ('http://127.0.0.1:8025', 'http://localhost:8025'):
            with self.subTest(address=address):
                self.worker.switching = False
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                             base_url=address) as client:
                    response = await client.post('/api/simulators/select', json={'engine_id': 'genesis'},
                                                 headers={'origin': address})
                    self.assertEqual(response.status_code, 200, response.text)
        executed = self.worker.execute.await_count
        for origin in ('https://example.com', 'http://127.0.0.1.evil.test:8025', 'http://127.0.0.1:8026', 'null'):
            with self.subTest(origin=origin):
                self.worker.switching = False
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                             base_url='http://127.0.0.1:8025') as client:
                    response = await client.post('/api/simulators/select', json={'engine_id': 'genesis'},
                                                 headers={'origin': origin})
                    self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(self.worker.execute.await_count, executed)

    async def test_stop_cancels_pending_inference_when_engine_failed_without_claiming_physical_stop(self):
        failure = 'BackendProcessError: isolated simulator exited with diagnostic 17'
        self.worker.failure = failure
        self.worker.thread = SimpleNamespace(is_alive=lambda: False)
        for kind in ('agent', 'policy'):
            with self.subTest(controller=kind):
                controller = self.controllers[kind]
                started, cancelled = asyncio.Event(), asyncio.Event()
                async def infer():
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                pending = asyncio.create_task(infer())
                await started.wait()
                controller.state.update(active=True, status='thinking')
                async def stop():
                    pending.cancel()
                    try: await pending
                    except asyncio.CancelledError: pass
                    controller.state.update(active=False, status='stopped')
                controller.stop = AsyncMock(side_effect=stop)
                try:
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                                 base_url='http://127.0.0.1:8000') as client:
                        response = await client.post(f'/api/{kind}/stop',
                            headers={'origin': 'http://127.0.0.1:8000'})
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()
                    self.assertFalse(result['active'])
                    self.assertEqual(result['status'], 'stopped')
                    self.assertFalse(result['engine_stopped'])
                    self.assertIn(failure, result['engine_error'])
                    self.assertTrue(pending.cancelled())
                    self.assertTrue(cancelled.is_set())
                    self.worker.execute.assert_not_awaited()
                finally:
                    if not pending.done():
                        pending.cancel()
                        try: await pending
                        except asyncio.CancelledError: pass


class WorkerSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.native_instances = []
        def native():
            simulation = Simulation(render=False)
            simulation.paused = True
            simulation.render = lambda: image_bytes()
            self.native_instances.append(simulation)
            return simulation
        item = patch.object(application, 'Simulation', side_effect=native)
        item.start()
        self.addCleanup(item.stop)
        self.worker = application.Worker()
        self.worker.thread.start()
        ready = await asyncio.to_thread(self.worker.ready.wait, 20)
        self.assertTrue(ready)
        self.assertIsNone(self.worker.failure)

    async def asyncTearDown(self):
        self.worker.shutdown.set()
        await asyncio.to_thread(self.worker.thread.join, 5)
        self.assertFalse(self.worker.thread.is_alive())

    async def test_missing_engine_does_not_close_or_replace_current_simulation(self):
        session, generation = self.worker.session_id, self.worker.snapshot['generation']
        with patch('server.backends.process_backend.ProcessBackend', side_effect=ValueError('missing optional environment')):
            with self.assertRaises(HTTPException) as failure:
                await self.worker.execute({'action': 'backend_select', 'engine_id': 'genesis'})
        self.assertEqual(failure.exception.status_code, 400)
        self.assertIn('missing optional environment', failure.exception.detail)
        self.assertEqual(self.worker.engine_id, 'mujoco')
        self.assertEqual(self.worker.session_id, session)
        self.assertEqual(self.worker.snapshot['generation'], generation)
        self.assertFalse(self.worker.switching)
        self.assertIsNone(self.worker.failure)
        before = self.worker.snapshot['time']
        await self.worker.execute({'action': 'step'})
        self.assertGreater(self.worker.snapshot['time'], before)

    async def test_not_ready_candidate_and_cleanup_failure_preserve_original_cause_and_worker(self):
        # An unavailable candidate may have a placeholder image and actuator list.
        # Neither is permission to commit a failed renderer/native initialization.
        candidate = SimpleNamespace(generation=1000001,
            state=lambda: {'ready': False, 'backend': backend_identity('genesis', 'fixture'),
                           'actuators': [{'index': 0}], 'timestep': .002},
            render=lambda: image_bytes('red'), close=Mock(side_effect=RuntimeError('candidate cleanup error')))
        session = self.worker.session_id
        with patch('server.backends.process_backend.ProcessBackend', return_value=candidate):
            with self.assertRaises(HTTPException) as failure:
                await self.worker.execute({'action': 'backend_select', 'engine_id': 'genesis'})
        self.assertEqual(failure.exception.status_code, 400)
        self.assertIn('cleanup', failure.exception.detail)
        self.assertIn('有效模型', failure.exception.detail)
        self.assertEqual(self.worker.engine_id, 'mujoco')
        self.assertEqual(self.worker.session_id, session)
        self.assertIsNone(self.worker.failure)
        await self.worker.execute({'action': 'step'})

    async def test_recording_locks_backend_until_record_stop(self):
        self.worker.recorder = SimpleNamespace(flush_trajectory=lambda: None, finalize_video=lambda: None)
        try:
            with self.assertRaises(HTTPException) as failure:
                await self.worker.execute({'action': 'backend_select', 'engine_id': 'genesis'})
            self.assertEqual(failure.exception.status_code, 400)
            self.assertEqual(self.worker.engine_id, 'mujoco')
            self.assertIsNone(self.worker.failure)
        finally:
            self.worker.recorder = None

    def idle_api(self):
        controllers = {}
        for name in ('agent', 'policy', 'comparison', 'composition', 'research', 'repair'):
            state = {'active': False}
            controller = SimpleNamespace(state=state, snapshot=lambda state=state: state,
                                         reset_observation=Mock())
            if name == 'agent': controller.lock = asyncio.Lock()
            controllers[name] = controller
        return patch.multiple(application, worker=self.worker, **controllers)

    async def select_engine(self, engine_id):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                     base_url='http://127.0.0.1:8025') as client:
            return await client.post('/api/simulators/select', json={'engine_id': engine_id},
                                     headers={'origin': 'http://127.0.0.1:8025'})

    async def test_fatal_isolated_backend_can_explicitly_reload_mujoco_and_reject_old_action(self):
        from server.backends.process_backend import BackendProcessError
        def isolated_fixture(engine_id, generation_base):
            # Real local physics under a fixture identity only exercises the
            # worker lifecycle; this test makes no Genesis integration claim.
            simulation = Simulation(render=False)
            simulation.generation += generation_base
            simulation.paused = True
            simulation.render = lambda: image_bytes('red')
            native_state, native_command = simulation.state, simulation.command
            def state():
                return {**native_state(), 'backend': backend_identity('genesis', 'fixture')}
            def command(payload):
                if payload['action'] == 'fixture_fatal':
                    raise BackendProcessError('isolated child exited with fixture diagnostic')
                return native_command(payload)
            simulation.state, simulation.command = state, command
            return simulation
        with patch('server.backends.process_backend.ProcessBackend', side_effect=isolated_fixture), \
                patch('server.app.traceback.print_exc'):
            await self.worker.execute({'action': 'backend_select', 'engine_id': 'genesis'})
            old_generation, old_session = self.worker.snapshot['generation'], self.worker.session_id
            await self.worker.execute({'action': 'home'})
            self.assertEqual(self.worker.snapshot['task']['status'], 'running')
            with self.assertRaises(HTTPException):
                await self.worker.execute({'action': 'fixture_fatal'})
            await asyncio.to_thread(self.worker.thread.join, 5)
            self.assertFalse(self.worker.thread.is_alive())
            self.assertIn('fixture diagnostic', self.worker.failure)
            self.assertFalse(self.worker.snapshot['ready'])
            self.assertEqual(self.worker.snapshot['task']['status'], 'running')
            old_thread = self.worker.thread
            with self.idle_api():
                response = await self.select_engine('mujoco')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['restarted'])
            self.assertIsNot(self.worker.thread, old_thread)
            self.assertTrue(self.worker.thread.is_alive())
            self.assertEqual(self.worker.engine_id, 'mujoco')
            self.assertIsNone(self.worker.failure)
            self.assertTrue(self.worker.snapshot['ready'])
            self.assertNotEqual(self.worker.session_id, old_session)
            self.assertGreater(self.worker.snapshot['generation'], old_generation)
            time_before = self.worker.snapshot['time']
            await self.worker.execute({'action': 'step'})
            self.assertGreater(self.worker.snapshot['time'], time_before)
            with self.assertRaises(HTTPException) as stale:
                await self.worker.execute({'action': 'move_tcp', 'generation': old_generation, 'pose': {}})
            self.assertEqual(stale.exception.status_code, 400)
            self.assertIn('失效', stale.exception.detail)

    async def test_failed_explicit_reload_has_no_fallback_and_can_then_select_another_engine(self):
        self.worker.shutdown.set()
        await asyncio.to_thread(self.worker.thread.join, 5)
        native_count = len(self.native_instances)
        with self.idle_api(), patch('server.app.traceback.print_exc'), \
                patch('server.backends.process_backend.ProcessBackend', side_effect=ValueError('missing genesis fixture environment')):
            failed = await self.select_engine('genesis')
            self.assertEqual(failed.status_code, 503, failed.text)
            self.assertIn('missing genesis fixture environment', failed.json()['detail'])
            self.assertEqual(self.worker.engine_id, 'genesis')
            self.assertFalse(self.worker.snapshot['ready'])
            self.assertEqual(len(self.native_instances), native_count, 'A failed Genesis load must not start MuJoCo automatically.')
            recovered = await self.select_engine('mujoco')
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertTrue(self.worker.snapshot['ready'])
            self.assertEqual(self.worker.engine_id, 'mujoco')
            self.assertIsNone(self.worker.failure)

    async def test_restart_rejects_a_healthy_running_thread(self):
        thread, session = self.worker.thread, self.worker.session_id
        with self.assertRaises(HTTPException) as rejected:
            await self.worker.restart('mujoco')
        self.assertEqual(rejected.exception.status_code, 409)
        self.assertIs(self.worker.thread, thread)
        self.assertEqual(self.worker.session_id, session)

    async def test_slow_old_cleanup_does_not_leave_reload_permanently_locked(self):
        self.worker.failure = 'failed backend is still cleaning up'
        self.worker.switching = True
        with patch.object(self.worker.thread, 'join') as wait_for_cleanup:
            with self.assertRaises(HTTPException) as rejected:
                await self.worker.restart('mujoco')
        self.assertEqual(rejected.exception.status_code, 409)
        wait_for_cleanup.assert_called_once_with(5)
        self.assertFalse(self.worker.switching)

    async def test_disconnect_during_explicit_reload_does_not_abandon_native_initialization(self):
        self.worker.shutdown.set()
        await asyncio.to_thread(self.worker.thread.join, 5)
        entered, release = Event(), Event()
        native = application.Simulation
        def slow_native():
            entered.set()
            if not release.wait(5): raise RuntimeError('test fixture was not released')
            return native()
        with self.idle_api(), patch.object(application, 'Simulation', side_effect=slow_native):
            request = asyncio.create_task(self.select_engine('mujoco'))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                request.cancel()
                with self.assertRaises(asyncio.CancelledError): await request
                self.assertTrue(self.worker.switching)
                self.assertFalse(self.worker.snapshot['ready'])
            finally:
                release.set()
            self.assertTrue(await asyncio.to_thread(self.worker.ready.wait, 10))
            self.assertIsNone(self.worker.failure)
            self.assertTrue(self.worker.snapshot['ready'])
            self.assertFalse(self.worker.switching)


class FrozenBackendContractTests(unittest.TestCase):
    def test_posix_install_detection_uses_independent_environment_python_version(self):
        with tempfile.TemporaryDirectory(prefix='backend-install-') as folder:
            venv = Path(folder)
            python = venv / 'bin' / 'python'
            python.parent.mkdir()
            python.write_bytes(b'fixture only')
            other_version = '3.11' if tuple(sys.version_info[:2]) != (3, 11) else '3.12'
            distribution = venv / 'lib' / f'python{other_version}' / 'site-packages' / 'genesis_world-1.4.5.dist-info'
            distribution.mkdir(parents=True)
            with patch.object(simulators, 'runtime_python', return_value=python), patch.object(simulators.sys, 'platform', 'linux'):
                entry = next(item for item in simulators.catalog()['backends'] if item['id'] == 'genesis')
            self.assertTrue(entry['installed'])
            self.assertEqual(entry['version'], '1.4.5')

    def test_genesis_catalog_is_specific_and_does_not_claim_other_scenes_or_robots(self):
        models, scenes, tasks = supported_catalog('genesis')
        self.assertEqual(set(models), {'panda'})
        self.assertEqual(set(scenes), {'tabletop'})
        self.assertEqual(set(tasks), {'lift_red', 'place_red', 'reach_up'})
        for robot, scene, task in [('so101', 'tabletop', 'lift_red'), ('panda', 'drawer', 'open_drawer')]:
            with self.assertRaises(ValueError): validate_selection('genesis', robot, scene, task)

    def test_checkpoint_and_trajectory_reject_foreign_backend_without_mutating_physics(self):
        simulation = Simulation(render=False)
        try:
            saved, frame = simulation.checkpoint(), simulation.trajectory_frame()
            initial = simulation.data.qpos.copy()
            for record, restore in [(saved, simulation.restore), (frame, simulation.replay_frame)]:
                for backend in [backend_identity('genesis', 'fixture'),
                                {**backend_identity(), 'engine_version': 'other-version'},
                                {**backend_identity(), 'state_schema': 'other-schema'}]:
                    with self.subTest(backend=backend, restore=restore.__name__):
                        with self.assertRaises(ValueError): restore({**record, 'backend': backend})
                        np.testing.assert_array_equal(simulation.data.qpos, initial)
            # Records made before engine metadata was introduced remain MuJoCo.
            legacy = {key: value for key, value in saved.items() if key != 'backend'}
            simulation.restore(legacy)
            self.assertTrue(simulation.paused)
        finally:
            simulation.close()

    def test_success_requires_current_engine_privileged_source(self):
        for engine_id in ('mujoco', 'genesis'):
            snapshot = {'ready': True, 'backend': backend_identity(engine_id),
                        'evaluation': {'success': True, 'source': f'{engine_id}_privileged_state'}}
            self.assertTrue(environment_success(snapshot))
            for source in ('model_claim', 'unknown', f'{"genesis" if engine_id == "mujoco" else "mujoco"}_privileged_state'):
                self.assertFalse(environment_success({**snapshot, 'evaluation': {'success': True, 'source': source}}))
            self.assertFalse(environment_success({**snapshot, 'ready': False}))
        self.assertFalse(environment_success({'ready': True, 'backend': {'engine_id': 'unregistered'},
            'evaluation': {'success': True, 'source': 'unregistered_privileged_state'}}))

    def test_old_v1_protocol_id_is_unchanged_and_v2_freezes_engine(self):
        with tempfile.TemporaryDirectory(prefix='backend-protocol-') as folder, \
                patch.object(research, 'PROTOCOLS', Path(folder)):
            spec = research.ResearchProtocol(name='existing MuJoCo protocol').model_dump()
            spec.pop('engine_id')
            content = {'spec': spec, 'contract': research.evaluation_contract(),
                       'task': research.TASKS[spec['task_id']], 'schema_version': 'embodied-protocol-v1'}
            encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
            old_id = 'protocol-' + hashlib.sha256(encoded).hexdigest()[:24]
            manifest = {**content, 'protocol_id': old_id, 'created_at': '2026-10-23T00:00:00Z'}
            path = Path(folder) / (old_id + '.json')
            path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(research.get_protocol(old_id), manifest)
            self.assertNotIn('engine_id', research.get_protocol(old_id)['spec'])
            current = research.register_protocol({'name': spec['name'], 'robot': 'panda'})
            self.assertEqual(current['schema_version'], 'embodied-protocol-v2')
            self.assertEqual(current['spec']['engine_id'], 'mujoco')
            self.assertEqual(research.get_protocol(current['protocol_id']), current)
            changed = deepcopy(current)
            changed['spec']['engine_id'] = 'genesis'
            target = Path(folder) / (current['protocol_id'] + '.json')
            target.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '编号'):
                research.get_protocol(current['protocol_id'])


class CompletionBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_backend_completion_source_is_logged_and_rejected(self):
        controller = SimpleNamespace(state={}, _accepting=True)
        worker = SimpleNamespace(lock=Lock(), snapshot={'ready': True, 'backend': backend_identity('genesis')},
            execute=AsyncMock(return_value={'success': True, 'source': 'mujoco_privileged_state'}))
        log = SimpleNamespace(write=Mock())
        self.assertFalse(await verify_completion(controller, worker, log, 1, 'model says done', 12))
        self.assertEqual(controller.state['status'], 'replan')
        self.assertFalse(log.write.call_args[0][1]['accepted'])

    async def test_comparison_does_not_allocate_trials_for_other_engine(self):
        worker = SimpleNamespace(lock=Lock(), snapshot={'backend': backend_identity('mujoco')})
        controller = ComparisonController(worker, None, None)
        request = ComparisonStart(engine_id='genesis', robot='panda', methods=['agent', 'vla'])
        with self.assertRaisesRegex(ValueError, '后端'):
            await controller.start(request)
        self.assertFalse(controller.state['active'])
        self.assertEqual(controller.state['rows'], [])


if __name__ == '__main__':
    unittest.main()
