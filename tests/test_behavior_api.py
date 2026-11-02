"""Public benchmark lifecycle and dynamic native view contracts; no SDK launch."""
import asyncio
import importlib
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

application = importlib.import_module('server.app')


class BehaviorApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.controllers = {}
        for name in ('agent', 'policy', 'comparison', 'composition', 'research', 'repair',
                     'benchmark', 'behavior_benchmark', 'harnesspai'):
            state = {'active': False}
            controller = SimpleNamespace(state=state, snapshot=lambda state=state: state,
                stop=AsyncMock(return_value=state), start=AsyncMock(return_value={'active': True}))
            if name == 'agent': controller.lock = asyncio.Lock()
            self.controllers[name] = controller
        self.worker = SimpleNamespace(lock=Lock(), switching=False, execute=AsyncMock(return_value='ok'),
            snapshot={'task': {'status': 'idle'}, 'backend': {'engine_id': 'mujoco'}})
        for name, value in {**self.controllers, 'worker': self.worker}.items():
            item = patch.object(application, name, value)
            item.start()
            self.addCleanup(item.stop)

    async def request(self, path, body=None, origin='http://127.0.0.1:8000'):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app),
                                     base_url='http://127.0.0.1:8000') as client:
            return await client.post(path, json=body or {}, headers={'origin': origin})

    async def test_launch_requires_idle_and_releases_heavy_interactive_engine_first(self):
        route = '/api/benchmarks/behavior/start'
        for name, controller in self.controllers.items():
            controller.state['active'] = True
            with self.subTest(controller=name): self.assertEqual((await self.request(route)).status_code, 409)
            controller.state['active'] = False
        self.worker.snapshot['backend']['engine_id'] = 'omnigibson'
        self.assertEqual((await self.request(route)).status_code, 409)
        self.controllers['behavior_benchmark'].start.assert_not_awaited()
        self.worker.snapshot['backend']['engine_id'] = 'mujoco'
        self.assertEqual((await self.request(route)).status_code, 200)
        self.controllers['behavior_benchmark'].start.assert_awaited_once()

    async def test_active_behavior_blocks_simulator_switch_and_is_globally_stoppable(self):
        self.controllers['behavior_benchmark'].state['active'] = True
        response = await self.request('/api/simulators/select', {'engine_id': 'omnigibson'})
        self.assertEqual(response.status_code, 409)
        self.worker.execute.assert_not_awaited()
        self.assertEqual((await self.request('/api/command', {'action': 'stop'})).status_code, 200)
        self.controllers['behavior_benchmark'].stop.assert_awaited()

    async def test_plan_is_read_only_and_origin_guard_applies(self):
        with patch.object(application, 'behavior_plan', return_value={'ready': False, 'commands': []}) as plan:
            response = await self.request('/api/benchmarks/behavior/plan')
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()['ready'])
            plan.assert_called_once()
            self.controllers['behavior_benchmark'].start.assert_not_awaited()
        self.assertEqual((await self.request('/api/benchmarks/behavior/start', origin='https://example.org')).status_code, 403)
        self.assertEqual((await self.request('/api/benchmarks/behavior/start', {'instance_indices': [20]})).status_code, 422)

    async def test_declared_native_camera_alias_reaches_backend_and_path_is_rejected(self):
        alias = 'robot0_head_camera_123456abcd'
        response = await self.request('/api/command', {'action': 'view', 'view': alias})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.worker.execute.await_args.args[0]['view'], alias)
        self.assertEqual((await self.request('/api/command', {'action': 'view', 'view': '../private'})).status_code, 422)
