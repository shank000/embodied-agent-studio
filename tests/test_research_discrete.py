"""Discrete research requests fail before control/recording; old records remain readable."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from server import research
from server.simulators import backend_identity, supported_catalog


DISCRETE = {
    'alfworld': ('alfworld_agent', 'alfworld_unseen', 'pick_and_place_simple'),
    'alfworld_thor': ('alfworld_thor_agent', 'alfworld_thor_unseen', 'pick_and_place_simple'),
    'ai2thor': ('thor_agent', 'FloorPlan1', 'thor_open_object'),
}


class DiscreteResearchPreflightTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='research-discrete-')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)/'protocols'
        configured = patch.object(research, 'PROTOCOLS', self.folder)
        configured.start(); self.addCleanup(configured.stop)

    def spec(self, engine):
        robot, scene, task = DISCRETE[engine]
        return research.ResearchProtocol(name='Legacy discrete protocol fixture',
            engine_id=engine, robot=robot, scene_id=scene, task_id=task)

    def legacy_record(self, engine):
        spec = self.spec(engine)
        content = {'spec': spec.model_dump(), 'contract': research.evaluation_contract(engine),
                   'task': supported_catalog(engine)[2][spec.task_id], 'schema_version':'embodied-protocol-v2'}
        encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        identity = 'protocol-'+hashlib.sha256(encoded).hexdigest()[:24]
        record = {**content, 'protocol_id':identity, 'created_at':'2026-10-25T00:00:00Z'}
        self.folder.mkdir(exist_ok=True)
        (self.folder/(identity+'.json')).write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
        return record

    def test_new_discrete_protocol_is_rejected_before_creating_a_registry_record(self):
        for engine in DISCRETE:
            with self.subTest(engine=engine), self.assertRaisesRegex(ValueError, '普通 Agent'):
                research.register_protocol(self.spec(engine))
        self.assertFalse(self.folder.exists())

    async def test_existing_discrete_protocol_is_readable_but_cannot_allocate_or_mutate_a_run(self):
        for engine in DISCRETE:
            with self.subTest(engine=engine):
                record = self.legacy_record(engine)
                self.assertEqual(research.get_protocol(record['protocol_id']), record)
                worker = SimpleNamespace(lock=Lock(), snapshot={'backend':backend_identity(engine)}, execute=AsyncMock())
                runner = SimpleNamespace(snapshot=lambda: {'active':False}, task=None)
                controller = research.ResearchController(worker, runner)
                before = deepcopy(controller.state)
                request = research.ResearchExperiment(protocol_id=record['protocol_id'], method_ids=['fixture-parent', 'fixture-child'])
                with patch.object(research, 'verify_method') as methods, patch.object(research, 'RunLog') as log:
                    with self.assertRaisesRegex(ValueError, '连续机械臂动作'):
                        await controller.start(request)
                worker.execute.assert_not_called(); methods.assert_not_called(); log.assert_not_called()
                self.assertEqual(controller.state, before)
                self.assertIsNone(controller.task)
                self.assertIsNone(controller._active_trial)
        self.assertEqual(len(research.list_protocols()), len(DISCRETE))

    def test_continuous_robot_protocol_registration_remains_available(self):
        record = research.register_protocol({'name':'Continuous protocol fixture'})
        self.assertEqual(record['spec']['engine_id'], 'mujoco')
        self.assertEqual(research.get_protocol(record['protocol_id']), record)


if __name__ == '__main__': unittest.main()
