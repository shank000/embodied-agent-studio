"""Public platform integration checks; these do not certify the native SDK."""
import base64
import importlib
import json
from pathlib import Path
import tempfile
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from server import research, simulators
from server.comparison import ComparisonController, ComparisonStart
from server.policy import PolicyConfig, PolicyController, PolicyStart, predict_policy
from server.task_verification import environment_failure, environment_success

app = importlib.import_module('server.app')


class RoboDojoIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def snapshot(self):
        return {'ready': True, 'backend': simulators.backend_identity('robodojo'),
                'model': 'arx_x5', 'timestep': .04,
                'evaluation': {'source': 'robodojo_official_state', 'success': False, 'terminated': False}}

    async def test_native_camera_names_are_served_for_agent_and_policy(self):
        # Bytes stand in for a recorded camera payload, not a rendered native image.
        payload = b'camera-payload-fixture'
        observation = {'cameras': [{'name': name, 'mime_type': 'image/png',
            'image_base64': base64.b64encode(payload).decode()} for name in ('head', 'left_wrist', 'right_wrist')]}
        with patch.object(app, 'agent', SimpleNamespace(observation=observation)), \
             patch.object(app, 'policy', SimpleNamespace(observation=observation)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='http://127.0.0.1:8000') as client:
                for owner in ('agent', 'policy'):
                    for camera in ('head', 'left_wrist', 'right_wrist'):
                        response = await client.get(f'/api/{owner}/camera/{camera}')
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.content, payload)
                        self.assertEqual(response.headers['content-type'], 'image/png')
                    self.assertEqual((await client.get(f'/api/{owner}/camera/front')).status_code, 404)

    def test_runtime_readiness_prevents_launch_before_native_sdk_is_ready(self):
        status = {'ready': False, 'checks': [{'label': 'Isaac Sim 5.1', 'status': 'missing'}]}
        with patch('server.robodojo.readiness', return_value=status), patch('server.robodojo.runtime_config', return_value=None):
            info = simulators.runtime_info('robodojo')
            self.assertFalse(info['installed'])
            with self.assertRaisesRegex(ValueError, 'Isaac Sim 5.1'):
                simulators.runtime_command('robodojo')

    def test_trusted_terminal_is_required_to_stop_failed_episode(self):
        state = self.snapshot()
        self.assertFalse(environment_failure(state))
        state['evaluation']['terminated'] = True
        self.assertTrue(environment_failure(state))
        state['evaluation']['success'] = True
        self.assertFalse(environment_failure(state))
        self.assertTrue(environment_success(state))
        state['evaluation'].update(source='model_claim', success=False)
        self.assertFalse(environment_failure(state))
        state['evaluation']['source'] = 'robodojo_official_state'
        state['ready'] = False
        self.assertFalse(environment_failure(state))

    async def test_paired_experiment_rejected_before_any_control_or_record_creation(self):
        worker = SimpleNamespace(lock=Lock(), snapshot=self.snapshot(), execute=AsyncMock())
        comparison = ComparisonController(worker, None, None)
        request = ComparisonStart(engine_id='robodojo', robot='arx_x5', scene_id='stack_bowls',
                                  task_id='stack_bowls', methods=['agent', 'vla'])
        with self.assertRaisesRegex(ValueError, '恢复完整物理初态'):
            await comparison.start(request)
        self.assertFalse(comparison.state['active'])
        self.assertEqual(comparison.state['rows'], [])
        worker.execute.assert_not_called()
        with tempfile.TemporaryDirectory() as folder, patch.object(research, 'PROTOCOLS', Path(folder)/'protocols'):
            with self.assertRaisesRegex(ValueError, '恢复完整物理初态'):
                research.register_protocol({'name': 'RoboDojo pair', 'engine_id': 'robodojo',
                    'robot': 'arx_x5', 'scene_id': 'stack_bowls', 'task_id': 'stack_bowls'})
            self.assertFalse((Path(folder)/'protocols').exists())

    async def test_metadata_trajectory_keeps_video_without_offering_physical_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = {'trajectory': 'trajectory.jsonl', 'trajectory_frames': 2,
                'checkpoints': {'initial': {'path': 'initial.json'}}, 'video': {'path': 'video.mp4'}}
            (root/'summary.json').write_text(json.dumps({'backend': simulators.backend_identity('robodojo'),
                                                        'artifacts': artifacts}), encoding='utf-8')
            with patch.object(app, 'get_run', return_value=root):
                result = await app.run_episode('fixture')
            self.assertFalse(result['available'])
            self.assertEqual(result['video'], artifacts['video'])
            self.assertEqual(result['trajectory_frames'], 2)
            self.assertIn('视频', result['message'])

    async def test_invalid_policy_cadence_does_not_call_api_or_move_robot(self):
        worker = SimpleNamespace(lock=Lock(), snapshot=self.snapshot(), execute=AsyncMock())
        policy = PolicyController(worker)
        config = PolicyConfig(kind='vla', endpoint='http://127.0.0.1:8011/predict', dt_s=.05)
        policy.configure(config)
        with self.assertRaisesRegex(ValueError, '0.04'):
            await policy.start(PolicyStart(kind='vla', task='stack bowls'))
        worker.execute.assert_not_called()
        self.assertFalse(policy.state['active'])
        client = SimpleNamespace(post=AsyncMock())
        with self.assertRaisesRegex(ValueError, '0.04'):
            await predict_policy(config, {'observation': self.snapshot()}, client)
        client.post.assert_not_called()
        policy.configure(config.model_copy(update={'dt_s': .08}))
        policy.validate_start('vla', 'arx_x5')


if __name__ == '__main__': unittest.main()
