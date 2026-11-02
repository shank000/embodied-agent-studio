"""Native observation archives preserve camera identity and typed depth without SDKs."""
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from server.runlog import RunLog


def camera(name, depth=None):
    image = io.BytesIO()
    Image.new('RGB', (3, 2), (30, 90, 150)).save(image, format='JPEG')
    result = {'name': name, 'image_base64': base64.b64encode(image.getvalue()).decode(),
              'intrinsics': [[1, 0, 1.5], [0, 1, 1], [0, 0, 1]], 'width': 3, 'height': 2}
    if depth is not None:
        data = io.BytesIO()
        np.save(data, depth, allow_pickle=False)
        result['depth'] = {'encoding': 'npy-float32', 'unit': 'm', 'shape': list(depth.shape),
                           'data_base64': base64.b64encode(data.getvalue()).decode(),
                           'validity': 'zero_is_invalid'}
    return result


class ObservationArchiveContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='native-observation-contract-')
        self.addCleanup(self.tmp.cleanup)
        self.log = RunLog('agent', {}, directory=Path(self.tmp.name))
        self.addCleanup(lambda: self.log.finish({'status': 'complete'}))

    def test_declared_native_camera_names_roundtrip_with_calibration_and_depth(self):
        names = ['head', 'left_wrist', 'right_wrist']
        depth = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        original = {'backend': {'engine_id': 'omnigibson', 'action_mode': 'native', 'camera_names': names},
                    'cameras': [camera(name, depth) for name in reversed(names)]}
        saved = self.log.save_observation(original, 1)
        self.assertEqual([item['name'] for item in saved['cameras']], list(reversed(names)))
        for source, item in zip(original['cameras'], saved['cameras']):
            self.assertEqual(item['intrinsics'], source['intrinsics'])
            image = (self.log.folder / item['image_ref']).read_bytes()
            self.assertEqual(image, base64.b64decode(source['image_base64']))
            raw = (self.log.folder / item['depth']['depth_ref']).read_bytes()
            self.assertEqual(item['depth']['depth_sha256'], hashlib.sha256(raw).hexdigest())
            np.testing.assert_array_equal(np.load(io.BytesIO(raw), allow_pickle=False), depth)
            self.assertNotIn('data_base64', item['depth'])
            self.assertIn('data_base64', source['depth'])
            self.assertEqual(item['depth']['validity'], 'zero_is_invalid')
        manifest = (self.log.folder / 'observations/manifest.jsonl').read_text(encoding='utf-8')
        self.assertNotIn('image_base64', manifest)
        self.assertNotIn('data_base64', manifest)
        self.assertEqual(json.loads(manifest)['backend']['camera_names'], names)

    def test_legacy_camera_contracts_remain_explicit(self):
        for step, (mode, names) in enumerate((('tcp_pose', ['front', 'side', 'wrist']),
                                            ('discrete', ['front']), ('text', [])), 1):
            saved = self.log.save_observation({'backend': {'action_mode': mode},
                                               'cameras': [camera(name) for name in names]}, step)
            self.assertEqual([item['name'] for item in saved['cameras']], names)
        with self.assertRaises(ValueError):
            self.log.save_observation({'backend': {'action_mode': 'tcp_pose'},
                                       'cameras': [camera('head')]}, 4)

    def test_invalid_declarations_and_camera_sets_are_rejected_before_any_write(self):
        declarations = (['../outside'], ['head/left'], ['head\\left'], ['head.jpg'],
                        ['CON'], ['lpt1'], ['head', 'HEAD'], ['head', 'head'],
                        ['x' * 65], [None], [{}], [], 'head')
        for declared in declarations:
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                self.log.save_observation({'backend': {'action_mode': 'native', 'camera_names': declared},
                                           'cameras': [camera('head')]}, 1)
        for actual in ([], [camera('head'), camera('head')], [camera('unlisted')],
                       [{'name': {}}], [None], {}):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                self.log.save_observation({'backend': {'action_mode': 'native', 'camera_names': ['head']},
                                           'cameras': actual}, 1)
        self.assertFalse((self.log.folder / 'observations').exists())

    def test_invalid_depth_does_not_create_partial_observation(self):
        original = camera('head', np.ones((2, 3), dtype=np.float32))
        invalid = [camera('head', np.ones((2, 3), dtype=np.float64)),
                   camera('head', np.array([[np.nan]], dtype=np.float32)),
                   camera('head', np.array([[-1]], dtype=np.float32))]
        for change in ({'shape': [1, 1]}, {'unit': 'mm'}, {'encoding': 'jpeg'},
                       {'data_base64': 'not-base64'}, {'shape': [True, 3]}):
            value = deepcopy(original)
            value['depth'].update(change)
            invalid.append(value)
        for value in invalid:
            with self.subTest(depth=value['depth']), self.assertRaisesRegex(ValueError, '深度'):
                self.log.save_observation({'backend': {'action_mode': 'native', 'camera_names': ['head']},
                                           'cameras': [value]}, 1)
        self.assertFalse((self.log.folder / 'observations').exists())

    def test_bad_rgb_and_duplicate_archive_do_not_overwrite_existing_round(self):
        observation = {'backend': {'action_mode': 'native', 'camera_names': ['head']},
                       'cameras': [camera('head')]}
        saved = self.log.save_observation(observation, 1)
        path = self.log.folder / saved['cameras'][0]['image_ref']
        original_bytes = path.read_bytes()
        with self.assertRaises(FileExistsError): self.log.save_observation(observation, 1)
        broken = deepcopy(observation)
        broken['cameras'][0]['image_base64'] = base64.b64encode(b'not a JPEG').decode()
        with self.assertRaisesRegex(ValueError, 'JPEG'): self.log.save_observation(broken, 2)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertFalse((self.log.folder / 'observations/000002').exists())


if __name__ == '__main__':
    unittest.main()
