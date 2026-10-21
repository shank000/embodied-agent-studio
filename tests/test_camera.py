"""View navigation changes only the interactive MuJoCo camera."""
import base64
import io
import unittest

import numpy as np
from pydantic import ValidationError
from PIL import Image

from server.app import Command
from server.simulation import Simulation


class CameraNavigationTests(unittest.TestCase):
    def test_pan_changes_lookat_without_changing_physics_or_calibrated_cameras(self):
        for robot in ('panda', 'so101', 'rm65_gaia'):
            with self.subTest(robot=robot):
                sim = Simulation(robot, render=False)
                try:
                    sim.command({'action': 'pause', 'value': True})
                    before = sim.checkpoint()
                    lookat = sim.cam.lookat.copy()
                    angles = [sim.cam.azimuth, sim.cam.elevation, sim.cam.distance]
                    camera_positions = sim.model.cam_pos.copy()
                    camera_quaternions = sim.model.cam_quat.copy()
                    steps = sim.physics_steps
                    sim.command({'action': 'camera', 'pan_dx': .15, 'pan_dy': .1})
                    self.assertGreater(np.linalg.norm(sim.cam.lookat-lookat), .01*sim.config['scale'])
                    self.assertEqual([sim.cam.azimuth, sim.cam.elevation, sim.cam.distance], angles)
                    self.assertEqual(sim.view, 'custom')
                    self.assertEqual(sim.checkpoint(), before)
                    self.assertEqual(sim.physics_steps, steps)
                    self.assertTrue(sim.paused)
                    np.testing.assert_array_equal(sim.model.cam_pos, camera_positions)
                    np.testing.assert_array_equal(sim.model.cam_quat, camera_quaternions)
                finally:
                    sim.close()

    def test_top_view_vertical_pan_moves_across_table_and_presets_reset_target(self):
        sim = Simulation('so101', render=False)
        try:
            sim.set_view('top')
            before = sim.cam.lookat.copy()
            sim.command({'action': 'camera', 'pan_dy': .2})
            # A world-Z-only translation would leave the top view's table framing unchanged.
            self.assertGreater(np.linalg.norm((sim.cam.lookat-before)[:2]), .01)
            self.assertLess(abs(sim.cam.lookat[2]-before[2]), .01)
            sim.set_view('perspective')
            np.testing.assert_allclose(sim.cam.lookat, np.asarray([.24, 0., .30])*sim.config['scale'])
            sim.command({'action': 'camera', 'dx': 15., 'dy': 4., 'zoom': .1})
            self.assertEqual(sim.cam.azimuth, 150.)
            self.assertEqual(sim.cam.elevation, -22.)
            self.assertGreater(sim.cam.distance, 2.15*sim.config['scale'])
        finally:
            sim.close()

    def test_native_user_view_moves_but_agent_camera_geometry_and_pixels_stay_stable(self):
        sim = Simulation('so101', scene_id='drawer', render=True)
        try:
            sim.command({'action': 'settle'})
            before_observation = sim.observe()
            before_frame = sim.render()
            before_state = (sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.ctrl.copy(), sim.data.time)
            sim.command({'action': 'camera', 'pan_dx': .15, 'pan_dy': .1})
            after_frame = sim.render()
            after_observation = sim.observe()
            self.assertNotEqual(before_frame, after_frame)
            for before, after in zip(before_observation['cameras'], after_observation['cameras']):
                self.assertEqual({key: value for key, value in before.items() if key != 'image_base64'},
                                 {key: value for key, value in after.items() if key != 'image_base64'})
                pixels = [np.asarray(Image.open(io.BytesIO(base64.b64decode(camera['image_base64']))), dtype=np.int16)
                          for camera in (before, after)]
                difference = np.abs(pixels[0]-pixels[1])
                # Repeated unchanged native renders also differ by up to 2/255
                # in a few decoded JPEG pixels on this GPU; geometry stays exact.
                self.assertLessEqual(int(difference.max()), 3)
                self.assertLess(float(difference.mean()), .01)
            self.assertEqual({key: value for key, value in before_observation.items() if key != 'cameras'},
                             {key: value for key, value in after_observation.items() if key != 'cameras'})
            for before, after in zip(before_state[:3], (sim.data.qpos, sim.data.qvel, sim.data.ctrl)):
                np.testing.assert_array_equal(before, after)
            self.assertEqual(sim.data.time, before_state[3])
        finally:
            sim.close()

    def test_camera_api_accepts_bounded_pan_and_rejects_invalid_values(self):
        command = Command.model_validate({'action': 'camera', 'pan_dx': .2, 'pan_dy': -.3})
        self.assertEqual(command.pan_dx, .2)
        self.assertEqual(command.pan_dy, -.3)
        for values in ({'pan_dx': float('nan')}, {'pan_dy': float('inf')}, {'pan_dx': 3.}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Command.model_validate({'action': 'camera', **values})


if __name__ == '__main__': unittest.main()
