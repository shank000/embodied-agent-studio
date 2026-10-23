"""MJPEG framing works with a paused worker; no native simulation is started."""
import io
from email import policy
from email.parser import BytesParser
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from server.app import stream


PART_PREFIX = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'


def jpeg(color):
    output = io.BytesIO()
    Image.new('RGB', (12, 8), color).save(output, 'JPEG')
    return output.getvalue()


class Connection:
    disconnected = False

    async def is_disconnected(self):
        return self.disconnected


class MjpegStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_paused_first_frame_is_complete_without_waiting_for_another_image(self):
        image = jpeg('#668877')
        worker = SimpleNamespace(lock=Lock(), frame=7, jpeg=image, failure=None)
        connection = Connection()
        with patch('server.app.worker', worker):
            response = await stream(connection)
            chunks = response.body_iterator
            try:
                self.assertIn('boundary=frame', response.headers['content-type'])
                opening = await anext(chunks)
                first = await anext(chunks)
                # Chromium's multipart image callback publishes the previous
                # JPEG only once the next part's complete headers arrive.
                completed, pending = (opening + first).split(b'\r\n--frame\r\n', 1)
                self.assertEqual(pending, b'Content-Type: image/jpeg\r\n\r\n')
                headers, body = completed.split(b'\r\n\r\n', 1)
                self.assertEqual(opening, PART_PREFIX)
                self.assertNotIn(b'Content-Length:', headers)
                self.assertEqual(body, image)
                self.assertEqual((opening + first).count(image), 1)
                with Image.open(io.BytesIO(body)) as decoded:
                    self.assertEqual(decoded.size, (12, 8))
                    decoded.verify()
                connection.disconnected = True
                with self.assertRaises(StopAsyncIteration):
                    await anext(chunks)
            finally:
                await chunks.aclose()

    async def test_consecutive_frames_form_one_multipart_sequence_without_empty_parts(self):
        first_image, second_image = jpeg('#446688'), jpeg('#aa7755')
        worker = SimpleNamespace(lock=Lock(), frame=1, jpeg=first_image, failure=None)
        connection = Connection()
        with patch('server.app.worker', worker):
            response = await stream(connection)
            chunks = response.body_iterator
            try:
                wire = await anext(chunks) + await anext(chunks)
                worker.frame, worker.jpeg = 2, second_image
                wire += await anext(chunks)
                self.assertTrue(wire.endswith(PART_PREFIX))
                # Complete the otherwise live stream for the standard MIME parser.
                wire = wire[:-len(PART_PREFIX)] + b'--frame--\r\n'
                message = BytesParser(policy=policy.default).parsebytes(
                    b'Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n' + wire)
                parts = list(message.iter_parts())
                self.assertEqual(len(parts), 2)
                self.assertEqual([part.get_content_type() for part in parts], ['image/jpeg'] * 2)
                self.assertEqual([part.get_payload(decode=True) for part in parts], [first_image, second_image])
                self.assertTrue(all(part['Content-Length'] is None for part in parts))
                connection.disconnected = True
                with self.assertRaises(StopAsyncIteration):
                    await anext(chunks)
            finally:
                await chunks.aclose()
