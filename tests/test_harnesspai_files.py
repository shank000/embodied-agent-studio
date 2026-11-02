import ctypes
import json
import os
from pathlib import Path
import tempfile
from threading import Thread, Event
import time
import unittest

from tools.harnesspai_runner import write_json


class SnapshotTests(unittest.TestCase):
    def test_status_keeps_failure_cause_and_full_log_remains_available(self):
        from server.harnesspai import runner_error
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'runner.log'
            content='[rollout] '+('command '*800)+'\nTraceback (most recent call last):\n  File "test.py", line 1\nPermissionError: [WinError 5] access denied\n'
            path.write_text(content)
            message=runner_error(path,1)
            self.assertIn('PermissionError: [WinError 5]',message)
            self.assertLess(len(message),150)
            self.assertEqual(path.read_text(),content)

    def test_invalid_serialization_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            write_json(path,{'stage':'baseline'})
            with self.assertRaises(TypeError): write_json(path,{'value':object()})
            self.assertEqual(json.loads(path.read_text()),{'stage':'baseline'})
            self.assertEqual(list(Path(directory).iterdir()),[path])


@unittest.skipUnless(os.name=='nt','Windows delete-sharing reproducer')
class WindowsSnapshotTests(unittest.TestCase):
    def reader_lock(self,path):
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
            wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        kernel.CloseHandle.restype=wintypes.BOOL
        # Read/write sharing, deliberately no FILE_SHARE_DELETE, as an ordinary
        # Windows reader may hold while the evolution subprocess publishes.
        handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,0,None)
        if handle==ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
        return lambda: kernel.CloseHandle(handle)

    def test_short_reader_lock_does_not_abort_progress_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';write_json(path,{'stage':'rollout'})
            close=self.reader_lock(path);done=Event();errors=[]
            def publish():
                try: write_json(path,{'stage':'diagnosis','round':2})
                except Exception as exc: errors.append(exc)
                finally: done.set()
            thread=Thread(target=publish);thread.start()
            try:
                self.assertFalse(done.wait(.12))
                self.assertEqual(json.loads(path.read_text()),{'stage':'rollout'})
            finally: close();thread.join(2)
            self.assertTrue(done.is_set());self.assertEqual(errors,[])
            self.assertEqual(json.loads(path.read_text()),{'stage':'diagnosis','round':2})
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_persistent_access_denial_stops_with_real_error_and_keeps_previous_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json';write_json(path,{'stage':'rollout'})
            close=self.reader_lock(path);started=time.monotonic()
            try:
                with self.assertRaises(PermissionError) as error: write_json(path,{'stage':'diagnosis'})
                self.assertIn(error.exception.winerror,(5,32,33))
                self.assertIn('旧快照保留',' '.join(error.exception.__notes__))
                self.assertLess(time.monotonic()-started,2)
                self.assertEqual(json.loads(path.read_text()),{'stage':'rollout'})
            finally: close()
            self.assertEqual(list(Path(directory).iterdir()),[path])
