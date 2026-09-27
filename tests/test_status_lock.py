import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.recorder.phase4_session import status


class StatusLockTests(unittest.TestCase):
    def test_temporary_lock_retries_then_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'session.json'
            real_replace=os.replace
            calls=[]
            def replace(src,dst):
                calls.append(1)
                if len(calls)<3:
                    raise PermissionError('reader has file open')
                real_replace(src,dst)
            with patch.dict(os.environ,{'P4_SESSION_CONFIG':str(path)}), patch('src.recorder.phase4_session.os.replace',side_effect=replace), patch('src.recorder.phase4_session.time.sleep'):
                self.assertTrue(status({'PHASE3_TARGET_OBJECT':'scalpel'},'recording'))
            self.assertEqual(len(calls),3)
            self.assertTrue(path.with_suffix('.status.json').exists())

    def test_persistent_lock_is_nonfatal_and_warning_is_not_spammed(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'session.json'
            ns={'PHASE3_TARGET_OBJECT':'scalpel'}
            with patch.dict(os.environ,{'P4_SESSION_CONFIG':str(path)}), patch('src.recorder.phase4_session.os.replace',side_effect=PermissionError('locked')), patch('src.recorder.phase4_session.time.sleep'), patch('builtins.print') as log:
                self.assertFalse(status(ns,'recording'))
                self.assertFalse(status(ns,'recording'))
                self.assertEqual(log.call_count,1)
            with patch.dict(os.environ,{'P4_SESSION_CONFIG':str(path)}):
                self.assertTrue(status(ns,'recording'))
                self.assertNotIn('_p4_status_write_warned',ns)


if __name__=='__main__':
    unittest.main()
