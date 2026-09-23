import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase4_feedback import transfer_detour


class TransferRouteTests(unittest.TestCase):
    def test_far_side_path_avoids_base_corridor(self):
        for x in (.22, .42):
            for y in (.2, .4):
                start, end = np.array([x, y, .16]), np.array([.15, -.70, .18])
                via = transfer_detour(start, end)
                self.assertIsNotNone(via)
                for a, b in ((start, via), (via, end)):
                    points = a + np.linspace(0, 1, 101)[:, None] * (b-a)
                    self.assertGreater(np.linalg.norm(points[:, :2], axis=1).min(), .29)
                self.assertGreaterEqual(via[2], max(start[2], end[2]))

    def test_far_side_routes_cover_every_instrument_slot(self):
        # The outer scissor/scalpel slots can miss a narrow straight-line radius
        # test despite their transfer still running into a wrist branch limit.
        for x in (.22,.42):
            for y in (.2,.4):
                for slot_x in (.358,.254,.150,.046,-.058):
                    with self.subTest(x=x,y=y,slot_x=slot_x):
                        self.assertIsNotNone(transfer_detour([x,y,.16],[slot_x,-.6976,.18]))

    def test_near_and_tray_side_paths_unchanged(self):
        for start in ([.22, 0, .16], [.42, -.2, .16], [.22, -.4, .16]):
            self.assertIsNone(transfer_detour(start, [.15, -.70, .18]))

    def test_nonfinite_pose_is_rejected(self):
        with self.assertRaises(ValueError):
            transfer_detour([float('nan'), .2, .16], [.15, -.7, .18])


if __name__ == '__main__':
    unittest.main()
