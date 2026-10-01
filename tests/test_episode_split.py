import unittest
from src.recorder.episode_split import choose_split, SPLITS


class SplitTests(unittest.TestCase):
    def sequence(self, count, cells):
        history = []
        for i in range(count):
            cell = i % cells
            split = choose_split(history, cell, (i * 137) % 360)
            history.append(dict(dataset_split=split, split_cell_id=cell,
                                split_yaw_bin=int(((i * 137) % 360) // 90)))
        return history

    def test_small_collection_priorities(self):
        for cells in (1,10):
            self.assertEqual([r['dataset_split'] for r in self.sequence(3,cells)], list(SPLITS))

    def test_every_grid_has_seven_two_one(self):
        for cells in (1,3,10,15):
            for blocks in (1,2,5):
                history = self.sequence(cells*10*blocks,cells)
                for cell in range(cells):
                    counts = [sum(r['dataset_split']==s and r['split_cell_id']==cell for r in history) for s in SPLITS]
                    self.assertEqual(counts,[7*blocks,2*blocks,blocks])

    def test_retry_and_resume_do_not_change_assignment(self):
        history = self.sequence(27,10)
        self.assertEqual(choose_split(history,7,35),choose_split(list(history),7,35))

    def test_first_cycle_spreads_splits_across_grid(self):
        history = self.sequence(10,10)
        self.assertEqual([sum(r['dataset_split']==s for r in history) for s in SPLITS],[7,2,1])


if __name__ == '__main__':
    unittest.main()
