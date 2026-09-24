import unittest

import numpy as np

from error_ledger import candidate_rank, classify_patch, patch_signal


class ErrorLedgerTests(unittest.TestCase):
    def test_candidate_rank_uses_first_match(self) -> None:
        candidates = [[100, 100, 0.9], [10, 20, 0.8], [11, 20, 0.7]]
        self.assertEqual(candidate_rank(candidates, (10, 20, 0), 12.0), 2)
        self.assertEqual(candidate_rank(candidates, None, 12.0), 0)

    def test_failure_stages_are_distinct(self) -> None:
        truth = (10, 20, 1)
        self.assertEqual(classify_patch(truth, None, 0, 0, 12.0), "retrieval_missing")
        self.assertEqual(classify_patch(truth, None, 2, 0, 12.0), "presence_missed")
        self.assertEqual(
            classify_patch(truth, (100, 100, 0), 2, 0, 12.0), "location_selection_error"
        )
        self.assertEqual(classify_patch(truth, (10, 20, 0), 1, 1, 12.0), "membership_missed")
        self.assertEqual(classify_patch(None, (10, 20, 0), 0, 0, 12.0), "false_positive")

    def test_signal_detects_center_point(self) -> None:
        patch = np.full((32, 32), 20, dtype=np.uint8)
        patch[16, 16] = 180
        metrics = patch_signal(patch)
        self.assertGreater(metrics["peak_over_noise"], 100)
        self.assertLess(metrics["star_width_px"], 2)


if __name__ == "__main__":
    unittest.main()
