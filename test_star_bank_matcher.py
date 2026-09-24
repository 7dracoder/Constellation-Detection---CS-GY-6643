from __future__ import annotations

import unittest

import numpy as np

from star_bank_matcher import merge_candidate_lists, subpixel_centroid


class StarBankMatcherTests(unittest.TestCase):
    def test_subpixel_centroid_tracks_fractional_peak(self) -> None:
        yy, xx = np.mgrid[:11, :11]
        expected_x, expected_y = 5.35, 4.65
        response = np.exp(-((xx - expected_x) ** 2 + (yy - expected_y) ** 2) / (2 * 1.1**2))
        actual_x, actual_y = subpixel_centroid(response.astype(np.float32), 5, 5, radius=3)
        self.assertAlmostEqual(actual_x, expected_x, delta=0.12)
        self.assertAlmostEqual(actual_y, expected_y, delta=0.12)

    def test_merge_preserves_fractional_coordinates(self) -> None:
        primary = [[10.0, 10.0, 0.8, 0.1]]
        secondary = [(0.7, 40.25, 50.75, 0.05)]
        merged = merge_candidate_lists(primary, secondary, top_k=2)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1][0], 40.25)
        self.assertEqual(merged[1][1], 50.75)


if __name__ == "__main__":
    unittest.main()
