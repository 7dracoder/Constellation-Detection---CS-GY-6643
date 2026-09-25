import unittest

from clear_gap_refiner import candidate_relative_gap


class RelativeGapTests(unittest.TestCase):
    def test_relative_gap_normalizes_by_remaining_score_headroom(self) -> None:
        self.assertAlmostEqual(
            candidate_relative_gap([[0, 0, 0.9], [10, 10, 0.88]]), 0.2
        )

    def test_short_group_is_not_clear(self) -> None:
        self.assertEqual(candidate_relative_gap([[0, 0, 0.9]]), 0.0)


if __name__ == "__main__":
    unittest.main()
