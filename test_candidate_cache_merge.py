import unittest

from candidate_cache_merge import merge_group


class CandidateCacheMergeTests(unittest.TestCase):
    def test_keeps_raw_first_and_distinct_alternate_hypothesis(self) -> None:
        raw = [[10, 10, 0.7, 0.1], [40, 40, 0.6, 0.0]]
        alternate = [[80, 80, 0.95, 0.2], [10, 11, 0.8, 0.1]]
        merged = merge_group(raw, alternate, 4)
        self.assertEqual([(row[0], row[1]) for row in merged],
                         [(10, 10), (80, 80), (40, 40)])
        self.assertEqual(merged[0][2], 0.7)
        self.assertLess(merged[1][2], merged[0][2])
        self.assertEqual(merged[1][3], 0.0)

    def test_rejects_empty_baseline(self) -> None:
        with self.assertRaises(ValueError):
            merge_group([], [[1, 2, 0.8, 0.0]], 2)


if __name__ == "__main__":
    unittest.main()
