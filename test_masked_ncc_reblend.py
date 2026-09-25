import unittest

from masked_ncc_reblend import reblend_group


class ReblendTests(unittest.TestCase):
    def test_weight_changes_candidate_order(self) -> None:
        group = [
            [1, 1, 0, 0, 0.9, 0.1],
            [2, 2, 0, 0, 0.7, 0.8],
        ]
        self.assertEqual(reblend_group(group, 0.0)[0][:2], [1.0, 1.0])
        self.assertEqual(reblend_group(group, 1.0)[0][:2], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
