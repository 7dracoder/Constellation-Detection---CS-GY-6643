import unittest

from ownership_refiner import rank_cost, refine_row


class FixedNodeOwnershipTests(unittest.TestCase):
    def test_reassigns_only_existing_members_and_nodes(self) -> None:
        row = {
            "Id": "scene",
            "n_patches": "3",
            "patch_01": "(100, 100, 1)",
            "patch_02": "(200, 200, 1)",
            "patch_03": "(300, 300, 0)",
            "constellation": "orion",
        }
        raw = [
            [[200, 200, 0.9], [100, 100, 0.8]],
            [[100, 100, 0.9], [200, 200, 0.8]],
            [[300, 300, 0.9]],
        ]
        gaussian = raw
        refined, changed, gain = refine_row(row, raw, gaussian)
        self.assertEqual(changed, 2)
        self.assertGreater(gain, 0)
        self.assertEqual(refined["patch_01"], "(200, 200, 1)")
        self.assertEqual(refined["patch_02"], "(100, 100, 1)")
        self.assertEqual(refined["patch_03"], row["patch_03"])
        self.assertEqual(refined["constellation"], row["constellation"])
        blocked, changed, _ = refine_row(row, raw, gaussian, min_gain=gain + 0.1)
        self.assertEqual(changed, 0)
        self.assertEqual(blocked, row)

    def test_missing_view_is_not_rank_one(self) -> None:
        self.assertGreater(rank_cost(0, 24), rank_cost(1, 24))


if __name__ == "__main__":
    unittest.main()
