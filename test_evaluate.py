import unittest

from evaluate import greedy_reward, score_scene


class CorrespondenceDiagnosticTests(unittest.TestCase):
    def test_geometry_matches_globally_nearest_pairs_first(self) -> None:
        # The first truth point's nearest prediction belongs exactly to the
        # second point, so row-order greedy matching would produce a different
        # result from the competition's nearest-pairs-first rule.
        self.assertAlmostEqual(
            greedy_reward([(0, 0), (20, 0)], [(20, 0), (-30, 0)]),
            (1.0 + 0.25) / 2.0,
        )

    def test_swapped_member_patches_are_visible(self) -> None:
        truth = {
            "Id": "orion", "n_patches": "2",
            "patch_01": "(100, 100, 1)", "patch_02": "(200, 200, 1)",
        }
        prediction = {
            "Id": "orion", "n_patches": "2", "constellation": "orion",
            "patch_01": "(200, 200, 1)", "patch_02": "(100, 100, 1)",
        }
        result = score_scene(truth, prediction)
        self.assertEqual(result["geometry_strict"], 1.0)
        self.assertEqual(result["member_patch_f1"], 0.0)
        self.assertEqual(result["localisation"], 0.0)


if __name__ == "__main__":
    unittest.main()
