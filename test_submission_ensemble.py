import unittest

from submission_ensemble import changed_patch_count


class ChangedPatchCountTests(unittest.TestCase):
    def test_counts_only_active_patch_cells(self) -> None:
        prior = {
            "Id": "scene",
            "n_patches": "2",
            "patch_01": "-1",
            "patch_02": "(1, 2, 0)",
            "patch_03": "-1",
            "constellation": "orion",
        }
        candidate = {
            **prior,
            "patch_02": "(3, 4, 0)",
            "patch_03": "(9, 9, 1)",
            "constellation": "taurus",
        }
        self.assertEqual(changed_patch_count(prior, candidate), 1)


if __name__ == "__main__":
    unittest.main()
