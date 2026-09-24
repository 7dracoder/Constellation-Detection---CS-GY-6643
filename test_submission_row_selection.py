import unittest

from submission_row_selection import select_rows


class RowSelectionTests(unittest.TestCase):
    def test_uses_complete_alternative_row_only_for_selected_scene(self) -> None:
        old = [
            {"Id": "a", "n_patches": "1", "patch_01": "-1", "constellation": "pavo"},
            {"Id": "b", "n_patches": "1", "patch_01": "-1", "constellation": "orion"},
        ]
        new = [
            {"Id": "a", "n_patches": "1", "patch_01": "(10, 20, 1)", "constellation": "orion"},
            {**old[1], "constellation": "lupus"},
        ]
        result = select_rows(old, new, {"a"})
        self.assertEqual(result, [new[0], old[1]])

    def test_rejects_unknown_scene(self) -> None:
        rows = [{"Id": "a", "n_patches": "1", "patch_01": "-1", "constellation": "pavo"}]
        with self.assertRaises(ValueError):
            select_rows(rows, rows, {"missing"})


if __name__ == "__main__":
    unittest.main()
