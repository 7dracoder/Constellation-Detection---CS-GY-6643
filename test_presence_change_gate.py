import unittest

from presence_change_gate import select_rows


class PresenceChangeGateTests(unittest.TestCase):
    def test_accepts_small_addition_and_rejects_large_rate(self) -> None:
        old = [
            {"Id": "a", "n_patches": "2", "constellation": "orion", "patch_01": "-1", "patch_02": "-1"},
            {"Id": "b", "n_patches": "2", "constellation": "orion", "patch_01": "-1", "patch_02": "-1"},
        ]
        new = [
            {**old[0], "patch_01": "(10, 20, 0)"},
            {**old[1], "patch_01": "(10, 20, 0)", "patch_02": "(30, 40, 0)"},
        ]
        selected, audit = select_rows(old, new, 0.5)
        self.assertEqual(selected[0], new[0])
        self.assertEqual(selected[1], old[1])
        self.assertEqual([entry[3] for entry in audit], [True, False])

    def test_rejects_membership_or_location_rewrite(self) -> None:
        old = [{"Id": "a", "n_patches": "1", "constellation": "orion", "patch_01": "(10, 20, 0)"}]
        new = [{**old[0], "patch_01": "(11, 20, 0)"}]
        with self.assertRaises(ValueError):
            select_rows(old, new, 1.0)


if __name__ == "__main__":
    unittest.main()
