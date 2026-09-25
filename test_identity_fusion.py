import unittest

from identity_fusion import fuse_identities


class IdentityFusionTests(unittest.TestCase):
    def test_changes_only_name(self) -> None:
        row = {
            "Id": "constellation_01",
            "n_patches": "1",
            "patch_01": "(10, 20, 1)",
            "constellation": "hydra",
        }
        fused = fuse_identities([row], [{"constellation_01": "orion"}])[0]
        self.assertEqual(fused["constellation"], "orion")
        self.assertEqual(fused["patch_01"], row["patch_01"])
        self.assertEqual(row["constellation"], "hydra")

    def test_rejects_unknown_scene(self) -> None:
        row = {"Id": "constellation_01", "constellation": "hydra"}
        with self.assertRaises(ValueError):
            fuse_identities([row], [{"constellation_99": "orion"}])


if __name__ == "__main__":
    unittest.main()
