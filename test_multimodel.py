"""Focused regression tests for calibration, fold isolation and promotion."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import structural_refiner as sr
import joint_geometric_solver as jg
import presence_refiner as pr
from shared_geometry_search import choose_both
from annular_refiner import masks, normalized_masked, refine
from multimodel_pipeline import CandidateModel, promotion, rerank, ranking_diagnostic, fit_candidate_model


class EnsembleChecks(unittest.TestCase):
    def test_annular_descriptor_excludes_bright_core(self):
        rng = np.random.default_rng(6643)
        original = rng.uniform(10, 30, (1, 32, 32)).astype(np.float32)
        changed = original.copy()
        changed[:, 13:19, 13:19] = 255.
        np.testing.assert_allclose(normalized_masked(original, masks()[1]),
                                   normalized_masked(changed, masks()[1]), atol=1e-6)

    def test_annular_refiner_preserves_presence_identity_and_members(self):
        source = {"Id": "s", "constellation": "fixture", "n_patches": "3",
                  "patch_01": "(10, 20, 1)", "patch_02": "-1", "patch_03": "(1, 1, 0)"}
        saved = {"columns": ["patch_01", "patch_02", "patch_03"],
                 "candidates": [[[i+100, i+200, 1-i/20., 0.] for i in range(16)]] * 3}
        class FlatModel:
            def predict(self, x):
                return np.zeros(x.shape[:2])
        result = refine(source, (saved, None, np.zeros((3, 16, 1)), None), FlatModel())
        for key in ("Id", "constellation", "n_patches", "patch_01", "patch_02"):
            self.assertEqual(source[key], result[key])
        self.assertEqual(result["patch_03"], "(100, 200, 0)")
        self.assertEqual(source["patch_03"], "(1, 1, 0)")

    def test_shared_geometry_preserves_independent_branches(self):
        points = np.array([[0, 0], [30, 0], [15, 40], [60, 55], [20, 80], [80, 90]], np.float32)
        patterns = [sr.Pattern("fixture", points)]
        candidates = [jg.Candidate(i, int(4*x+100), int(4*y+200), .9)
                      for i, (x, y) in enumerate(points)]
        context = jg.FitContext(len(candidates), 3000.*3000., 6.)
        shared = choose_both(patterns, candidates, 200, 6650, context, 2)
        for mode in ("similarity", "affine"):
            separate, _ = jg.choose_fit_consensus(patterns, candidates, 200, 6650, context, 2, mode)
            self.assertIsNotNone(separate)
            self.assertEqual(shared[mode].pattern.name, separate.pattern.name)
            self.assertEqual(shared[mode].assignments, separate.assignments)
            self.assertAlmostEqual(shared[mode].quality, separate.quality, places=7)
            np.testing.assert_allclose(shared[mode].mapped_points, separate.mapped_points)

    def test_candidate_probabilities_and_fusion_endpoints(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 3))
        model = CandidateModel(x, (x[:, 0] > 0).astype(float))
        self.assertGreater(model.predict(np.array([[2., 0, 0]]))[0], .8)
        self.assertLess(model.predict(np.array([[-2., 0, 0]]))[0], .2)
        saved = {"candidates": [[[i, i, 1-i/20., 0.] for i in range(16)]]}
        features = np.zeros((1, 16, 3))
        features[0, :, 0] = np.arange(16)
        self.assertEqual(rerank(saved, features, model, 0.)[0][0].x, 0)
        self.assertEqual(rerank(saved, features, model, 1.)[0][0].x, 15)

    def test_held_scene_is_never_read_by_membership_training(self):
        fake = {"held": {"patch_01": (0, 0, 1)},
                "other": {"patch_01": (0, 0, 1), "patch_02": (0, 0, 0)}}
        paths = []
        def imread(path, flags):
            paths.append(path)
            return np.full((32, 32), 200 if "01" in path else 20, np.uint8)
        with patch.object(sr, "parse_truth_rows", return_value=fake), patch.object(sr.cv2, "imread", side_effect=imread):
            sr.train_membership_classifier(Path("/fixture"), excluded_scene="held")
        self.assertEqual(len(paths), 2)
        self.assertFalse(any("held" in p for p in paths))

    def test_candidate_training_skips_entire_held_scene(self):
        coords = np.array([[[100*i, 100*i] for i in range(16)]])
        features = np.arange(32, dtype=float).reshape(1, 16, 2)
        data = {"held": None,
                "other": ({"columns": ["patch_01"]}, coords, features, (3000, 3000))}
        truth = {"held": None, "other": {"patch_01": "(0, 0, 1)"}}
        model = fit_candidate_model(data, truth, exclude="held")
        self.assertTrue(np.isfinite(model.predict(features)).all())

    def test_presence_training_skips_entire_held_scene(self):
        rows = [{"Id": "held"}, {"Id": "other", "n_patches": "2",
                                    "patch_01": "(0, 0, 1)", "patch_02": "-1"}]
        with patch.object(pr, "read_csv_rows", return_value=rows), patch.object(
            pr, "scene_features", return_value=(np.array([[0.], [1.]]), {})
        ) as read_features:
            pr.train_classifier(Path("/fixture"), Path("/cache"), excluded_scene="held")
        self.assertEqual(read_features.call_count, 1)
        self.assertEqual(read_features.call_args.args[2], "other")

    def test_promotion_rejects_identity_regression(self):
        base = dict(mean=dict(total_loose=.8, total_strict=.78, identity=1.),
                    per_scene={"s": dict(total_loose=.8)})
        worse = dict(mean=dict(total_loose=.85, total_strict=.83, identity=.66),
                     per_scene={"s": dict(total_loose=.85)})
        self.assertFalse(promotion(base, worse))
        worse["mean"]["identity"] = 1.
        self.assertTrue(promotion(base, worse))

    def test_promotion_rejects_large_individual_scene_loss(self):
        base = dict(mean=dict(total_loose=.8, total_strict=.78, identity=1.),
                    per_scene={"s": dict(total_loose=.8)})
        challenger = dict(mean=dict(total_loose=.85, total_strict=.83, identity=1.),
                          per_scene={"s": dict(total_loose=.75)})
        self.assertFalse(promotion(base, challenger))

    def test_ranking_diagnostic_ignores_absent_patches(self):
        saved = {"columns": ["patch_01", "patch_02"],
                 "candidates": [[[100*i, 100*i, 1-i/20., 0.] for i in range(16)]] * 2}
        coords = np.array([[[100*i, 100*i] for i in range(16)]] * 2)
        features = np.zeros((2, 16, 1))
        class FlatModel:
            def predict(self, x):
                return np.zeros(x.shape[:2])
        result = ranking_diagnostic((saved, coords, features, (3000, 3000)),
                                    {"patch_01": "(0, 0, 1)", "patch_02": "-1"}, FlatModel())
        self.assertEqual(result["present"], 1)
        self.assertEqual(result["raw_top1"], 1)
        self.assertEqual(result["ensemble_top1"], 1)
        self.assertEqual(result["figure"], 1)


if __name__ == "__main__":
    unittest.main()
