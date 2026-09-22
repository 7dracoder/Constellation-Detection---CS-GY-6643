"""Focused regression tests for calibration, fold isolation and promotion."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import structural_refiner as sr
from multimodel_pipeline import CandidateModel, promotion, rerank, ranking_diagnostic


class EnsembleChecks(unittest.TestCase):
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
