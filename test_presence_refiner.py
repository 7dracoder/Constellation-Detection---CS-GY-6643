import unittest

import numpy as np

from presence_refiner import _agreement_features, _score_features


class PresenceFusionFeatureTests(unittest.TestCase):
    def test_score_feature_shape_and_finiteness(self) -> None:
        scores = np.linspace(1.0, 0.0, 32, dtype=np.float32).reshape(2, 16)
        features = _score_features(scores)
        self.assertEqual(features.shape, (2, 12))
        self.assertTrue(np.isfinite(features).all())

    def test_agreement_features_recognise_matching_top_location(self) -> None:
        group_a = [[10.0 + index, 20.0, 1.0 - index / 100.0] for index in range(16)]
        group_b = [[10.0, 20.0 + index, 0.9 - index / 100.0] for index in range(16)]
        features = _agreement_features(
            [{"candidates": [group_a]}, {"candidates": [group_b]}], 0
        )
        self.assertEqual(features.shape, (6,))
        self.assertEqual(float(features[0]), 0.0)
        self.assertEqual(float(features[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
