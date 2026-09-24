import unittest

import numpy as np

from constellation_pipeline import MatcherConfig, denoise_for_matching
from joint_geometric_solver import Candidate, assign_to_mapped, graph_query_targets


class GraphQueryTargetTests(unittest.TestCase):
    def test_disabled_expansion_preserves_one_width(self) -> None:
        self.assertEqual(graph_query_targets(9.23, 2.0, 1.0, 12, 30, 41), (18,))

    def test_expansion_adds_bounded_second_width(self) -> None:
        self.assertEqual(graph_query_targets(9.23, 2.0, 1.5, 12, 30, 41), (18, 27))
        self.assertEqual(graph_query_targets(15.0, 2.0, 1.5, 12, 30, 87), (30,))

    def test_scene_patch_count_caps_widths(self) -> None:
        self.assertEqual(graph_query_targets(9.0, 2.0, 1.5, 12, 30, 20), (18, 20))


class DenoisingTests(unittest.TestCase):
    def test_none_preserves_pixels(self) -> None:
        image = np.arange(25, dtype=np.float32).reshape(5, 5)
        self.assertIs(denoise_for_matching(image, MatcherConfig()), image)

    def test_gaussian_reduces_impulse_variance(self) -> None:
        image = np.zeros((9, 9), dtype=np.float32)
        image[4, 4] = 255.0
        config = MatcherConfig(denoise_method="gaussian", denoise_sigma=0.65)
        filtered = denoise_for_matching(image, config)
        self.assertLess(float(filtered.var()), float(image.var()))
        self.assertGreater(float(filtered[4, 4]), 0.0)


class FinalAssignmentRankTests(unittest.TestCase):
    def test_rank_prior_can_prefer_nearby_top_candidate(self) -> None:
        mapped = np.asarray([[100.0, 100.0]], dtype=np.float32)
        candidates = [
            Candidate(0, 102.0, 100.0, 0.8),
            Candidate(0, 100.0, 100.0, 0.8),
        ]
        self.assertEqual(assign_to_mapped(mapped, candidates, 30.0)[0].x, 100.0)
        self.assertEqual(assign_to_mapped(mapped, candidates, 30.0, rank_weight=0.4)[0].x, 102.0)

if __name__ == "__main__":
    unittest.main()
