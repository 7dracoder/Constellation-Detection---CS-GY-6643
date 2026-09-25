import unittest

import numpy as np

from constellation_pipeline import MatcherConfig, bandpass_for_matching, denoise_for_matching
from joint_geometric_solver import (
    Candidate,
    FitContext,
    GraphFit,
    _support_tail_significance,
    _triangle_index,
    assign_to_mapped,
    duplicate_candidate_scores,
    duplicate_coverage_log_prior,
    final_assignment_queries,
    graph_query_targets,
    merge_dense_candidate_views,
    patch_budget_log_prior,
    propose_triangle_transforms,
    recalibrate_fit_for_clutter,
    unmatched_node_star_evidence,
)
from structural_refiner import Pattern


class GraphQueryTargetTests(unittest.TestCase):
    def test_disabled_expansion_preserves_one_width(self) -> None:
        self.assertEqual(graph_query_targets(9.23, 2.0, 1.0, 12, 30, 41), (18,))

    def test_expansion_adds_bounded_second_width(self) -> None:
        self.assertEqual(graph_query_targets(9.23, 2.0, 1.5, 12, 30, 41), (18, 27))
        self.assertEqual(graph_query_targets(15.0, 2.0, 1.5, 12, 30, 87), (30,))

    def test_scene_patch_count_caps_widths(self) -> None:
        self.assertEqual(graph_query_targets(9.0, 2.0, 1.5, 12, 30, 20), (18, 20))

    def test_all_queries_are_only_added_to_final_assignment(self) -> None:
        selected = [1, 3]
        self.assertIs(final_assignment_queries("selected", selected, 5), selected)
        self.assertEqual(final_assignment_queries("all", selected, 5), [0, 1, 2, 3, 4])


class PatchBudgetPriorTests(unittest.TestCase):
    def test_expected_ratio_has_no_penalty(self) -> None:
        self.assertAlmostEqual(patch_budget_log_prior(297, 100), 0.0, places=12)

    def test_multiplicative_errors_are_symmetric(self) -> None:
        low = patch_budget_log_prior(30, 20, center=3.0, width=0.5)
        high = patch_budget_log_prior(30, 5, center=3.0, width=0.5)
        self.assertAlmostEqual(low, high, places=12)

    def test_extreme_penalty_is_bounded(self) -> None:
        self.assertEqual(patch_budget_log_prior(1_000, 1, clip=20.0), -20.0)


class DuplicateEvidenceTests(unittest.TestCase):
    def test_duplicate_coverage_is_bounded(self) -> None:
        self.assertEqual(duplicate_coverage_log_prior(20, 2, clip=7.0), -7.0)

    def test_matching_annuli_receive_high_score(self) -> None:
        image = np.zeros((80, 80), dtype=np.float32)
        rng = np.random.default_rng(17)
        texture = rng.normal(size=(20, 20)).astype(np.float32)
        image[10:30, 10:30] = texture
        image[50:70, 50:70] = texture
        groups = [[
            Candidate(0, 20.0, 20.0, 0.9),
            Candidate(0, 60.0, 60.0, 0.89),
        ]]
        score = duplicate_candidate_scores(
            image, groups, top_k=2, radius=10, inner_radius=3.0
        )[0]
        self.assertGreater(score, 0.99)


class UnmatchedStarEvidenceTests(unittest.TestCase):
    def test_assigned_nodes_are_excluded_from_star_evidence(self) -> None:
        mapped = np.asarray([[10, 10], [30, 10], [50, 10]], dtype=np.float32)
        assigned = {0: Candidate(0, 10.0, 10.0, 0.9)}
        context = FitContext(
            cloud_size=3,
            image_area=1_000_000.0,
            expected_figure=3.0,
            image_width=1_000.0,
            image_height=1_000.0,
            star_points=np.asarray([[10, 10], [30, 10]], dtype=np.float32),
            star_tolerance=5.0,
        )
        hits, trials, likelihood = unmatched_node_star_evidence(mapped, assigned, context)
        self.assertEqual((hits, trials), (1, 2))
        self.assertGreater(likelihood, 0.0)

    def test_missing_unmatched_stars_are_negative_evidence(self) -> None:
        mapped = np.asarray([[10, 10], [30, 10]], dtype=np.float32)
        context = FitContext(
            cloud_size=2,
            image_area=1_000_000.0,
            expected_figure=2.0,
            image_width=1_000.0,
            image_height=1_000.0,
            star_points=np.asarray([[900, 900]], dtype=np.float32),
            star_tolerance=5.0,
        )
        hits, trials, likelihood = unmatched_node_star_evidence(mapped, {}, context)
        self.assertEqual((hits, trials), (0, 2))
        self.assertLess(likelihood, 0.0)


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

    def test_bandpass_suppresses_constant_background_without_erasing_star(self) -> None:
        image = np.full((33, 33), 20.0, dtype=np.float32)
        image[16, 16] = 100.0
        config = MatcherConfig(bandpass_weight=0.6)
        filtered = bandpass_for_matching(image, config)
        self.assertGreater(float(filtered[16, 16]), 0.0)
        self.assertAlmostEqual(float(filtered.mean()), 0.0, places=4)

    def test_bandpass_rejects_invalid_scales(self) -> None:
        image = np.zeros((9, 9), dtype=np.float32)
        with self.assertRaises(ValueError):
            bandpass_for_matching(image, MatcherConfig(bandpass_weight=1.1))
        with self.assertRaises(ValueError):
            bandpass_for_matching(image, MatcherConfig(
                bandpass_weight=0.6, bandpass_sigma_small=2.0,
                bandpass_sigma_large=1.0,
            ))


class FinalAssignmentRankTests(unittest.TestCase):
    def test_rank_prior_can_prefer_nearby_top_candidate(self) -> None:
        mapped = np.asarray([[100.0, 100.0]], dtype=np.float32)
        candidates = [
            Candidate(0, 102.0, 100.0, 0.8),
            Candidate(0, 100.0, 100.0, 0.8),
        ]
        self.assertEqual(assign_to_mapped(mapped, candidates, 30.0)[0].x, 100.0)
        self.assertEqual(assign_to_mapped(mapped, candidates, 30.0, rank_weight=0.4)[0].x, 102.0)


class DenseCandidateViewTests(unittest.TestCase):
    def test_alternate_view_adds_only_spatially_new_candidates(self) -> None:
        primary = [[Candidate(0, 10.0, 10.0, 0.9)]]
        alternate = [[
            Candidate(0, 12.0, 12.0, 0.95),
            Candidate(0, 40.0, 40.0, 0.8),
        ]]
        merged = merge_dense_candidate_views(primary, [alternate], minimum_separation=6.0)
        self.assertEqual([(item.x, item.y) for item in merged[0]], [(10.0, 10.0), (40.0, 40.0)])
        self.assertAlmostEqual(merged[0][1].score, primary[0][0].score - 0.001)
        self.assertEqual(merged[0][1].margin, 0.0)


class TriangleProposalTests(unittest.TestCase):
    def test_signature_is_invariant_to_similarity_and_reflection(self) -> None:
        source = np.asarray([[0, 0], [20, 0], [4, 13]], dtype=np.float32)
        transformed = source @ np.asarray([[0, -3], [-3, 0]], dtype=np.float32).T + (80, 40)
        _, source_signature = _triangle_index(source, 1.0)
        _, transformed_signature = _triangle_index(transformed, 1.0)
        np.testing.assert_allclose(source_signature, transformed_signature, atol=1e-6)

    def test_triangle_proposals_recover_transformed_points_in_clutter(self) -> None:
        source = np.asarray(
            [[0, 0], [20, 0], [5, 13], [25, 18], [10, 31]], dtype=np.float32
        )
        angle = np.deg2rad(37.0)
        matrix = 4.0 * np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        target = source @ matrix.T + np.asarray((300, 500), dtype=np.float32)
        candidates = [
            Candidate(index, float(point[0]), float(point[1]), 0.9)
            for index, point in enumerate(target)
        ]
        candidates.extend(
            Candidate(index + len(source), float(80 + 61 * index), float(90 + 37 * index), 0.5)
            for index in range(6)
        )
        matrices, offsets = propose_triangle_transforms(source, candidates, 300, 7)
        self.assertGreater(len(matrices), 0)
        mapped = np.einsum("bij,pj->bpi", matrices, source) + offsets[:, None, :]
        error = np.linalg.norm(mapped - target[None, :, :], axis=2).mean(axis=1)
        self.assertLess(float(error.min()), 1e-3)


class ClutterCalibrationTests(unittest.TestCase):
    def test_overdispersed_null_penalizes_clustered_hits(self) -> None:
        independent = np.tile(np.asarray([2, 3, 3, 4], dtype=np.float32), 64)
        clustered = np.tile(np.asarray([0, 0, 6, 6], dtype=np.float32), 64)
        clean_significance, clean_p, clean_rho = _support_tail_significance(
            8, 10, independent
        )
        clutter_significance, clutter_p, clutter_rho = _support_tail_significance(
            8, 10, clustered
        )
        self.assertAlmostEqual(clean_p, clutter_p, places=6)
        self.assertEqual(clean_rho, 0.0)
        self.assertGreater(clutter_rho, 0.0)
        self.assertLess(clutter_significance, clean_significance)

    def test_scene_calibration_is_deterministic(self) -> None:
        mapped = np.asarray(
            [[-20, 0], [20, 0], [0, -20], [0, 20], [-14, -14], [14, 14]],
            dtype=np.float32,
        ) + 100.0
        pattern = Pattern("test", mapped - 100.0)
        candidates = [
            Candidate(index, float(point[0]), float(point[1]), 0.8)
            for index, point in enumerate(mapped)
        ]
        fit = GraphFit(
            pattern=pattern,
            mapped_points=mapped,
            assignments={index: candidate for index, candidate in enumerate(candidates)},
            support=6,
            mean_error=0.0,
            tolerance=12.0,
            quality=1.0,
            coverage=1.0,
            significance=1.0,
        )
        context = FitContext(6, 40_000.0, 6.0, 200.0, 200.0)
        first = recalibrate_fit_for_clutter(fit, candidates, context, 128, 19)
        second = recalibrate_fit_for_clutter(fit, candidates, context, 128, 19)
        self.assertEqual(first.quality, second.quality)
        self.assertGreater(first.null_hit_probability, 0.0)
        self.assertGreater(first.significance, 0.0)

if __name__ == "__main__":
    unittest.main()
