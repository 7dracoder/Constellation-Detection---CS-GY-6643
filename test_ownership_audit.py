import unittest

from ownership_audit import evidence, rank_direction


class OwnershipEvidenceTests(unittest.TestCase):
    def test_evidence_returns_first_nearby_candidate(self) -> None:
        group = [[100.0, 100.0, 0.9], [200.0, 200.0, 0.8]]
        self.assertEqual(evidence(group, (201, 199, 1), 8.0), (2, 0.8))
        self.assertEqual(evidence(group, (500, 500, 1), 8.0)[0], 0)
        self.assertEqual(evidence(group, None, 8.0)[0], 0)

    def test_zero_rank_is_not_treated_as_best(self) -> None:
        self.assertEqual(rank_direction(0, 1), "unmatched")
        self.assertEqual(rank_direction(4, 1), "better")
        self.assertEqual(rank_direction(1, 4), "worse")


if __name__ == "__main__":
    unittest.main()
