import unittest

import numpy as np

from masked_ncc_reranker import masked_ncc


class MaskedNccTests(unittest.TestCase):
    def test_identical_masked_patch_scores_one(self) -> None:
        query = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        masks = np.ones_like(query)
        patches = np.concatenate((query, -query), axis=0)
        scores = masked_ncc(query, masks, patches)
        self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(scores[0, 1]), -1.0, places=5)

    def test_invalid_pixels_do_not_affect_score(self) -> None:
        query = np.asarray([[[1, 2], [3, 99]]], dtype=np.float32)
        masks = np.asarray([[[1, 1], [1, 0]]], dtype=np.float32)
        patches = np.asarray([[[1, 2], [3, -500]]], dtype=np.float32)
        self.assertAlmostEqual(float(masked_ncc(query, masks, patches)[0, 0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
