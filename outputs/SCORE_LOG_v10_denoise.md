# Denoise pass (v10)

Star-preserving denoise added to the live image path:

- impulse/hot-pixel median replace outside star cores
- mild bilateral grain cleanup
- star cores written back so localisation stays sharp

Wired into `normalize_image`, `SceneMatcher`, GPU matcher queries, and star-bank enrichment.

| File | vs v4 | Notes |
| --- | --- | --- |
| `submission_v10_denoise_gated.csv` | 0 label / 60 cell | Denoised starbank + affine + presence, gated to v4 labels |

Train LOSO after denoise: **0.8625 / 0.8458** (same as prior similarity+presence base — no regression).

Submit: `kaggle_submission_v10_denoise_gated.csv`

Realistic note: denoise alone cannot jump public LB from ~0.72 to 0.94; localisation ceiling and identity on the public 38% still dominate.
