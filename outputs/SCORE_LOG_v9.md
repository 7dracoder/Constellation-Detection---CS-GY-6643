# Public-first score climb — artifact log

Date: 2026-09-23

Base (known public LB): `submission_v4_presence.csv` = **0.71544**

| Phase | File | vs v4 | Notes |
| --- | --- | --- | --- |
| 1 | `submission_v5_affine_gated.csv` | 0 label / 60 cell | Submit first public probe. Promote as base only if LB > 0.71544. |
| 2 | `submission_v9_starbank_gated.csv` | 0 label / 61 cell | Star-bank enrich + affine + presence; gated to v4 labels (12 accept / 4 keep). |
| 3 | `submission_v9_presence65.csv` | 0 label / 42 cell | present-rate 0.65 + presence refine; gated to v4 labels. Train LOSO: loose **0.8734** > base 0.8625, strict **0.8484** > 0.8458 → submit OK. |
| 4 private | `submission_v8_catalog_forceunique_tight.csv` | `_15` eridanus→scorpius only | Weak auriga blocked. Leaves canis-major dup. Private-endgame only. |

Desktop / artifacts copies:
- `kaggle_submission_v5_affine_gated.csv`
- `kaggle_submission_v9_starbank_gated.csv`
- `kaggle_submission_v9_presence65.csv`
- `kaggle_submission_v8_private_endgame.csv` (same as tight force-unique)

Suggested upload order for public climb: **v5 gated → v9 starbank → v9 presence65**.
Private final: **v8 private endgame**.
