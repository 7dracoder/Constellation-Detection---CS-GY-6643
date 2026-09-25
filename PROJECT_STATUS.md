# Project status — Constellation Detection, CS-GY 6643

Last updated: September 24, 2026

## Scope and compliance

The scored submissions through v4 use the course-provided materials:

- supplied training and validation sky images;
- supplied 32×32 query patches;
- supplied constellation-pattern diagrams; and
- supplied training labels.

The multi-model experiment now also uses public ESO observations after the
user reported instructor approval for external data. Sources, credits and
preprocessing are recorded in `EXTERNAL_TRAINING_DATA.md`. NYU's guidance requires
coursework to use Cloud Bursting/Open OnDemand instead of the researcher-facing
Torch cluster.

## Score record

| Submission stage | Kaggle score | Notes |
| --- | ---: | --- |
| Initial validated baseline | approximately 0.26710 | Correct schema, but weak localisation and all-`unknown` identity. |
| Earlier geometric submission | 0.58634 | Added candidate shortlists and geometric identity fitting. |
| A100 wide-search submission | 0.66890 | Superseded: `outputs/submission_a100_wide_fixed.csv`. |
| RTX 5070 Ti rebuild | 0.68196 | Superseded: `outputs/submission_v2.csv`. |
| Assignment-robustness pass | 0.68431 | Superseded: `outputs/submission_v3.csv`. |
| Presence refinement | 0.71544 | Former best: `outputs/submission_v4_presence.csv`. |
| Gaussian sigma=0.65 hybrid + multi-width gate | 0.73803 | Former best: `outputs/submission_hybrid_gaussian_multiwidth_gated.csv`; user-reported Kaggle result September 23. |
| Gaussian sigma=0.85 hybrid + same identity gate | **0.74973** | Current scored best: `outputs/submission_hybrid_gaussian085_multiwidth_gated.csv`; user-reported Kaggle result September 23. |
| Affine gating | Not submitted | `outputs/submission_v5_affine_gated.csv`; absent from Kaggle history checked September 22. |
| Teammate graphguard | 0.70102 | `submission_v6_graphguard.csv`; below the v4 best, so not promoted. |

The submission history above was checked directly on the signed-in Kaggle
Submissions page on September 22, 2026. The graphguard artifact is not present
in this checkout; its filename and score are recorded from Kaggle, not an
assumed reconstruction of its code.

Kaggle provides a single hidden-label score, not component scores. A valid CSV
only proves that Kaggle can read it; it does not predict a leaderboard score.
No implementation or experiment in this repository establishes a 0.90 or 0.96
score guarantee.

## Final-day submission triage (September 24, 2026)

The user supplied the signed-in Fall 2026 Evaluation text: scene scores are
averaged equally, with 0.25 presence macro-F1, 0.20 patch-specific
localisation, 0.25 set-based figure-star recovery, and 0.30 identity. Point
reward is full through 12 px and falls to zero at 36 px. Figure stars match
all reported points one-to-one, nearest pairs first. The public leaderboard
uses roughly 40% of hidden scenes, so a one-scene change can be invisible
publicly while still affecting the private score. `evaluate.py` now uses the
specified nearest-pairs-first geometric matcher. The Fall 2025 competition
page had a different metric; do not use it to decide Fall 2026 submissions.

Three daily submission slots remained when this triage began. Do **not** spend
one first on the fixed-node ownership-only file: it changes only the 20%
patch-localisation component for one scene. It is not score-identical under
the 2026 metric, but its upside is narrower than the alternatives below.

Candidate A, `outputs/submission_scene01_15_hydra.csv`, replaces complete
algorithm-produced rows in scenes 01 and 15 only, leaving all other scenes
identical to the scored 0.74973 file. Raw course-data geometry supports Hydra
with 8 and 12 stars respectively. A read-only public-catalog consistency
check agrees with Hydra at its normal 12,000-proposal budget: scene 01 has
support 7, quality 16.80, runner quality 10.76; scene 15 has support 11,
quality 26.80, runner quality 18.08. The fallback Sagittarius and Eridanus
rows did not receive catalog agreement at the same budget. The catalog is
corroboration, not hidden truth. This is the main higher-upside *unscored*
candidate, SHA-256
`82CF76BAB34D898CA943D3E2F18CF9A13FDF7E9E2DB9FEF42CE4C9F0B5F14D3F`.
An additional 4,000-proposal robustness audit kept Hydra first for both
scenes under two different affine random seeds and a similarity transform.
The independent catalog fit gave support 7 for scene 01 in all checks, and
support 10-11 for scene 15. This tests fit stability, not hidden accuracy;
older course-data solver variants disagreed on these identities, especially
scene 15.

Candidate B, `outputs/submission_fusion085_065_t54_rate15_gated.csv`, keeps
all scored labels and graph-member cells. It accepts two-filter presence
additions only when at most 15% of a scene's patches are newly marked
present. The general rate rule rejects a 30-addition outlier scene and leaves
14 added nonmember detections across six scenes. A strict scene-held-out
three-scene check scored 0.9059 versus 0.9038 for the established candidate,
from one extra correctly detected training patch. That small check does not
guarantee a hidden gain. The file is valid and unscored, SHA-256
`A58900CFFEE6F06A9F48D68D732E43E949B6BBAF59F7C5E53CC64DA96218307B`.

Candidate C, `outputs/submission_scene01_15_hydra_fusion_rate15.csv`, combines
A's complete rows with B's bounded additions elsewhere. It is valid and
unscored, SHA-256
`9FE20F420CF444132E3D1B12DE18DB3F647417B5ECABB75DD7F09C288C361FB3`.
The scored 0.74973 file remains intact. Reserve the third daily slot until
the first two results are known; public feedback may be uninformative for
an individual scene, and final selection should weigh method evidence as
well as public score.

The two-scene Hydra candidate was submitted and received the **same 0.74973
public score**. Because the public split contains only about 40% of hidden
scenes, this does not establish whether either changed scene was scored. The
scored baseline is still the safe fallback; do not treat this flat score as
positive evidence for Hydra.

### Weighted template matching and noise audit (September 24)

An optional, default-off dual-view matcher now tests NCC on both the image and
a Difference-of-Gaussians bandpass view. The exact suggested 0.4 raw + 0.6
bandpass weighting retrieved the correct location among 24 candidates for
68/71 labelled-present patches (raw: 63/71), but ranked it first for only
53/71 (Gaussian-0.85: 58/71). After full five-seed affine geometry and the
same held-out presence refinement, it scored **0.8811** versus the established
**0.9038**. The constellation names were 3/3 correct but localisation fell
from 0.687 to 0.556.

A conservative Gaussian-0.85 + 0.2 bandpass variant retrieved 69/71 but
ranked only 56/71 first. Full geometry misidentified Taurus as Orion and
scored **0.7296**. A separate multi-hypothesis cache kept the raw top match
while interleaving distinct bandpass alternatives for RANSAC. It retrieved
70/71, maintained 3/3 correct identities and 1.000 loose geometry at the
equal 12,000-proposal budget, but scored only **0.8792**; localisation was
0.562 and patch-member F1 0.227. The 3,000-proposal screen scored 0.8866.

Diagnostic overlays in `outputs/diagnostics/` were visually inspected.
Bright structured backgrounds and mosaic seams can attract matches; long
lines from reported points to labelled truth show that correspondence, not
candidate retrieval alone, is the principal failure in this ablation. No
validation CSV was built from these failed methods. The default matcher and
scored 0.74973 artifact were not replaced.

## Gaussian-hybrid progression

Previous file: `outputs/submission_hybrid_gaussian_multiwidth_gated.csv` (Kaggle
**0.73803**, reported September 23, 2026).

This pass introduced two independently gated improvements:

- multi-width affine geometry evaluates both a narrow membership-ranked query
  cloud and a 1.5x broader cloud, selecting with the existing density-adjusted
  quality rather than scene identity or labels;
- matched Gaussian denoising (`sigma=0.65`) is applied to both scene and query
  images for a second candidate cache. Raw candidates still drive geometry;
  denoised candidates only refine non-member presence and localisation.

On the three labelled scenes, Gaussian matching raised top-1 candidate recall
from 49/71 to 58/71 and retained-list recall from 63/71 to 67/71. The hybrid
leave-one-scene-out diagnostic reached 0.8993 strict/loose, 3/3 identities,
1.000 strict geometry, 0.858 presence, and 0.674 localisation. Using denoised
candidates for geometry scored only 0.8617 strict, so that variant was rejected.
The final identity gate keeps the complete established v4 row wherever the
new and established predicted constellation names disagree.

Artifact audit: 16 rows, 90 columns, SHA-256
`FE259FAA89BF05DD3AB05D2D9CAAFC06BAEFAEE70F2C7B2A2D1457EFED8822A0`.

### Noise-removal ablation

Matched Gaussian smoothing at `sigma=0.65` is the retained denoiser. A second
September 23 experiment subtracted a slowly varying Gaussian background after
smoothing. It was rejected on the labelled candidate-retrieval audit: exact
top-1 recall fell from 58/71 to 28/71, and retained-list recall fell from 67/71
to 48/71. The low-frequency patch context is therefore informative to the
matcher, not disposable noise. No validation submission was built from this
failed variant.

A controlled Gaussian-strength sweep then measured:

| Candidate cache | Top-1 recall | Retained recall | End-to-end strict/loose |
| --- | ---: | ---: | ---: |
| Raw | 49/71 | 63/71 | not the denoised hybrid |
| Gaussian `sigma=0.45` | 54/71 | 65/71 | not promoted |
| Gaussian `sigma=0.65` | 58/71 | 67/71 | 0.8993 |
| Gaussian `sigma=0.85` | 58/71 | 67/71 | **0.9014** |

At `sigma=0.85`, presence rose from 0.858 to 0.866 and localisation from
0.674 to 0.675 relative to `sigma=0.65`; identity stayed 3/3 and strict
geometry stayed 1.000. The validation candidate
`outputs/submission_hybrid_gaussian085_multiwidth_gated.csv` preserves the
same nine accepted identities and seven complete v4 fallback rows as the
0.73803 file. Its 24 changed cells are all non-member cells: 6 additions,
9 removals, and 9 `m=0` relocations; no `m=1` geometry changed. The candidate
is schema-valid, has 16 rows and 90 columns, and has SHA-256
`8A36B9E3E8294815EE0DFCB54C9C24863BE3F6C039B5A502F4B067180E70044D`.
Kaggle scored this `sigma=0.85` artifact at **0.74973**, a gain of 0.01170
over `sigma=0.65` and 0.03429 over v4. It is now the current scored best.

### Multi-width distinct-identity experiment

The validation geometry frequently predicts the same label for several scenes
(especially Hydra), even though the competition split is consistent with one
scene per supplied constellation. `distinct_label_solver.py` now evaluates the
same narrow and 1.5x-expanded membership clouds as the successful multi-width
geometry pass, retains each pattern's own winning cloud, and applies a guarded
one-to-one assignment only when an alternative fit is within 1.25 quality
units of that scene's independent best.

On the three labelled scenes, the solver retained Pisces, Scorpius, and Taurus
with best-to-runner quality margins of 3.84, 7.91, and 7.11 respectively. After
the established Gaussian presence/localisation refinement, the full diagnostic
remained 0.8993 strict/loose with 3/3 identities and 1.000 strict geometry.
This establishes non-regression on known data; it does not prove the distinct
validation-label assumption, so any validation output remains a candidate for
comparison rather than an automatic promotion.

The validation audit also showed why strict uniqueness must not be assumed:
both scenes 04 and 11 had strong Canis Major fits, ahead of their runners by
5.14 and 6.74 quality units. Guarded assignment changed only three low-margin
collisions (03 to Sagittarius, 05 to Serpens Caput, and 16 to Scorpius), but
the resulting file differs from the scored identity gate on seven scenes and
is not promoted. Broad query clouds also over-selected Hydra on several
ambiguous scenes, so future identity work should add width/seed stability and
pattern-specific null-fit calibration before changing scored identities.

### Rejected two-Gaussian confidence fusion

`presence_refiner.py` now supports multiple candidate caches and exposes
cross-filter location-agreement features while preserving exact single-cache
behavior. The single-cache `sigma=0.85` compatibility run reproduced the
existing CSV byte-for-byte. A scene-held-out fusion of `sigma=0.85` and
`sigma=0.65` initially scored 0.8988 at the established 0.58 threshold. A
bounded threshold sweep found 0.54 reached 0.9034, changing only one labelled
patch and preserving 3/3 identities plus 1.000 strict geometry.

That apparent gain did not survive the validation-distribution audit. Relative
to the scored 0.74973 file, the 0.54 fusion candidate added 44 non-member
detections, concentrated heavily in scene 04. A one-patch change across the
three labelled scenes does not support that scale of validation change. The
candidate `outputs/submission_hybrid_fusion085_065_t54_multiwidth_gated.csv`
is therefore rejected and must not be submitted. This is evidence that
absolute multi-view probability calibration is unstable with only three
labelled scenes; future fusion should use a bounded change budget or
rank/stability gate learned without relying on a global probability threshold.

### Corrected scene-held-out evaluation and error ledger

The earlier 0.9014 train diagnostic held out each scene from presence-model
training, but `joint_geometric_solver.py` had trained its patch membership
selector on all three labelled scenes. The new
`--cross-validated-membership` option trains that selector on the other two
scenes for each train prediction. It cannot be used on validation. Running the
existing raw adaptive candidate cache through two-width affine geometry,
then the established `sigma=0.85` cache through scene-held-out presence
refinement, gives **0.9038** strict/loose, 3/3 identities, and 1.000 set-based
strict geometry. The member selector used 44, 45, and 53 labelled-present
patches in the Pisces, Scorpius, and Taurus folds respectively. All cache
generation is label-free; matcher settings and search hyperparameters were
chosen in earlier experiments using these same three scenes, so this is still
a regression check, not an unbiased leaderboard estimate.

`evaluate.py` now also prints patch-specific `memberF1` without changing the
historical total. It is **0.471** mean (Pisces 0.583, Scorpius 0.455, Taurus
0.375), versus 1.000 for set-based geometry. The latter matches the set of
predicted figure points to the set of true points and can hide a swap between
query patches. Only 15 of 36 predicted `m=1` patches have the correct query,
membership bit, and position within 12 pixels; eight `m=1` predictions are
on patches that are actually absent.

The patch-level diagnostic is in
`outputs/train_strict_heldout_error_ledger.csv` (116 rows, one per labelled
patch). Among 71 truly present patches: 48 are correct on both location and
membership, 15 have a correct candidate in at least one cache but a wrong
final location, 4 lack a correct candidate in both caches, 2 are rejected by
presence despite having a correct candidate, and 2 have the right location
but the wrong membership bit. Nine of 45 absent patches are false positives.
Twelve of the 15 location-selection errors are `m=1` graph assignments; in
eight of those, the right location is top-ranked by at least one matcher.
Three of four retrieval misses are in the lower half of the query-patch
signal-to-noise proxy, but the higher-signal half has more overall errors
(14/36 versus 9/35). Dim-star retrieval deserves a targeted check, while
patch-to-star assignment and false-positive control are larger measured
opportunities. Brightness alone cannot separate absent from present patches:
some absent patches have high center contrast.

### Final graph-assignment rank prior

`joint_geometric_solver.py` now offers `--final-rank-weight` (default `0.0`).
It adds a small log-rank penalty to candidate costs only for the final dense
patch-to-star assignment; RANSAC search, identity selection, and candidate
retrieval are unchanged. The ablation at weight `0.20` used the same caches,
search settings, and fully scene-held-out downstream classifiers as the strict
diagnostic above. The three-scene total rose from **0.9038** to **0.9093**;
patch-specific member F1 rose from **0.471** to **0.529**. Identity remained
3/3 and set-based geometry remained 1.000. Across 116 labelled patches, five
categories improved and two regressed; location-selection errors fell from 15
to 14, false positives from 9 to 8, and correct locations rose from 48 to 50.
Taurus had no category-level gain. This is a small, mixed diagnostic gain,
not a validated hidden-label improvement. The existing scored artifact and
the zero-weight default are unchanged.

The ablation outputs are `outputs/train_heldout_finalrank020_geometry.csv`,
`outputs/train_heldout_finalrank020_presence.csv`, and
`outputs/train_heldout_finalrank020_error_ledger.csv`. The validation audit
does not justify automatic promotion: after the same Gaussian-0.85 refinement
and established identity gate, the candidate
`outputs/submission_finalrank020_gated.csv` retains the same nine accepted
constellation identities and seven complete v4 fallback rows as the scored
file, but changes **42 patch cells across eight scenes**. Every change touches
`m=1` ownership: 18 member-to-member relocations, 8 `m=0` to `m=1`, 9 `m=1`
to `m=0`, 3 absent to `m=1`, and 4 `m=1` to absent. Of the 18 relocations,
16 move to a better raw candidate rank and two to a worse rank; this is
expected from the prior but is not proof that the new locations are true.
Scenes 10 and 11 alone account for 18 changed cells, while scene 02 loses
one assigned member. The scale of validation reassignment is large relative
to the five improved and two regressed labelled patches, so this remains an
**experimental, unscored candidate**; the 0.74973 file remains the recommended
submission. The new CSV is schema-valid (16 rows, 90 columns), SHA-256
`9E7CC598AE85D29C94CD102332C59027BEE3A890B37D3591A896759206A8E19C`.
Do not infer a Kaggle score from the training proxy.

### Fixed-node, two-view ownership refinement

`ownership_audit.py` checks every changed graph-member patch against both the
raw adaptive and Gaussian-0.85 candidate caches. The earlier rank-prior
candidate moved 42 member-related cells; among its 18 member-to-member
relocations, raw ranks improved in 16 and Gaussian ranks improved in 11 (six
Gaussian matches were absent from the retained list). Scene 11 had strong
agreement between views, while scene 10 had contradictory evidence and lost
an assigned member. These ranks are diagnostics, not ground truth.

`ownership_refiner.py` is a narrower experimental alternative. It holds the
predicted constellation, existing `m=1` query set, and graph-star coordinate
set fixed; it only uses one-to-one assignment to permute which already-member
patch owns which existing node. Its cost uses log ranks from both candidate
views, with a penalty for missing Gaussian support. The optional `--min-gain`
is the required total cost improvement before any scene-level permutation is
accepted. No validation truth, scene name, or constellation label enters the
cost.

On the same scene-held-out three-scene diagnostic, unrestricted two-view
ownership scored **0.9091** versus the fixed-node input's **0.9038**. An
exploratory `--min-gain 3.0` gate retained only the Scorpius permutation and
scored **0.9115**, with localisation 0.726 and patch-specific member F1 0.562
(input: 0.471). Pisces and Taurus were unchanged; identity remained 3/3 and
strict geometry 1.000. The gate was chosen during this small diagnostic, so
this is not an independent validation estimate.

Applying the same gate to the raw validation geometry **before** the
established identity fallback changes only 10 `m=1` cells, all in scene 11.
The resulting `outputs/submission_fixednode_ownership_gain3_gated.csv` keeps
all 16 labels, every membership count, and every graph-star coordinate set
from the scored 0.74973 file. Seven of the 10 moved patches improve raw rank
and nine improve Gaussian rank; the remaining ranks are a reason for caution,
not evidence of hidden correctness. This is a schema-valid, **unscored test
candidate**, SHA-256
`29BDE258EAD41B0806F59B017018F6B827D4201DE3B8559577E3DF0B9706B0AA`.
The scored 0.74973 file remains the default recommendation until Kaggle tests
the candidate. The diagnostic CSV is
`outputs/validation_fixednode_ownership_gain3_audit.csv`.

Reproduce the fixed-node candidate from the existing caches in WSL Ubuntu:

```bash
PY=/opt/constellation-venv/bin/python
$PY ownership_refiner.py --root . \
  --input outputs/submission_hybrid_gaussian085_multiwidth.csv \
  --output outputs/submission_fixednode_ownership_gain3_raw.csv \
  --split validation --raw-cache outputs/cache_validation_adaptive \
  --gaussian-cache outputs/cache_validation_gaussian085 --min-gain 3.0
$PY submission_ensemble.py --root . \
  --established outputs/submission_v4_presence.csv \
  --candidate outputs/submission_fixednode_ownership_gain3_raw.csv \
  --output outputs/submission_fixednode_ownership_gain3_gated.csv
```

To reproduce the earlier `--final-rank-weight 0.20` candidate from the saved
validation cache in WSL Ubuntu:

```bash
PY=/opt/constellation-venv/bin/python
$PY joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide.json \
  --output outputs/submission_finalrank020_geometry.csv \
  --split validation --device cpu --top-k 24 --graph-top-k 3 \
  --proposals 12000 --cache-dir outputs/cache_validation_adaptive \
  --presence-mode quantile --present-rate 0.625 \
  --graph-query-factor 2.0 --graph-query-expansion-factor 1.5 \
  --max-graph-queries 30 --consensus-trials 5 --transform-model affine \
  --final-rank-weight 0.20
$PY presence_refiner.py --root . \
  --input outputs/submission_finalrank020_geometry.csv \
  --output outputs/submission_finalrank020_gaussian085_presence.csv \
  --split validation --cache-dir outputs/cache_validation_gaussian085 \
  --train-cache-dir outputs/cache_train_gaussian085
$PY submission_ensemble.py --root . \
  --established outputs/submission_v4_presence.csv \
  --candidate outputs/submission_finalrank020_gaussian085_presence.csv \
  --output outputs/submission_finalrank020_gated.csv
```

Reproduce the held-out diagnostic from existing caches in WSL Ubuntu:

```bash
PY=/opt/constellation-venv/bin/python
$PY joint_geometric_solver.py --root . --config matcher_config_gpu_wide.json \
  --output outputs/train_strict_heldout_geometry.csv --split train --device cpu \
  --top-k 24 --graph-top-k 3 --proposals 12000 \
  --cache-dir outputs/cache_train_adaptive --presence-mode quantile \
  --present-rate 0.625 --graph-query-factor 2.0 \
  --graph-query-expansion-factor 1.5 --max-graph-queries 30 \
  --consensus-trials 5 --transform-model affine \
  --cross-validated-membership
$PY presence_refiner.py --root . \
  --input outputs/train_strict_heldout_geometry.csv \
  --output outputs/train_strict_heldout_gaussian085_presence.csv \
  --split train --cache-dir outputs/cache_train_gaussian085 \
  --train-cache-dir outputs/cache_train_gaussian085 --cross-validated
$PY evaluate.py --root . \
  --predictions outputs/train_strict_heldout_gaussian085_presence.csv
$PY error_ledger.py --root . \
  --predictions outputs/train_strict_heldout_gaussian085_presence.csv \
  --raw-cache outputs/cache_train_adaptive \
  --filtered-cache outputs/cache_train_gaussian085 \
  --output outputs/train_strict_heldout_error_ledger.csv
```

### Triangle-seeded RANSAC and empirical clutter calibration

The geometric solver now has a default-off `--proposal-mode hybrid` option.
Half of the fixed RANSAC budget remains ordinary candidate/node pairs and half
uses scale-free triangle side-ratio signatures. Triangle vertices are ordered
canonically, so the seeds are invariant to translation, rotation, uniform
scale, and reflection. On the strict scene-held-out diagnostic, 3,000 hybrid
proposals reproduced the established **0.9038** result that previously used
12,000 pair proposals. This is an efficiency and search-robustness gain, not
by itself a score gain.

`--clutter-trials 256` replaces the uniform candidate-density significance
only for final fit ranking. It rotates and translates the fitted diagram over
the actual scene candidate cloud, measures the random support distribution,
and fits an over-dispersed beta-binomial null. This charges patterns for
clustered false-star structure without using labels or scene identities. The
option defaults to zero, preserving all established outputs.

With hybrid 3,000-proposal search, 256 clutter trials, fully scene-held-out
membership, and the established Gaussian-0.85 presence pass, two independent
runs produced the same diagnostic:

| Metric | Established strict baseline | Triangle + clutter |
| --- | ---: | ---: |
| Presence | 0.866 | 0.866 |
| Patch localisation | 0.687 | **0.706** |
| Set geometry | 1.000 | 1.000 |
| Identification | 3/3 | 3/3 |
| Patch-specific member F1 | 0.471 | **0.513** |
| Total | 0.9038 | **0.9075** |

The gain is an ownership correction, not a new star location: on Taurus one
patch relinquishes an incorrectly owned constellation node and another patch
takes that same node. Pisces is byte-identical; Scorpius changes a three-patch
ownership cycle without changing its scene score. All 42 unit tests pass.

The unrestricted validation candidate was deliberately not promoted. Even
after identity agreement with the scored best it changed 49 patch cells in
seven scenes. Of 17 member relocations, raw candidate rank improved 10 and
worsened 7, while Gaussian rank improved only 4, worsened 8, and did not
retain 5 locations. That is too much distribution shift for three labelled
scenes.

`submission_ensemble.py` therefore supports an optional, default-off
`--max-patch-changes` scene budget. A budget of three is the maximum change
observed in any labelled scene (0, 3, and 2), so it preserves the full 0.9075
diagnostic. Applied against the current scored 0.74973 CSV, it changes only
scene 10 and exactly two cells:

- patch 42: `(1558, 2302, 1)` to its raw/Gaussian top-1 location
  `(2308, 1654, 0)`;
- patch 45: `(1603, 1384, 0)` to the released constellation node
  `(1559, 2302, 1)`.

This is the same ownership-swap structure as the measured Taurus gain. The
constellation point set and every scene label remain unchanged. The schema-
valid, unscored primary candidate is
`outputs/submission_triangle_clutter256_change3_gated.csv`, SHA-256
`61D2FF2DC6BFD410B2BC0D7481C10AF56D0BF5DA59338BE1363AF32A346D23AE`.
No Kaggle score is inferred from the local result.

Reproduce it from the existing caches in WSL Ubuntu:

```bash
PY=/opt/constellation-venv/bin/python
$PY joint_geometric_solver.py --root . --config matcher_config_gpu_wide.json \
  --output outputs/submission_triangle_clutter256_geometry.csv \
  --split validation --device cpu --top-k 24 --graph-top-k 3 \
  --proposals 3000 --cache-dir outputs/cache_validation_adaptive \
  --presence-mode quantile --present-rate 0.625 \
  --graph-query-factor 2.0 --graph-query-expansion-factor 1.5 \
  --max-graph-queries 30 --consensus-trials 5 --transform-model affine \
  --proposal-mode hybrid --clutter-trials 256
$PY presence_refiner.py --root . \
  --input outputs/submission_triangle_clutter256_geometry.csv \
  --output outputs/submission_triangle_clutter256_gaussian085_presence.csv \
  --split validation --cache-dir outputs/cache_validation_gaussian085 \
  --train-cache-dir outputs/cache_train_gaussian085
$PY submission_ensemble.py --root . \
  --established outputs/submission_hybrid_gaussian085_multiwidth_gated.csv \
  --candidate outputs/submission_triangle_clutter256_gaussian085_presence.csv \
  --output outputs/submission_triangle_clutter256_change3_gated.csv \
  --max-patch-changes 3
$PY constellation_pipeline.py validate --root . \
  --output outputs/submission_triangle_clutter256_change3_gated.csv
```

## Previous v3 submission

File: `outputs/submission_v3.csv` (Kaggle **0.68431**, September 21, 2026).

The gain over the previous best is small: **+0.00235** (0.68196 to 0.68431).
That is consistent with what the assignment-robustness pass itself predicted
it could show: 11 of 16 validation labels were unchanged from
`outputs/submission_v2.csv`, and of the 5 that changed, most were already
flagged as seed-dependent under the *old* code (see that section below), so
a small net movement - rather than a large swing in either direction - was
the expected outcome, not a surprise. `outputs/submission_v2.csv` (0.68196)
is superseded but retained for comparison, alongside the A100 file below.

## Presence-refined submission

File: `outputs/submission_v4_presence.csv` (Kaggle **0.71544**).

`presence_refiner.py` trains a small regularised logistic classifier from the
supplied training labels, the raw matcher's top-16 score distribution, and
simple query-patch brightness/contrast features. It never changes a
graph-supported `m=1` cell or a constellation label; it only revises the
presence and raw top-1 location of non-members.

Under leave-one-scene-out prediction, this raised the labelled diagnostic from
0.8369 to **0.8709** with loose geometry, and from 0.8119 to **0.8459** with
strict geometry. Presence rose from 0.764 to 0.861 and localisation from 0.562
to 0.612. Kaggle confirmed the direction of the diagnostic: v4 improved the
public score by 0.03113 over v3 (0.68431 -> 0.71544).

## Superseded RTX 5070 Ti rebuild

File: `outputs/submission_v2.csv` (Kaggle 0.68196).  Superseded by
`outputs/submission_v3.csv` above; see "September 21, 2026 rebuild (local
RTX 5070 Ti)" for its method.

## Superseded A100 submission

File: `outputs/submission_a100_wide_fixed.csv`

The file has been checked with `constellation_pipeline.py validate` and has:

- 16 data rows plus the header;
- 90 columns (`Id`, `n_patches`, 87 patch columns, `constellation`);
- no null or blank-like cells;
- `-1` padding for non-applicable patch columns; and
- cells formatted as `-1` or `(x, y, m)`.

The A100 output selected the following labels. These are predictions, not
ground truth.

| Scene | Predicted present stars | `m=1` stars | Predicted label |
| --- | ---: | ---: | --- |
| constellation_01 | 18 | 9 | eridanus |
| constellation_02 | 21 | 11 | eridanus |
| constellation_03 | 30 | 6 | carina |
| constellation_04 | 21 | 9 | canis-major |
| constellation_05 | 47 | 7 | pisces |
| constellation_06 | 37 | 4 | orion |
| constellation_07 | 13 | 7 | perseus |
| constellation_08 | 20 | 7 | orion |
| constellation_10 | 54 | 13 | centaurus |
| constellation_11 | 49 | 12 | canis-major |
| constellation_12 | 21 | 11 | lupus |
| constellation_13 | 30 | 7 | aquila |
| constellation_14 | 36 | 6 | pisces |
| constellation_15 | 44 | 6 | hydra |
| constellation_16 | 18 | 9 | eridanus |
| constellation_17 | 27 | 7 | ursa-minor |

## Method used by the current best run

### 1. Wide GPU patch localisation

`gpu_matcher.py` searches each query patch over the supplied sky image with a
coarse-to-fine CUDA matcher. It applies background normalization, a 36-angle
and 10-scale transform bank, coarse peak retrieval, and local refinement.

The configuration in `matcher_config_gpu_wide.json` uses scales from 0.58 to
1.46, a 10-pixel refinement radius, and weighted intensity, gradient, and SSIM
similarity. Rather than committing to the first visually plausible location,
it preserves the 16 best spatially distinct candidates for each patch.

On all 71 labelled-present training patches, the candidate-retrieval diagnostic
was:

| Diagnostic | Earlier CPU matcher | A100 wide matcher |
| --- | ---: | ---: |
| Exact top-1 location | 36 / 71 | 49 / 71 |
| Exact location within top 16 | 56 / 71 | 63 / 71 |

These figures measure training candidate retrieval only; they are not a
leaderboard score.

### 2. Constellation geometry and membership

`joint_geometric_solver.py` reads the supplied pattern diagrams, extracts their
star nodes, and tests each pattern against the query-candidate point cloud.

It evaluates scale, translation, rotation, and reflection using RANSAC-style
similarity-transform proposals. It then uses a one-to-one assignment between
patch queries and pattern nodes, preventing multiple patches from claiming the
same star. Graph-supported assignments become `m=1`; direct but non-graph
matches may remain `m=0`.

`structural_refiner.py` supplies a small regularized membership proposal model
based only on simple patch brightness and contrast features from the labelled
training patches. It is not a pretrained model.

### 3. A100 execution and recovery

The first CUDA solver run stalled because its geometry phase forked worker
processes after CUDA was initialized. Forking a live CUDA process can deadlock.
The solver was changed to use CUDA-safe threaded geometry workers. Candidate
results were cached per scene, and the corrected A100 run completed all 16
scenes before the final CSV was validated and downloaded.

## NYU compute history

The project was initially explored through NYU HPC access. The course
announcement stated that coursework must not run on the Torch research cluster,
so the supported NYU option is Cloud Bursting/Open OnDemand with NYU VPN. HPC
bootstrap and Slurm scripts remain under `hpc/`, but the current best score came
from the Colab A100 run.

## Known limitations and work still required

Status as of the September 21, 2026 rebuild.  Items marked RESOLVED were
fixed in that rebuild and are kept here so the record stays readable.

1. RESOLVED - GPU presence calibration failed, so the 0.66890 run used the
   default threshold of 0.46 and marked 72.8% of patches present against a
   61.2% training rate.  Presence is now a per-scene quantile
   (`--presence-mode quantile --present-rate 0.625`); the current output sits
   at 65.3% with a 59-78% per-scene range.
2. RESOLVED - `score_train_predictions` could not evaluate identity and
   substituted `constellation == "unknown"`, so its number rose as
   identification got worse.  It now reports localisation components only,
   and `evaluate.py` scores all four components with identity corrected.
3. RESOLVED - the geometry scorer's coverage expression degenerated to 1.0,
   reducing ranking to raw support and favouring large diagrams.  Replaced
   with a size-fair significance score plus true coverage.
4. OPEN - eight of the 71 labelled-present training patches have no candidate
   at their true location, so geometry cannot place a point that was never
   proposed.  This is now the binding constraint on the localisation term,
   which `evaluate.py` puts at 0.574 against 0.933 for geometry.
5. PARTLY RESOLVED - scene-held-out evaluation now exists, but only three
   scenes are labelled and the membership classifier is fitted on those same
   three.  The train score is a regression guard, not an unbiased estimate,
   and the scoring weights remain tuned on three scenes.
6. OPEN - `constellation_06` (tucana, support 4, reached only via the
   fallback path) and `constellation_03` (sagittarius, coverage 0.24) are the
   weakest fits in the current submission.
7. OPEN - two labels still repeat (`canis-major`, `hydra`).  If the 16
   validation scenes are 16 distinct constellations, a one-to-one assignment
   would resolve this, but that is an assumption about dataset construction
   and has not been tested.

## Recommended next iteration

Ordered by measured headroom rather than by effort.

1. **Localisation is now the weakest component.** `evaluate.py` reports 0.574
   on the labelled scenes against 0.933 for geometry and 1.00 for identity.
   It carries 0.20 of the metric, so the realistic gain is larger than
   anything remaining in identity.  Eight of the 71 labelled-present patches
   still have no candidate within tolerance at all; geometry cannot place a
   point that was never proposed.
2. **Re-check the presence rate against the leaderboard.** 0.625 was chosen on
   three scenes against a combined presence+localisation objective, and the
   curve was flat from 0.60 to 0.675.  One submission at 0.65 would show
   whether that flatness holds out of sample.
3. **The scoring weights remain unvalidated.** `SCORE_COVERAGE_WEIGHT` and
   `SCORE_COUNT_WEIGHT` were set on three scenes.  If a future submission
   regresses, reduce them before touching the proposal count, which is the
   only change with direct evidence behind it.
4. **Two weak scenes.** `constellation_06` (tucana, support 4, reached only
   through the fallback path) and `constellation_03` (sagittarius, coverage
   0.24) are the least-supported fits in the current output.
5. **Distinct-label assignment is still untested.** If the 16 validation
   scenes are 16 distinct constellations, a Hungarian assignment over the
   16x48 quality matrix would resolve the two remaining duplicate labels
   (`canis-major`, `hydra`).  This is an assumption about how the dataset was
   built, so it belongs in its own A/B submission.

## September 21, 2026 rebuild (local RTX 5070 Ti)

Candidate localisation is unchanged; every change below is in the geometry,
scoring, and calibration stages.  A held-out-style check now exists:
`evaluate.py` scores a train-split prediction with the identity term
**corrected**.  On the three labelled scenes the weighted score moved from
**0.7030 to 0.8393** (loose geometry) and **0.6336 to 0.8143** (strict), with
identity going from 2/3 to **3/3**.

That train figure is optimistic: the membership classifier is fitted on those
same three scenes, so it is a regression guard, not a leaderboard forecast.

### 1. RANSAC was under-sampled (the largest defect)

`--proposals` defaulted to 6000.  Re-fitting the true pattern on a labelled
scene across six seeds at that budget gave supports of 9, 6, 9, 7, 9, 9 - it
missed the correct fit about a third of the time.  From 12000 proposals it
returned 9 on every seed.  Constellation identity was therefore being decided
partly by sampling luck.  The default is now **20000**.

### 2. The coverage term was dead code

`coverage = support / min(len(pattern.points), support)` is identically 1.0,
because support never exceeds the node count.  Ranking reduced to raw support,
which rewards large diagrams: a free similarity transform finds more
coincidences the more nodes it can place.  Every one of the 16 published
labels came from the larger half of the 48 diagrams (median size rank 7 of
48); `eridanus`, the largest, was chosen three times.

Ranking is now `significance + coverage - error/tolerance - count_penalty`,
where significance is `-log10 P(Binomial(nodes, p) >= support)` under a
uniform-null cloud of the observed density.  This charges a 27-node diagram
more for the same support than an 11-node one.

### 3. Small diagrams then won instead

A four-node diagram matching four points reaches coverage 1.0 for free, and a
similarity transform has only four degrees of freedom.  A fit must now explain
at least half the stars the figure is expected to contribute
(`FitContext.minimum_support`).  If nothing clears that floor the best
`support >= 4` fit is still returned, because the identity term is scored as
accuracy and abstaining earns exactly what a wrong name earns.

### 4. Graph fitting is fed a sparse, high-precision cloud

Spurious support grows with candidate density.  `--graph-top-k` (default 3)
now controls the cloud used to *choose* the pattern, while `--top-k` (16) is
still used afterwards to *place* the stars: once a transform wins, the
one-to-one assignment is re-run against every retained candidate.

### 5. Membership selection is rank-based

The classifier's absolute probability scale shifts between scenes.  The old
`0.60 * threshold` floor selected 12-17 queries on each labelled scene but as
few as 4 on several validation scenes, which caps support at 4 and makes every
pattern look like a coincidence.  Selection now takes a scene-adaptive number
of the most figure-like queries.

### 6. Presence is calibrated per scene

The 0.66890 run used `presence_threshold = 0.46`, the dataclass default,
because GPU calibration had failed.  It marked 72.8% of patches present
against a 61.2% training rate, with six scenes above 90% and one at 100%.
Presence is now a per-scene quantile (`--present-rate`, default 0.625, chosen
against the combined presence+localisation contribution).

### 7. Corrected and repaired

- `score_train_predictions` awarded the identity term for answering
  `unknown`, so its "approximate train score" rose as identification got
  worse.  It now reports localisation components only and points to
  `evaluate.py`.
- `geometric_ensemble.py` passed a tuple to `format_cell`, which reads
  `.x/.y/.m`; it raised `AttributeError` on the first scene.

### Result

`outputs/submission_v2.csv` passes the validator.  Against the
training priors, compared with the 0.66890 submission:

| Statistic | Train prior | 0.66890 run | Rebuild |
| --- | --- | ---: | ---: |
| Present rate | 61.2% | 72.8% (range 39-100%) | 65.3% (range 59-78%) |
| Figure / present | 33-38% | 31.9% | 32.8% |
| Figure / nodes | 55-77% | 53.6% | 65.5% |
| Distinct labels | - | 11 / 16 | 14 / 16 |
| Median diagram size rank | - | 7 of 48 | 13 of 48 |

These are consistency checks against three labelled scenes, not a predicted
score.  Kaggle's result is authoritative; keep whichever CSV scores better.

### Kaggle result: 0.68196

`outputs/submission_v2.csv` scored **0.68196**, against 0.66890 for
the previous best: **+0.01306**.  The changes generalised to the held-out
scenes, so this file is now the one to beat.

The gain is much smaller than the train-split movement (0.7030 to 0.8393)
implied, which is the expected outcome and worth recording:

* the train figure covers three scenes and the membership classifier is
  fitted on those same three, so it was always an upper bound;
* Kaggle returns one aggregate number, so the presence, localisation,
  membership, and identity contributions cannot be separated from it.  The
  +0.01306 is consistent with roughly one additional correct identity plus
  small presence gains, but that decomposition is not observable and must not
  be asserted;
* identity is 0.30 of the metric across 16 scenes, so each additional correct
  scene is worth about 0.019.  A move of this size is one or two scenes, not a
  broad improvement.

Treat the size-fair scoring weights as unvalidated beyond "did not regress".
The proposal-count fix is the one change backed by a direct measurement
(support 6-9 across seeds at 6000 proposals, stable 9 from 12000) rather than
by leaderboard movement.

### Reproduce

```sh
.venv/Scripts/python.exe joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide.json \
  --output outputs/submission_v2.csv \
  --split validation --device cuda --top-k 16 --graph-top-k 3 \
  --proposals 20000 --cache-dir outputs/cache_validation \
  --presence-mode quantile --present-rate 0.625
```

Swap `--split train` and run `evaluate.py --predictions <csv>` to re-check the
labelled scenes.  Candidates are cached per scene, so geometry-only changes
re-run in minutes without repeating GPU matching.

## September 21, 2026 assignment-robustness pass

Follow-up to the rebuild above, prompted by a specific request to audit the
solver for hard-coded values that would not hold on unseen scenes.  Candidate
localisation and the scoring formula are unchanged; every change is in how a
transform is proposed, refit, and used to place stars.  Submitted as
`outputs/submission_v3.csv` (generated under the working filename
`submission_consensus.csv`, then renamed before upload) and scored
**0.68431** on Kaggle, **+0.00235** over the previous best.

### What was audited, and what changed

1. **The RANSAC scale-ratio filter was a hard rejection, not a soft
   preference.** `propose_transforms` only accepted a candidate-pair/node-pair
   length ratio in `[2, 10]`, chosen because the three labelled scenes'
   fitted scale factors are 4.4-6.5.  That is a search-time filter only -
   correctness is still enforced downstream by tolerance and significance -
   so narrowing it bought nothing, while a held-out scene whose true scale
   fell outside `[2, 10]` would have had its correct transform silently
   never proposed, with no signal that anything was missed.  Widened to
   `[1.3, 18]`.  Verified on an isolated ablation to change nothing on the
   three labelled scenes (identical winners, identical scores); it only
   removes a failure mode that could not have been exercised by three
   examples.
2. **A single assignment pass was frozen at the seed proposal's accuracy.**
   Traced one figure star (`pisces patch_11`) whose true candidate sat 2px
   from truth but 37.6px from its nearest mapped pattern node against a
   29.5px tolerance - correct, but rejected, with nothing else contesting
   that node.  `assign_queries` now refits the transform from its own
   inliers and reassigns repeatedly (bounded at 5 rounds), keeping whichever
   round scores highest by the same `quality` measure this file already
   uses to rank RANSAC proposals against each other - not simply the last
   round computed, since a refit that tightens the transform is not
   guaranteed to raise the raw assignment count.  Verified by ablation to
   reproduce the prior single-refit behaviour exactly when capped at one
   refit round, and to change nothing further on the three labelled scenes
   when allowed to iterate up to 5 rounds (it already converges by round 2
   on all three).
3. **Match margin was computed and immediately discarded.** The matcher
   already computes each candidate's margin over the next-best candidate at
   its rank (`Prediction.margin`), but it was dropped the moment a
   `Candidate` was built, before even reaching the cache.  Threaded end to
   end: `Candidate` now carries it, the cache stores it (a cache written
   before this field existed reads back as margin `0.0`, never a fabricated
   value), and the one-to-one assignment cost can weight it.
   - **This was tested and found not to help, and is disabled by default.**
     A four-tuple collision was traced directly (`taurus patch_08` vs
     `patch_22`, two different patches whose candidates land within 2px of
     the same star): a naive min-max normalisation of margin was swamped by
     a couple of outliers (per-candidate margins in this scene cluster
     within 0.001-0.009 against a scene-wide range of -0.15 to 0.12);
     switching to percentile-rank normalisation fixed that specific flaw,
     but a clean, isolated ablation still showed a small regression on the
     three labelled scenes (train total 0.8393/0.8143 to 0.8331/0.8081 at
     margin weight 0.15), and in the traced collision itself the true
     candidate's own margin was *smaller* than the wrong one's.  Margin
     does not reliably separate correct from coincidental matches in this
     data, at least not on this little evidence.  `ASSIGNMENT_MARGIN_WEIGHT`
     defaults to `0.0`; the plumbing is kept for a better use of the signal
     later, or re-evaluation once more labelled scenes exist.
4. **A single arbitrary RANSAC seed decided some scenes' identity.**
   Re-fitting `constellation_08` at 6 independent seeds gave `orion` in 5
   and `hydra` in 1 - a clear majority, but the one fixed seed the rest of
   this pipeline used for that scene happened to be the dissenting draw, so
   the submitted CSV read `hydra`.  Added `choose_fit_consensus`: the final
   pattern choice now runs `--consensus-trials` (default 5) independent
   seeds and keeps the plurality-winning pattern, using its own
   highest-quality fit among the trials that agreed with it.  Confirmed
   this recovers `orion` for `constellation_08`.  On the three labelled
   scenes, all three patterns won unanimously across all 5 trials (train
   total 0.8369/0.8119, an untroubled, noise-level difference from the
   single-seed 0.8393/0.8143).

### Net effect on the actual submission: Kaggle 0.68431 (+0.00235)

11 of 16 validation labels are unchanged from `outputs/submission_v2.csv`.
5 changed: `constellation_01`, `03`, `06`, `15`, `16`.  A stability check
(6 independent seeds each, using the code *before* this pass) had already
flagged `01`, `03`, `06`, and `16` as scenes where the winning pattern
depended on which seed ran - so these were not cases of a settled answer
being disturbed; they were already unsettled.  `constellation_15`'s change
(`hydra` to `eridanus`) traces to the wider scale band alone and was not
seed-dependent in the same check.

The +0.00235 result is consistent with that picture, not a surprise: with
identity worth 0.30 of the metric across 16 scenes, one full scene flipping
correct is worth about 0.019, so a movement this small is compatible with a
mix of small gains and losses across the changed scenes rather than a clean
win or a clean loss on any one of them.  This still cannot be decomposed
further - Kaggle returns one aggregate number, and only three scenes are
labelled, with the membership classifier and scoring weights already fitted
to those same three.  Kaggle's result is the only authoritative signal
either submission has received; the small, positive movement is the reason
`outputs/submission_v3.csv` is recorded as the current best above, not proof
that every individual change in this section was itself correct.

### Reproduce

```sh
.venv/Scripts/python.exe joint_geometric_solver.py --root . \
  --config matcher_config_gpu_wide.json \
  --output outputs/submission_v3.csv \
  --split validation --device cuda --top-k 16 --graph-top-k 3 \
  --proposals 20000 --cache-dir outputs/cache_validation \
  --presence-mode quantile --present-rate 0.625 --consensus-trials 5
```

This takes roughly 5x longer than the single-seed run above, since the final
pattern choice is now voted across 5 independent RANSAC seeds per scene.

## September 21-22, 2026 affine and external-catalog pass

The assignment says diagram aspect ratio is unrelated to the scene, but the
solver still used only rotation + uniform scale + translation. A full affine
transform was therefore added as a safeguarded refinement after the existing
similarity-RANSAC initialization. Blind three-point affine RANSAC was avoided:
any three non-collinear points define an affine map and would create too many
false hypotheses in these dense candidate clouds.

Five-seed train regression results after the same presence refinement:

| Model | Presence | Localisation | Loose geometry | Strict geometry | Identity | Loose total | Strict total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Similarity | 0.861 | 0.611 | 0.900 | 0.833 | 3/3 | 0.8625 | 0.8458 |
| Affine | 0.850 | 0.623 | 1.000 | 0.900 | 3/3 | **0.8871** | **0.8621** |

The affine validation run disagreed with v4 identity on scenes 01, 03, 06,
and 16. All four were weak/ambiguous fits. `submission_ensemble.py` therefore
uses affine rows only on the other 12 scenes and retains the complete v4 row
on disagreements. Copying only the old name would be invalid reasoning because
the retained `m=1` points would still describe the affine solver's other
pattern. The resulting upload candidate is
`outputs/submission_v5_affine_gated.csv`; schema validation reports 16 rows,
90 columns, and zero blank/null cells. SHA-256:
`726c1af12d267315c883f208255c867daf300b559a33ffceea88f957c1efd9b1`.

### Public external catalog check

The signed-in Fall 2026 Kaggle rules permit public, equally accessible external
data/models, although the older handout in this repo still says external data
is prohibited. Keep the instructor's written clarification. No validation row
was hand-labeled.

Two public repositories are pinned as submodules:

- `ofrohn/d3-celestial` (BSD-3-Clause), used for constellation line geometry;
- `MarcvdSluys/ConstellationLines` (CC BY 4.0), retained as a second documented
  catalog source but not used to rewrite v5.

`external_catalog_validator.py` is read-only. It independently recovers
Pisces, Scorpius, and Taurus from the exact labeled `m=1` coordinates (3/3),
then agrees with 10/16 v5 identities. It supports several strong predictions
including Corona Australis, Canis Major, Hydra, Aries, Perseus, Orion, Lupus,
Corona Borealis, and Eridanus. Weak disagreements are not automatically
substituted because catalog line conventions differ from the supplied pattern
diagrams and the validator's own runner-up is sometimes the submitted label.

## September 22, 2026 external-image multi-model experiment

Training executed in the user's live Colab session on an **NVIDIA
A100-SXM4-40GB**, not the local laptop. The run started from commit
`14131744f169b31649c412eedd9a809c8268f494` with 4,000 updates for each of two
CNN patch encoders (intensity and high-pass). Pretraining used 18,362 crops
from four downloaded ESO observations and 12,000 synthetic star patches.
Exact source credits and preprocessing are in `EXTERNAL_TRAINING_DATA.md`.

The downstream pipeline combines cached image-correlation candidates, both
CNN descriptors, radial photometry, a regularized candidate classifier,
presence/membership classifiers, and similarity/affine geometric consensus.
Unlike earlier train regression numbers, each evaluated scene is excluded
from **all three downstream classifiers**, not only the presence classifier.
CNN pretraining uses no competition images. Geometry uses 12,000 proposals
per pattern and three seeds per transformation family.

| Scene-held-out metric | Classical ranking | Neural ensemble ranking |
| --- | ---: | ---: |
| Presence macro-F1 | 0.86096 | 0.86096 |
| Localisation | **0.62346** | 0.61111 |
| Loose geometry | 1.00000 | 1.00000 |
| Strict geometry | 0.90000 | 0.90000 |
| Identity | 3/3 | 3/3 |
| Loose total proxy | **0.88993** | 0.88746 |
| Strict total proxy | **0.86493** | 0.86246 |

**The ensemble failed its promotion gate.** Lower contrastive training loss
did not translate into better held-out localisation. These are three-scene
diagnostics under an approximate scorer, not Kaggle scores. This experiment
does not establish an improvement over the public best of 0.71544 and must
not be advertised as a 0.93 model.

`colab_multimodel.ipynb` makes the distinction explicit: the separately named
experimental CSV contains new predictions, while the recommended CSV falls
back to the established v4 file on a failed gate. The exporter preserves
checkpoints, candidate features, train predictions, report, source manifest,
hashes and a fixed-geometry ranking-weight diagnostic so the run can be
audited without retraining. Training-manifest bytes are preserved separately
from a corrected-credit attribution copy to keep checkpoint fingerprints
reproducible.
