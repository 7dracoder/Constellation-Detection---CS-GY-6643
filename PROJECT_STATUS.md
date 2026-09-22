# Project status — Constellation Detection, CS-GY 6643

Last updated: September 21, 2026

## Scope and compliance

This project is intentionally restricted to the course-provided materials:

- supplied training and validation sky images;
- supplied 32×32 query patches;
- supplied constellation-pattern diagrams; and
- supplied training labels.

It does **not** use web images, external astronomical data, external labels,
pretrained weights, or external models. NYU's course guidance also requires
coursework to use Cloud Bursting/Open OnDemand instead of the researcher-facing
Torch cluster.

## Score record

| Submission stage | Kaggle score | Notes |
| --- | ---: | --- |
| Initial validated baseline | approximately 0.26710 | Correct schema, but weak localisation and all-`unknown` identity. |
| Earlier geometric submission | approximately 0.58 | Added candidate shortlists and geometric identity fitting. |
| A100 wide-search submission | 0.66890 | Superseded: `outputs/submission_a100_wide_fixed.csv`. |
| RTX 5070 Ti rebuild | 0.68196 | Superseded: `outputs/submission_v2.csv`. |
| Assignment-robustness pass | **0.68431** | Current best: `outputs/submission_v3.csv`. See "September 21, 2026 assignment-robustness pass". |

Kaggle provides a single hidden-label score, not component scores. A valid CSV
only proves that Kaggle can read it; it does not predict a leaderboard score.
No implementation or experiment in this repository establishes a 0.90 or 0.96
score guarantee.

## Current best submission

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
