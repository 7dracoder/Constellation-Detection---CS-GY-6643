# Colab submission workflow

`constellation_colab_bundle.zip` is the original self-contained,
competition-data-only package:

- supplied `train`, `validation`, and `patterns` images;
- the supplied `sample_submission.csv` and training labels;
- the matcher and tested joint geometric solver; and
- `colab_constellation_submission.ipynb`.

It deliberately contains no external images, labels, datasets, or pretrained
models. Experimental external-catalog work is separate and must remain fully
reproducible from public, no-cost sources.

## Run it in Colab

1. Download `constellation_colab_bundle.zip` from this project.
2. Open [Google Colab](https://colab.research.google.com/) and upload
   `colab_constellation_submission.ipynb` from the archive (or open it locally
   and choose **Open in Colab**).
3. A standard CPU runtime can generate the original baseline. For the current
   best wide-search path, use a GPU runtime; the recorded best run used an A100.
4. Upload the same ZIP when the notebook asks, or place it in Google Drive and
   set `ARCHIVE` in the notebook to its Drive path.
5. Run the cells in order.  The standard path creates
   `outputs/submission_final_colab.csv`, validates it against the exact
   sample-submission schema, and downloads it.

## Current v5 path

The scored best v4 is **0.71544**. The unscored v5 candidate adds affine
refinement and conservative identity gating. Colab/A100 produced the cached
top-16 image matches; once those caches exist, affine fitting is CPU geometry
and does not benefit from repeating image matching on the GPU.

For a fresh runtime, clone the catalog submodules and run the commands in the
"Affine-gated v5 candidate" section of [README.md](README.md):

```sh
!git clone --recurse-submodules \
  https://github.com/7dracoder/Constellation-Detection---CS-GY-6643.git
%cd Constellation-Detection---CS-GY-6643
!python -m pip install -q -r requirements.txt
```

The older notebook still works as a valid course-data-only route, but it
predates the geometry, scoring, presence, and calibration fixes recorded in
[PROJECT_STATUS.md](PROJECT_STATUS.md).

## What to upload to Kaggle

Upload only one CSV. The scored fallback is `submission_v4_presence.csv`; the
new candidate to test is `submission_v5_affine_gated.csv`.

Every path validates the header, 16 rows, all 87 patch fields, `-1` padding,
coordinate ranges, and null/blank cells before download.  Do not upload a JSON
diagnostics file or a ZIP.

## Score expectations

The CSV being valid only proves that Kaggle can read it; it does not predict a
leaderboard score.

| Run | Kaggle |
| --- | ---: |
| Initial baseline | 0.26710 |
| Earlier geometric run | ~0.58 |
| A100 wide search | 0.66890 |
| Local RTX 5070 Ti rebuild | 0.68196 |
| Assignment-robustness pass | 0.68431 |
| Presence-refined pass | **0.71544** |
| Affine-gated candidate | not submitted |

No run here has established a 0.90 or 0.96 score.  Use `evaluate.py` on a
train-split prediction to compare configurations before spending a submission;
do not use the old `score_train_predictions` number, which awarded the identity
term for answering `unknown`.  Submit the generated CSV and retain the version
with the better Kaggle score.

See `PROJECT_STATUS.md` for the full method, current output, and known issues.

## External-data experiments

The signed-in Fall 2026 Kaggle rules currently permit public, equally
accessible external data and models. The course handout in this repository
still says external data is not allowed, so retain the instructor's written
clarification and document every source and licence. Never hand-label or
manually predict validation/test records; Kaggle explicitly prohibits that.
