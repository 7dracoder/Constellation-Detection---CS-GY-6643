# Colab submission workflow

`constellation_colab_bundle.zip` is a self-contained, course-data-only package:

- supplied `train`, `validation`, and `patterns` images;
- the supplied `sample_submission.csv` and training labels;
- the matcher and tested joint geometric solver; and
- `colab_constellation_submission.ipynb`.

It deliberately contains no external images, labels, datasets, or pretrained
models.  That keeps the workflow within the competition/course constraints.

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

## Colab is no longer the best path

The current best submission (**0.68196**) was produced on a local CUDA GPU, not
in Colab.  See the "Reproducing the current best" section of
[README.md](README.md).  This notebook still works and remains a valid
course-data-only route, but it predates the geometry, scoring, and calibration
fixes recorded in [PROJECT_STATUS.md](PROJECT_STATUS.md), and in particular it
defaults to 6000 RANSAC proposals, which was measured to be under-sampled:
the true pattern's support on a labelled scene varied between 6 and 9 across
seeds at that budget.  Pass `--proposals 20000` if you run the solver here.

## What to upload to Kaggle

Upload only the single generated CSV.  For the current best that is
`submission_v2.csv`.

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
| Local RTX 5070 Ti rebuild | **0.68196** |

No run here has established a 0.90 or 0.96 score.  Use `evaluate.py` on a
train-split prediction to compare configurations before spending a submission;
do not use the old `score_train_predictions` number, which awarded the identity
term for answering `unknown`.  Submit the generated CSV and retain the version
with the better Kaggle score.

See `PROJECT_STATUS.md` for the full method, current output, and known issues.

## Do not do this

Do not add web images, external constellation datasets, external annotations,
or pretrained models.  The bundle is designed to use only the project data.
