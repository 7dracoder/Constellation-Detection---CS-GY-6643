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

## What to upload to Kaggle

Upload only the generated file named:

`submission_a100_wide_fixed.csv`

The final path validates the header, 16 rows, all 87 patch fields, `-1`
padding, coordinate ranges, and null/blank cells before download. Do not upload
a JSON diagnostics file or a ZIP.

## Score expectations

The CSV being valid only proves that Kaggle can read it; it does not predict a
leaderboard score. The earlier baseline scored about 0.26710, an earlier
geometric run scored about 0.58, and the A100 wide-search run scored 0.66890.
No run here has established a 0.90 or 0.96 score. The old local approximate
metric does not correctly evaluate constellation identity and must not be used
as a leaderboard forecast. Submit the generated CSV and retain the version with
the better Kaggle score.

See `PROJECT_STATUS.md` for the full method, current output, and known issues.

## Do not do this

Do not add web images, external constellation datasets, external annotations,
or pretrained models.  The bundle is designed to use only the project data.
