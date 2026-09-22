# External training sources

The user reports instructor approval for external training data on September
22, 2026. The multi-model experiment downloads the following observation
images directly from the ESO public image archive. The older v4/v5 submission
files do not use these trained neural models.

| Observation | Credit (as supplied by ESO) | Source |
| --- | --- | --- |
| Milky Way star field around CS 31082-001 | AURA | https://www.eso.org/public/images/eso0106a/ |
| Dark cloud Lupus 4 | ESO | https://www.eso.org/public/images/eso1427a/ |
| Star cluster NGC 3532 | ESO/G. Beccari | https://www.eso.org/public/images/eso1439a/ |
| VISTA central Milky Way mosaic | ESO/VVV Survey/D. Minniti. Acknowledgement: Ignacio Toledo, Martin Kornmesser | https://www.eso.org/public/images/eso1242a/ |

License: [CC BY 4.0 under ESO's usage terms](https://www.eso.org/public/copyright/).
No ESO endorsement is implied. The downloader records the exact download URL,
credit, license, image dimensions, byte count, and SHA-256 in
`external/training_images/manifest.json`.

Processing: convert to grayscale; extract star-centered 32×32 windows at two
resolutions; independently apply rotation, scale, translation, blur, gamma,
brightness and noise to form positive correspondence pairs. These altered
crops train patch descriptors, not constellation identity labels. Crops from
the same source image are related examples, not independent whole-scene
validation cases. The first Colab run extracted 18,362 external crops and
added 12,000 procedurally generated star patches. Neither CNN's pretraining
reads competition train or validation images.

Whole competition training scenes are held out for the downstream candidate,
presence and membership models. Reports from these three folds are still
small-sample regression diagnostics; only Kaggle measures the hidden-label
competition score.
