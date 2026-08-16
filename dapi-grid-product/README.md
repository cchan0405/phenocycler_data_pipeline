# DAPI Grid

This project converts the one-ROI proof of concept into a resumable whole-tissue
pipeline. It uses DAPI only, segments nuclei with StarDist2D, summarises nuclear
shape within overlapping spatial neighbourhoods, fits an unsupervised model on
reliable windows, predicts sparse windows with explicit confidence, and renders
small display tiles across the full tissue.

## What the clusters mean

The clustering unit is an **analysis window**. It is not an individual cell.
The pipeline first learns morphology-only nuclear phenotypes, then represents
each grid by its phenotype composition, robust shape distributions and local
tissue architecture. Density is retained only as a down-weighted context
feature by default. The resulting label describes a local nuclear-morphology
environment rather than a cell type.

This two-stage design distinguishes a mixture of round and elongated nuclei
from a uniformly intermediate population. It also prevents duplicate
count/density variables from driving a sparse-versus-dense split. DAPI
intensity is excluded from clustering by default because each processing chunk
is normalized independently.

Version 2 separates the analysis window from the display stride. A 512-pixel
window can move in 256-pixel steps, so each coloured tile uses a larger local
neighbourhood and adjacent measurements overlap. Only well-populated windows
fit the model; sparse windows are predicted afterward and shown with confidence-
weighted opacity.

## Important safeguards

- The level-0 image is never loaded in full.
- Low-resolution masking skips empty chunks.
- StarDist chunks include a halo so nuclei are not cut at boundaries.
- Only nuclei whose centres belong to the non-overlapping chunk core are kept,
  preventing duplicate counting.
- Every completed chunk is checkpointed as a CSV file. Re-running the same
  command resumes automatically.
- One model is fitted across reliable tissue windows, so cluster labels are
  consistent across the complete image.
- Sparse windows cannot move the fitted centroids, PCA or feature scaling.
- Empty windows remain transparent and low-confidence predictions appear pale.

## Installation

Python 3.10 or 3.11 is recommended because TensorFlow/StarDist compatibility is
usually simplest there.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If the QPTIFF compression codec is unavailable, install `imagecodecs`. On Linux,
the system may also require `libaec-dev`.

## Configure

Copy `config.example.yaml` to `config.yaml` and change:

```yaml
input_qptiff: "D:/data/slide.qptiff"
output_dir: "D:/data/slide_dapi_grid"
```

The first run should use the defaults. Because pixel size is currently unknown,
the initial grid is 512 × 512 level-0 pixels. Once calibration is available,
set both `pixel_size_um` and `grid_size_um`; this overrides `grid_size_px`.

## Run or resume

```bash
dapi-grid run config.yaml
```

Interrupted runs can be resumed with the same command. To deliberately
re-segment completed chunks:

```bash
dapi-grid run config.yaml --force
```

## Open the viewer

```bash
dapi-grid view "D:/data/slide_dapi_grid"
```

The viewer displays the whole-tissue overlay, allows clusters to be hidden,
changes overlay opacity, reports cluster sizes and exports the grid table.

## Main outputs

| File | Meaning |
| --- | --- |
| `tissue_mask.npz` | Low-resolution DAPI image and tissue mask |
| `chunks/nuclei_*.csv` | Resumable per-chunk nucleus measurements |
| `nuclei.csv` | All QC-passed nuclei in global coordinates |
| `nuclear_phenotype_model.joblib` | Morphology-only per-nucleus phenotype model |
| `grid_features.csv` | Shape distributions for all prediction-eligible windows |
| `window_features.csv` | Alias with version-2 terminology for the same feature table |
| `grid_clusters.csv` | Final grid features and cluster labels |
| `training_windows.csv` | Reliable windows used to fit the model |
| `predicted_windows.csv` | Sparse windows predicted after fitting |
| `density_residualization_report.csv` | Density dependence removed from each feature |
| `cluster_summary.csv` | Cluster size, reliability and confidence summary |
| `cluster_feature_profiles.csv` | Features distinguishing each cluster |
| `representative_windows.csv` | Twelve centroid-nearest windows per cluster |
| `cluster_selection.json` | Candidate cluster scores and selected solution |
| `grid_cluster_model.joblib` | Saved scaler, PCA and clustering model |
| `whole_tissue_cluster_overlay.png` | Confidence-weighted final map |
| `whole_tissue_unweighted_overlay.png` | All predictions at equal opacity |
| `whole_tissue_confidence_overlay.png` | Continuous confidence map |
| `whole_tissue_training_vs_predicted.png` | Training, prediction and low-confidence status |

## Cluster-number selection

By default, the pipeline evaluates `k=3` through `k=10`. Selection combines
sampled silhouette, agreement across repeated fits (adjusted Rand index), and a
small complexity term that prevents automatic collapse to the broadest two-way
split. To force a particular result after inspection:

```yaml
clustering:
  fixed_k: 5
```

Density handling can be changed without regenerating the feature table:

```yaml
clustering:
  density_mode: "controlled"  # exclude, controlled, or full
  density_weight: 0.20
```

## Overlapping windows and confidence

```yaml
spatial:
  analysis_window_px: 512
  display_stride_px: 256

coverage:
  min_nuclei_predict: 1
  min_tissue_fraction_predict: 0.005

training:
  min_nuclei_train: 8
  min_tissue_fraction_train: 0.10

density_correction:
  residualize: true

confidence:
  confidence_weighted_alpha: true
```

The analysis window must be at least as large as the display stride. Windows
below the training thresholds never influence preprocessing or cluster fitting.
Sampling confidence is `min(1, n_nuclei / min_nuclei_train)` and is combined
with separation from the second-nearest centroid. This makes uncertainty
visible rather than presenting every coloured tile as equally trustworthy.

Unsupervised does not mean biologically validated. Inspect representative areas
from each cluster and test whether the result is stable when grid size, minimum
nuclei and cluster number are varied.

## Recommended first validation

Before committing to the full run:

1. Confirm that the configured DAPI channel name matches the QPTIFF metadata.
2. Run with a copied test image or temporarily constrained image.
3. Inspect segmentation in sparse, dense, edge and artefact-heavy regions.
4. Check `grid_features.csv` for plausible nucleus counts and measurements.
5. Compare several grid sizes; 512 pixels is only a starting value without
   physical calibration.

## QPTIFF coordinates

The pipeline reads exact pyramid dimensions directly from TIFF metadata without
allocating the level-0 image. Region requests follow the current MxTiffFile API:
`pos=(x, y)` and `shape=(width, height)`.
