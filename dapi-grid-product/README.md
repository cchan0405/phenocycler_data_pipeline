# DAPI Grid

This project converts the one-ROI proof of concept into a resumable whole-tissue
pipeline. It uses DAPI only, segments nuclei with StarDist2D, summarises nuclear
shape within spatial grid squares, performs one global unsupervised clustering,
and renders every valid grid square on the full tissue.

## What the clusters mean

The clustering unit is a **spatial grid square**. It is not an individual cell.
Each square is represented by distributions of nuclear area, perimeter,
circularity, eccentricity, solidity, axis lengths, aspect ratio and DAPI
intensity, plus nuclear density. The resulting label describes a local
nuclear-shape phenotype.

## Important safeguards

- The level-0 image is never loaded in full.
- Low-resolution masking skips empty chunks.
- StarDist chunks include a halo so nuclei are not cut at boundaries.
- Only nuclei whose centres belong to the non-overlapping chunk core are kept,
  preventing duplicate counting.
- Every completed chunk is checkpointed as a CSV file. Re-running the same
  command resumes automatically.
- One model is fitted across all valid tissue grids, so cluster labels are
  consistent across the complete image.
- Empty or low-nucleus grids remain transparent.

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
| `grid_features.csv` | Shape distributions for valid grid squares |
| `grid_clusters.csv` | Final grid features and cluster labels |
| `cluster_selection.json` | Candidate cluster scores and selected solution |
| `grid_cluster_model.joblib` | Saved scaler, PCA and clustering model |
| `whole_tissue_cluster_overlay.png` | Final coloured tissue |

## Cluster-number selection

By default, the pipeline evaluates `k=2` through `k=10` and selects the highest
sampled silhouette score. To force a particular result after inspection:

```yaml
clustering:
  fixed_k: 5
```

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
