# DAPI Grid

A compact whole-slide pipeline for unsupervised mapping of nuclear morphology
from Phenocycler QPTIFF images.

## What it does

1. Reads the DAPI channel in overlapping level-0 chunks.
2. Segments nuclei with StarDist and checkpoints every completed chunk.
3. Measures nuclear shape without filtering detections by morphology or density.
4. Summarises median and IQR morphology in overlapping spatial windows.
5. Selects the number of window clusters automatically using silhouette and
   elbow analysis, with minimum-size and repeated-seed stability safeguards.
6. Produces a whole-tissue cluster map and a pyramidal OME-TIFF containing the
   actual StarDist boundaries over the original level-0 DAPI image.

Because DAPI labels nuclei, the outlines are **nuclear segmentations**, not
complete cell boundaries.

## Install

Python 3.10 is required by the locked TensorFlow/StarDist environment. Check it
before creating the environment:

```bash
python3.10 --version
python3.10 -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Copy `config.example.yaml` to `config.yaml`, set the input and output paths,
then run:

```bash
dapi-grid run config.yaml
```

The same command resumes completed chunks. Use `--force` only to deliberately
repeat segmentation:

```bash
dapi-grid run config.yaml --force
```

## Automatic cluster selection

Candidate values from `k_min` to `k_max` are fitted. Silhouette is calculated
on spatially separated windows so strongly overlapping neighbours do not make
the clusters look artificially clean. The inertia curve supplies an automatic
elbow. The elbow is selected when its silhouette is at least 95% of the best
silhouette; otherwise the silhouette optimum is selected. Solutions containing
a cluster smaller than `minimum_cluster_fraction` are excluded when possible.
Repeated fits report whether the selected solution is stable.

## Outputs

| Output | Purpose |
| --- | --- |
| `nuclei.csv` | Level-0 nuclear measurements and coordinates |
| `grid_clusters.csv` | Window features and final cluster labels |
| `whole_tissue_cluster_overlay.png` | Whole-slide cluster overview |
| `nuclear_segmentation_overlay.ome.tif` | Zoomable level-0 DAPI with true StarDist outlines |
| `cluster_selection.png` | Combined elbow and silhouette diagnostic |
| `run_summary.json` | Selected k, stability, counts and configuration |
| `chunks/` | Resumable measurements and polygon checkpoints |

Open `nuclear_segmentation_overlay.ome.tif` in QuPath or another pyramidal
OME-TIFF viewer. Its base layer is created directly from level 0; lower pyramid
levels are included only for efficient navigation.

## Compare one control with multiple experimentals

First run every QPTIFF independently, giving each image its own output folder:

```bash
dapi-grid run control.yaml
dapi-grid run experimental_01.yaml
dapi-grid run experimental_02.yaml
```

These runs perform the expensive level-0 segmentation once. Copy
`comparison.example.yaml` to `comparison.yaml`, list the completed result
folders, and run:

```bash
dapi-grid compare comparison.yaml
```

The comparison does not compare the independently learned cluster numbers.
Instead, it reads the original morphology features, gives every image equal
influence during training, automatically selects a joint cluster number, and
assigns one shared set of cluster meanings and colours to all images. It does
not repeat StarDist segmentation.

All inputs must have the same window size, stride, pixel size, DAPI channel,
StarDist model and StarDist thresholds. The command stops if these settings or
the feature schemas differ.

### Comparison outputs

| Output | Purpose |
| --- | --- |
| `comparison_overview.png` | Cluster proportion, abundance and spatial changes versus control |
| `comparison_results.csv` | Exact per-sample, per-cluster comparisons |
| `joint_grid_clusters.csv` | All windows with comparable shared labels |
| `joint_cluster_selection.png` | Elbow and silhouette result for the shared model |
| `joint_overlays/` | Same cluster definitions and colours on every QPTIFF |
| `comparison_summary.json` | Compatibility checks, inputs and joint-model details |

The joint overlays reopen the low-resolution DAPI pyramid from each original
QPTIFF. Keep the original files at the paths recorded by their individual
`run_summary.json` files. The existing level-0 segmentation OME-TIFFs are reused
as QC results and are not regenerated.

With one negative-control image, these are exploratory image-level differences,
not formal biological-replicate significance tests. Patterns repeated across
several independent experimental images are more persuasive than a change seen
in only one image.
