# DAPI Grid

A compact whole-slide pipeline for unsupervised mapping of nuclear morphology
from Phenocycler QPTIFF images.

## What it does

1. Builds an expanded, permissive tissue/no-tissue mask from the DAPI preview.
2. Skips background chunks and runs StarDist only on tissue-containing level-0 chunks.
3. Removes detections whose centres fall outside the tissue mask.
4. Clusters individual nuclei from their own morphology measurements.
5. Outputs silhouette and elbow diagnostics and, by default, asks you to choose
   the number of nuclear phenotypes in the terminal.
6. Summarises the mixture of cell phenotypes in spatial windows.
7. Produces a pyramidal OME-TIFF containing the actual StarDist masks over the
   original level-0 DAPI image. Each nucleus is coloured from its own morphology,
   so square grid boundaries cannot determine its colour.

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

If the output folder was created by the earlier grid-coloured version, run once
with `--force`. This rebuilds the chunk checkpoints using tissue-first
segmentation; future runs can resume normally without the flag.

## Choosing the cluster number

Candidate values from `k_min` to `k_max` are fitted on a bounded random sample
of retained tissue nuclei. The inertia curve supplies an automatic
elbow. A near-linear inertia curve is reported as having no reliable elbow
instead of selecting the first tested value. A supported elbow is selected when
its silhouette is at least 95% of the best silhouette; otherwise the silhouette
optimum is selected. Solutions containing
a cluster smaller than `minimum_cluster_fraction` are excluded when possible.
Repeated fits report whether the selected solution is stable.

The default YAML uses `cluster_selection_mode: prompt`. The pipeline first
writes `cluster_selection.png`, prints the numerical scores and graph path, and
then pauses at:

```text
Choose the number of clusters [2, 3, 4, 5, ...]:
```

Open the PNG while the command is waiting, inspect both curves, and enter one
of the displayed values. The selected value is then marked on the final graph.

For an unattended run, set `cluster_selection_mode: automatic`. To always use a
scientifically motivated value, set `cluster_selection_mode: fixed` and
`fixed_k: 5`. The diagnostic curves are produced in all three modes, and the
summary records how the number was selected.

## Outputs

| Output | Purpose |
| --- | --- |
| `tissue_mask.png` | Low-resolution mask used to separate tissue from background |
| `nuclei.csv` | Level-0 measurements, coordinates and individual phenotype label |
| `window_phenotype_composition.csv` | Per-window phenotype fractions and morphology summaries |
| `spatial_dominant_phenotype_overlay.png` | Explicitly grid-based spatial summary |
| `nuclear_segmentation_overlay.ome.tif` | Zoomable level-0 DAPI with cluster-coloured StarDist masks |
| `cluster_selection.png` | Combined elbow and silhouette diagnostic |
| `run_summary.json` | Selected k, stability, counts and configuration |
| `chunks/` | Resumable measurements and polygon checkpoints |

Open `nuclear_segmentation_overlay.ome.tif` in QuPath or another pyramidal
OME-TIFF viewer. Its base layer is created directly from level 0; lower pyramid
levels are included only for efficient navigation. Colour is assigned directly
from each nucleus's morphology cluster, rather than inherited from a square window.

To rebuild only this coloured OME-TIFF after changing clustering or mask
settings, reuse the completed StarDist chunks with:

```bash
dapi-grid run config.yaml --rerender-segmentation
```

This does not repeat level-0 segmentation. `--force` is only for deliberately
rerunning StarDist.

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
Instead, it reads the individual nuclei, gives every image equal influence
during training, asks for one joint phenotype count, and assigns one
shared set of nuclear-phenotype meanings and colours to all images. It does not
repeat StarDist segmentation.

All inputs must have the same window size, stride, pixel size, DAPI channel,
StarDist model and StarDist thresholds. The command stops if these settings or
the feature schemas differ.

### Comparison outputs

| Output | Purpose |
| --- | --- |
| `comparison_overview.png` | Cell-phenotype proportion, abundance and spatial changes versus control |
| `comparison_results.csv` | Exact per-sample, per-cluster comparisons |
| `joint_nuclei/` | Individual nuclei with comparable shared phenotype labels |
| `joint_window_phenotype_composition.csv` | Window mixtures based on shared phenotypes |
| `joint_cluster_selection.png` | Elbow and silhouette result for the shared model |
| `joint_overlays/` | Shared-phenotype OME-TIFFs and spatial summaries for every QPTIFF |
| `comparison_summary.json` | Compatibility checks, inputs and joint-model details |

The joint overlays reopen each original QPTIFF. Keep the original files at the paths recorded by their individual
`run_summary.json` files. The existing level-0 segmentation OME-TIFFs are reused
as QC results and are not regenerated.

With one negative-control image, these are exploratory image-level differences,
not formal biological-replicate significance tests. Patterns repeated across
several independent experimental images are more persuasive than a change seen
in only one image.
