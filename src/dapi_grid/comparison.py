from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .clustering import _elbow, _plot_selection, choose_cluster_count
from .config import ComparisonConfig
from .render import render_cluster_overlay, render_segmentation_ometiff
from .window_features import MORPHOLOGY, aggregate_windows


LOG = logging.getLogger("dapi_grid")


def _load_sample(sample, group):
    table_path = sample.results_dir / "nuclei.csv"
    summary_path = sample.results_dir / "run_summary.json"
    if not table_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"{sample.name}: expected nuclei.csv and run_summary.json in "
            f"{sample.results_dir}"
        )
    frame = pd.read_csv(table_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("feature_schema_version") != "2.0":
        raise ValueError(
            f"{sample.name}: results were created by the earlier window-clustering "
            "pipeline. Run the updated single-image pipeline before comparison."
        )
    frame["sample"] = sample.name
    frame["group"] = group
    return frame, summary


def _compatibility_signature(summary):
    config = summary["configuration"]
    sd = config["stardist"]
    return {
        "window_size_px": config["window_size_px"],
        "stride_px": config["stride_px"],
        "pixel_size_um": config.get("pixel_size_um"),
        "channel": config["channel"],
        "stardist_model": sd["model"],
        "stardist_prob_thresh": sd["prob_thresh"],
        "stardist_nms_thresh": sd["nms_thresh"],
        "minimum_tissue_fraction": config.get("minimum_tissue_fraction", 0.0),
        "minimum_nuclei_per_cluster_window": config.get(
            "minimum_nuclei_per_cluster_window", 1
        ),
        "mask_rescue_nuclei": config.get("mask_rescue_nuclei", 1),
        "tissue_mask_expansion_radius": config.get("tissue_mask_expansion_radius", 0),
        "minimum_chunk_tissue_fraction": config.get(
            "minimum_chunk_tissue_fraction", 0.0
        ),
    }


def _validate_compatibility(records):
    reference_name, _, reference_summary = records[0]
    reference = _compatibility_signature(reference_summary)
    reference_features = [feature for feature in MORPHOLOGY if feature in records[0][1]]
    for name, frame, summary in records[1:]:
        signature = _compatibility_signature(summary)
        differences = [key for key in reference if signature[key] != reference[key]]
        if differences:
            raise ValueError(
                f"{name} is incompatible with {reference_name}: different "
                + ", ".join(differences)
            )
        if [feature for feature in MORPHOLOGY if feature in frame] != reference_features:
            raise ValueError(f"{name} uses a different nuclear-feature schema")
    return reference, reference_features


def _balanced_training(records, signature, maximum, seed):
    rng = np.random.default_rng(seed)
    candidates = [np.arange(len(frame)) for _, frame, _ in records]
    per_image = min(maximum, min(map(len, candidates)))
    if per_image < 3:
        raise ValueError("Every image needs at least three retained tissue nuclei")
    selections = [
        rng.choice(index, per_image, replace=False) if len(index) > per_image else index
        for index in candidates
    ]
    return per_image, selections


def _fit_joint(records, features, signature, cfg, output):
    per_image, selections = _balanced_training(
        records, signature, cfg.max_training_nuclei_per_image, cfg.random_seed
    )
    training = pd.concat(
        [frame.iloc[index][features] for (_, frame, _), index in zip(records, selections)],
        ignore_index=True,
    )
    preprocess = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    x = preprocess.fit_transform(training)
    full_pca = PCA(random_state=cfg.random_seed).fit(x)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = min(x.shape[1], max(2, int(np.searchsorted(cumulative, .90) + 1)))
    pca = PCA(n_components=n_components, random_state=cfg.random_seed).fit(x)
    xp = pca.transform(x)
    max_k = min(cfg.k_max, len(xp) - 1)
    if max_k < cfg.k_min:
        raise ValueError("Not enough balanced training windows for the requested k range")
    minimum_size = max(2, math.ceil(cfg.minimum_cluster_fraction * len(xp)))
    rng = np.random.default_rng(cfg.random_seed)
    score_index = (
        rng.choice(len(xp), cfg.silhouette_sample, replace=False)
        if len(xp) > cfg.silhouette_sample else np.arange(len(xp))
    )
    rows, models = [], {}
    for k in range(cfg.k_min, max_k + 1):
        model = MiniBatchKMeans(
            n_clusters=k, random_state=cfg.random_seed, batch_size=2048, n_init=10
        ).fit(xp)
        sampled_labels = model.labels_[score_index]
        score = (
            silhouette_score(xp[score_index], sampled_labels)
            if 1 < len(np.unique(sampled_labels)) < len(sampled_labels) else -1.0
        )
        smallest = int(np.bincount(model.labels_, minlength=k).min())
        rows.append({"k": k, "inertia": float(model.inertia_),
                     "silhouette": float(score), "smallest_cluster": smallest,
                     "valid_size": smallest >= minimum_size})
        models[k] = model
    scores = pd.DataFrame(rows)
    valid = scores.loc[scores.valid_size]
    if valid.empty:
        valid = scores
    elbow_k = _elbow(valid.k.to_numpy(), valid.inertia.to_numpy())
    best = valid.loc[valid.silhouette.idxmax()]
    graph_path = output / "joint_cluster_selection.png"
    selected_k, selection_method = choose_cluster_count(
        scores, valid, models, cfg, elbow_k, best, graph_path
    )
    model = models[selected_k]
    repeat_ari = []
    for repeat in range(1, max(1, cfg.stability_repeats)):
        repeated = MiniBatchKMeans(
            n_clusters=selected_k, random_state=cfg.random_seed + repeat,
            batch_size=2048, n_init=10,
        ).fit_predict(xp)
        repeat_ari.append(adjusted_rand_score(model.labels_, repeated))
    stability = float(np.mean(repeat_ari)) if repeat_ari else 1.0
    order = np.argsort(model.cluster_centers_[:, 0])
    relabel = np.empty(selected_k, dtype=int)
    relabel[order] = np.arange(selected_k)
    _plot_selection(
        scores, selected_k, graph_path, elbow_k=elbow_k
    )

    labelled = []
    for name, frame, _ in records:
        transformed = pca.transform(preprocess.transform(frame[features]))
        result = frame.copy()
        result["local_cluster"] = result.pop("cluster") if "cluster" in result else -1
        result["cluster"] = relabel[model.predict(transformed)]
        labelled.append(result)
    return labelled, {
        "selected_k": selected_k, "selection_method": selection_method,
        "elbow_k": elbow_k,
        "best_silhouette_k": int(best.k), "stability_ari": stability,
        "stable": stability >= .80, "training_nuclei_per_image": per_image,
        "total_balanced_training_nuclei": len(training),
        "pca_components": n_components,
        "pca_variance_explained": float(cumulative[n_components - 1]),
    }


def _spatial_metrics(frame):
    height = int(frame.grid_row.max()) + 1
    width = int(frame.grid_col.max()) + 1
    grid = np.full((height, width), -1, dtype=int)
    grid[frame.grid_row.astype(int), frame.grid_col.astype(int)] = frame.cluster.astype(int)
    agreements = []
    for first, second in [(grid[:, :-1], grid[:, 1:]), (grid[:-1], grid[1:])]:
        valid = (first >= 0) & (second >= 0)
        agreements.extend((first[valid] == second[valid]).tolist())
    patch_sizes = []
    for cluster in sorted(frame.cluster.unique()):
        labelled, count = connected_components(grid == cluster)
        if count:
            patch_sizes.extend(np.bincount(labelled.ravel())[1:].tolist())
    return {
        "neighbour_agreement": float(np.mean(agreements)) if agreements else np.nan,
        "median_patch_windows": float(np.median(patch_sizes)) if patch_sizes else np.nan,
    }


def _summary_table(labelled, windows, control_name):
    spatial = {
        frame["sample"].iloc[0]: _spatial_metrics(
            window.rename(columns={"dominant_phenotype": "cluster"})
        )
        for frame, window in zip(labelled, windows)
    }
    proportions = []
    all_clusters = sorted(pd.concat(labelled).cluster.unique())
    for frame, window in zip(labelled, windows):
        name = frame["sample"].iloc[0]
        counts = frame.cluster.value_counts(normalize=True)
        for cluster in all_clusters:
            proportions.append({
                "sample": name, "group": frame["group"].iloc[0], "cluster": cluster,
                "proportion": float(counts.get(cluster, 0)),
                "median_nuclei_per_window": float(window.n_nuclei.median()),
                **spatial[name],
            })
    result = pd.DataFrame(proportions)
    control = result.loc[result["sample"].eq(control_name)].set_index("cluster")
    result["control_proportion"] = result.cluster.map(control.proportion)
    result["proportion_change"] = result.proportion - result.control_proportion
    control_abundance = float(control.median_nuclei_per_window.iloc[0])
    result["abundance_change_percent"] = np.where(
        control_abundance > 0,
        100 * (result.median_nuclei_per_window - control_abundance) / control_abundance,
        np.nan,
    )
    control_agreement = float(control.neighbour_agreement.iloc[0])
    result["neighbour_agreement_change"] = result.neighbour_agreement - control_agreement
    return result


def _overview(results, control_name, path):
    experimental = results.loc[~results["sample"].eq(control_name)]
    heat = experimental.pivot(index="sample", columns="cluster", values="proportion_change")
    per_sample = experimental.drop_duplicates("sample").set_index("sample")
    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, .45 * len(heat))))
    limit = max(abs(heat.to_numpy()).max(), .01)
    image = axes[0].imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0].set(title="Cluster proportion change", xlabel="Joint cluster", ylabel="Experimental")
    axes[0].set_xticks(range(len(heat.columns)), heat.columns)
    axes[0].set_yticks(range(len(heat.index)), heat.index)
    fig.colorbar(image, ax=axes[0], label="Experiment − control")
    per_sample.abundance_change_percent.plot.barh(ax=axes[1], color="#0072B2")
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set(title="Nuclear abundance", xlabel="Change from control (%)", ylabel="")
    per_sample.neighbour_agreement_change.plot.barh(ax=axes[2], color="#009E73")
    axes[2].axvline(0, color="black", linewidth=.8)
    axes[2].set(title="Spatial organisation", xlabel="Neighbour-agreement change", ylabel="")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_comparison(cfg: ComparisonConfig):
    from .io import SlideReader

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = cfg.output_dir / "joint_overlays"
    overlay_dir.mkdir(exist_ok=True)
    samples = [(cfg.control, "control")] + [
        (sample, "experimental") for sample in cfg.experimentals
    ]
    records = []
    for sample, group in samples:
        frame, summary = _load_sample(sample, group)
        records.append((sample.name, frame, summary))
    signature, features = _validate_compatibility(records)
    labelled, selection = _fit_joint(records, features, signature, cfg, cfg.output_dir)
    joint_nuclei_dir = cfg.output_dir / "joint_nuclei"
    joint_nuclei_dir.mkdir(exist_ok=True)
    windows = []
    for (name, _, summary), frame in zip(records, labelled):
        run_cfg = summary["configuration"]
        window = aggregate_windows(
            frame,
            image_shape=tuple(summary["image_shape_level0"]),
            window_size=run_cfg["window_size_px"],
            stride=run_cfg["stride_px"],
            minimum_nuclei=run_cfg.get("minimum_nuclei_per_cluster_window", 1),
        )
        window["sample"] = name
        window["group"] = frame["group"].iloc[0]
        windows.append(window)
        frame.to_csv(joint_nuclei_dir / f"{name}.csv", index=False)
    combined_windows = pd.concat(windows, ignore_index=True)
    combined_windows.to_csv(
        cfg.output_dir / "joint_window_phenotype_composition.csv", index=False
    )
    results = _summary_table(labelled, windows, cfg.control.name)
    results.to_csv(cfg.output_dir / "comparison_results.csv", index=False)
    _overview(results, cfg.control.name, cfg.output_dir / "comparison_overview.png")

    for (name, _, summary), frame, window in zip(records, labelled, windows):
        run_cfg = summary["configuration"]
        slide = SlideReader(run_cfg["input_qptiff"], run_cfg["channel"])
        preview = slide.read_level(run_cfg["preview_level"])
        dominant = window.rename(columns={"dominant_phenotype": "cluster"})
        render_cluster_overlay(
            preview, dominant, image_shape=tuple(summary["image_shape_level0"]),
            stride=run_cfg["stride_px"], alpha=cfg.overlay_alpha,
            path=overlay_dir / f"{name}.png",
        )
        if cfg.render_joint_segmentation:
            render_segmentation_ometiff(
                slide, Path(run_cfg["output_dir"]) / "chunks",
                overlay_dir / f"{name}.ome.tif", preview=preview,
                chunk_size=run_cfg["chunk_size_px"],
                outline_width=run_cfg.get("segmentation_outline_width", 2),
                nuclei=frame,
                fill_alpha=run_cfg.get("segmentation_fill_alpha", .35),
            )
    comparison_summary = {
        "control": cfg.control.name,
        "experimentals": [sample.name for sample in cfg.experimentals],
        "compatibility_signature": signature,
        "features": features,
        "cluster_selection": selection,
    }
    (cfg.output_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2), encoding="utf-8"
    )
    LOG.info("Comparison complete: %s", cfg.output_dir)
    return cfg.output_dir
