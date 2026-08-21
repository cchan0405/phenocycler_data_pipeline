from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap

from .render import COLORS


def main() -> None:
    st.set_page_config(page_title="DAPI Grid Viewer", layout="wide")
    default = sys.argv[-1] if len(sys.argv) > 1 and not sys.argv[-1].startswith("-") else ""
    results_dir = Path(st.sidebar.text_input("Results folder", value=default))
    st.title("Whole-tissue DAPI nuclear-shape clusters")
    needed = results_dir / "grid_clusters.csv"
    if not needed.exists():
        st.info("Enter a completed results folder.")
        return
    grid = pd.read_csv(needed)
    meta = json.loads((results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    overlay_options = {
        "Confidence-weighted clusters": "whole_tissue_cluster_overlay.png",
        "Unweighted clusters": "whole_tissue_unweighted_overlay.png",
        "Prediction confidence": "whole_tissue_confidence_overlay.png",
        "Training versus predicted": "whole_tissue_training_vs_predicted.png",
    }
    overlay_name = st.sidebar.selectbox("Saved overlay", list(overlay_options))
    overlay = results_dir / overlay_options[overlay_name]
    if overlay.exists():
        st.image(str(overlay), caption=overlay_name, use_container_width=True)

    clusters = sorted(grid["cluster"].unique())
    selected = st.sidebar.multiselect("Visible clusters", clusters, default=clusters)
    alpha = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)
    subset = grid[grid["cluster"].isin(selected)]
    gh = int(np.ceil(meta["level0_shape"][0] / meta["grid_size_px"]))
    gw = int(np.ceil(meta["level0_shape"][1] / meta["grid_size_px"]))
    arr = np.full((gh, gw), np.nan)
    for row in subset.itertuples():
        arr[int(row.grid_row), int(row.grid_col)] = row.cluster
    dapi = plt.imread(results_dir / "dapi_preview.png")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(dapi, cmap="gray", extent=[0, gw, gh, 0])
    cmap = ListedColormap(COLORS[: max(clusters) + 1])
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-0.5, max(clusters) + 1.5), max(clusters) + 1)
    ax.imshow(arr, cmap=cmap, norm=norm, alpha=alpha, extent=[0, gw, gh, 0])
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Valid grid squares", f"{len(grid):,}")
    c2.metric("Clusters", len(clusters))
    c3.metric("Median nuclei/grid", f"{grid['n_nuclei'].median():.0f}")
    if "prediction_status" in grid:
        st.subheader("Prediction reliability")
        st.dataframe(
            grid["prediction_status"].value_counts().rename_axis("status").reset_index(name="windows"),
            use_container_width=True,
        )
    st.subheader("Cluster summary")
    st.dataframe(
        grid.groupby("cluster")
        .agg(grids=("cluster", "size"), median_nuclei=("n_nuclei", "median"))
        .reset_index(),
        use_container_width=True,
    )
    st.download_button(
        "Download grid table",
        grid.to_csv(index=False),
        "grid_clusters.csv",
        "text/csv",
    )


main()
