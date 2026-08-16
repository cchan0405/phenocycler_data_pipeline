from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PHENOTYPE_FEATURES = [
    "area",
    "circularity",
    "eccentricity",
    "solidity",
    "aspect_ratio",
]


def assign_nuclear_phenotypes(nuclei: pd.DataFrame, cfg):
    """Fit morphology-only nuclear phenotypes and label every nucleus."""
    result = nuclei.copy()
    if not cfg.enabled or len(result) < cfg.n_phenotypes:
        result["nuclear_phenotype"] = 0
        return result, None

    xdf = result[PHENOTYPE_FEATURES].copy()
    xdf["area"] = np.log1p(xdf["area"].clip(lower=0))
    preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    x = preprocess.fit_transform(xdf)
    rng = np.random.default_rng(cfg.random_seed)
    if len(x) > cfg.fit_sample:
        fit_idx = rng.choice(len(x), cfg.fit_sample, replace=False)
    else:
        fit_idx = np.arange(len(x))

    model = MiniBatchKMeans(
        n_clusters=cfg.n_phenotypes,
        random_state=cfg.random_seed,
        batch_size=cfg.minibatch_size,
        n_init=10,
    )
    model.fit(x[fit_idx])
    result["nuclear_phenotype"] = model.predict(x).astype(int)
    artifact = {
        "features": PHENOTYPE_FEATURES,
        "preprocess": preprocess,
        "cluster_model": model,
    }
    return result, artifact
