"""Dimensionality reduction functions."""

import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from cno_project.config import SEED


def apply_reduction(x_train, x_test, method=None, n_components=5):
    """Scale data and optionally apply PCA or SVD."""
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    if method is None:
        return x_train_scaled, x_test_scaled

    max_components = min(
        n_components,
        x_train_scaled.shape[0],
        x_train_scaled.shape[1],
    )

    if method == "pca":
        reducer = PCA(n_components=max_components, random_state=SEED)
    elif method == "svd":
        reducer = TruncatedSVD(n_components=max_components, random_state=SEED)
    else:
        raise ValueError("method must be None, 'pca', or 'svd'.")

    x_train_reduced = reducer.fit_transform(x_train_scaled)
    x_test_reduced = reducer.transform(x_test_scaled)

    return x_train_reduced, x_test_reduced


def pca_summary(model_df, feature_cols, n_components=5):
    """Create PCA explained variance summary."""
    x_values = model_df[feature_cols].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_values)

    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(x_scaled)

    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": pca.explained_variance_ratio_.cumsum(),
        }
    )