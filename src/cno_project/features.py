"""Feature engineering functions."""

import numpy as np
import pandas as pd

from cno_project.config import KAPPA, N_LAGS


def make_monthly_lags(df, value_col, lags=N_LAGS):
    """Create monthly lag features."""
    output = df[["date", value_col]].copy()

    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output[value_col] = pd.to_numeric(output[value_col], errors="coerce")
    output["date"] = output["date"].dt.to_period("M").dt.to_timestamp()

    output = (
        output.groupby("date", as_index=False)[value_col]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )

    for lag in range(1, lags + 1):
        output[f"{value_col}_lag{lag}"] = output[value_col].shift(lag)

    lag_cols = [f"{value_col}_lag{lag}" for lag in range(1, lags + 1)]

    return output[["date"] + lag_cols]


def build_model_dataset(cleaned):
    """Build final monthly modeling dataset."""
    model_df = cleaned["cno"].copy()

    feature_tables = [
        make_monthly_lags(cleaned["cpo"], "cpo_price", lags=4),
        make_monthly_lags(cleaned["usd_php"], "usd_php", lags=4),
        make_monthly_lags(cleaned["oni"], "oni", lags=1),
        make_monthly_lags(cleaned["copra"], "copra_farmgate", lags=4),
        make_monthly_lags(cleaned["production"], "coconut_prod", lags=6),
    ]

    for feature_table in feature_tables:
        model_df = model_df.merge(feature_table, on="date", how="left")

    model_df["copra_usd_mt"] = (
        model_df["copra_farmgate_lag1"] * 1000 / model_df["usd_php_lag1"]
    )

    model_df["scfm"] = (
        model_df["cno_price"].shift(1)
        - model_df["copra_usd_mt"] / KAPPA
    )

    model_df["scfm_lag1"] = model_df["scfm"].shift(1)
    model_df["scfm_mean6"] = model_df["scfm"].shift(1).rolling(6).mean()
    model_df["scfm_dev"] = model_df["scfm"] - model_df["scfm_mean6"]

    model_df["target_next_cno_log_return"] = np.log(
        model_df["cno_price"].shift(-1) / model_df["cno_price"]
    )

    model_df["target_next_cno_direction"] = (
        model_df["target_next_cno_log_return"] > 0
    ).astype(int)

    model_df = model_df.drop(columns=["copra_usd_mt"])
    model_df = model_df.dropna().sort_values("date").reset_index(drop=True)

    return model_df


def get_feature_columns(model_df):
    """Return feature columns with and without SCFM variables."""
    target_cols = {
        "date",
        "cno_price",
        "target_next_cno_log_return",
        "target_next_cno_direction",
    }

    feature_cols = [
        col for col in model_df.columns
        if col not in target_cols
    ]

    feature_cols_no_scfm = [
        col for col in feature_cols
        if "scfm" not in col
    ]

    return feature_cols, feature_cols_no_scfm