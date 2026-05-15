"""Main end-to-end CNO forecasting pipeline."""

import json

import pandas as pd

from cno_project.cleaning import clean_all
from cno_project.config import (
    BACKTEST_DIR,
    FINAL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_FILES,
    TABLES_DIR,
)
from cno_project.dimensionality import pca_summary
from cno_project.features import build_model_dataset, get_feature_columns
from cno_project.models import (
    CLASSIFICATION_MODEL_TYPES,
    MODEL_LABELS,
    REGRESSION_MODEL_TYPES,
    evaluate_classification,
    evaluate_regression,
    naive_persistence_baseline,
    rolling_classification_backtest,
    rolling_regression_backtest,
)
from cno_project.scraping import scrape_oni_with_bs4


def save_csv(df, path):
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)


def load_raw_data():
    """Load all raw datasets."""
    raw = {}

    for key, filename in RAW_FILES.items():
        path = RAW_DIR / filename

        if key == "oni" and not path.exists():
            raw[key] = scrape_oni_with_bs4()
            save_csv(raw[key], path)
        else:
            raw[key] = pd.read_csv(path)

    return raw


def run_regression_experiments(model_df, feature_cols, feature_cols_no_scfm):
    """Run raw, PCA, and SVD regression experiments."""
    results = {}
    rows = []

    experiments = [
        ("Raw", "with_scfm", feature_cols, None),
        ("Raw", "no_scfm", feature_cols_no_scfm, None),
        ("PCA", "with_scfm", feature_cols, "pca"),
        ("PCA", "no_scfm", feature_cols_no_scfm, "pca"),
        ("SVD", "with_scfm", feature_cols, "svd"),
        ("SVD", "no_scfm", feature_cols_no_scfm, "svd"),
    ]

    for method, feature_set, columns, reduction in experiments:
        for model_type in REGRESSION_MODEL_TYPES:
            result_key = f"{method}_{feature_set}_{model_type}"

            backtest = rolling_regression_backtest(
                df=model_df,
                feature_cols=columns,
                model_type=model_type,
                reduction=reduction,
            )

            results[result_key] = backtest

            metrics = evaluate_regression(
                backtest,
                MODEL_LABELS[model_type],
            )

            metrics["method"] = method
            metrics["feature_set"] = feature_set
            metrics["model_type"] = model_type
            metrics["result_key"] = result_key

            rows.append(metrics)

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["DA", "RMSE"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return results, comparison


def run_classification_experiments(
    model_df,
    feature_cols,
    feature_cols_no_scfm,
):
    """Run classification experiments."""
    results = {}
    rows = []

    experiments = [
        ("Raw", "with_scfm", feature_cols),
        ("Raw", "no_scfm", feature_cols_no_scfm),
    ]

    for method, feature_set, columns in experiments:
        for model_type in CLASSIFICATION_MODEL_TYPES:
            result_key = f"{method}_{feature_set}_{model_type}"

            backtest = rolling_classification_backtest(
                df=model_df,
                feature_cols=columns,
                model_type=model_type,
            )

            results[result_key] = backtest

            metrics = evaluate_classification(
                backtest,
                MODEL_LABELS[model_type],
            )

            metrics["method"] = method
            metrics["feature_set"] = feature_set
            metrics["model_type"] = model_type
            metrics["result_key"] = result_key

            rows.append(metrics)

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["DA", "balanced_accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return results, comparison


def run_full_pipeline():
    """Run the full CNO project pipeline."""
    raw = load_raw_data()
    cleaned = clean_all(raw)

    for name, df in cleaned.items():
        save_csv(df, PROCESSED_DIR / f"{name}_cleaned.csv")

    model_df = build_model_dataset(cleaned)
    save_csv(model_df, FINAL_DIR / "df_model_final.csv")

    feature_cols, feature_cols_no_scfm = get_feature_columns(model_df)

    pca_table = pca_summary(model_df, feature_cols)
    save_csv(pca_table, TABLES_DIR / "pca_summary.csv")

    regression_results, regression_comparison = run_regression_experiments(
        model_df,
        feature_cols,
        feature_cols_no_scfm,
    )

    classification_results, classification_comparison = (
        run_classification_experiments(
            model_df,
            feature_cols,
            feature_cols_no_scfm,
        )
    )

    for key, df in regression_results.items():
        save_csv(df, BACKTEST_DIR / f"regression_{key}.csv")

    for key, df in classification_results.items():
        save_csv(df, BACKTEST_DIR / f"classification_{key}.csv")

    naive = naive_persistence_baseline(model_df)

    save_csv(naive, TABLES_DIR / "naive_persistence_baseline.csv")
    save_csv(regression_comparison, TABLES_DIR / "regression_comparison.csv")
    save_csv(
        classification_comparison,
        TABLES_DIR / "classification_comparison.csv",
    )

    best_row = regression_comparison.iloc[0]

    summary = {
        "rows": len(model_df),
        "columns": list(model_df.columns),
        "feature_count_with_scfm": len(feature_cols),
        "feature_count_without_scfm": len(feature_cols_no_scfm),
        "baseline_da": naive["correct_direction"].mean(),
        "best_model": best_row["model"],
        "best_method": best_row["method"],
        "best_feature_set": best_row["feature_set"],
        "best_da": best_row["DA"],
        "best_rmse": best_row["RMSE"],
        "best_mae": best_row["MAE"],
        "note": "This project uses local public CSV datasets and ONI web scraping.",
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)

    print("=" * 70)
    print("CNO PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Model dataset shape: {model_df.shape}")
    print(f"Features with SCFM: {len(feature_cols)}")
    print(f"Features without SCFM: {len(feature_cols_no_scfm)}")
    print(f"Baseline DA: {summary['baseline_da']:.2%}")
    print(f"Best model: {summary['best_model']}")
    print(f"Best method: {summary['best_method']}")
    print(f"Best feature set: {summary['best_feature_set']}")
    print(f"Best DA: {summary['best_da']:.2%}")
    print(f"Outputs saved to: {OUTPUT_DIR}")

    return {
        "raw": raw,
        "cleaned": cleaned,
        "model_df": model_df,
        "feature_cols": feature_cols,
        "feature_cols_no_scfm": feature_cols_no_scfm,
        "regression_results": regression_results,
        "classification_results": classification_results,
        "regression_comparison": regression_comparison,
        "classification_comparison": classification_comparison,
        "naive": naive,
        "summary": summary,
    }


if __name__ == "__main__":
    run_full_pipeline()