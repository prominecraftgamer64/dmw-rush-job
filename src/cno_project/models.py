"""Modeling and evaluation functions."""

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cno_project.config import SEED, TRAIN_SIZE
from cno_project.dimensionality import apply_reduction


REGRESSION_MODEL_TYPES = [
    "linear",
    "ridge",
    "lasso",
    "knn_reg",
    "svm_reg",
    "dt_reg",
    "rf_reg",
    "gbm_reg",
]

CLASSIFICATION_MODEL_TYPES = [
    "logistic",
    "knn_clf",
    "svm_clf",
    "dt_clf",
    "rf_clf",
    "gbm_clf",
]

MODEL_LABELS = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "knn_reg": "KNN Regressor",
    "svm_reg": "SVM Regressor",
    "dt_reg": "Decision Tree Regressor",
    "rf_reg": "Random Forest Regressor",
    "gbm_reg": "GBM Regressor",
    "logistic": "Logistic Regression",
    "knn_clf": "KNN Classifier",
    "svm_clf": "SVM Classifier",
    "dt_clf": "Decision Tree Classifier",
    "rf_clf": "Random Forest Classifier",
    "gbm_clf": "GBM Classifier",
}


def get_regressor(model_type):
    """Return a regression model."""
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01, max_iter=10000),
        "knn_reg": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "svm_reg": SVR(kernel="rbf", C=1.0, epsilon=0.01),
        "dt_reg": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        ),
        "rf_reg": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        ),
        "gbm_reg": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            random_state=SEED,
        ),
    }

    return models[model_type]


def get_classifier(model_type):
    """Return a classification model."""
    models = {
        "logistic": LogisticRegression(max_iter=10000),
        "knn_clf": KNeighborsClassifier(n_neighbors=5),
        "svm_clf": SVC(kernel="rbf", C=1.0),
        "dt_clf": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        ),
        "rf_clf": RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        ),
        "gbm_clf": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=SEED,
        ),
    }

    return models[model_type]


def rolling_regression_backtest(
    df,
    feature_cols,
    model_type,
    target_col="target_next_cno_log_return",
    train_size=TRAIN_SIZE,
    reduction=None,
):
    """Run rolling-window regression backtest."""
    rows = []

    for start in range(0, len(df) - train_size):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + 1]

        x_train = train[feature_cols].values
        x_test = test[feature_cols].values
        y_train = train[target_col].values

        x_train, x_test = apply_reduction(
            x_train,
            x_test,
            method=reduction,
        )

        model = get_regressor(model_type)
        model.fit(x_train, y_train)

        predicted = model.predict(x_test)[0]
        actual = test[target_col].iloc[0]

        rows.append(
            {
                "date": test["date"].iloc[0],
                "actual": actual,
                "predicted": predicted,
                "actual_direction": int(actual > 0),
                "pred_direction": int(predicted > 0),
                "correct_direction": int((actual > 0) == (predicted > 0)),
            }
        )

    return pd.DataFrame(rows)


def rolling_classification_backtest(
    df,
    feature_cols,
    model_type,
    target_col="target_next_cno_direction",
    train_size=TRAIN_SIZE,
):
    """Run rolling-window classification backtest."""
    rows = []

    for start in range(0, len(df) - train_size):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + 1]

        x_train = train[feature_cols].values
        x_test = test[feature_cols].values
        y_train = train[target_col].values
        y_test = test[target_col].iloc[0]

        x_train, x_test = apply_reduction(x_train, x_test, method=None)

        if len(np.unique(y_train)) < 2:
            predicted = y_train[-1]
        else:
            model = get_classifier(model_type)
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)[0]

        actual_return = test["target_next_cno_log_return"].iloc[0]

        rows.append(
            {
                "date": test["date"].iloc[0],
                "actual": actual_return,
                "actual_direction": int(y_test),
                "pred_direction": int(predicted),
                "predicted": 1 if int(predicted) == 1 else -1,
                "correct_direction": int(predicted == y_test),
            }
        )

    return pd.DataFrame(rows)


def evaluate_regression(backtest, model_name):
    """Evaluate regression backtest."""
    return {
        "model": model_name,
        "DA": backtest["correct_direction"].mean(),
        "RMSE": mean_squared_error(
            backtest["actual"],
            backtest["predicted"],
        ) ** 0.5,
        "MAE": mean_absolute_error(
            backtest["actual"],
            backtest["predicted"],
        ),
        "n_predictions": len(backtest),
    }


def evaluate_classification(backtest, model_name):
    """Evaluate classification backtest."""
    y_true = backtest["actual_direction"]
    y_pred = backtest["pred_direction"]

    return {
        "model": model_name,
        "DA": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_predictions": len(backtest),
    }


def naive_persistence_baseline(df):
    """Create naive baseline from previous target return."""
    output = df[["date", "target_next_cno_log_return"]].copy()

    output["actual"] = output["target_next_cno_log_return"]
    output["predicted"] = output["target_next_cno_log_return"].shift(1)

    output = output.dropna().reset_index(drop=True)

    output["actual_direction"] = (output["actual"] > 0).astype(int)
    output["pred_direction"] = (output["predicted"] > 0).astype(int)
    output["correct_direction"] = (
        output["actual_direction"] == output["pred_direction"]
    ).astype(int)

    return output[
        [
            "date",
            "actual",
            "predicted",
            "actual_direction",
            "pred_direction",
            "correct_direction",
        ]
    ]