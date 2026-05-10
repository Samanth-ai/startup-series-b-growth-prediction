"""Train the four classifiers across splits and feature specs."""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "target_series_b_36m"
TIME_COL = "first_funding_at"


@dataclass
class FitResult:
    model_name: str
    split: str
    spec: str
    metrics: dict
    thresholds: dict
    feature_names: list
    y_test: np.ndarray
    proba_test: np.ndarray
    preds_test: np.ndarray
    X_test_transformed: np.ndarray
    pipeline: Pipeline
    importance_df: pd.DataFrame = field(default=None)


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def build_models(random_state=42):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            early_stopping=True,
            max_iter=200,
            random_state=random_state,
        ),
    }


def score(y_true, y_pred, proba):
    two_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if two_classes else np.nan,
        "pr_auc": average_precision_score(y_true, proba) if two_classes else np.nan,
    }


def pick_threshold_by_f1(y_val, proba_val):
    """Sweep thresholds in 0.01 steps and return the one that maximises F1."""
    best_thr = 0.5
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    for thr in np.arange(0.01, 1.00, 0.01):
        preds = (proba_val >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = round(float(thr), 2)
            best_p = float(precision_score(y_val, preds, zero_division=0))
            best_r = float(recall_score(y_val, preds, zero_division=0))
    return best_thr, best_f1, best_p, best_r


def feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols):
    names = list(numeric_cols)
    if not categorical_cols:
        return names
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    names.extend(ohe.get_feature_names_out(categorical_cols).tolist())
    return names


def compute_permutation_importance(pipe, X_test, y_test, top_k=15):
    """Permutation importance over the raw input columns (not the one-hot
    expansion). Runs the full Pipeline so sklearn permutes the columns the
    user sees."""
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
    )
    imp = pd.DataFrame({
        "feature": list(X_test.columns),
        "importance": result.importances_mean,
    })
    return imp.sort_values("importance", ascending=False).head(top_k)


def _make_split(df, split, random_state):
    if split == "random":
        return train_test_split(
            df, test_size=0.2, stratify=df[TARGET], random_state=random_state
        )
    if split == "temporal":
        ordered = df.sort_values(TIME_COL)
        n_test = int(np.ceil(len(ordered) * 0.2))
        return ordered.iloc[:-n_test].copy(), ordered.iloc[-n_test:].copy()
    raise ValueError(f"unknown split: {split}")


def train_for_spec(df, feature_spec, split, spec_name, random_state=42):
    numeric_cols = feature_spec["numeric"]
    categorical_cols = feature_spec["categorical"]
    use_cols = numeric_cols + categorical_cols

    work = df[use_cols + [TARGET, TIME_COL]].copy()
    train_df, test_df = _make_split(work, split, random_state)

    X_train = train_df[use_cols]
    y_train = train_df[TARGET].to_numpy()
    X_test = test_df[use_cols]
    y_test = test_df[TARGET].to_numpy()

    # Validation slice inside training, used for threshold tuning.
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    results = []
    for name, estimator in build_models(random_state=random_state).items():
        pre = build_preprocessor(numeric_cols, categorical_cols)
        pipe = Pipeline([("preprocessor", pre), ("model", estimator)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_sub, y_sub)

        proba_val = pipe.predict_proba(X_val)[:, 1]
        thr, thr_f1, thr_p, thr_r = pick_threshold_by_f1(y_val, proba_val)

        # Refit on sub + val before scoring on the test set.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        proba_test = pipe.predict_proba(X_test)[:, 1]
        preds_test = (proba_test >= thr).astype(int)
        metrics = score(y_test, preds_test, proba_test)
        metrics.update({
            "split": split,
            "spec": spec_name,
            "model": name,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "threshold": thr,
        })

        feature_names = feature_names_from_preprocessor(
            pipe.named_steps["preprocessor"], numeric_cols, categorical_cols
        )
        importance_df = compute_permutation_importance(pipe, X_test, y_test)

        results.append(FitResult(
            model_name=name,
            split=split,
            spec=spec_name,
            metrics=metrics,
            thresholds={
                "split": split,
                "spec": spec_name,
                "model": name,
                "best_f1_threshold": thr,
                "best_f1": thr_f1,
                "best_precision": thr_p,
                "best_recall": thr_r,
            },
            feature_names=feature_names,
            y_test=y_test,
            proba_test=proba_test,
            preds_test=preds_test,
            X_test_transformed=pipe.named_steps["preprocessor"].transform(X_test),
            pipeline=pipe,
            importance_df=importance_df,
        ))

    return results
