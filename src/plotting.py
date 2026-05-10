"""Figure and table output helpers."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


sns.set_theme(style="whitegrid")


def _save(fig, path, dpi=200):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_target_distribution(df, outpath):
    counts = df["target_series_b_36m"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["No Series B<=36m", "Series B<=36m"], counts.values)
    ax.set_title("Target Class Distribution")
    _save(fig, outpath)


def save_success_by_founding_year(df, outpath):
    # Require at least 30 firms per year so the rate isn't dominated by noise.
    clipped = df[df["founded_year"].between(1985, 2020, inclusive="both")].copy()
    grouped = clipped.groupby("founded_year").agg(
        n=("target_series_b_36m", "size"),
        rate=("target_series_b_36m", "mean"),
    ).reset_index()
    grouped = grouped[grouped["n"] >= 30]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grouped["founded_year"], grouped["rate"], marker="o")
    ax.set_title("Success Rate by Founding Year")
    ax.set_ylabel("Series B within 36 months")
    ax.set_xlabel("Founded year")
    _save(fig, outpath)


def save_missingness(df, outpath, csv_out):
    miss = df.isna().mean().sort_values(ascending=False).head(15)
    miss.to_csv(csv_out, header=["missing_rate"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(miss.index[::-1], miss.values[::-1])
    ax.set_title("Top Missingness Rates")
    _save(fig, outpath)


def save_top_markets(df, outpath, csv_out):
    top = df["market"].fillna("Unknown").value_counts().head(10)
    top.to_csv(csv_out, header=["count"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title("Top 10 Markets in Analysis Dataset")
    _save(fig, outpath)


def save_corr(df, outpath, csv_out):
    candidates = [
        "founded_year", "funding_rounds", "funding_total_usd",
        "months_to_first_funding", "seed", "angel", "grant", "convertible_note",
        "debt_financing", "equity_crowdfunding", "category_count",
        "target_series_b_36m",
    ]
    cols = [c for c in candidates if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(csv_out)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    _save(fig, outpath)


def save_roc_pr(results, figs_dir):
    splits = sorted({r.split for r in results})
    specs = sorted({r.spec for r in results})

    for split in splits:
        for spec in specs:
            subset = [r for r in results if r.split == split and r.spec == spec]
            if not subset:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            for r in subset:
                RocCurveDisplay.from_predictions(
                    r.y_test, r.proba_test, name=r.model_name, ax=ax
                )
            ax.set_title(f"ROC Curves ({split}, {spec})")
            _save(fig, figs_dir / f"roc_{split}_{spec}.png")

            fig, ax = plt.subplots(figsize=(8, 6))
            for r in subset:
                PrecisionRecallDisplay.from_predictions(
                    r.y_test, r.proba_test, name=r.model_name, ax=ax
                )
            ax.set_title(f"Precision-Recall Curves ({split}, {spec})")
            _save(fig, figs_dir / f"pr_{split}_{spec}.png")


def save_confusion_and_importance(results, figs_dir, tables_dir):
    for r in results:
        cm = confusion_matrix(r.y_test, r.preds_test)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {r.model_name} ({r.split}, {r.spec})")
        _save(fig, figs_dir / f"confusion_{r.model_name}_{r.split}_{r.spec}.png")

        if r.importance_df is None:
            continue
        r.importance_df.to_csv(
            tables_dir / f"importance_{r.model_name}_{r.split}_{r.spec}.csv",
            index=False,
        )
        show = r.importance_df.sort_values("importance").tail(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(show["feature"], show["importance"])
        ax.set_title(f"Permutation Importance: {r.model_name} ({r.split}, {r.spec})")
        _save(fig, figs_dir / f"importance_{r.model_name}_{r.split}_{r.spec}.png")


def save_model_comparisons(metrics_df, figs_dir):
    label_of = lambda row: f"{row['model']} ({row['split']}, {row['spec']})"
    for metric in ["roc_auc", "pr_auc", "f1"]:
        ordered = metrics_df.sort_values(metric)
        labels = ordered.apply(label_of, axis=1)
        pretty = "F1" if metric == "f1" else metric.upper()

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(labels, ordered[metric])
        ax.set_title(f"{pretty} by Model / Split / Spec")
        ax.set_xlabel(pretty)
        _save(fig, figs_dir / f"model_comparison_{metric}.png")
