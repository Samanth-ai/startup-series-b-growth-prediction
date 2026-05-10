"""Write out the tables and the short human-readable summary."""

import json

import pandas as pd


METRIC_COLS = [
    "split", "spec", "model", "threshold",
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
]


def export_tables(metadata, metrics_df, thresholds_df, tables_dir):
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Metadata goes to both JSON (for programs) and a one-row CSV (for humans).
    with open(tables_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    pd.DataFrame([metadata]).to_csv(tables_dir / "dataset_summary.csv", index=False)

    metrics_df.to_csv(tables_dir / "metrics.csv", index=False)
    thresholds_df.to_csv(tables_dir / "thresholds.csv", index=False)

    # Compact table that goes into the paper.
    compact = (
        metrics_df[METRIC_COLS]
        .sort_values(["split", "spec", "f1"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    compact.to_csv(tables_dir / "paper_results_table.csv", index=False)
    with open(tables_dir / "metrics.md", "w") as f:
        f.write(compact.to_markdown(index=False))

    # Best model per (split, spec) cell by each headline metric.
    best_rows = []
    for (split, spec), group in metrics_df.groupby(["split", "spec"]):
        best_rows.append({
            "split": split,
            "spec": spec,
            "best_model_by_f1": group.loc[group["f1"].idxmax(), "model"],
            "best_model_by_pr_auc": group.loc[group["pr_auc"].idxmax(), "model"],
            "best_model_by_roc_auc": group.loc[group["roc_auc"].idxmax(), "model"],
        })
    pd.DataFrame(best_rows).to_csv(
        tables_dir / "best_model_by_split_spec.csv", index=False
    )


def write_summary(metadata, metrics_df, reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)
    best = metrics_df.sort_values("f1", ascending=False).iloc[0]
    text = (
        f"# Summary\n\n"
        f"Eligible rows: {metadata['eligible_rows']} of {metadata['raw_rows']} raw rows.\n\n"
        f"Positive rate: {metadata['positive_rate']:.3%}.\n\n"
        f"Best overall by F1: {best['model']} on {best['split']} split / {best['spec']} spec.\n\n"
        f"Target definition: {metadata['target_definition']}\n\n"
        f"Limitation: {metadata['important_limitation']}\n"
    )
    (reports_dir / "summary.md").write_text(text)
