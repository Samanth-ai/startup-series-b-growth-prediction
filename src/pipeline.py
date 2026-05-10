"""End-to-end pipeline: load data, train all models, write artifacts."""

import json
import shutil

import pandas as pd

from .data import load_and_prepare
from .modeling import train_for_spec
from .plotting import (
    save_confusion_and_importance,
    save_corr,
    save_missingness,
    save_model_comparisons,
    save_roc_pr,
    save_success_by_founding_year,
    save_target_distribution,
    save_top_markets,
)
from .reporting import export_tables, write_summary
from .shap_utils import make_shap_beeswarm


PAPER_TABLES = [
    "paper_results_table.csv",
    "best_model_by_split_spec.csv",
    "dataset_summary.csv",
    "dataset_metadata.json",
    "metrics.csv",
    "thresholds.csv",
    "split_summary.csv",
]
PAPER_FIGURES = [
    "target_distribution.png",
    "success_by_founding_year.png",
    "missingness_top15.png",
    "top_markets.png",
    "correlation_heatmap.png",
    "model_comparison_roc_auc.png",
    "model_comparison_pr_auc.png",
    "model_comparison_f1.png",
]


def _descriptive_artifacts(df, figs_dir, tables_dir):
    save_target_distribution(df, figs_dir / "target_distribution.png")
    save_success_by_founding_year(df, figs_dir / "success_by_founding_year.png")
    save_missingness(df, figs_dir / "missingness_top15.png",
                     tables_dir / "missingness_top15.csv")
    save_top_markets(df, figs_dir / "top_markets.png",
                     tables_dir / "top_markets.csv")
    save_corr(df, figs_dir / "correlation_heatmap.png",
              tables_dir / "correlation_matrix.csv")


def _train_everything(df, feature_specs, random_state):
    results = []
    for spec_name, spec in feature_specs.items():
        for split in ("random", "temporal"):
            print(f"[train] spec={spec_name} split={split}")
            results.extend(
                train_for_spec(df, spec, split=split,
                               spec_name=spec_name, random_state=random_state)
            )
    return results


def _shap_for_best(results, metrics_df, figs_dir, max_samples):
    picks = set()
    for metric in ("f1", "pr_auc"):
        picks.add(metrics_df[metric].idxmax())

    for idx in picks:
        row = metrics_df.loc[idx]
        match = next(
            (r for r in results
             if r.model_name == row["model"]
             and r.split == row["split"]
             and r.spec == row["spec"]),
            None,
        )
        if match is not None:
            make_shap_beeswarm(match, figs_dir, max_samples=max_samples)


def _copy_paper_snapshot(metrics_df, figs_dir, tables_dir, paper_dir):
    files_tables = list(PAPER_TABLES)
    files_figs = list(PAPER_FIGURES)

    best = metrics_df.sort_values("f1", ascending=False).iloc[0]
    stem = f"{best['model']}_{best['split']}_{best['spec']}"
    files_figs.append(f"confusion_{stem}.png")
    files_figs.append(f"importance_{stem}.png")

    for sf in figs_dir.glob("shap_beeswarm_*.png"):
        files_figs.append(sf.name)

    for name in files_tables:
        src = tables_dir / name
        if src.exists():
            shutil.copy2(src, paper_dir / name)

    for name in files_figs:
        src = figs_dir / name
        if src.exists():
            shutil.copy2(src, paper_dir / name)


def run_pipeline(csv_path, root, encoding="latin1", random_state=42,
                 skip_shap=False, max_shap_samples=300):
    artifacts = root / "artifacts"
    figs_dir = artifacts / "figures"
    tables_dir = artifacts / "tables"
    reports_dir = root / "reports"
    paper_dir = reports_dir / "paper_artifacts"
    for d in (figs_dir, tables_dir, paper_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {csv_path}")
    prepared = load_and_prepare(csv_path, encoding=encoding)
    print(f"[load] eligible rows: {prepared.metadata['eligible_rows']}, "
          f"positive rate: {prepared.metadata['positive_rate']:.3%}")

    _descriptive_artifacts(prepared.df, figs_dir, tables_dir)

    results = _train_everything(prepared.df, prepared.feature_specs, random_state)

    metrics_df = pd.DataFrame([r.metrics for r in results])
    thresholds_df = pd.DataFrame([r.thresholds for r in results])

    export_tables(prepared.metadata, metrics_df, thresholds_df, tables_dir)
    write_summary(prepared.metadata, metrics_df, reports_dir)

    save_roc_pr(results, figs_dir)
    save_confusion_and_importance(results, figs_dir, tables_dir)
    save_model_comparisons(metrics_df, figs_dir)

    split_summary = (
        metrics_df.groupby(["split", "spec"])
        .agg(best_f1=("f1", "max"),
             best_pr_auc=("pr_auc", "max"),
             best_roc_auc=("roc_auc", "max"))
        .reset_index()
    )
    split_summary.to_csv(tables_dir / "split_summary.csv", index=False)

    if not skip_shap:
        _shap_for_best(results, metrics_df, figs_dir, max_shap_samples)

    _copy_paper_snapshot(metrics_df, figs_dir, tables_dir, paper_dir)

    manifest = {
        "tables": sorted(p.name for p in tables_dir.glob("*")),
        "figures": sorted(p.name for p in figs_dir.glob("*.png")),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[done]")
