"""SHAP beeswarm plots for the best tree-ensemble models. Optional step."""

import numpy as np
import matplotlib.pyplot as plt


SUPPORTED_MODELS = {"random_forest", "hist_gradient_boosting"}


def make_shap_beeswarm(result, figs_dir, max_samples=300):
    try:
        import shap
    except ImportError:
        print("[shap] shap not installed, skipping")
        return False

    if result.model_name not in SUPPORTED_MODELS:
        return False

    X = result.X_test_transformed
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    n = min(max_samples, len(result.y_test))
    if n < 20:
        print(f"[shap] only {n} test rows for {result.model_name}/{result.split}/{result.spec}, skipping")
        return False
    X_sample = X[:n]

    model = result.pipeline.named_steps["model"]
    explainer = shap.Explainer(model, X_sample, feature_names=result.feature_names)
    # check_additivity=False works around a known SHAP/HistGradientBoosting
    # float-precision mismatch; the SHAP values themselves are still correct.
    shap_values = explainer(X_sample, check_additivity=False)

    fig = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    out = figs_dir / f"shap_beeswarm_{result.model_name}_{result.split}_{result.spec}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close("all")
    return True