"""SHAP explanations with a native XGBoost TreeSHAP compatibility fallback."""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def _tree_contributions(model, transformed):
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).reshape(-1)[-1])
        return values, float(base)
    except (ValueError, TypeError, AttributeError) as error:
        if not hasattr(model, "get_booster"):
            raise
        LOGGER.warning("SHAP tree loader is incompatible with this XGBoost model; using native TreeSHAP: %s", error)
        import xgboost as xgb

        contributions = model.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)
        return contributions[:, :-1], contributions[:, -1]


def explain_one(pipeline, row):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(row)
    values, base = _tree_contributions(model, transformed)
    if values.ndim == 2:
        values = values[0]
    names = preprocessor.get_feature_names_out()
    ranked = sorted(zip(names, values), key=lambda item: abs(float(item[1])), reverse=True)[:8]
    return {
        "base_value": float(np.asarray(base).reshape(-1)[0]),
        "top_factors": [{"feature": str(name), "contribution": float(value)} for name, value in ranked],
        "interpretation": "Feature contributions explain this model output; they do not establish causes.",
    }


def explain_global(pipeline, features, max_rows=500):
    sample = features.sample(min(len(features), max_rows), random_state=42)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(sample)
    values, _ = _tree_contributions(model, transformed)
    names = preprocessor.get_feature_names_out()
    importance = np.abs(values).mean(axis=0)
    ranked = sorted(zip(names, importance), key=lambda item: float(item[1]), reverse=True)[:20]
    return {
        "rows_sampled": int(len(sample)),
        "features": [{"feature": str(name), "mean_absolute_contribution": float(value)}
                     for name, value in ranked],
        "interpretation": "Global SHAP values summarize model contributions, not causal effects.",
    }
