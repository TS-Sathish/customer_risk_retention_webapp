"""Train churn classifiers and customer segments from the bundled snapshot."""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_DIR, RANDOM_STATE, REPORT_DIR
from .data import ID_COL, TARGET, dataset_summary, load_data, segmentation_frame, split_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("No usable model features found")
    return ColumnTransformer(transformers, remainder="drop")


def build_models(features: pd.DataFrame) -> dict[str, Pipeline]:
    from xgboost import XGBClassifier

    models = {
        "xgboost": XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.9,
            colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.0,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4,
        ),
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
    }
    return {name: Pipeline([("preprocessor", build_preprocessor(features)), ("model", model)])
            for name, model in models.items()}


def choose_threshold(labels: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(labels, probabilities >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_metrics(labels: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "test_rows": int(len(labels)),
    }


def train_segments(df: pd.DataFrame) -> dict:
    raw = segmentation_frame(df)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputer.fit_transform(raw))
    upper_k = min(8, max(2, len(df) - 1))
    candidates = range(2, upper_k + 1)
    evaluations = []
    fitted = []
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(scaled)
        evaluations.append({"k": k, "inertia": float(model.inertia_),
                            "silhouette": float(silhouette_score(scaled, labels))})
        fitted.append((model, labels))
    best_index = int(np.argmax([row["silhouette"] for row in evaluations]))
    model, labels = fitted[best_index]
    profiles = raw.assign(segment=labels).groupby("segment").mean(numeric_only=True).round(2)
    result = {"columns": raw.columns.tolist(), "imputer": imputer, "scaler": scaler,
              "model": model, "evaluation": evaluations,
              "profiles": {str(index): row for index, row in profiles.to_dict(orient="index").items()}}
    return result


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = split_features(df)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, stratify=y_holdout, random_state=RANDOM_STATE)
    scores = {}
    fitted_models = {}
    for name, pipeline in build_models(X).items():
        LOGGER.info("Fitting %s", name)
        pipeline.fit(X_train, y_train)
        validation_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold(y_valid, validation_probabilities)
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        scores[name] = {"validation_threshold": threshold,
                        "validation_f1": float(f1_score(y_valid, validation_probabilities >= threshold,
                                                         zero_division=0)),
                        "test": classification_metrics(y_test, test_probabilities, threshold)}
        fitted_models[name] = pipeline

    selected_model = max(scores, key=lambda name: scores[name]["validation_f1"])
    joblib.dump(fitted_models[selected_model], MODEL_DIR / "churn_pipeline.joblib")
    joblib.dump(fitted_models["logistic_regression"], MODEL_DIR / "logistic_pipeline.joblib")
    segment_artifact = train_segments(df)
    assignments = segment_artifact["model"].predict(
        segment_artifact["scaler"].transform(segment_artifact["imputer"].transform(segmentation_frame(df))))
    joblib.dump(segment_artifact, MODEL_DIR / "segment_model.joblib")
    pd.DataFrame({ID_COL: df[ID_COL], "segment": assignments}).to_csv(
        REPORT_DIR / "customer_segments.csv", index=False)
    summary = dataset_summary(df)
    summary["total_charges_coerced_missing"] = int(df["TotalCharges"].isna().sum())
    metrics = {"dataset": summary, "models": scores, "selected_model": selected_model,
               "splits": {"train": len(y_train), "validation": len(y_valid), "test": len(y_test)},
               "segmentation": {"selected_k": int(segment_artifact["model"].n_clusters),
                                "evaluation": segment_artifact["evaluation"],
                                "profiles": segment_artifact["profiles"]},
               "interpretation": "Metrics use a held-out test split. SHAP values describe model output contributions and are not causal.",
               "threshold_selection": "F1 threshold selected on validation data only."}
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Training complete; reports saved under %s", REPORT_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
