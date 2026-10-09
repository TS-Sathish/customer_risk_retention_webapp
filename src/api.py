"""FastAPI interface for the customer risk analysis project."""

import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import LOW_RISK_MAX, MEDIUM_RISK_MAX, MODEL_DIR, REPORT_DIR
from .data import ID_COL, TARGET, dataset_summary, load_data, segmentation_frame, split_features
from .explain import explain_global, explain_one
from .narrative import generate_narrative
from .storage import recent_predictions, save_prediction

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Customer Risk & Retention Intelligence API", version="1.0.0")
pipeline = baseline = segments = customers = None
metrics_report: dict = {}
selected_threshold = 0.5


class PredictionRequest(BaseModel):
    customerID: str = Field(min_length=1, max_length=64, description="Known customer identifier")


def risk_label(probability: float) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    return "Low" if probability < LOW_RISK_MAX else ("Medium" if probability < MEDIUM_RISK_MAX else "High")


def _require_models() -> None:
    if pipeline is None or customers is None or segments is None:
        raise HTTPException(status_code=503, detail="Models are unavailable. Run `python -m src.train` first.")


def _customer(customer_id: str) -> pd.DataFrame:
    _require_models()
    found = customers.loc[customers[ID_COL].astype(str) == customer_id]
    if found.empty:
        raise HTTPException(status_code=404, detail="Customer ID not found")
    return found.iloc[[0]]


@app.on_event("startup")
def startup() -> None:
    global pipeline, baseline, segments, customers, metrics_report, selected_threshold
    try:
        pipeline = joblib.load(MODEL_DIR / "churn_pipeline.joblib")
        segment_artifact = joblib.load(MODEL_DIR / "segment_model.joblib")
        baseline_path = MODEL_DIR / "logistic_pipeline.joblib"
        baseline = joblib.load(baseline_path) if baseline_path.exists() else None
        segments = segment_artifact
        customers = load_data()
        metrics_path = REPORT_DIR / "metrics.json"
        if metrics_path.exists():
            metrics_report = json.loads(metrics_path.read_text(encoding="utf-8"))
            selected_name = metrics_report.get("selected_model", "xgboost")
            selected_threshold = float(metrics_report.get("models", {}).get(selected_name, {}).get("validation_threshold", 0.5))
        LOGGER.info("Models and %d customer rows loaded", len(customers))
    except FileNotFoundError:
        LOGGER.warning("Model artifacts not found; train with `python -m src.train`")
        pipeline = baseline = segments = customers = None


@app.get("/")
def root() -> dict:
    return {"message": "Customer Risk & Retention Intelligence API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if pipeline is not None else "not_ready", "models_loaded": pipeline is not None}


@app.get("/dataset/summary")
def get_dataset_summary() -> dict:
    _require_models()
    return dataset_summary(customers)


@app.get("/metrics")
def get_metrics() -> dict:
    if not metrics_report:
        raise HTTPException(status_code=404, detail="Evaluation report is unavailable; run training first.")
    return metrics_report


@app.get("/portfolio/summary")
def portfolio_summary() -> dict:
    _require_models()
    features, labels = split_features(customers)
    probabilities = pipeline.predict_proba(features)[:, 1]
    risks = pd.Series([risk_label(float(value)) for value in probabilities]).value_counts()
    segment_values = segments["imputer"].transform(
        segmentation_frame(customers)[segments["columns"]])
    assignments = segments["model"].predict(segments["scaler"].transform(segment_values))
    segment_counts = pd.Series(assignments).value_counts().sort_index()
    return {
        "customer_count": int(len(customers)),
        "historical_churn_rate": float(labels.mean()),
        "historical_churn_counts": {"No": int((labels == 0).sum()), "Yes": int((labels == 1).sum())},
        "predicted_risk_counts": {risk: int(risks.get(risk, 0)) for risk in ["Low", "Medium", "High"]},
        "segment_counts": {str(segment): int(count) for segment, count in segment_counts.items()},
    }


@app.get("/customers")
def list_customers(limit: int = Query(default=100, ge=1, le=7043)) -> list[dict]:
    _require_models()
    records = customers.head(limit).astype(object).where(pd.notna(customers.head(limit)), None)
    return records.to_dict(orient="records")


@app.get("/customers/{customer_id}")
def customer_lookup(customer_id: str) -> dict:
    row = _customer(customer_id).iloc[0]
    return row.astype(object).where(pd.notna(row), None).to_dict()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    global selected_threshold
    row = _customer(request.customerID)
    features = row.drop(columns=[TARGET, ID_COL])
    probability = float(pipeline.predict_proba(features)[0, 1])
    risk = risk_label(probability)
    explanation = explain_one(pipeline, features)
    segment_row = segmentation_frame(row)[segments["columns"]]
    segment_values = segments["imputer"].transform(segment_row)
    segment = int(segments["model"].predict(segments["scaler"].transform(segment_values))[0])
    narrative = generate_narrative(probability, risk, explanation["top_factors"], segment)
    try:
        save_prediction(request.customerID, probability, risk, segment, explanation, narrative)
    except Exception:
        LOGGER.exception("Could not persist prediction history")
    return {"customerID": request.customerID, "churn_probability": probability, "risk": risk,
            "threshold": selected_threshold, "predicted_churn": bool(probability >= selected_threshold),
            "segment": segment, "historical_churn": int(row[TARGET].iloc[0]),
            "explanation": explanation, "narrative": narrative}


@app.get("/segments")
def segment_summary() -> dict:
    _require_models()
    return {"selected_k": int(segments["model"].n_clusters), "evaluation": segments["evaluation"],
            "profiles": segments["profiles"], "features": segments["columns"]}


@app.get("/segments/{customer_id}")
def customer_segment(customer_id: str) -> dict:
    row = _customer(customer_id)
    values = segmentation_frame(row)[segments["columns"]]
    transformed = segments["scaler"].transform(segments["imputer"].transform(values))
    return {"customerID": customer_id, "segment": int(segments["model"].predict(transformed)[0])}


@app.get("/explanations/global")
def global_explanation() -> dict:
    _require_models()
    features, _ = split_features(customers)
    try:
        return explain_global(pipeline, features)
    except Exception as error:
        LOGGER.exception("Global SHAP explanation failed")
        raise HTTPException(status_code=500, detail="Global explanation could not be generated") from error


@app.get("/explanations/{customer_id}")
def customer_explanation(customer_id: str) -> dict:
    row = _customer(customer_id)
    features = row.drop(columns=[TARGET, ID_COL])
    try:
        return explain_one(pipeline, features)
    except Exception as error:
        LOGGER.exception("Local SHAP explanation failed")
        raise HTTPException(status_code=500, detail="Customer explanation could not be generated") from error


@app.get("/predictions/recent")
def history(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    return recent_predictions(limit)
