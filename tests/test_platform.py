import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import api
from src.data import load_data, split_features
from src.train import build_preprocessor


@pytest.fixture
def sample_frame():
    return pd.DataFrame({
        "customerID": [f"ID{i}" for i in range(8)],
        "Contract": ["Month-to-month", "One year"] * 4,
        "tenure": [1, 12, 3, 24, 5, 36, 8, 48],
        "TotalCharges": [None, 120.0, 90.0, 240.0, 75.0, 360.0, 80.0, 480.0],
        "Churn": ["Yes", "No"] * 4,
    })


def test_loader_validates_ids_and_encodes_target(tmp_path, sample_frame):
    path = tmp_path / "customers.csv"
    sample_frame.to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded["Churn"].tolist() == [1, 0] * 4
    assert loaded["TotalCharges"].isna().sum() == 1
    assert "customerID" not in split_features(loaded)[0]
    assert "Churn" not in split_features(loaded)[0]

    sample_frame.loc[1, "customerID"] = sample_frame.loc[0, "customerID"]
    sample_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_data(path)


def test_preprocessing_handles_missing_values_and_unseen_categories(sample_frame):
    features, labels = split_features(sample_frame.assign(Churn=sample_frame["Churn"].map({"Yes": 1, "No": 0})))
    model = Pipeline([("preprocessor", build_preprocessor(features)),
                      ("model", LogisticRegression(max_iter=500))])
    model.fit(features, labels)
    transformed = model.named_steps["preprocessor"].transform(
        pd.DataFrame({"Contract": ["Two year"], "tenure": [9], "TotalCharges": [None]}))
    names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    assert transformed.shape[1] == len(names)
    assert all("customerID" not in name and "Churn" not in name for name in names)


def test_risk_labels_validate_probability():
    assert api.risk_label(0.0) == "Low"
    assert api.risk_label(0.30) == "Medium"
    assert api.risk_label(0.65) == "High"
    with pytest.raises(ValueError):
        api.risk_label(1.1)


def test_api_health_and_request_validation(monkeypatch):
    monkeypatch.setattr(api, "pipeline", None)
    monkeypatch.setattr(api, "baseline", None)
    monkeypatch.setattr(api, "segments", None)
    monkeypatch.setattr(api, "customers", None)
    monkeypatch.setattr(api.app.router, "on_startup", [])
    with TestClient(api.app) as client:
        assert client.get("/health").json()["models_loaded"] is False
        assert client.post("/predict", json={"customerID": ""}).status_code == 422
        assert client.post("/predict", json={"customerID": "missing"}).status_code == 503
