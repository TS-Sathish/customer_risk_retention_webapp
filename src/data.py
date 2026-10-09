from pathlib import Path

import pandas as pd
from .config import DATA_PATH
TARGET, ID_COL = "Churn", "customerID"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True)
    missing = {ID_COL, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[ID_COL].isna().any() or df[ID_COL].astype(str).str.strip().eq("").any():
        raise ValueError("customerID must be present and non-empty")
    if df[ID_COL].duplicated().any():
        raise ValueError("customerID values must be unique")
    if df[TARGET].isna().any() or not set(df[TARGET].astype(str).unique()) <= {"Yes", "No"}:
        raise ValueError("Churn must contain only Yes/No values")
    df = df.copy()
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df:
        raise ValueError(f"Missing target column: {TARGET}")
    return df.drop(columns=[TARGET, ID_COL], errors="ignore"), df[TARGET].astype(int)


def dataset_summary(df: pd.DataFrame) -> dict:
    target_counts = df[TARGET].value_counts().to_dict()
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "duplicates": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[ID_COL].duplicated().sum()),
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "target_counts": {"No": int(target_counts.get(0, 0)), "Yes": int(target_counts.get(1, 0))},
        "churn_rate": float(df[TARGET].mean()),
        "schema": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
    }


def segmentation_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df[[column for column in ["tenure", "MonthlyCharges", "TotalCharges"]
                   if column in df]].copy()
    service_columns = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                       "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    service_count = [df[column].astype(str).isin(["Yes", "DSL", "Fiber optic"]).astype(int)
                     for column in service_columns if column in df]
    features["service_count"] = sum(service_count) if service_count else 0
    return features
