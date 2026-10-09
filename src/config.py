from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DB_PATH = ROOT / "customer_risk.sqlite3"
RANDOM_STATE = 42
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.65
