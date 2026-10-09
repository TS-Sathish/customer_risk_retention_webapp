# Customer Risk & Retention Intelligence

A local analytics prototype for the bundled IBM telco churn snapshot. It trains XGBoost and logistic regression models, evaluates them on a held-out test split, segments customers with K-Means, serves results through FastAPI, and provides a Streamlit dashboard. The original CSV is preserved.

## Setup

Python 3.10 or newer is recommended. Run these commands from the repository root.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

OpenAI is optional. Set `OPENAI_API_KEY` in `.env` to enable generated wording. The API sends only risk, estimated probability, and segment; local fallback wording works without a key or when the service fails.

## Train and run

```bash
python -m src.train
uvicorn src.api:app --reload
```

In another terminal, start the dashboard:

```bash
streamlit run dashboard.py
```

FastAPI docs are at `http://127.0.0.1:8000/docs`. Set `API_BASE_URL` to use another API address. Model artifacts and reports are generated locally in `models/` and `reports/`; SQLite history is stored in `customer_risk.sqlite3`. These files are ignored by Git.

Run automated checks from the repository root with `pytest -q`.

## API overview

- `GET /health`, `GET /dataset/summary`, `GET /metrics`, `GET /portfolio/summary`
- `GET /customers`, `GET /customers/{customer_id}`
- `POST /predict` with `{"customerID":"..."}`
- `GET /segments`, `GET /segments/{customer_id}`
- `GET /explanations/global`, `GET /explanations/{customer_id}`
- `GET /predictions/recent`

## Data and research limits

The supplied CSV contains 7,043 customer rows, 21 columns, 11 blank `TotalCharges` entries (coerced to missing and imputed within the fitted pipeline), zero exact duplicate rows, and zero duplicate customer identifiers. Churn is 1,869 Yes (26.5%) and 5,174 No (73.5%). Train/validation/test partitions are stratified and reproducible. Classification thresholds are selected on validation data only; reported test metrics include accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix.

This is a single historical telco snapshot, without longitudinal or intervention evidence. It does not establish current customer state, broad external validity, or retention impact. Model quality can vary across time, providers, and populations. Demographic and correlated service variables can create disparate errors; review subgroup performance and decision impact before operational use. SHAP describes contributions to model output and does not establish causes. Any retention action remains a human-reviewed proposal; an intervention study is needed to measure its effect.

See [MODEL_CARD.md](MODEL_CARD.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for further details.
