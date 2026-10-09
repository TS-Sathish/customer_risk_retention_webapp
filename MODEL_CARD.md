# Model Card

## Intended use

Educational and exploratory analysis of the bundled IBM Telco Customer Churn snapshot. Outputs can support human review of churn patterns and model behavior. They are not a real-time customer status or an automated decision to retain, contact, price, or deny service.

## Data

The input contains 7,043 records and 21 columns; the binary target is `Churn`. There are 1,869 positive and 5,174 negative labels. Eleven blank `TotalCharges` cells are parsed as missing and imputed in the model pipeline. Exact duplicate rows and duplicate IDs were not observed. The source is a single historical telco snapshot; precise collection dates and representativeness are not established here.

`customerID` and `Churn` are excluded from model inputs. A stratified, reproducible split allocates 60% for fitting, 20% for validation threshold selection, and 20% for final test metrics. XGBoost is the primary pipeline; logistic regression is a baseline. K-Means uses tenure, monthly and total charges, and a derived service count after imputation and scaling.

## Evaluation

Run `python -m src.train` to create `reports/metrics.json`. Thresholds are selected by validation F1 and fixed before test evaluation. Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix are reported on the test partition. A random split within one snapshot may overstate future performance; a time-based or external validation set is needed for deployment claims.

## Limitations and risks

- Churn patterns may depend on one provider, product mix, and historical collection period; transfer to other populations is unknown.
- Demographic fields and their proxies may produce uneven errors or consequences. Evaluate calibration and error rates by appropriate, legally reviewed groups before operational decisions.
- Features and SHAP rankings describe associations the model uses, not causal churn drivers.
- Risk scores estimate model probabilities, not certain outcomes. No retention intervention effect has been measured.
- OpenAI narrative generation is optional and advisory. It receives only probability, risk band, and segment, and local fallback wording remains available.

## Monitoring and governance

Reassess data quality, drift, subgroup performance, calibration, and operational impact before each deployment. Keep a human reviewer responsible for any customer-facing action. Do not treat this artifact as a validated production model.
