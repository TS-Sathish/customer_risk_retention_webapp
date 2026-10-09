# Architecture

```mermaid
flowchart LR
    CSV[Historical telco CSV] --> Validate[Validation and type normalization]
    Validate --> Split[Stratified train / validation / test]
    Split --> Preprocess[Impute and one-hot encode]
    Preprocess --> XGB[XGBoost]
    Preprocess --> LR[Logistic regression baseline]
    Validate --> Cluster[Impute, scale, evaluate K-Means]
    XGB --> Artifacts[Local model artifacts and evaluation report]
    LR --> Artifacts
    Cluster --> Artifacts
    Artifacts --> API[FastAPI]
    API --> SHAP[Local and global SHAP]
    API --> SQLite[(SQLite prediction history)]
    API --> UI[Streamlit dashboard]
    API -. minimal score evidence .-> OpenAI[Optional OpenAI narrative]
```

Training writes local, git-ignored artifacts. FastAPI loads those artifacts and the source snapshot for lookup and scoring. The dashboard consumes API summaries and individual results. SQLite stores prediction history locally. Optional OpenAI receives only risk band, estimated probability, and segment; API failure returns local fallback text.
