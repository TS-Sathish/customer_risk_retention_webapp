"""Streamlit dashboard for portfolio, customer, and model review."""

import html
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Customer Intelligence", page_icon="◈", layout="wide")
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#15243b; --muted:#738198; --blue:#3968d8; --line:#e6ebf2; --surface:#ffffff; }
.stApp { background:linear-gradient(180deg,#f6f8fc 0%,#f4f7fb 45%,#f8fafc 100%); color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.025em; }
h2 { font-size:1.35rem !important; margin-top:.55rem !important; }
[data-testid="stSidebar"] { background:#111f35; }
[data-testid="stSidebar"] * { color:#edf3ff; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#aebbd0; }
[data-testid="stMetric"] { background:#fff; padding:17px 19px; border:1px solid var(--line); border-radius:14px; box-shadow:0 3px 12px rgba(20,38,67,.035); }
[data-testid="stMetricLabel"] { color:#75839a; font-size:.82rem; }
[data-testid="stMetricValue"] { color:#172942; font-family:'Manrope',sans-serif; font-weight:700; }
[data-testid="stTabs"] button { font-weight:600; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:15px; background:rgba(255,255,255,.78); }
.hero { padding:27px 31px; border-radius:19px; color:white; background:linear-gradient(112deg,#132542 0%,#1c3a69 58%,#3159a2 100%); box-shadow:0 14px 35px rgba(28,54,96,.16); margin:2px 0 23px 0; }
.hero-kicker { color:#a9c3ff; font-size:.75rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.hero-title { color:white; font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; letter-spacing:-.045em; margin:.35rem 0 .3rem 0; }
.hero-copy { color:#d0dbee; font-size:.94rem; margin:0; }
.section-note { color:#77869c; font-size:.88rem; margin-top:-.45rem; margin-bottom:.9rem; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; font-size:.75rem; font-weight:700; }
.narrative { border-left:3px solid #6385dc; background:#f4f7ff; padding:13px 16px; border-radius:5px 12px 12px 5px; color:#263a5a; }
div.stButton > button { border-radius:9px; font-weight:650; min-height:2.65rem; }
</style>
""", unsafe_allow_html=True)


def get_json(path: str, timeout: int = 15):
    response = requests.get(f"{API}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def risk_color(risk: str) -> tuple[str, str]:
    return {"Low": ("#e8f7ef", "#198754"),
            "Medium": ("#fff4df", "#a96b00"),
            "High": ("#fdebec", "#c33f4c")}.get(risk, ("#edf1f7", "#51627a"))


st.sidebar.markdown("### ◈ Customer Intelligence")
st.sidebar.caption("RISK & RETENTION WORKSPACE")
st.sidebar.divider()
st.sidebar.markdown("**Data source**")
st.sidebar.caption("Historical telco customer snapshot")
st.sidebar.markdown("**Analysis**")
st.sidebar.caption("Churn estimates · Segments · Model explanations")

try:
    health = get_json("/health", timeout=4)
except requests.RequestException:
    st.error("The analysis API is unavailable. Start it with `uvicorn src.api:app --reload`.")
    st.stop()

if not health.get("models_loaded"):
    st.warning("Model files are not loaded. Run `python -m src.train`, then restart the API.")
    st.stop()

try:
    portfolio = get_json("/portfolio/summary", timeout=60)
    customers = pd.DataFrame(get_json("/customers?limit=7043", timeout=20))
    metrics = get_json("/metrics")
    segments = get_json("/segments")
    summary = get_json("/dataset/summary")
except (requests.RequestException, ValueError) as error:
    st.error(f"Could not load dashboard data: {error}")
    st.stop()

st.sidebar.success("API connected · Models ready")
st.sidebar.caption(f"{portfolio['customer_count']:,} records available")
st.sidebar.divider()
st.sidebar.caption("Predictions are estimates from a historical snapshot. Review outputs before taking action.")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Customer analytics platform</div>
  <div class="hero-title">Risk & Retention Intelligence</div>
  <p class="hero-copy">Explore historical churn patterns, review model estimates, and understand the signals behind each prediction.</p>
</div>
""", unsafe_allow_html=True)

overview_tab, customer_tab, model_tab = st.tabs(["◫  Portfolio overview", "◎  Customer review", "⌁  Model & data"])

with overview_tab:
    st.subheader("Portfolio at a glance")
    st.markdown('<div class="section-note">A summary of observed churn labels and current model estimates across the supplied snapshot.</div>',
                unsafe_allow_html=True)
    total, churn, high_risk, cluster_count = st.columns(4)
    total.metric("Customers in snapshot", f"{portfolio['customer_count']:,}")
    churn.metric("Historical churn rate", f"{portfolio['historical_churn_rate']:.1%}")
    high_risk.metric("Estimated high risk", f"{portfolio['predicted_risk_counts']['High']:,}")
    cluster_count.metric("Behavior segments", str(segments["selected_k"]))

    historical_col, risk_col = st.columns(2, gap="large")
    with historical_col, st.container(border=True):
        st.markdown("#### Historical churn labels")
        st.caption("Observed outcomes in the dataset")
        historical = pd.DataFrame({"Customers": portfolio["historical_churn_counts"]})
        st.bar_chart(historical, color="#7184a4", height=235)
    with risk_col, st.container(border=True):
        st.markdown("#### Model-estimated risk bands")
        st.caption("Risk bands assigned from model probability estimates")
        risks = pd.DataFrame({"Customers": portfolio["predicted_risk_counts"]})
        st.bar_chart(risks, color="#4776df", height=235)

    st.subheader("Customer segments")
    segment_col, profile_col = st.columns([1, 2], gap="large")
    with segment_col, st.container(border=True):
        st.markdown("#### Segment size")
        st.bar_chart(pd.DataFrame({"Customers": portfolio["segment_counts"]}), color="#38a58b", height=260)
    with profile_col, st.container(border=True):
        st.markdown("#### Segment profiles")
        profiles = pd.DataFrame.from_dict(segments["profiles"], orient="index")
        profiles.index.name = "Segment"
        st.dataframe(profiles, use_container_width=True, height=260)
    st.caption("Cluster profiles show average feature values in original units. Clusters are descriptive groups, not customer quality labels.")

    try:
        history = get_json("/predictions/recent?limit=8")
        if history:
            st.subheader("Recent customer reviews")
            recent = pd.DataFrame(history).drop(columns=["explanation", "narrative"], errors="ignore")
            st.dataframe(recent, use_container_width=True, hide_index=True)
    except requests.RequestException:
        st.info("Recent analysis history is unavailable.")

with customer_tab:
    st.subheader("Review an individual customer")
    st.markdown('<div class="section-note">View a model estimate alongside the historical outcome and local explanation.</div>',
                unsafe_allow_html=True)
    selector, action = st.columns([4, 1], vertical_alignment="bottom")
    customer_ids = customers["customerID"].astype(str).tolist()
    with selector:
        customer_id = st.selectbox("Customer identifier", customer_ids, label_visibility="visible")
    with action:
        analyse = st.button("Run analysis", type="primary", use_container_width=True)
    if analyse:
        try:
            response = requests.post(f"{API}/predict", json={"customerID": customer_id}, timeout=60)
            response.raise_for_status()
            st.session_state["result"] = response.json()
        except requests.RequestException as error:
            st.error(f"Analysis failed: {error}")

    result = st.session_state.get("result")
    if result:
        background, foreground = risk_color(result["risk"])
        st.markdown(f'<span class="pill" style="background:{background};color:{foreground}">{result["risk"].upper()} RISK</span>',
                    unsafe_allow_html=True)
        probability, segment, prediction, observed = st.columns(4)
        probability.metric("Estimated churn probability", f"{result['churn_probability']:.1%}")
        segment.metric("Behavior segment", str(result["segment"]))
        prediction.metric("Model classification", "Likely churn" if result["predicted_churn"] else "Not flagged")
        observed.metric("Historical outcome", "Churn" if result["historical_churn"] else "No churn")

        explanation_col, narrative_col = st.columns([1.15, .85], gap="large")
        with explanation_col, st.container(border=True):
            st.markdown("#### What influenced this estimate")
            st.caption(result["explanation"]["interpretation"])
            factors = pd.DataFrame(result["explanation"]["top_factors"])
            factors = factors.sort_values("contribution")
            st.bar_chart(factors.set_index("feature")[["contribution"]], color="#4776df", height=330)
        with narrative_col, st.container(border=True):
            st.markdown("#### Review notes")
            narrative = html.escape(result["narrative"])
            st.markdown(f'<div class="narrative">{narrative}</div>', unsafe_allow_html=True)
            st.caption("Decision support only. Review customer context before considering any action.")
    else:
        st.info("Choose a customer and run an analysis to see their estimate and explanation.")

with model_tab:
    st.subheader("Held-out model evaluation")
    st.markdown('<div class="section-note">Thresholds were selected on validation data. The following scores are from a separate test split.</div>',
                unsafe_allow_html=True)
    model_rows = []
    for name, evaluation in metrics.get("models", {}).items():
        test_metrics = evaluation["test"]
        model_rows.append({
            "Model": name.replace("_", " ").title(),
            "Threshold": test_metrics["threshold"],
            "Accuracy": test_metrics["accuracy"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "F1": test_metrics["f1"],
            "ROC-AUC": test_metrics["roc_auc"],
            "PR-AUC": test_metrics["pr_auc"],
        })
    st.dataframe(pd.DataFrame(model_rows).set_index("Model").style.format("{:.3f}"),
                 use_container_width=True)
    with st.expander("Confusion matrices"):
        for name, evaluation in metrics.get("models", {}).items():
            st.markdown(f"**{name.replace('_', ' ').title()}** — rows: actual class, columns: predicted class `[No, Yes]`")
            st.dataframe(pd.DataFrame(evaluation["test"]["confusion_matrix"],
                                      index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"]),
                         use_container_width=True)

    st.subheader("K-Means evaluation")
    cluster_scores = pd.DataFrame(segments["evaluation"])
    st.dataframe(cluster_scores.set_index("k").style.format("{:.3f}"), use_container_width=True)
    st.caption("Silhouette compares cohesion and separation; inertia decreases as K grows. Cluster labels are arbitrary.")

    st.subheader("Dataset quality")
    data_one, data_two, data_three = st.columns(3)
    data_one.metric("Rows", f"{summary['rows']:,}")
    data_two.metric("Columns", str(summary["columns"]))
    data_three.metric("Duplicate rows", str(summary["duplicates"]))
    with st.expander("Missing values and target balance"):
        missing = {name: count for name, count in summary["missing_values"].items() if count}
        st.write("Missing values")
        st.json(missing or {"status": "No missing values"})
        st.write("Historical target counts")
        st.json(summary["target_counts"])
        st.caption("This is one historical telco snapshot. It does not establish current customer status or future retention impact.")

    with st.expander("Global SHAP summary"):
        st.caption("Global SHAP summarizes model contributions across a sample; it does not establish causes.")
        if st.button("Calculate global explanation"):
            try:
                st.session_state["global_shap"] = get_json("/explanations/global", timeout=90)
            except requests.RequestException as error:
                st.error(f"Global explanation failed: {error}")
        global_explanation = st.session_state.get("global_shap")
        if global_explanation:
            importance = pd.DataFrame(global_explanation["features"]).set_index("feature")
            st.bar_chart(importance, color="#4776df", height=450)
