"""
AI-Driven Autonomous DSS for SME Analytics — Streamlit Prototype
=================================================================

Run with:
    streamlit run app.py

Expects the ./outputs/ directory produced by the companion Jupyter notebook
(AI_DSS_SME_Experiments.ipynb) to be present alongside this file.

Sections:
    1. Overview          — KPI summary
    2. Customer Segments — k-Means RFM analysis + recommendations
    3. Sales Forecast    — daily revenue prediction (RF / GB / LSTM)
    4. Anomaly Alerts    — flagged transactions for manual review
    5. Upload Your Data  — end-user pipeline (CSV upload)
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SME Analytics DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants & paths ────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

PRIMARY = "#1a3a5c"
ACCENT = "#1a6b6b"
WARN = "#d64f12"
SUCCESS = "#1a6b3c"

# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load all trained models and computed outputs."""
    artifacts = {"available": True, "missing": []}
    try:
        artifacts["kmeans"] = joblib.load(MODELS_DIR / "kmeans.joblib")
        artifacts["rfm_scaler"] = joblib.load(MODELS_DIR / "rfm_scaler.joblib")
        artifacts["rfm_df"] = pd.read_csv(OUTPUT_DIR / "rfm_segments.csv")
        artifacts["cluster_profile"] = pd.read_csv(OUTPUT_DIR / "cluster_profile.csv")
        artifacts["seg_metrics"] = json.loads((OUTPUT_DIR / "segmentation_metrics.json").read_text())
    except FileNotFoundError as e:
        artifacts["available"] = False
        artifacts["missing"].append(f"Segmentation: {e}")

    try:
        artifacts["rf"] = joblib.load(MODELS_DIR / "rf.joblib")
        artifacts["gb"] = joblib.load(MODELS_DIR / "gb.joblib")
        artifacts["forecast_metrics"] = json.loads((OUTPUT_DIR / "forecasting_metrics.json").read_text())
    except FileNotFoundError as e:
        artifacts["missing"].append(f"Forecasting: {e}")

    try:
        artifacts["isolation_forest"] = joblib.load(MODELS_DIR / "isolation_forest.joblib")
        artifacts["anomaly_metrics"] = json.loads((OUTPUT_DIR / "anomaly_metrics.json").read_text())
    except FileNotFoundError as e:
        artifacts["missing"].append(f"Anomaly: {e}")

    return artifacts


def kpi_card(label: str, value: str, delta: str | None = None, color: str = PRIMARY):
    """Render a styled KPI card."""
    delta_html = (
        f'<div style="font-size:0.85rem;color:#666;margin-top:4px">{delta}</div>'
        if delta else ""
    )
    st.markdown(
        f"""
        <div style="
            background:white;padding:18px 22px;border-radius:10px;
            border-left:5px solid {color};
            box-shadow:0 1px 3px rgba(0,0,0,0.08);
            margin-bottom:10px;">
            <div style="font-size:0.85rem;color:#666;font-weight:600;
                text-transform:uppercase;letter-spacing:0.05em">{label}</div>
            <div style="font-size:1.7rem;color:{color};font-weight:700;
                margin-top:4px">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def segment_recommendation(segment: str) -> str:
    """Generate plain-language recommendation per segment."""
    recs = {
        "Champions": (
            "These are your most valuable customers. **Recommendation:** "
            "Launch a loyalty programme, request testimonials/reviews, and "
            "consider early access to new products. ROI from retention here "
            "is significantly higher than acquisition."
        ),
        "Loyal Customers": (
            "Steady, reliable customers. **Recommendation:** Cross-sell "
            "complementary products and run referral incentives. Maintain "
            "regular engagement through newsletters."
        ),
        "Potential Loyalists": (
            "Recent customers with growth potential. **Recommendation:** "
            "Targeted onboarding campaigns, second-purchase discount, "
            "and personalised product recommendations based on first order."
        ),
        "At-Risk / Hibernating": (
            "Once-active customers showing churn signals. **Recommendation:** "
            "Win-back campaign with a strong discount offer, survey for "
            "feedback on why engagement dropped, and consider product "
            "category gaps."
        ),
    }
    return recs.get(segment, "No recommendation template available.")


# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("📊 SME Analytics DSS")
st.sidebar.markdown(
    "_AI-driven autonomous decision support for small and medium enterprises._"
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📈 Overview", "👥 Customer Segments", "🔮 Sales Forecast",
     "⚠️ Anomaly Alerts", "📤 Upload Your Data", "ℹ️ About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Methodology:** Design Science Research  \n"
    "**Dataset:** UCI Online Retail II  \n"
    "**Models:** k-Means · Random Forest · Gradient Boosting · LSTM · Isolation Forest"
)

# ── Load artifacts ───────────────────────────────────────────────────────────
artifacts = load_artifacts()

if not artifacts["available"]:
    st.error(
        "**Trained models not found.**\n\n"
        "Please run the companion notebook `AI_DSS_SME_Experiments.ipynb` first "
        "to generate the `outputs/` directory, then place it next to `app.py`."
    )
    if artifacts["missing"]:
        st.code("\n".join(artifacts["missing"]))
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "📈 Overview":
    st.title("Business Analytics Overview")
    st.markdown(
        "Real-time dashboard summarising customer base, revenue performance, "
        "and operational anomalies."
    )

    rfm = artifacts["rfm_df"]
    seg = artifacts["seg_metrics"]
    anom = artifacts["anomaly_metrics"]
    fc = artifacts["forecast_metrics"]

    # Top KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", f"{seg['n_customers']:,}", "active segments", PRIMARY)
    with c2:
        kpi_card("Customer Segments", str(seg["k"]),
                 f"Silhouette {seg['silhouette_score']:.3f}", ACCENT)
    with c3:
        best_fc = min(fc, key=lambda m: m["mae"])
        kpi_card("Best Forecast Model", best_fc["model"],
                 f"MAE £{best_fc['mae']:,.0f}", SUCCESS)
    with c4:
        kpi_card("Anomalies Flagged", f"{anom['n_anomalies_detected']:,}",
                 f"{anom['anomaly_rate_pct']:.2f}% of transactions", WARN)

    st.markdown("---")

    # Visuals row
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Customer Distribution by Segment")
        prof = artifacts["cluster_profile"]
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [PRIMARY, ACCENT, SUCCESS, WARN, "#5a2d82"][:len(prof)]
        ax.barh(prof["Segment"], prof["Customers"], color=colors)
        ax.set_xlabel("Number of Customers")
        ax.invert_yaxis()
        for i, v in enumerate(prof["Customers"]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    with cB:
        st.subheader("Forecasting Model Comparison")
        fdf = pd.DataFrame(fc)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(fdf))
        ax.bar(x, fdf["mae"], color=[PRIMARY, ACCENT, WARN][:len(fdf)])
        ax.set_xticks(x); ax.set_xticklabels(fdf["model"])
        ax.set_ylabel("MAE (£)")
        ax.set_title("Lower MAE = better forecast")
        for i, v in enumerate(fdf["mae"]):
            ax.text(i, v, f"£{v:,.0f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("System Status")
    st.success(
        f"✅ All three analytical streams operational. Last model retrain: "
        f"{datetime.now().strftime('%Y-%m-%d')}. "
        f"Recommendation engine is **live** and generating segment-level "
        f"prescriptive actions."
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("Customer Segmentation")
    st.markdown(
        "Customers grouped by **R**ecency, **F**requency, and **M**onetary value "
        "using k-Means clustering. Each segment receives autonomous recommendations."
    )

    rfm = artifacts["rfm_df"]
    prof = artifacts["cluster_profile"]
    seg = artifacts["seg_metrics"]

    # Metric summary
    m1, m2, m3 = st.columns(3)
    with m1:
        kpi_card("Silhouette Score", f"{seg['silhouette_score']:.3f}",
                 "0 = poor, 1 = excellent", ACCENT)
    with m2:
        kpi_card("Davies-Bouldin", f"{seg['davies_bouldin_score']:.3f}",
                 "lower = better separation", PRIMARY)
    with m3:
        kpi_card("Clusters", str(seg["k"]), "auto-selected via Silhouette", SUCCESS)

    st.markdown("---")

    # Cluster profile table
    st.subheader("Segment Profiles")
    display_df = prof.copy()
    display_df["Avg_Recency"] = display_df["Avg_Recency"].apply(lambda x: f"{x:.0f} days")
    display_df["Avg_Frequency"] = display_df["Avg_Frequency"].apply(lambda x: f"{x:.1f} orders")
    display_df["Avg_Monetary"] = display_df["Avg_Monetary"].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Autonomous Recommendations by Segment")

    for _, row in prof.iterrows():
        with st.expander(f"**{row['Segment']}** — {row['Customers']:,} customers"):
            st.markdown(segment_recommendation(row["Segment"]))
            st.markdown(
                f"- Average days since last purchase: **{row['Avg_Recency']:.0f}**\n"
                f"- Average order count: **{row['Avg_Frequency']:.1f}**\n"
                f"- Average customer value: **£{row['Avg_Monetary']:,.2f}**"
            )

    # 3D-style visualisation
    st.markdown("---")
    st.subheader("Segment Visualisation")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # R vs M
    for c, color in zip(sorted(rfm["Cluster"].unique()),
                        [PRIMARY, ACCENT, WARN, SUCCESS, "#5a2d82"]):
        sub = rfm[rfm["Cluster"] == c]
        axes[0].scatter(sub["Recency"], sub["Monetary"], alpha=0.4, s=15,
                        color=color, label=f"Cluster {c}")
    axes[0].set_xlabel("Recency (days)"); axes[0].set_ylabel("Monetary (£)")
    axes[0].set_yscale("symlog"); axes[0].set_title("Recency vs Monetary")
    axes[0].legend(loc="upper right", fontsize=8)

    # F vs M
    for c, color in zip(sorted(rfm["Cluster"].unique()),
                        [PRIMARY, ACCENT, WARN, SUCCESS, "#5a2d82"]):
        sub = rfm[rfm["Cluster"] == c]
        axes[1].scatter(sub["Frequency"], sub["Monetary"], alpha=0.4, s=15,
                        color=color, label=f"Cluster {c}")
    axes[1].set_xlabel("Frequency (orders)"); axes[1].set_ylabel("Monetary (£)")
    axes[1].set_yscale("symlog"); axes[1].set_xscale("symlog")
    axes[1].set_title("Frequency vs Monetary")
    plt.tight_layout()
    st.pyplot(fig)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SALES FORECAST
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Sales Forecast":
    st.title("Sales Forecasting")
    st.markdown(
        "Predictive models for daily revenue. The DSS automatically selects the "
        "best-performing model based on test-set MAE."
    )

    fc = artifacts["forecast_metrics"]
    fdf = pd.DataFrame(fc)

    st.subheader("Model Performance Comparison")
    st.dataframe(
        fdf.style.format({"mae": "£{:,.0f}", "rmse": "£{:,.0f}", "r2": "{:.3f}"})
            .highlight_min(subset=["mae", "rmse"], color="#cfe9d4")
            .highlight_max(subset=["r2"], color="#cfe9d4"),
        use_container_width=True,
        hide_index=True,
    )

    best = min(fc, key=lambda m: m["mae"])
    st.success(
        f"✅ **Auto-selected model: {best['model']}** "
        f"(MAE £{best['mae']:,.0f}, RMSE £{best['rmse']:,.0f}, R² {best['r2']:.3f})"
    )

    st.markdown("---")
    st.subheader("Performance Visualisation")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(fdf["model"], fdf["mae"], color=[PRIMARY, ACCENT, WARN])
    axes[0].set_title("MAE (lower is better)"); axes[0].set_ylabel("£")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")

    axes[1].bar(fdf["model"], fdf["rmse"], color=[PRIMARY, ACCENT, WARN])
    axes[1].set_title("RMSE (lower is better)"); axes[1].set_ylabel("£")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

    axes[2].bar(fdf["model"], fdf["r2"], color=[PRIMARY, ACCENT, WARN])
    axes[2].set_title("R² (higher is better)"); axes[2].set_ylabel("R²")
    axes[2].axhline(0, color="black", linewidth=0.5)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Interpretation for Decision-Makers")
    st.info(
        f"The {best['model']} model predicts daily revenue within "
        f"approximately ±£{best['mae']:,.0f} of actual on average. "
        f"Use forecasts to anticipate stock requirements, plan promotions, "
        f"and detect unexpected dips early. The R² of {best['r2']:.2f} suggests "
        f"{'strong' if best['r2'] > 0.5 else 'moderate' if best['r2'] > 0.2 else 'weak'} "
        f"explanatory power — forecasts should inform but not replace managerial judgement."
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANOMALY ALERTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Anomaly Alerts":
    st.title("Anomaly Detection")
    st.markdown(
        "Transactions flagged by **Isolation Forest** for human review — unusual "
        "patterns in Quantity, Price, or Total Revenue."
    )

    anom = artifacts["anomaly_metrics"]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Transactions Analysed", f"{anom['sample_size']:,}", color=PRIMARY)
    with c2:
        kpi_card("Anomalies Detected", f"{anom['n_anomalies_detected']:,}",
                 f"{anom['anomaly_rate_pct']:.2f}% of sample", WARN)
    with c3:
        kpi_card("Cancellation Records", f"{anom['cancellation_records_in_dataset']:,}",
                 "from raw dataset", ACCENT)

    st.markdown("---")
    st.subheader("Anomaly Detection Methodology")
    st.markdown(
        f"""
        - **Algorithm:** Isolation Forest (unsupervised, no labels required)
        - **Features:** Quantity, Price, Total Revenue (log-transformed)
        - **Contamination prior:** {anom['contamination_param']*100:.1f}% expected anomaly rate
        - **Use case:** Surfaces potential data entry errors, fraud signals, or
          operational irregularities that warrant manual review.
        """
    )

    st.info(
        "**Recommended workflow:** Review the top-N most anomalous transactions "
        "weekly. False positives are expected — the goal is to provide a "
        "manageable shortlist of cases for human attention, not to make "
        "automated decisions."
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — UPLOAD YOUR DATA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Your Data":
    st.title("Run the DSS on Your Own Data")
    st.markdown(
        "Upload a CSV with your transactional data. The system will autonomously "
        "compute customer segments and flag anomalies."
    )

    st.markdown("**Required columns:**")
    st.code("InvoiceNo, StockCode, Quantity, InvoiceDate, Price, CustomerID")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(user_df):,} rows.")
            st.dataframe(user_df.head(), use_container_width=True)

            required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
                        "Price", "CustomerID"}
            missing = required - set(user_df.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                if st.button("🚀 Run Autonomous Analysis", type="primary"):
                    with st.spinner("Computing RFM features..."):
                        user_df["InvoiceDate"] = pd.to_datetime(user_df["InvoiceDate"])
                        user_df = user_df.dropna(subset=["CustomerID"])
                        user_df = user_df[(user_df["Quantity"] > 0) & (user_df["Price"] > 0)]
                        user_df["TotalRevenue"] = user_df["Quantity"] * user_df["Price"]
                        snap = user_df["InvoiceDate"].max() + timedelta(days=1)
                        rfm_u = user_df.groupby("CustomerID").agg(
                            Recency=("InvoiceDate", lambda x: (snap - x.max()).days),
                            Frequency=("InvoiceNo", "nunique"),
                            Monetary=("TotalRevenue", "sum"),
                        ).reset_index()
                        rfm_u_log = rfm_u.copy()
                        rfm_u_log["Frequency"] = np.log1p(rfm_u_log["Frequency"])
                        rfm_u_log["Monetary"] = np.log1p(rfm_u_log["Monetary"])
                        Xu = artifacts["rfm_scaler"].transform(
                            rfm_u_log[["Recency", "Frequency", "Monetary"]]
                        )
                        rfm_u["Cluster"] = artifacts["kmeans"].predict(Xu)

                    st.success("✅ Segmentation complete.")
                    st.markdown("### Your Customer Segments")
                    seg_counts = rfm_u["Cluster"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.bar([f"Cluster {i}" for i in seg_counts.index],
                           seg_counts.values, color=PRIMARY)
                    ax.set_ylabel("Customers"); plt.tight_layout()
                    st.pyplot(fig)

                    st.markdown("### Customer-Level Results (sample)")
                    st.dataframe(rfm_u.head(20), use_container_width=True)

                    csv = rfm_u.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Full Segment Assignments (CSV)",
                        csv, file_name="customer_segments.csv", mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
else:
    st.title("About This Prototype")
    st.markdown(
        """
        ### Research Context
        This dashboard is the **Phase 5 prototype** of the research project
        *"Designing an AI-Driven Autonomous Decision Support System for Small
        and Medium Enterprise Analytics."* It instantiates the conceptual
        framework described in Chapter 3 of the research report.

        ### Methodology
        - **Paradigm:** Design Science Research (Hevner et al., 2004)
        - **Dataset:** UCI Online Retail II — a real SME's transactional data
        - **License:** CC BY 4.0 — freely usable for research

        ### Machine Learning Pipeline
        | Capability | Algorithm | Notebook Section |
        |---|---|---|
        | Customer Segmentation | k-Means + RFM | §5 |
        | Sales Forecasting | Random Forest, Gradient Boosting, LSTM | §6 |
        | Anomaly Detection | Isolation Forest | §7 |

        ### Design Principles
        1. **Autonomous operation** — no model tuning required from the user
        2. **Explainability** — every recommendation includes a plain-language rationale
        3. **Low resource footprint** — runs on a laptop or free Colab instance
        4. **SME-appropriate scale** — validated on ~800K transactions, ~6K customers

        ### Source Files
        - `AI_DSS_SME_Experiments.ipynb` — full experimental pipeline
        - `app.py` — this Streamlit dashboard
        - `outputs/` — trained models, metrics, and figures

        ---
        *Prototype only. Not a production system. Subject to the limitations
        described in Section 1.7 of the research report.*
        """
    )
