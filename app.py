"""
Customer Retention Intelligence Dashboard  (Streamlit)
=======================================================
Run with:   streamlit run app.py

Put these 3 files in the SAME folder as app.py (or in a ./models sub-folder):
    best_churn_model.joblib   -> Gradient Boosting classifier
    feature_scaler.joblib     -> StandardScaler (used here for retention-strength scoring)
    model_metadata.joblib     -> feature names, decision threshold, metrics

Modules
    1. Engagement vs churn overview
    2. Product utilization impact analysis
    3. High-value disengaged customer detector
    4. Retention strength scoring panels
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.special import ndtr  # standard-normal CDF: turns a z-score into a 0-1 percentile

st.set_page_config(page_title="Retention Intelligence", page_icon="🏦", layout="wide")

# Newer Streamlit versions replaced use_container_width=True with width="stretch".
_ver = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

# The model was trained with "high balance / high salary" = above the median of the
# training data. These are the medians of the public Bank Churn dataset.
HIGH_BALANCE_CUTOFF = 97198.54
HIGH_SALARY_CUTOFF = 100193.915

REQUIRED_COLS = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                 "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
AGE_LABELS = ["≤30", "31–40", "41–50", "51–60", "60+"]

STRONG_CUTOFF, WEAK_CUTOFF = 70, 45  # retention-score bands
BAND_COLORS = {"Strong": "#2E8B57", "Moderate": "#E0A100", "Weak": "#C8372D"}


# ----------------------------------------------------------------------------
# LOADING ARTIFACTS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model files…")
def load_artifacts():
    def find(name):
        for folder in (APP_DIR, APP_DIR / "models", Path.cwd()):
            if (folder / name).exists():
                return folder / name
        raise FileNotFoundError(f"Could not find '{name}' next to app.py or in ./models")

    model = joblib.load(find("best_churn_model.joblib"))
    scaler = joblib.load(find("feature_scaler.joblib"))
    meta = joblib.load(find("model_metadata.joblib"))
    return model, scaler, meta


# ----------------------------------------------------------------------------
# DATA: demo generator + feature engineering
# ----------------------------------------------------------------------------
@st.cache_data
def make_demo_data(n=5000, seed=42):
    """Synthetic customers with realistic ranges, so the app works without a CSV."""
    rng = np.random.default_rng(seed)
    geo = rng.choice(["France", "Germany", "Spain"], n, p=[0.50, 0.25, 0.25])
    age = np.clip(18 + rng.gamma(3.96, 5.28, n), 18, 92).round()
    balance = np.where(rng.random(n) < 0.36, 0.0, rng.normal(119800, 30000, n).clip(3000))
    balance = np.where(geo == "Germany", rng.normal(120000, 30000, n).clip(3000), balance)
    active = (rng.random(n) < np.clip(0.60 - (age - 38) * 0.004, 0.2, 0.8)).astype(int)
    return pd.DataFrame({
        "CustomerId": 15_600_000 + np.arange(n),
        "CreditScore": np.clip(rng.normal(650, 97, n), 350, 850).round(),
        "Geography": geo,
        "Gender": rng.choice(["Female", "Male"], n, p=[0.45, 0.55]),
        "Age": age,
        "Tenure": rng.integers(0, 11, n),
        "Balance": balance.round(2),
        "NumOfProducts": rng.choice([1, 2, 3, 4], n, p=[0.51, 0.46, 0.027, 0.003]),
        "HasCrCard": (rng.random(n) < 0.70).astype(int),
        "IsActiveMember": active,
        "EstimatedSalary": rng.uniform(12, 200000, n).round(2),
    })


def engineer_features(df):
    """Rebuild the 24 model features from the 10 raw customer columns."""
    d = df.copy()
    d["Geography_enc"] = d["Geography"].map({"France": 0, "Germany": 1, "Spain": 2})
    d["Gender_enc"] = d["Gender"].map({"Female": 0, "Male": 1})

    d["HighBalance"] = (d["Balance"] > HIGH_BALANCE_CUTOFF).astype(int)
    d["HighSalary"] = (d["EstimatedSalary"] > HIGH_SALARY_CUTOFF).astype(int)
    d["ZeroBalance"] = (d["Balance"] == 0).astype(int)

    d["EngagementScore"] = d["IsActiveMember"] * d["NumOfProducts"]
    d["BalancePerProduct"] = d["Balance"] / (d["NumOfProducts"] + 1)
    d["ActiveHighBalance"] = d["IsActiveMember"] * d["HighBalance"]
    d["SalaryBalanceMismatch"] = ((d["HighBalance"] == 1) & (d["HighSalary"] == 0)).astype(int)
    d["AtRiskPremium"] = ((d["HighBalance"] == 1) & (d["HighSalary"] == 1)
                          & (d["IsActiveMember"] == 0)).astype(int)

    d["AgeGroup"] = pd.cut(d["Age"], [-np.inf, 30, 40, 50, 60, np.inf], labels=False).astype(int)
    d["TenureGroup"] = pd.cut(d["Tenure"], [-np.inf, 2, 5, 7, np.inf], labels=False).astype(int)
    d["CreditScoreBin"] = pd.cut(d["CreditScore"], [-np.inf, 500, 600, 700, 800, np.inf],
                                 labels=False).astype(int)

    d["ProductTenureInteraction"] = d["NumOfProducts"] * d["Tenure"]
    d["AgeBalanceInteraction"] = d["AgeGroup"] * d["HighBalance"]
    d["ActiveProductScore"] = d["IsActiveMember"] * d["NumOfProducts"] * (d["Tenure"] > 5)
    return d


@st.cache_data(show_spinner="Scoring customers…")
def score_customers(raw: pd.DataFrame) -> pd.DataFrame:
    """Add churn probability, risk flag and retention components to every customer."""
    model, scaler, meta = load_artifacts()
    feats = engineer_features(raw)

    # The Gradient Boosting model was trained on RAW (unscaled) features -> no scaling here.
    X = feats[list(meta["feature_names"])]
    feats["Churn_Prob"] = model.predict_proba(X)[:, 1]
    feats["Flagged"] = feats["Churn_Prob"] >= float(meta["optimal_threshold"])

    # Retention components: where does this customer rank vs. the training population?
    # z = (value - training mean) / training std  ->  normal CDF -> 0-100 percentile.
    stats = dict(zip(scaler.feature_names_in_, zip(scaler.mean_, scaler.scale_)))
    parts = {"Engagement": "EngagementScore", "Loyalty (tenure)": "Tenure",
             "Financial commitment": "Balance", "Credit health": "CreditScore"}
    for label, col in parts.items():
        mean, std = stats[col]
        feats[f"C_{label}"] = ndtr((feats[col] - mean) / std) * 100

    feats["Engagement_Status"] = np.where(feats["IsActiveMember"] == 1, "Active", "Inactive")
    feats["Age_Band"] = pd.Categorical.from_codes(feats["AgeGroup"], AGE_LABELS)
    return feats


def add_retention_score(d: pd.DataFrame, model_weight: float) -> pd.DataFrame:
    d = d.copy()
    comp_cols = [c for c in d.columns if c.startswith("C_")]
    behaviour = d[comp_cols].mean(axis=1)
    d["Retention_Score"] = model_weight * (1 - d["Churn_Prob"]) * 100 + (1 - model_weight) * behaviour
    d["Retention_Band"] = pd.cut(d["Retention_Score"], [-np.inf, WEAK_CUTOFF, STRONG_CUTOFF, np.inf],
                                 labels=["Weak", "Moderate", "Strong"], right=False).astype(str)
    return d


# ----------------------------------------------------------------------------
# SMALL HELPERS
# ----------------------------------------------------------------------------
def summarize(d: pd.DataFrame, by) -> pd.DataFrame:
    g = d.groupby(by, observed=True)
    out = g.agg(Customers=("Churn_Prob", "size"),
                Avg_Risk=("Churn_Prob", "mean"),
                Flagged_Rate=("Flagged", "mean"),
                Avg_Balance=("Balance", "mean")).reset_index()
    if "Exited" in d.columns:
        out["Actual_Churn"] = g["Exited"].mean().values
    return out


def comparison_bars(summary, x, title):
    """Grouped bars: predicted risk, flagged share, and (if available) actual churn."""
    cols = {"Avg_Risk": "Avg predicted risk", "Flagged_Rate": "Flagged at-risk"}
    if "Actual_Churn" in summary:
        cols["Actual_Churn"] = "Actual churn"
    long = summary.melt(id_vars=x, value_vars=list(cols), var_name="Metric", value_name="Value")
    long["Metric"] = long["Metric"].map(cols)
    long["Value"] *= 100
    fig = px.bar(long, x=x, y="Value", color="Metric", barmode="group", text_auto=".1f", title=title,
                 labels={"Value": "% of customers"})
    fig.update_layout(legend_title_text="", yaxis_ticksuffix="%")
    return fig


def pct(x):
    return f"{x * 100:.1f}%"


# ----------------------------------------------------------------------------
# LOAD MODEL + DATA
# ----------------------------------------------------------------------------
try:
    model, scaler, meta = load_artifacts()
except FileNotFoundError as err:
    st.error(f"{err}. Place the three .joblib files next to app.py and restart.")
    st.stop()

st.title("🏦 Customer Retention Intelligence")
st.caption(f"Model: **{meta['model_name']}** · ROC-AUC {meta['roc_auc']:.3f} · "
           f"Recall {meta['recall']:.1f}% · Decision threshold {float(meta['optimal_threshold']):.2f}")

with st.sidebar:
    st.header("📂 Data")
    upload = st.file_uploader("Upload customer CSV", type="csv",
                              help="Needs columns: " + ", ".join(REQUIRED_COLS)
                              + ". An optional 'Exited' column (0/1) enables actual-churn comparisons.")

if upload is not None:
    raw = pd.read_csv(upload)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"Your CSV is missing these columns: {', '.join(missing)}")
        st.stop()
    num_cols = [c for c in REQUIRED_COLS if c not in ("Geography", "Gender")]
    raw[num_cols] = raw[num_cols].apply(pd.to_numeric, errors="coerce")
    before = len(raw)
    raw = raw[(raw["Geography"].isin(["France", "Germany", "Spain"]))
              & (raw["Gender"].isin(["Female", "Male"]))].dropna(subset=num_cols).reset_index(drop=True)
    if len(raw) < before:
        st.sidebar.warning(f"Skipped {before - len(raw)} rows with missing/unsupported values "
                           "(the model knows France/Germany/Spain and Female/Male).")
    data_note = f"Uploaded file · {len(raw):,} customers"
else:
    raw = make_demo_data()
    data_note = "⚠️ Demo mode: synthetic customers. Upload your CSV in the sidebar for real results."
st.sidebar.caption(data_note)

if raw.empty:
    st.error("No usable rows found in the data.")
    st.stop()

scored = score_customers(raw)

# ----------------------------------------------------------------------------
# SIDEBAR: FILTERS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Engagement filters")
    status = st.multiselect("Engagement status", ["Active", "Inactive"], default=["Active", "Inactive"])
    max_eng = int(scored["EngagementScore"].max())
    eng_range = st.slider("Engagement score (active × products)", 0, max_eng, (0, max_eng),
                          help="0 = inactive member. Higher = active customer using more products.")
    geos = st.multiselect("Geography", sorted(scored["Geography"].unique()),
                          default=sorted(scored["Geography"].unique()))
    genders = st.multiselect("Gender", sorted(scored["Gender"].unique()),
                             default=sorted(scored["Gender"].unique()))
    age_lo, age_hi = int(scored["Age"].min()), int(scored["Age"].max())
    age_range = st.slider("Age", age_lo, age_hi, (age_lo, age_hi))

    st.header("📦 Product count")
    prod_range = st.slider("Number of products", 1, int(max(4, scored["NumOfProducts"].max())), (1, 4))

    st.header("💰 Balance & salary thresholds")
    bal_max = int(np.ceil(scored["Balance"].max() / 1000) * 1000) or 1000
    sal_max = int(np.ceil(scored["EstimatedSalary"].max() / 1000) * 1000) or 1000
    min_bal = st.slider("Minimum balance", 0, bal_max, 100_000, step=5_000)
    min_sal = st.slider("Minimum estimated salary", 0, sal_max, 80_000, step=5_000)
    st.caption("Thresholds define who counts as 'high-value' in the detector tab.")

    st.header("⚙️ Scoring")
    risk_threshold = st.slider("Flag as at-risk above probability", 0.05, 0.95,
                               float(meta["optimal_threshold"]), 0.01,
                               help="Default is the model's tuned decision threshold.")
    model_weight = st.slider("Weight of model in retention score", 0.0, 1.0, 0.6, 0.05,
                             help="Rest of the score comes from engagement, tenure, balance, credit health.")

view = scored[
    scored["Engagement_Status"].isin(status)
    & scored["EngagementScore"].between(*eng_range)
    & scored["Geography"].isin(geos)
    & scored["Gender"].isin(genders)
    & scored["Age"].between(*age_range)
    & scored["NumOfProducts"].between(*prod_range)
].copy()

if view.empty:
    st.warning("No customers match the current filters. Loosen them in the sidebar.")
    st.stop()

view["Flagged"] = view["Churn_Prob"] >= risk_threshold  # let the sidebar threshold override
view = add_retention_score(view, model_weight)
has_actual = "Exited" in view.columns

tab1, tab2, tab3, tab4 = st.tabs(["📊 Engagement vs Churn", "📦 Product Utilization",
                                  "🚨 High-Value Disengaged", "🛡️ Retention Strength"])

# ============================================================================
# TAB 1 — ENGAGEMENT VS CHURN
# ============================================================================
with tab1:
    c = st.columns(5)
    c[0].metric("Customers in view", f"{len(view):,}")
    c[1].metric("Active members", pct(view["IsActiveMember"].mean()))
    c[2].metric("Avg churn probability", pct(view["Churn_Prob"].mean()))
    c[3].metric("Flagged at-risk", pct(view["Flagged"].mean()))
    c[4].metric("Actual churn rate", pct(view["Exited"].mean()) if has_actual else "n/a")

    by_status = summarize(view, "Engagement_Status")
    left, right = st.columns(2)
    left.plotly_chart(comparison_bars(by_status, "Engagement_Status", "Churn risk by engagement status"),
                      **STRETCH)
    hist = px.histogram(view, x="Churn_Prob", color="Engagement_Status", nbins=40, barmode="overlay",
                        opacity=0.65, title="Distribution of churn probability",
                        labels={"Churn_Prob": "Predicted churn probability"})
    hist.add_vline(x=risk_threshold, line_dash="dash", annotation_text="risk threshold")
    right.plotly_chart(hist, **STRETCH)

    left, right = st.columns(2)
    heat = view.pivot_table(index="Age_Band", columns="Engagement_Status", values="Churn_Prob",
                            aggfunc="mean", observed=True)
    left.plotly_chart(px.imshow(heat, text_auto=".0%", aspect="auto", color_continuous_scale="RdYlGn_r",
                                title="Avg churn probability: age band × engagement",
                                labels={"color": "Risk"}), **STRETCH)
    by_geo = summarize(view, ["Geography", "Engagement_Status"])
    right.plotly_chart(px.bar(by_geo, x="Geography", y="Avg_Risk", color="Engagement_Status",
                              barmode="group", text_auto=".1%", title="Avg churn probability by country",
                              labels={"Avg_Risk": "Avg predicted risk"}), **STRETCH)

    act = by_status.set_index("Engagement_Status")["Avg_Risk"]
    if {"Active", "Inactive"} <= set(act.index) and act["Active"] > 0:
        st.info(f"💡 In this view, inactive customers carry **{act['Inactive'] / act['Active']:.1f}×** "
                f"the average churn risk of active customers "
                f"({pct(act['Inactive'])} vs {pct(act['Active'])}).")

# ============================================================================
# TAB 2 — PRODUCT UTILIZATION
# ============================================================================
with tab2:
    by_prod = summarize(view, "NumOfProducts")
    left, right = st.columns(2)
    left.plotly_chart(comparison_bars(by_prod, "NumOfProducts", "Churn risk by number of products"),
                      **STRETCH)
    right.plotly_chart(px.bar(by_prod, x="NumOfProducts", y="Customers", text_auto=",",
                              title="How many customers hold each product count"), **STRETCH)

    left, right = st.columns(2)
    heat = view.pivot_table(index="NumOfProducts", columns="Engagement_Status", values="Churn_Prob",
                            aggfunc="mean")
    left.plotly_chart(px.imshow(heat, text_auto=".0%", aspect="auto", color_continuous_scale="RdYlGn_r",
                                title="Avg churn probability: products × engagement",
                                labels={"color": "Risk"}), **STRETCH)
    view["Credit_Card"] = np.where(view["HasCrCard"] == 1, "Has credit card", "No credit card")
    by_card = summarize(view, ["NumOfProducts", "Credit_Card"])
    right.plotly_chart(px.bar(by_card, x="NumOfProducts", y="Avg_Risk", color="Credit_Card",
                              barmode="group", text_auto=".1%", title="Does a credit card change the picture?",
                              labels={"Avg_Risk": "Avg predicted risk"}), **STRETCH)

    st.subheader("Product summary table")
    table = by_prod.rename(columns={"NumOfProducts": "Products", "Avg_Risk": "Avg risk",
                                    "Flagged_Rate": "Flagged %", "Avg_Balance": "Avg balance",
                                    "Actual_Churn": "Actual churn %"})
    fmt = {"Avg risk": "{:.1%}", "Flagged %": "{:.1%}", "Avg balance": "${:,.0f}", "Actual churn %": "{:.1%}"}
    st.dataframe(table.style.format({k: v for k, v in fmt.items() if k in table.columns}),
                 **STRETCH, hide_index=True)
    worst = by_prod.loc[by_prod["Avg_Risk"].idxmax()]
    best = by_prod.loc[by_prod["Avg_Risk"].idxmin()]
    st.info(f"💡 Riskiest product count: **{int(worst['NumOfProducts'])}** ({pct(worst['Avg_Risk'])} avg risk, "
            f"{int(worst['Customers']):,} customers). Safest: **{int(best['NumOfProducts'])}** "
            f"({pct(best['Avg_Risk'])}).")

# ============================================================================
# TAB 3 — HIGH-VALUE DISENGAGED DETECTOR
# ============================================================================
with tab3:
    st.markdown(f"Finds customers with **balance ≥ ${min_bal:,}** and **salary ≥ ${min_sal:,}** "
                "(set in the sidebar) who are disengaged.")
    o1, o2, o3 = st.columns([2, 1, 1])
    mode = o1.radio("Disengaged means…", ["Inactive members only", "Inactive OR single-product"],
                    horizontal=True)
    only_flagged = o2.checkbox("Only model-flagged", value=False)
    top_n = o3.number_input("Rows to show", 10, 500, 50, step=10)

    disengaged = (view["IsActiveMember"] == 0)
    if mode.startswith("Inactive OR"):
        disengaged |= (view["NumOfProducts"] == 1)
    hits = view[(view["Balance"] >= min_bal) & (view["EstimatedSalary"] >= min_sal) & disengaged].copy()
    if only_flagged:
        hits = hits[hits["Flagged"]]
    hits["Expected_Balance_At_Risk"] = hits["Balance"] * hits["Churn_Prob"]

    if hits.empty:
        st.warning("No customers meet these thresholds. Try lowering the balance/salary minimums.")
    else:
        k = st.columns(4)
        k[0].metric("High-value disengaged", f"{len(hits):,}", f"{len(hits) / len(view):.1%} of view")
        k[1].metric("Balance they hold", f"${hits['Balance'].sum():,.0f}")
        k[2].metric("Expected balance at risk", f"${hits['Expected_Balance_At_Risk'].sum():,.0f}",
                    help="Sum of balance × churn probability.")
        k[3].metric("Avg churn probability", pct(hits["Churn_Prob"].mean()))

        sample = hits.sample(min(len(hits), 3000), random_state=0)
        st.plotly_chart(px.scatter(sample, x="Balance", y="Churn_Prob", color="EstimatedSalary",
                                   hover_data=["Age", "NumOfProducts", "Geography"],
                                   color_continuous_scale="Viridis",
                                   title="Who should the retention team call first? (top-right = valuable & likely to leave)",
                                   labels={"Churn_Prob": "Churn probability"}), **STRETCH)

        show = hits.sort_values("Expected_Balance_At_Risk", ascending=False).head(int(top_n))
        id_col = "CustomerId" if "CustomerId" in show.columns else None
        cols = ([id_col] if id_col else []) + ["Geography", "Gender", "Age", "Tenure", "Balance",
                "EstimatedSalary", "NumOfProducts", "Churn_Prob", "Expected_Balance_At_Risk", "Retention_Score"]
        st.dataframe(show[cols], hide_index=True, **STRETCH, column_config={
            "Balance": st.column_config.NumberColumn(format="$%.0f"),
            "EstimatedSalary": st.column_config.NumberColumn("Salary", format="$%.0f"),
            "Churn_Prob": st.column_config.ProgressColumn("Churn prob", min_value=0, max_value=1,
                                                          format="%.2f"),
            "Expected_Balance_At_Risk": st.column_config.NumberColumn("Exp. $ at risk", format="$%.0f"),
            "Retention_Score": st.column_config.NumberColumn("Retention score", format="%.0f"),
        })
        st.download_button("⬇️ Download full list (CSV)",
                           hits.sort_values("Expected_Balance_At_Risk", ascending=False)[cols].to_csv(index=False),
                           file_name="high_value_disengaged.csv", mime="text/csv")

# ============================================================================
# TAB 4 — RETENTION STRENGTH SCORING
# ============================================================================
with tab4:
    st.markdown("**Retention Strength Score (0–100)** blends the model's *'will stay'* probability with "
                "four behaviour components, each ranked against your training population: "
                "engagement, tenure, balance and credit health. Higher = stickier customer.")
    avg_score = view["Retention_Score"].mean()

    g1, g2 = st.columns([1, 2])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=avg_score, title={"text": "Average retention strength"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, WEAK_CUTOFF], "color": BAND_COLORS["Weak"]},
                         {"range": [WEAK_CUTOFF, STRONG_CUTOFF], "color": BAND_COLORS["Moderate"]},
                         {"range": [STRONG_CUTOFF, 100], "color": BAND_COLORS["Strong"]}]}))
    gauge.update_layout(height=300, margin=dict(t=60, b=10))
    g1.plotly_chart(gauge, **STRETCH)
    g2.plotly_chart(px.histogram(view, x="Retention_Score", color="Retention_Band", nbins=40,
                                 color_discrete_map=BAND_COLORS, title="Retention score distribution",
                                 category_orders={"Retention_Band": ["Weak", "Moderate", "Strong"]}),
                    **STRETCH)

    st.subheader("Segment panels")
    panels = st.columns(3)
    for col, band in zip(panels, ["Strong", "Moderate", "Weak"]):
        seg = view[view["Retention_Band"] == band]
        with col:
            st.markdown(f"<h4 style='color:{BAND_COLORS[band]}'>{band} retention</h4>", unsafe_allow_html=True)
            st.metric("Customers", f"{len(seg):,}", f"{len(seg) / len(view):.1%} of view")
            if len(seg):
                st.metric("Avg churn probability", pct(seg["Churn_Prob"].mean()))
                st.metric("Avg balance", f"${seg['Balance'].mean():,.0f}")
                st.metric("Active members", pct(seg["IsActiveMember"].mean()))

    comp_cols = [c for c in view.columns if c.startswith("C_")]
    comp = (view.groupby("Retention_Band")[comp_cols].mean().rename(columns=lambda c: c[2:])
            .reset_index().melt(id_vars="Retention_Band", var_name="Component", value_name="Score"))
    left, right = st.columns(2)
    left.plotly_chart(px.bar(comp, x="Component", y="Score", color="Retention_Band", barmode="group",
                             color_discrete_map=BAND_COLORS, text_auto=".0f",
                             category_orders={"Retention_Band": ["Weak", "Moderate", "Strong"]},
                             title="What separates strong from weak customers?"), **STRETCH)
    imp = (pd.Series(model.feature_importances_, index=model.feature_names_in_)
           .sort_values().tail(10).reset_index())
    imp.columns = ["Feature", "Importance"]
    right.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                              title="Top 10 drivers in the model"), **STRETCH)

    st.subheader("🔍 Score a single customer")
    ids = view["CustomerId"] if "CustomerId" in view.columns else pd.Series(view.index, index=view.index)
    choice = st.selectbox("Pick a customer", ids.head(2000).tolist())
    row = view[ids == choice].iloc[0]
    s1, s2 = st.columns([1, 2])
    one = go.Figure(go.Indicator(mode="gauge+number", value=float(row["Retention_Score"]),
                                 title={"text": f"{row['Retention_Band']} retention"},
                                 gauge={"axis": {"range": [0, 100]},
                                        "bar": {"color": BAND_COLORS[row["Retention_Band"]]}}))
    one.update_layout(height=260, margin=dict(t=60, b=10))
    s1.plotly_chart(one, **STRETCH)
    breakdown = pd.DataFrame({"Component": [c[2:] for c in comp_cols] + ["Model: likelihood to stay"],
                              "Score": [row[c] for c in comp_cols] + [(1 - row["Churn_Prob"]) * 100]})
    s2.plotly_chart(px.bar(breakdown, x="Score", y="Component", orientation="h", text_auto=".0f",
                           range_x=[0, 100], title=f"Churn probability: {row['Churn_Prob']:.1%}"),
                    **STRETCH)

st.divider()
st.caption("Predictions are statistical estimates for prioritising outreach, not guarantees. "
           "Retention Strength is a blended heuristic score, not a model output.")
