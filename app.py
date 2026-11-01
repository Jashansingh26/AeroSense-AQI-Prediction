from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AeroSense | AQI Intelligence", page_icon="🌿", layout="wide")
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
REQUIRED = FEATURES + ["AQI"]

def category(v):
    if v <= 50: return ("Good", "Air quality is considered satisfactory.")
    if v <= 100: return ("Satisfactory", "Minor breathing discomfort may occur for sensitive people.")
    if v <= 200: return ("Moderate", "People with lung, asthma or heart conditions may experience discomfort.")
    if v <= 300: return ("Poor", "Prolonged exposure may cause breathing discomfort.")
    if v <= 400: return ("Very Poor", "Prolonged exposure may cause respiratory illness.")
    return ("Severe", "Health effects may occur even in healthy people.")

@st.cache_data
def read_csv(path):
    return pd.read_csv(path)

def clean_data(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = raw[REQUIRED].copy()
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df[REQUIRED] >= 0).all(axis=1)]
    return df.reset_index(drop=True)

def train_models(df):
    X, y = df[FEATURES], df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2, n_jobs=-1),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Linear Regression": LinearRegression()
    }
    trained, rows = {}, []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        trained[name] = model
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R²": r2_score(y_test, pred)})
    return trained, pd.DataFrame(rows)

st.markdown("""
<style>
.main {background:#f5f8f7;}
.block-container {padding-top:1.5rem; padding-bottom:2rem;}
.hero {padding:1.35rem 1.6rem; border-radius:18px; background:linear-gradient(115deg,#102c2b,#1d5a4e); color:white; margin-bottom:1rem;}
.hero h1 {font-size:2.1rem; margin:0;}
.hero p {color:#d8eee6; margin:.45rem 0 0 0;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🌿 AeroSense")
st.sidebar.caption("AI-powered air quality analysis")
page = st.sidebar.radio("NAVIGATION", ["Home Dashboard", "AQI Prediction", "Analytics & About"])
st.sidebar.divider()
st.sidebar.markdown("**Data source**")
uploaded = st.sidebar.file_uploader("Upload another AQI CSV (optional)", type=["csv"])

raw = None
source = ""
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        source = "User-uploaded CSV"
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded CSV: {e}")

real_path = DATA / "city_day.csv"
demo_path = DATA / "demo_aqi_data.csv"
if raw is None and real_path.exists():
    raw = read_csv(str(real_path))
    source = "India Air Quality Dataset — city_day.csv"
if raw is None and demo_path.exists():
    raw = read_csv(str(demo_path))
    source = "Built-in simulated demo dataset"
if raw is None:
    st.error("No dataset found. Put city_day.csv inside the project's data folder.")
    st.stop()

try:
    df = clean_data(raw)
except ValueError as e:
    st.error(str(e))
    st.info("CSV needs these columns: PM2.5, PM10, NO2, SO2, CO, O3 and AQI.")
    st.stop()

removed = len(raw) - len(df)
if len(df) < 30:
    st.error(f"Only {len(df)} complete rows remain after cleaning. At least 30 are needed.")
    st.stop()

models, scores = train_models(df)
best = scores.sort_values("RMSE").iloc[0]["Model"]
st.sidebar.caption("Selected source: " + source)
st.sidebar.caption(f"{len(df):,} complete rows used")
st.sidebar.caption(f"{removed:,} incomplete/invalid rows removed")

if page == "Home Dashboard":
    st.markdown('<div class="hero"><h1>Air Quality Intelligence</h1><p>Understand pollution patterns and estimate Air Quality Index with machine learning.</p></div>', unsafe_allow_html=True)
    if "simulated" in source.lower():
        st.warning("Demo mode: this bundled dataset is synthetic. Do not treat its predictions or metrics as real-world AQI results.")
    else:
        st.success("Historical AQI dataset is active. Results depend on data quality, source, coverage and measurement units.")
    a,b,c,d = st.columns(4)
    a.metric("Average AQI", f"{df.AQI.mean():.1f}")
    b.metric("Highest AQI", f"{df.AQI.max():.1f}")
    c.metric("Rows available", f"{len(df):,}")
    d.metric("Best test model", best)
    left,right = st.columns([1.3,1])
    with left:
        st.subheader("AQI distribution")
        st.plotly_chart(px.histogram(df, x="AQI", nbins=28, title="Distribution of AQI values"), use_container_width=True)
    with right:
        st.subheader("Average pollutant levels")
        avg = df[FEATURES].mean().rename("Average").reset_index().rename(columns={"index":"Pollutant"})
        st.plotly_chart(px.bar(avg, x="Pollutant", y="Average", title="Mean concentration in dataset"), use_container_width=True)
    st.subheader("Dataset preview")
    cols = [c for c in ["City","Date"] + REQUIRED if c in raw.columns]
    st.dataframe(raw[cols].head(8), use_container_width=True, hide_index=True)

elif page == "AQI Prediction":
    st.markdown('<div class="hero"><h1>AQI Prediction Studio</h1><p>Enter pollutant measurements and estimate AQI using a trained regression model.</p></div>', unsafe_allow_html=True)
    st.info("Model trained on cleaned historical city-day records. Enter values in units consistent with the training dataset.")
    chosen = st.selectbox("Choose prediction algorithm", list(models.keys()), index=0)
    st.subheader("Pollutant measurements")
    defaults = {"PM2.5":60.0,"PM10":100.0,"NO2":30.0,"SO2":10.0,"CO":1.0,"O3":40.0}
    cols = st.columns(3)
    vals = {}
    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            vals[feature] = st.number_input(feature, min_value=0.0, max_value=1000.0,
                value=defaults[feature], step=1.0,
                help="Use the same concentration units as the training dataset.")
    if st.button("✨ Predict AQI", type="primary", use_container_width=True):
        row = pd.DataFrame([vals], columns=FEATURES)
        result = max(0.0, float(models[chosen].predict(row)[0]))
        label, advice = category(result)
        x,y = st.columns([1,2])
        with x:
            st.metric("Estimated AQI", f"{result:.1f}")
            st.markdown(f"### {label}")
        with y:
            st.write(advice)
            st.caption("This is an estimate, not an official monitoring-station reading.")
        st.dataframe(row, use_container_width=True)
        export = row.assign(Predicted_AQI=round(result,1), Model=chosen)
        st.download_button("Download this prediction (CSV)", export.to_csv(index=False),
                           "aqi_prediction.csv", "text/csv")
    st.caption("Prediction is not an official reading or medical advice.")

else:
    st.markdown('<div class="hero"><h1>Analytics & Project Insights</h1><p>Compare algorithms, inspect model errors and understand the project workflow.</p></div>', unsafe_allow_html=True)
    if "simulated" in source.lower():
        st.warning("Scores below are calculated from synthetic demo data, not real-world performance.")
    else:
        st.info("Scores use a held-out test split from the cleaned historical dataset; they do not guarantee future performance.")
    st.subheader("Model comparison (held-out test split)")
    st.dataframe(scores.sort_values("RMSE").round(3), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(scores.melt(id_vars="Model", value_vars=["MAE","RMSE"], var_name="Metric", value_name="Error"),
                           x="Model", y="Error", color="Metric", barmode="group", title="Error comparison (lower is better)"),
                    use_container_width=True)
    st.plotly_chart(px.bar(scores, x="Model", y="R²", title="R² comparison (closer to 1 is generally better)"), use_container_width=True)
    st.subheader("Pollutant relationships")
    st.plotly_chart(px.imshow(df[REQUIRED].corr(numeric_only=True), text_auto=".2f", aspect="auto",
                              title="Correlation matrix"), use_container_width=True)
    st.subheader("Project methodology")
    st.markdown("""
    1. **Data preparation:** load historical city-day records, convert pollutant columns to numeric values and remove incomplete records.
    2. **Feature selection:** use PM2.5, PM10, NO2, SO2, CO and O3 as input features.
    3. **Model training:** train Random Forest, Decision Tree and Linear Regression regressors.
    4. **Evaluation:** use a held-out 20% test split and report MAE, RMSE and R².
    5. **Prediction:** use the selected model to estimate AQI from user-entered measurements.

    **AI/ML note:** these are supervised machine-learning regression models, not generative AI.
    """)
    st.subheader("Expected CSV format")
    st.code("PM2.5,PM10,NO2,SO2,CO,O3,AQI\\n60,100,30,10,1,40,148", language="csv")
    st.caption("Keep pollutant units consistent. The sample row is illustrative only.")

st.sidebar.divider()
st.sidebar.caption("AeroSense AQI • Academic demonstration project")
