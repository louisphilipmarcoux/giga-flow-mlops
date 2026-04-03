import os
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

# --- Configuration ---
DB_USER = os.getenv("POSTGRES_USER", "gigaflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres_db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "sentiment_db")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

QUERY_RECENT = text(
    "SELECT processed_at, text, sentiment_label, sentiment_score"
    " FROM sentiment_predictions"
    " ORDER BY processed_at DESC"
    " LIMIT 100"
)

QUERY_COUNT = text("SELECT COUNT(*) as total FROM sentiment_predictions")

# --- Streamlit App ---

st.set_page_config(page_title="GigaFlow Live Sentiment", page_icon="🚀", layout="wide")

st.title("🚀 GigaFlow: Live Sentiment Analysis")

# --- Tabs ---
tab_predict, tab_dashboard, tab_models = st.tabs(["🔬 Test Model", "📊 Live Dashboard", "🏆 Model Registry"])


@st.cache_resource
def get_db_connection():
    """Creates a SQLAlchemy engine (cached across reruns)."""
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None


engine = get_db_connection()


def load_data():
    """Loads recent prediction data from the database."""
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(QUERY_RECENT, conn)
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return pd.DataFrame()


def load_total_count():
    """Gets total prediction count."""
    if engine is None:
        return 0
    try:
        with engine.connect() as conn:
            result = conn.execute(QUERY_COUNT)
            return result.scalar() or 0
    except Exception:
        return 0


def load_model_versions():
    """Fetches all model versions from MLflow API."""
    try:
        resp = requests.get(
            f"{MLFLOW_URI}/api/2.0/mlflow/model-versions/search",
            params={"filter": "name='giga-flow-sentiment'", "max_results": "100"},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("model_versions", [])
        return []
    except Exception:
        return []


def load_run_metrics(run_id):
    """Fetches metrics for a specific MLflow run."""
    try:
        resp = requests.get(
            f"{MLFLOW_URI}/api/2.0/mlflow/runs/get",
            params={"run_id": run_id},
            timeout=5,
        )
        if resp.status_code == 200:
            run = resp.json().get("run", {})
            metrics = run.get("data", {}).get("metrics", [])
            params = run.get("data", {}).get("params", [])
            return {
                "metrics": {m["key"]: float(m["value"]) for m in metrics},
                "params": {p["key"]: p["value"] for p in params},
            }
        return {"metrics": {}, "params": {}}
    except Exception:
        return {"metrics": {}, "params": {}}


def get_champion_version():
    """Gets the current champion model version."""
    try:
        resp = requests.get(
            f"{MLFLOW_URI}/api/2.0/mlflow/registered-models/alias",
            params={"name": "giga-flow-sentiment", "alias": "champion"},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("model_version", {}).get("version")
        return None
    except Exception:
        return None


# --- Tab 1: Test Model ---
with tab_predict:
    st.header("🔬 Test the Model Live")
    user_text = st.text_input("Enter text to analyze:", "I love this product!")

    if st.button("Analyze Sentiment"):
        if user_text:
            try:
                response = requests.post(
                    "http://model_service:8000/predict",
                    json={"text": user_text},
                    timeout=10,
                )
                if response.status_code == 200:
                    pred = response.json()
                    emoji = "😊" if pred["sentiment_label"] == "Positive" else "😡"
                    st.subheader(f"Result: {emoji} **{pred['sentiment_label']}** (Score: {pred['sentiment_score']:.4f})")
                elif response.status_code == 503:
                    st.error("Model service is still loading. Please try again.")
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect: {e}")
        else:
            st.warning("Please enter some text.")

# --- Tab 2: Live Dashboard ---
with tab_dashboard:
    dashboard_placeholder = st.empty()

# --- Tab 3: Model Registry ---
with tab_models:
    st.header("🏆 Model Registry")

    # Show currently loaded model
    try:
        model_resp = requests.get("http://model_service:8000/model", timeout=5)
        if model_resp.status_code == 200:
            current = model_resp.json()
            st.info(f"🔵 **Currently loaded:** {current.get('version', 'unknown')} — `{current.get('uri', 'N/A')}`")
    except Exception:
        st.warning("Could not fetch current model info.")

    champion_version = get_champion_version()
    versions = load_model_versions()

    if not versions:
        st.info("No models registered yet. Train a model first.")
    else:
        rows = []
        for v in versions:
            run_data = load_run_metrics(v.get("run_id", ""))
            metrics = run_data["metrics"]
            params = run_data["params"]
            is_champion = str(v.get("version")) == str(champion_version)

            rows.append({
                "Version": v.get("version"),
                "Status": "👑 Champion" if is_champion else "",
                "Accuracy": f"{metrics.get('accuracy', 0):.4f}" if metrics.get("accuracy") else "N/A",
                "F1-Score": f"{metrics.get('f1_score', 0):.4f}" if metrics.get("f1_score") else "N/A",
                "Model": params.get("model_name", "N/A"),
                "Dataset Size": params.get("dataset_size", "-"),
                "Run ID": v.get("run_id", "")[:12],
            })

        df_models = pd.DataFrame(rows)
        st.dataframe(df_models, use_container_width=True, hide_index=True)

        st.markdown(f"**Current Champion:** Version {champion_version}" if champion_version else "**No champion set**")

        st.subheader("🔄 Load a Model Version")
        version_numbers = sorted([v.get("version") for v in versions], reverse=True)
        col_select, col_load, col_champion = st.columns([2, 1, 1])
        with col_select:
            selected_version = st.selectbox("Select version to load:", version_numbers)
        with col_load:
            st.write("")  # spacing
            if st.button("Load Version"):
                with st.spinner(f"Loading version {selected_version}... (large models may take ~60s)"):
                    try:
                        resp = requests.post(
                            "http://model_service:8000/reload",
                            json={"version": int(selected_version)},
                            timeout=120,
                        )
                        if resp.status_code == 200:
                            st.success(f"Version {selected_version} loaded!")
                        else:
                            st.error(f"Failed: {resp.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not reach model service: {e}")
        with col_champion:
            st.write("")  # spacing
            if st.button("Load Champion"):
                with st.spinner("Loading champion model... (large models may take ~60s)"):
                    try:
                        resp = requests.post("http://model_service:8000/reload", timeout=120)
                        if resp.status_code == 200:
                            st.success("Champion model loaded!")
                        else:
                            st.error(f"Failed: {resp.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not reach model service: {e}")

        st.markdown(f"📈 [View in MLflow UI]({MLFLOW_URI.replace('mlflow_server', 'localhost')})")

# --- Main Loop (updates dashboard tab) ---
while True:
    df = load_data()
    total_count = load_total_count()

    with dashboard_placeholder.container():
        if df.empty:
            st.info("No data received yet. Waiting for predictions...")
        else:
            latest = df.iloc[0]
            sentiment_emoji = "😊" if latest["sentiment_label"] == "Positive" else "😡"

            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Total Predictions", f"{total_count:,}")
            with col_metric2:
                pos_pct = (df["sentiment_label"] == "Positive").mean() * 100
                st.metric("Positive %", f"{pos_pct:.1f}%")
            with col_metric3:
                st.metric("Latest", f"{sentiment_emoji} {latest['sentiment_label']}")

            st.markdown(f'> *"{latest["text"]}"*')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Recent Predictions")
                st.dataframe(df[["processed_at", "text", "sentiment_label"]], use_container_width=True, height=400)
            with col2:
                st.subheader("Sentiment Distribution")
                sentiment_counts = df["sentiment_label"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment_label", "count"]

                color_scale = alt.Scale(domain=["Negative", "Positive"], range=["#FF4B4B", "#00F2A9"])

                chart = (
                    alt.Chart(sentiment_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("sentiment_label", title="Sentiment"),
                        y=alt.Y("count", title="Count"),
                        color=alt.Color("sentiment_label", scale=color_scale, legend=None),
                    )
                    .properties(height=350)
                )
                st.altair_chart(chart, use_container_width=True)

    time.sleep(5)
