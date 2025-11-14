import streamlit as st
import pandas as pd
import os
import time
import altair as alt
from sqlalchemy import create_engine, text
import requests

# --- Configuration ---
# Load from environment variables, with defaults for local testing
DB_USER = os.getenv("POSTGRES_USER", "gigaflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres_db") # Use service name
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "sentiment_db")

# Construct the database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQL query to fetch data
QUERY = text("""
    SELECT 
        processed_at, 
        text, 
        sentiment_label, 
        sentiment_score
    FROM 
        sentiment_predictions
    ORDER BY 
        processed_at DESC
    LIMIT 100;
""")

# --- Streamlit App ---

st.set_page_config(
    page_title="GigaFlow Live Sentiment",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 GigaFlow: Live Sentiment Analysis")

st.header("🔬 Test the Model Live")
user_text = st.text_input("Enter text to analyze:", "I love this product!")

if st.button("Analyze Sentiment"):
    if user_text:
        try:
            # The 'model_service' is the Docker Compose service name
            response = requests.post(
                "http://model_service:8000/predict", 
                json={"text": user_text},
                timeout=5  # Set a timeout
            )
            
            if response.status_code == 200:
                pred = response.json()
                emoji = "😊" if pred['sentiment_label'] == 'Positive' else "😡"
                st.subheader(f"Result: {emoji} **{pred['sentiment_label']}** (Score: {pred['sentiment_score']:.4f})")
            elif response.status_code == 503:
                st.error("Model service is still loading the model. Please try again in a moment.")
            else:
                st.error(f"Error from model service: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to model service: {e}")
            st.info("The model service might be restarting or loading. Please wait and try again.")
    else:
        st.warning("Please enter some text.")
st.markdown("---")  # Add a visual separator

# Placeholder for the data
data_placeholder = st.empty()

def get_db_connection():
    """Creates a SQLAlchemy engine."""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

engine = get_db_connection()

def load_data():
    """Loads prediction data from the database."""
    if engine is None:
        return pd.DataFrame() # Return empty frame if connection failed

    try:
        with engine.connect() as conn:
            df = pd.read_sql(QUERY, conn)
        return df
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return pd.DataFrame()

# --- Main App Loop ---

while True:
    df = load_data()

    with data_placeholder.container():
        st.header("📊 Live Dashboard")

        if df.empty:
            st.info("No data received yet. Waiting for predictions...")
        else:
            # --- Metrics ---
            st.subheader("Latest Predictions")

            # Get latest prediction
            latest = df.iloc[0]
            sentiment_emoji = "😊" if latest['sentiment_label'] == 'Positive' else "😡"

            st.markdown(f"### {sentiment_emoji} \"{latest['text']}\" -> **{latest['sentiment_label']}**")

            # --- Charts ---
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment_label'].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df, use_container_width=True)
            with col2:
                # Convert Series to DataFrame
                sentiment_counts_df = sentiment_counts.reset_index()
                sentiment_counts_df.columns = ['sentiment_label', 'count']

                # Define the custom color scale
                color_scale = alt.Scale(domain=['Negative', 'Positive'],
                                        range=['#FF4B4B', '#00F2A9']) # Red, Green

                # Create the Altair chart
                chart = alt.Chart(sentiment_counts_df).mark_bar().encode(
                    x=alt.X('sentiment_label', title='Sentiment'),
                    y=alt.Y('count', title='Count'),
                    color=alt.Color('sentiment_label', scale=color_scale, legend=None)
                ).properties(
                    title="Sentiment Distribution"
                )
                st.altair_chart(chart, use_container_width=True)

        # Auto-refresh interval
        time.sleep(5)