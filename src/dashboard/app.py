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
MODEL_SERVICE = "http://model_service:8000"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

QUERY_RECENT = text(
    "SELECT processed_at, text, sentiment_label, sentiment_score, top_emotion, language, is_toxic"
    " FROM sentiment_predictions"
    " ORDER BY processed_at DESC"
    " LIMIT 200"
)

QUERY_COUNT = text("SELECT COUNT(*) as total FROM sentiment_predictions")

QUERY_SENTIMENT_OVER_TIME = text(
    "SELECT date_trunc('minute', processed_at) as minute,"
    " sentiment_label, COUNT(*) as count"
    " FROM sentiment_predictions"
    " GROUP BY minute, sentiment_label"
    " ORDER BY minute DESC"
    " LIMIT 300"
)

QUERY_LANGUAGE_COUNTS = text(
    "SELECT language, COUNT(*) as count"
    " FROM sentiment_predictions"
    " WHERE language IS NOT NULL"
    " GROUP BY language"
    " ORDER BY count DESC"
    " LIMIT 10"
)

QUERY_EMOTION_COUNTS = text(
    "SELECT top_emotion, COUNT(*) as count"
    " FROM sentiment_predictions"
    " WHERE top_emotion IS NOT NULL"
    " GROUP BY top_emotion"
    " ORDER BY count DESC"
    " LIMIT 15"
)

# --- Multi-language examples ---
LANGUAGE_EXAMPLES = {
    "English": [
        "I absolutely love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The product works fine, nothing special.",
        "I'm curious about the new features.",
        "Wow, this completely blew my mind!",
    ],
    "French": [
        "J'adore ce produit, c'est fantastique!",
        "C'est vraiment terrible, je suis déçu.",
        "C'est correct, rien de spécial.",
        "Je me demande comment ça fonctionne.",
        "Quelle surprise incroyable!",
    ],
    "Spanish": [
        "¡Me encanta este producto, es increíble!",
        "Es el peor servicio que he recibido.",
        "Está bien, nada especial.",
        "Tengo curiosidad por las nuevas funciones.",
        "¡No puedo creer lo rápido que llegó!",
    ],
    "German": [
        "Ich liebe dieses Produkt, es ist fantastisch!",
        "Das ist wirklich schrecklich und enttäuschend.",
        "Es funktioniert, mehr nicht.",
        "Ich bin neugierig auf die neuen Funktionen.",
        "Wow, das hat mich total überrascht!",
    ],
}

# --- Language Names ---
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "sw": "Swahili",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "ca": "Catalan",
    "gl": "Galician",
    "eu": "Basque",
    "af": "Afrikaans",
    "sq": "Albanian",
    "mk": "Macedonian",
    "sr": "Serbian",
    "bs": "Bosnian",
    "is": "Icelandic",
    "mt": "Maltese",
    "ga": "Irish",
    "cy": "Welsh",
    "la": "Latin",
    "tl": "Tagalog",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
}

# --- Localized Labels ---
LOCALIZED_SENTIMENTS = {
    "en": {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"},
    "fr": {"Positive": "Positif", "Negative": "Négatif", "Neutral": "Neutre"},
    "es": {"Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutro"},
    "de": {"Positive": "Positiv", "Negative": "Negativ", "Neutral": "Neutral"},
    "it": {"Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutrale"},
    "pt": {"Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutro"},
    "nl": {"Positive": "Positief", "Negative": "Negatief", "Neutral": "Neutraal"},
    "ja": {"Positive": "ポジティブ", "Negative": "ネガティブ", "Neutral": "ニュートラル"},
    "zh": {"Positive": "积极", "Negative": "消极", "Neutral": "中性"},
    "ko": {"Positive": "긍정적", "Negative": "부정적", "Neutral": "중립적"},
    "ar": {"Positive": "إيجابي", "Negative": "سلبي", "Neutral": "محايد"},
    "ru": {"Positive": "Положительный", "Negative": "Отрицательный", "Neutral": "Нейтральный"},
}

LOCALIZED_EMOTIONS = {
    "en": {},  # default English labels
    "fr": {
        "joy": "joie",
        "love": "amour",
        "admiration": "admiration",
        "approval": "approbation",
        "anger": "colère",
        "sadness": "tristesse",
        "fear": "peur",
        "disgust": "dégoût",
        "surprise": "surprise",
        "neutral": "neutre",
        "curiosity": "curiosité",
        "confusion": "confusion",
        "excitement": "enthousiasme",
        "disappointment": "déception",
        "annoyance": "agacement",
        "gratitude": "gratitude",
        "optimism": "optimisme",
    },
    "es": {
        "joy": "alegría",
        "love": "amor",
        "admiration": "admiración",
        "approval": "aprobación",
        "anger": "ira",
        "sadness": "tristeza",
        "fear": "miedo",
        "disgust": "asco",
        "surprise": "sorpresa",
        "neutral": "neutro",
        "curiosity": "curiosidad",
        "confusion": "confusión",
        "excitement": "emoción",
        "disappointment": "decepción",
        "annoyance": "molestia",
        "gratitude": "gratitud",
        "optimism": "optimismo",
    },
    "de": {
        "joy": "Freude",
        "love": "Liebe",
        "admiration": "Bewunderung",
        "approval": "Zustimmung",
        "anger": "Wut",
        "sadness": "Traurigkeit",
        "fear": "Angst",
        "disgust": "Ekel",
        "surprise": "Überraschung",
        "neutral": "neutral",
        "curiosity": "Neugier",
        "confusion": "Verwirrung",
        "excitement": "Begeisterung",
        "disappointment": "Enttäuschung",
        "annoyance": "Ärger",
        "gratitude": "Dankbarkeit",
        "optimism": "Optimismus",
    },
}


def get_language_name(code):
    """Convert language code to full name."""
    if not code:
        return None
    return LANGUAGE_NAMES.get(code, code.upper())


def localize_sentiment(label, lang_code):
    """Get localized sentiment label."""
    lang = lang_code if lang_code in LOCALIZED_SENTIMENTS else "en"
    return LOCALIZED_SENTIMENTS.get(lang, LOCALIZED_SENTIMENTS["en"]).get(label, label)


def localize_emotion(emotion, lang_code):
    """Get localized emotion label."""
    lang = lang_code if lang_code in LOCALIZED_EMOTIONS else "en"
    translations = LOCALIZED_EMOTIONS.get(lang, {})
    return translations.get(emotion, emotion)


# --- Streamlit App ---

st.set_page_config(page_title="GigaFlow Live Sentiment", page_icon="🚀", layout="wide")

st.title("🚀 GigaFlow: Live Sentiment & Emotion Analysis")

tab_predict, tab_dashboard, tab_models = st.tabs(["🔬 Test Model", "📊 Live Dashboard", "🏆 Model Registry"])


@st.cache_resource
def get_db_connection():
    try:
        return create_engine(DATABASE_URL)
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None


engine = get_db_connection()


def load_data():
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(QUERY_RECENT, conn)
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return pd.DataFrame()


def load_total_count():
    if engine is None:
        return 0
    try:
        with engine.connect() as conn:
            return conn.execute(QUERY_COUNT).scalar() or 0
    except Exception:
        return 0


def load_sentiment_over_time():
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(QUERY_SENTIMENT_OVER_TIME, conn)
    except Exception:
        return pd.DataFrame()


def load_language_counts():
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(QUERY_LANGUAGE_COUNTS, conn)
    except Exception:
        return pd.DataFrame()


def load_emotion_counts():
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(QUERY_EMOTION_COUNTS, conn)
    except Exception:
        return pd.DataFrame()


def load_model_versions():
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


def predict_text(text_input):
    """Call model service prediction endpoint."""
    try:
        resp = requests.post(f"{MODEL_SERVICE}/predict", json={"text": text_input}, timeout=30)
        if resp.status_code == 200:
            return resp.json(), None
        if resp.status_code == 503:
            return None, "Model service is still loading. Please try again."
        return None, f"Error: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Failed to connect: {e}"


SENTIMENT_COLORS = alt.Scale(
    domain=["Negative", "Neutral", "Positive"],
    range=["#FF4B4B", "#FFD700", "#00F2A9"],
)
EMOJI_MAP = {"Positive": "😊", "Negative": "😡", "Neutral": "😐"}

# ============================================================
# TAB 1: Test Model
# ============================================================
with tab_predict:
    st.header("🔬 Test the Model")

    with st.expander("🌍 Supported Languages"):
        col_s, col_e = st.columns(2)
        with col_s:
            st.markdown("**Sentiment** (16+ languages)")
            st.markdown(
                "English, French, Spanish, German, Italian, Portuguese, "
                "Dutch, Russian, Chinese, Japanese, Korean, Arabic, "
                "Hindi, Turkish, Vietnamese, Polish"
            )
        with col_e:
            st.markdown("**Emotions** (6+ languages)")
            st.markdown("English, French, Spanish, German, Italian, Dutch")

    col_input, col_lang = st.columns([3, 1])
    with col_lang:
        language = st.selectbox("Language examples:", list(LANGUAGE_EXAMPLES.keys()))

    # Quick example buttons
    st.caption("Try an example:")
    for i, example in enumerate(LANGUAGE_EXAMPLES[language]):
        if st.button(example, key=f"ex_{language}_{i}"):
            st.session_state["analyze_text"] = example

    with col_input:
        default_text = st.session_state.get("analyze_text", "")
        user_text = st.text_area(
            "Enter text to analyze:", value=default_text, height=100, placeholder="Type anything in any language..."
        )
        if user_text != default_text:
            st.session_state["analyze_text"] = user_text

    if st.button("🔍 Analyze", type="primary") or user_text:
        if user_text:
            pred, error = predict_text(user_text)
            if error:
                st.error(error)
            elif pred:
                sentiment = pred["sentiment_label"]
                emoji = EMOJI_MAP.get(sentiment, "🤔")

                col_result, col_emotions = st.columns([1, 2])

                with col_result:
                    lang_code = pred.get("language", "en")
                    lang_name = get_language_name(lang_code)
                    localized_sent = localize_sentiment(sentiment, lang_code)

                    st.markdown(f"### {emoji} {localized_sent}")
                    if localized_sent != sentiment:
                        st.caption(f"({sentiment})")

                    top_emotion = pred.get("top_emotion", "")
                    if top_emotion:
                        localized_emo = localize_emotion(top_emotion, lang_code)
                        st.metric("Top Emotion", localized_emo)
                        if localized_emo != top_emotion:
                            st.caption(f"({top_emotion})")

                    if lang_name:
                        st.metric("Language", lang_name)

                    is_toxic = pred.get("is_toxic")
                    if is_toxic:
                        st.error("⚠️ Toxic content detected!")
                    elif is_toxic is not None:
                        st.success("✅ Non-toxic")

                with col_emotions:
                    emotions = pred.get("emotions", {})
                    if emotions:
                        # Localize emotion labels
                        emotion_df = pd.DataFrame(
                            [{"Emotion": localize_emotion(k, lang_code), "Score": v} for k, v in emotions.items()]
                        ).sort_values("Score", ascending=False)

                        chart = (
                            alt.Chart(emotion_df)
                            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                            .encode(
                                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Confidence"),
                                y=alt.Y("Emotion:N", sort="-x", title="", axis=alt.Axis(labelLimit=200)),
                                color=alt.Color(
                                    "Score:Q",
                                    scale=alt.Scale(scheme="viridis"),
                                    legend=None,
                                ),
                            )
                            .properties(height=250, title="Emotion Breakdown")
                        )
                        st.altair_chart(chart, use_container_width=True)

                # --- Explainability ---
                st.caption(
                    "*Word importance is approximate (leave-one-out perturbation). Results may vary for short or ambiguous texts.*"
                )
                if st.button("🔎 Explain this prediction", key="explain_btn"):
                    with st.spinner("Analyzing word importance..."):
                        try:
                            exp_resp = requests.post(
                                f"{MODEL_SERVICE}/explain",
                                json={"text": user_text},
                                timeout=60,
                            )
                            if exp_resp.status_code == 200:
                                exp = exp_resp.json()
                                words = exp.get("explanation", [])
                                if words:
                                    # Color words: green = positive contribution, red = negative
                                    html_parts = []
                                    for w in words:
                                        imp = w["importance"]
                                        if imp > 0.05:
                                            color = f"rgba(0, 200, 0, {min(abs(imp) * 5, 0.8)})"
                                        elif imp < -0.05:
                                            color = f"rgba(255, 0, 0, {min(abs(imp) * 5, 0.8)})"
                                        else:
                                            color = "transparent"
                                        html_parts.append(
                                            f'<span style="background-color:{color};padding:2px 4px;border-radius:3px">{w["word"]}</span>'
                                        )
                                    st.markdown("**Word importance** (green = positive, red = negative):")
                                    st.markdown(" ".join(html_parts), unsafe_allow_html=True)
                            else:
                                st.error(f"Explain failed: {exp_resp.text}")
                        except Exception as e:
                            st.error(f"Could not explain: {e}")

                # --- Feedback Section ---
                st.markdown("---")
                st.caption("Was this prediction correct?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("👍 Correct", key="fb_correct"):
                        try:
                            requests.post(
                                f"{MODEL_SERVICE}/feedback",
                                json={
                                    "original_text": user_text,
                                    "original_sentiment": sentiment,
                                    "original_emotion": pred.get("top_emotion", ""),
                                    "is_correct": True,
                                },
                                timeout=5,
                            )
                            st.success("Thanks for the feedback!")
                        except Exception:
                            st.error("Could not save feedback.")
                with col_no:
                    if st.button("👎 Incorrect", key="fb_incorrect"):
                        st.session_state["show_correction"] = True

                if st.session_state.get("show_correction"):
                    correct_sentiment = st.selectbox(
                        "What should the sentiment be?",
                        ["Positive", "Negative", "Neutral"],
                        key="correct_sent",
                    )
                    if st.button("Submit Correction", key="fb_submit"):
                        try:
                            requests.post(
                                f"{MODEL_SERVICE}/feedback",
                                json={
                                    "original_text": user_text,
                                    "original_sentiment": sentiment,
                                    "original_emotion": pred.get("top_emotion", ""),
                                    "corrected_sentiment": correct_sentiment,
                                    "is_correct": False,
                                },
                                timeout=5,
                            )
                            st.success("Correction saved! This helps improve the model.")
                            st.session_state["show_correction"] = False
                        except Exception:
                            st.error("Could not save feedback.")
        else:
            st.warning("Please enter some text.")

# ============================================================
# TAB 2: Live Dashboard
# ============================================================
with tab_dashboard:
    dashboard_placeholder = st.empty()

# ============================================================
# TAB 3: Model Registry
# ============================================================
with tab_models:
    st.header("🏆 Model Registry")

    try:
        model_resp = requests.get(f"{MODEL_SERVICE}/model", timeout=15)
        if model_resp.status_code == 200:
            current = model_resp.json()
            st.info(f"🔵 **Currently loaded:** {current.get('version', 'unknown')} — `{current.get('uri', 'N/A')}`")
        else:
            st.warning("Model service returned an error.")
    except Exception:
        st.warning("Model service is loading... refresh in a moment.")

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

            rows.append(
                {
                    "Version": v.get("version"),
                    "Status": "👑 Champion" if is_champion else "",
                    "Accuracy": f"{metrics.get('accuracy', 0):.4f}" if metrics.get("accuracy") else "N/A",
                    "F1-Score": f"{metrics.get('f1_score', 0):.4f}" if metrics.get("f1_score") else "N/A",
                    "Model": params.get("model_name") or params.get("sentiment_model", "N/A"),
                    "Dataset Size": params.get("dataset_size", "-"),
                    "Run ID": v.get("run_id", "")[:12],
                }
            )

        df_models = pd.DataFrame(rows)
        st.dataframe(df_models, use_container_width=True, hide_index=True)

        st.markdown(f"**Current Champion:** Version {champion_version}" if champion_version else "**No champion set**")

        st.subheader("🔄 Load a Model Version")
        version_numbers = sorted([v.get("version") for v in versions], reverse=True)
        col_select, col_load, col_champion = st.columns([2, 1, 1])
        with col_select:
            selected_version = st.selectbox("Select version to load:", version_numbers)
        with col_load:
            st.write("")
            if st.button("Load Version"):
                with st.spinner(f"Loading version {selected_version}... (large models may take ~60s)"):
                    try:
                        resp = requests.post(
                            f"{MODEL_SERVICE}/reload",
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
            st.write("")
            if st.button("Load Champion"):
                with st.spinner("Loading champion model... (large models may take ~60s)"):
                    try:
                        resp = requests.post(f"{MODEL_SERVICE}/reload", timeout=120)
                        if resp.status_code == 200:
                            st.success("Champion model loaded!")
                        else:
                            st.error(f"Failed: {resp.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not reach model service: {e}")

        st.markdown(f"📈 [View in MLflow UI]({MLFLOW_URI.replace('mlflow_server', 'localhost')})")

# ============================================================
# Main Loop — updates Live Dashboard tab
# ============================================================
while True:
    df = load_data()
    total_count = load_total_count()

    with dashboard_placeholder.container():
        if df.empty:
            st.info("No data received yet. Waiting for predictions...")
        else:
            latest = df.iloc[0]
            sentiment_emoji = EMOJI_MAP.get(latest["sentiment_label"], "🤔")

            # --- Feedback Stats ---
            try:
                fb_resp = requests.get(f"{MODEL_SERVICE}/feedback/stats", timeout=5)
                if fb_resp.status_code == 200:
                    fb = fb_resp.json()
                    if fb["total_feedback"] > 0:
                        col_fb1, col_fb2, col_fb3 = st.columns(3)
                        with col_fb1:
                            st.metric("User Feedback", fb["total_feedback"])
                        with col_fb2:
                            st.metric("Correct", f"{fb['user_accuracy']:.0%}")
                        with col_fb3:
                            st.metric("Corrections", fb["incorrect"])
            except Exception:
                pass

            # --- Metrics Row ---
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.metric("Total Predictions", f"{total_count:,}")
            with col_m2:
                pos_pct = (df["sentiment_label"] == "Positive").mean() * 100
                st.metric("Positive", f"{pos_pct:.1f}%")
            with col_m3:
                neu_pct = (df["sentiment_label"] == "Neutral").mean() * 100
                st.metric("Neutral", f"{neu_pct:.1f}%")
            with col_m4:
                neg_pct = (df["sentiment_label"] == "Negative").mean() * 100
                st.metric("Negative", f"{neg_pct:.1f}%")
            with col_m5:
                top_em = latest.get("top_emotion", "") if "top_emotion" in df.columns else ""
                st.metric("Latest Emotion", top_em or "-")

            st.markdown(f'> {sentiment_emoji} *"{latest["text"]}"* — **{latest["sentiment_label"]}**')

            # --- Row 1: Sentiment Distribution + Emotion Distribution ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentiment Distribution")
                sentiment_counts = df["sentiment_label"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment_label", "count"]

                chart = (
                    alt.Chart(sentiment_counts)
                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusTopLeft=5)
                    .encode(
                        x=alt.X("sentiment_label", title="", sort=["Positive", "Neutral", "Negative"]),
                        y=alt.Y("count", title="Count"),
                        color=alt.Color("sentiment_label", scale=SENTIMENT_COLORS, legend=None),
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

            with col2:
                st.subheader("Top Emotions")
                emotion_df = load_emotion_counts()
                if not emotion_df.empty:
                    chart = (
                        alt.Chart(emotion_df)
                        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                        .encode(
                            x=alt.X("count:Q", title="Count"),
                            y=alt.Y("top_emotion:N", sort="-x", title=""),
                            color=alt.Color("count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No emotion data yet.")

            # --- Row 2: Language Distribution + Sentiment Over Time ---
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Language Distribution")
                lang_df = load_language_counts()
                if not lang_df.empty:
                    lang_df["language"] = lang_df["language"].apply(get_language_name)
                    chart = (
                        alt.Chart(lang_df)
                        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                        .encode(
                            x=alt.X("count:Q", title="Count"),
                            y=alt.Y("language:N", sort="-x", title=""),
                            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No language data yet.")

            with col4:
                # Toxicity stats
                st.subheader("Content Safety")
                if "is_toxic" in df.columns:
                    toxic_count = df["is_toxic"].sum() if df["is_toxic"].notna().any() else 0
                    safe_count = len(df) - toxic_count
                    col_safe, col_toxic = st.columns(2)
                    with col_safe:
                        st.metric("Safe", f"{safe_count}")
                    with col_toxic:
                        st.metric("Toxic", f"{int(toxic_count)}", delta_color="inverse")
                else:
                    st.info("Toxicity detection not available.")

            # --- Row 3: Sentiment Over Time + Prediction History ---
            col5, col6 = st.columns(2)

            with col5:
                st.subheader("Sentiment Over Time")
                time_df = load_sentiment_over_time()
                if not time_df.empty:
                    chart = (
                        alt.Chart(time_df)
                        .mark_area(opacity=0.7)
                        .encode(
                            x=alt.X("minute:T", title="Time"),
                            y=alt.Y("count:Q", stack=True, title="Predictions"),
                            color=alt.Color("sentiment_label:N", scale=SENTIMENT_COLORS),
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Not enough data for time series.")

            with col6:
                st.subheader("Recent Predictions")
                display_cols = ["processed_at", "text", "sentiment_label"]
                if "top_emotion" in df.columns:
                    display_cols.append("top_emotion")
                if "language" in df.columns:
                    display_cols.append("language")
                display_df = df[display_cols].head(50).copy()
                if "language" in display_df.columns:
                    display_df["language"] = display_df["language"].apply(get_language_name)
                st.dataframe(display_df, use_container_width=True, height=300)

                # CSV Export
                csv_data = df[display_cols].to_csv(index=False)
                st.download_button(
                    "📥 Export CSV",
                    data=csv_data,
                    file_name="gigaflow_predictions.csv",
                    mime="text/csv",
                    key=f"csv_{int(time.time())}",
                )

    time.sleep(5)
