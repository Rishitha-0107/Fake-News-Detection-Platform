import streamlit as st
import numpy as np
import joblib
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Fake News Detection Platform",
    page_icon="📰",
    layout="wide"
)

# ==========================================
# LOAD FILES
# ==========================================

@st.cache_resource
def load_artifacts():

    tokenizer = joblib.load(
        "tokenizer.pkl"
    )

    encoder = joblib.load(
        "label_encoder.pkl"
    )

    MAX_WORDS = 10000
    MAX_LEN = 500

    model = Sequential()

    model.add(
        Embedding(
            MAX_WORDS,
            128,
            input_length=MAX_LEN
        )
    )

    model.add(
        LSTM(128)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.load_weights(
        "fake_news_weights.weights.h5"
    )

    return model, tokenizer, encoder

model, tokenizer, encoder = load_artifacts()

# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text
    )

    return text

# ==========================================
# HEADER
# ==========================================

st.title(
    "📰 AI Fake News Detection Platform"
)

st.markdown(
    "### LSTM-Based News Authenticity Analyzer"
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "📌 Project Information"
)

st.sidebar.info(
    """
    Model: LSTM

    Dataset:
    Fake & Real News Dataset

    Output:
    Fake / Real News
    """
)

# ==========================================
# INPUT AREA
# ==========================================

news_text = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)

# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🚀 Analyze News",
    use_container_width=True
):

    if len(news_text.strip()) == 0:

        st.warning(
            "Please enter a news article."
        )

    else:

        cleaned = clean_text(
            news_text
        )

        sequence = tokenizer.texts_to_sequences(
            [cleaned]
        )

        padded = pad_sequences(
            sequence,
            maxlen=500
        )

        probability = (
            model.predict(
                padded,
                verbose=0
            )[0][0]
        )

        if probability > 0.5:

            prediction = "Real"

            confidence = (
                probability * 100
            )

        else:

            prediction = "Fake"

            confidence = (
                (1 - probability)
                * 100
            )

        # ==========================
        # RESULTS
        # ==========================

        st.subheader(
            "📊 Prediction Results"
        )

        col1, col2 = st.columns(2)

        with col1:

            if prediction == "Real":

                st.success(
                    "✅ REAL NEWS"
                )

            else:

                st.error(
                    "🚨 FAKE NEWS"
                )

        with col2:

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

        st.progress(
            float(confidence / 100)
        )

        # ==========================
        # RISK LEVEL
        # ==========================

        st.subheader(
            "⚠ Risk Assessment"
        )

        if confidence > 90:

            st.success(
                "Very High Confidence Prediction"
            )

        elif confidence > 75:

            st.info(
                "High Confidence Prediction"
            )

        elif confidence > 60:

            st.warning(
                "Moderate Confidence Prediction"
            )

        else:

            st.error(
                "Low Confidence Prediction"
            )

        # ==========================
        # AI INSIGHTS
        # ==========================

        st.subheader(
            "🤖 AI Insights"
        )

        word_count = len(
            news_text.split()
        )

        char_count = len(
            news_text
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Words",
                word_count
            )

        with c2:

            st.metric(
                "Characters",
                char_count
            )

        with c3:

            st.metric(
                "Prediction",
                prediction
            )

        # ==========================
        # RECOMMENDATION
        # ==========================

        st.subheader(
            "💡 Recommendation"
        )

        if prediction == "Fake":

            st.error(
                """
                Verify this article using
                trusted news sources before
                sharing it.
                """
            )

        else:

            st.success(
                """
                This article appears to be
                from a reliable source.
                """
            )

# ==========================================
# COMPARISON SECTION
# ==========================================

st.divider()

st.subheader(
    "📈 ANN vs SimpleRNN vs LSTM"
)

comparison = {
    "Model":
    ["ANN", "SimpleRNN", "LSTM"],

    "Handles Long Text":
    ["❌", "⚠️", "✅"],

    "Memory":
    ["❌", "Limited", "Excellent"],

    "Context Understanding":
    ["Low", "Medium", "High"],

    "Accuracy":
    ["Medium", "Good", "Best"]
}

st.table(comparison)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "📰 AI Fake News Detection Platform | Deep Learning LSTM"
)
