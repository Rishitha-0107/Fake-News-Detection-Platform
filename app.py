import streamlit as st
import numpy as np
import joblib
import json
import re
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.text import (
    tokenizer_from_json
)

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Fake News Detection Platform",
    page_icon="📰",
    layout="wide"
)

# =====================================
# BASE DIRECTORY
# =====================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =====================================
# LOAD ARTIFACTS
# =====================================

@st.cache_resource
def load_artifacts():

    tokenizer_path = os.path.join(
        BASE_DIR,
        "tokenizer.json"
    )

    encoder_path = os.path.join(
        BASE_DIR,
        "label_encoder.pkl"
    )

    weights_path = os.path.join(
        BASE_DIR,
        "fake_news_weights.weights.h5"
    )

    # Load tokenizer JSON
    with open(
        tokenizer_path,
        "r",
        encoding="utf-8"
    ) as f:

        tokenizer = tokenizer_from_json(
            f.read()
        )

    # Load encoder
    encoder = joblib.load(
        encoder_path
    )

    # Rebuild model
    model = Sequential()

    model.add(
        Embedding(
            input_dim=10000,
            output_dim=128,
            input_length=500
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

    model.build(
        input_shape=(None, 500)
    )

    model.load_weights(
        weights_path
    )

    return model, tokenizer, encoder

# =====================================
# LOAD
# =====================================

try:

    model, tokenizer, encoder = (
        load_artifacts()
    )

except Exception as e:

    st.error(
        f"Loading Error: {e}"
    )

    st.stop()

# =====================================
# CLEAN TEXT
# =====================================

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

# =====================================
# HEADER
# =====================================

st.title(
    "📰 AI Fake News Detection Platform"
)

st.markdown(
    "### Deep Learning LSTM-Based News Verification System"
)

st.divider()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header(
    "📌 Project Information"
)

st.sidebar.success(
    """
    Model: LSTM

    Dataset:
    Fake & Real News Dataset

    Output:
    Fake / Real
    """
)

# =====================================
# INPUT
# =====================================

news_text = st.text_area(
    "Paste News Article",
    height=250
)

# =====================================
# PREDICT
# =====================================

if st.button(
    "🚀 Analyze News",
    use_container_width=True
):

    if len(news_text.strip()) == 0:

        st.warning(
            "Please enter a news article."
        )

    else:

        cleaned_text = clean_text(
            news_text
        )

        sequence = tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        padded = pad_sequences(
            sequence,
            maxlen=500
        )

        probability = float(
            model.predict(
                padded,
                verbose=0
            )[0][0]
        )

        if probability >= 0.5:

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

        # =========================
        # RESULTS
        # =========================

        st.subheader(
            "📊 Prediction Results"
        )

        c1, c2 = st.columns(2)

        with c1:

            if prediction == "Real":

                st.success(
                    "✅ REAL NEWS"
                )

            else:

                st.error(
                    "🚨 FAKE NEWS"
                )

        with c2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.progress(
            confidence / 100
        )

        # =========================
        # INSIGHTS
        # =========================

        st.subheader(
            "🤖 AI Insights"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Words",
                len(news_text.split())
            )

        with col2:

            st.metric(
                "Characters",
                len(news_text)
            )

        with col3:

            st.metric(
                "Prediction",
                prediction
            )

        # =========================
        # RECOMMENDATION
        # =========================

        st.subheader(
            "💡 Recommendation"
        )

        if prediction == "Fake":

            st.error(
                """
                Verify this article from
                trusted news sources before
                sharing.
                """
            )

        else:

            st.success(
                """
                Article appears genuine
                according to the model.
                """
            )

# =====================================
# MODEL COMPARISON
# =====================================

st.divider()

st.subheader(
    "📈 ANN vs SimpleRNN vs LSTM"
)

comparison_data = {
    "Model":
        ["ANN", "SimpleRNN", "LSTM"],

    "Long Text":
        ["❌", "⚠️", "✅"],

    "Memory":
        ["❌", "Limited", "Excellent"],

    "Accuracy":
        ["Medium", "Good", "Best"],

    "Context":
        ["Low", "Medium", "High"]
}

st.table(comparison_data)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "📰 AI Fake News Detection Platform | LSTM Deep Learning Model"
)
