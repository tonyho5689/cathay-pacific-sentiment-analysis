import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import time
import os


# --- Load Models (cached) ---
@st.cache_resource
def load_asr_pipeline():
    """Load Whisper Medium with task=translate (any language audio -> English text)."""
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        chunk_length_s=30,
        generate_kwargs={"task": "translate"},
    )


@st.cache_resource
def load_sentiment_pipeline():
    """Load the fine-tuned English sentiment model from HuggingFace Hub."""
    return pipeline(
        "text-classification",
        model="tonyho5689/cathay-pacific-sentiment-analysis",
    )


@st.cache_resource
def load_translator():
    """Load Helsinki-NLP/opus-mt-mul-en as a pre-processing utility for text input.
    This is NOT a core pipeline -- it simply translates non-English text to English
    before passing it to Pipeline 2 (Sentiment). For audio input, Whisper's built-in
    task=translate handles translation directly."""
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# --- Sentiment Display ---
SENTIMENT_CONFIG = {
    "Very Negative": {"color": "#DC3545", "emoji": "😡", "bg": "#FDE8EA"},
    "Negative":      {"color": "#E67E22", "emoji": "😞", "bg": "#FDF0E6"},
    "Neutral":       {"color": "#F5B041", "emoji": "😐", "bg": "#FEF7E8"},
    "Positive":      {"color": "#27AE60", "emoji": "🙂", "bg": "#E8F8EF"},
    "Very Positive": {"color": "#006564", "emoji": "😄", "bg": "#E6F2F2"},
}


def display_sentiment(result):
    """Display sentiment analysis result with color-coded card."""
    label = result[0]["label"]
    score = result[0]["score"]
    config = SENTIMENT_CONFIG.get(label, {"color": "#808080", "emoji": "?", "bg": "#F0F0F0"})

    st.markdown(
        f"""
        <div style="
            padding: 24px 28px;
            border-radius: 12px;
            background: {config['bg']};
            border-left: 6px solid {config['color']};
            margin: 16px 0;
        ">
            <div style="font-size: 2.2rem; margin-bottom: 4px;">
                {config['emoji']}
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {config['color']}; margin-bottom: 8px;">
                {label}
            </div>
            <div style="font-size: 0.95rem; color: #555;">
                Confidence: <strong>{score:.1%}</strong>
            </div>
            <div style="
                margin-top: 12px;
                background: #E0E0E0;
                border-radius: 6px;
                height: 10px;
                overflow: hidden;
            ">
                <div style="
                    width: {score * 100:.1f}%;
                    height: 100%;
                    background: {config['color']};
                    border-radius: 6px;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    # --- Page Config ---
    st.set_page_config(
        page_title="Cathay Pacific Customer Review Analyzer",
        page_icon="✈️",
        layout="wide",
    )

    # --- Custom CSS ---
    st.markdown(
        """
        <style>
        /* Hide default Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #006564 0%, #004D4C 100%);
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] a {
            color: #A8E6CF !important;
        }
        [data-testid="stSidebar"] code {
            color: #A8E6CF !important;
            background: rgba(255,255,255,0.15) !important;
        }
        [data-testid="stSidebar"] pre {
            background: rgba(0,0,0,0.3) !important;
            border: none !important;
        }
        [data-testid="stSidebar"] pre code {
            color: #E0F0EF !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.2) !important;
        }

        /* Main content cards */
        .stAlert {
            border-radius: 10px;
        }

        /* Pipeline summary cards */
        .pipeline-card {
            background: #F5F7F8;
            border-radius: 10px;
            padding: 16px 20px;
            border: 1px solid #E0E4E7;
        }
        .pipeline-card h4 {
            color: #006564;
            margin: 0 0 8px 0;
            font-size: 0.95rem;
        }
        .pipeline-card p {
            margin: 4px 0;
            font-size: 0.88rem;
            color: #555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            "<h1 style='font-size: 1.5rem; margin-bottom: 0;'>✈️ Cathay Pacific</h1>"
            "<p style='font-size: 0.85rem; opacity: 0.8; margin-top: 2px;'>Customer Review Analyzer</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            """
**How it works:**

1. **Translate** your input to English
2. **Classify** sentiment (5 levels)

**Architecture:**
```
Audio → Whisper (translate) → EN
Text  → Opus-MT (translate) → EN
              ↓
     Sentiment → Label
```

**Models:**
- **ASR:** Whisper Medium (`translate`)
- **Sentiment:** Fine-tuned DistilBERT
- **Translation:** opus-mt-mul-en

**Languages:** 99+ supported
            """
        )

        st.markdown("---")

        st.markdown(
            """
*ISOM5240(L1) Group 4*

*HO, Siu Hung*

*CHEUNG, Hiu Ling*
            """
        )

        st.markdown("---")

        st.markdown(
            "**Links:**\n"
            "- [Cathay Pacific](https://www.cathaypacific.com)\n"
            "- [HuggingFace Model](https://huggingface.co/tonyho5689/cathay-pacific-sentiment-analysis)\n"
            "- [GitHub Repo](https://github.com/tonyho5689/cathay-pacific-sentiment-analysis)\n"
        )

    # --- Title & Description ---
    st.markdown(
        "<h1 style='margin-bottom: 0;'>✈️ Cathay Pacific Customer Review Analyzer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style='font-size: 1.05rem; color: #555; margin-top: 8px;'>
        Analyze multilingual customer feedback using deep learning.
        Upload audio or type text in <strong>any language</strong> — the app translates to English and classifies sentiment.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Load pipelines
    with st.spinner("Loading models... This may take a moment on first run."):
        asr_pipe = load_asr_pipeline()
        sentiment_pipe = load_sentiment_pipeline()
        translator_tokenizer, translator_model = load_translator()

    st.success("All models loaded successfully!")

    # --- Input Mode Selection ---
    input_mode = st.radio(
        "Choose input method:",
        ["🎤 Upload Audio File", "⌨️ Type Text Directly"],
        horizontal=True,
    )

    st.markdown("")

    # --- Audio Input Mode ---
    if input_mode == "🎤 Upload Audio File":
        st.info(
            "Upload an audio file of a customer review **in any language**. "
            "Whisper will translate it to English automatically. "
            "Supported: WAV, MP3, FLAC, M4A, OGG"
        )

        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "m4a", "ogg"],
            help="Upload a customer review audio recording in any language",
        )

        if audio_file is not None:
            st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")

            if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
                # --- Pipeline 1: ASR (translate to English) ---
                st.markdown("#### Step 1: Speech-to-Text (Translate to English)")
                with st.spinner("Transcribing and translating audio to English..."):
                    start_time = time.time()

                    temp_path = f"temp_audio_{audio_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(audio_file.getbuffer())

                    try:
                        asr_result = asr_pipe(temp_path)
                        english_text = asr_result["text"]
                        asr_time = time.time() - start_time
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                st.success(f"Translated to English in {asr_time:.2f}s")
                st.text_area(
                    "English Translation",
                    value=english_text,
                    height=100,
                    disabled=True,
                )

                # --- Pipeline 2: Sentiment Analysis ---
                st.markdown("#### Step 2: Sentiment Analysis")
                with st.spinner("Analyzing sentiment..."):
                    start_time = time.time()
                    sentiment_result = sentiment_pipe(english_text)
                    sentiment_time = time.time() - start_time

                display_sentiment(sentiment_result)

                # --- Summary ---
                st.markdown("---")
                st.markdown("#### Pipeline Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        '<div class="pipeline-card">'
                        "<h4>Pipeline 1 — ASR + Translation</h4>"
                        f"<p>Model: Whisper Medium</p>"
                        f"<p>Runtime: {asr_time:.2f}s</p>"
                        f"<p>Output: {len(english_text)} chars (English)</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        '<div class="pipeline-card">'
                        "<h4>Pipeline 2 — Sentiment Analysis</h4>"
                        f"<p>Model: Fine-tuned DistilBERT</p>"
                        f"<p>Runtime: {sentiment_time:.2f}s</p>"
                        f"<p>Result: {sentiment_result[0]['label']} ({sentiment_result[0]['score']:.1%})</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

    # --- Text Input Mode ---
    else:
        st.info(
            "Type or paste a customer review **in any language**. "
            "Non-English text will be automatically translated to English before analysis."
        )

        text_input = st.text_area(
            "Enter review text",
            placeholder="e.g., The flight was excellent! The cabin crew was very friendly. "
            "/ 航班非常好，机组人员很友善。 / フライトは素晴らしかったです。",
            height=150,
        )

        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if text_input.strip():
                # --- Detect language and translate if needed ---
                try:
                    detected_lang = detect(text_input)
                except Exception:
                    detected_lang = "en"

                if detected_lang != "en":
                    st.markdown("#### Step 1: Translation to English")
                    with st.spinner(f"Detected language: **{detected_lang}** — Translating to English..."):
                        start_time = time.time()
                        inputs = translator_tokenizer(text_input, return_tensors="pt", max_length=512, truncation=True)
                        translated = translator_model.generate(**inputs, max_length=512)
                        english_text = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
                        translate_time = time.time() - start_time

                    st.success(f"Translated from **{detected_lang}** to English in {translate_time:.2f}s")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original:**")
                        st.text(text_input)
                    with col2:
                        st.markdown("**English Translation:**")
                        st.text(english_text)
                else:
                    english_text = text_input
                    translate_time = 0

                # --- Sentiment Analysis ---
                step_label = "Sentiment Analysis" if detected_lang == "en" else "Step 2: Sentiment Analysis"
                st.markdown(f"#### {step_label}")
                with st.spinner("Analyzing sentiment..."):
                    start_time = time.time()
                    sentiment_result = sentiment_pipe(english_text)
                    sentiment_time = time.time() - start_time

                display_sentiment(sentiment_result)

                # --- Summary ---
                st.markdown("---")
                st.markdown("#### Pipeline Summary")
                if detected_lang != "en":
                    st.markdown(
                        '<div class="pipeline-card">'
                        f"<h4>Translation</h4>"
                        f"<p>{detected_lang} → English ({translate_time:.2f}s)</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                st.markdown(
                    '<div class="pipeline-card">'
                    f"<h4>Sentiment Analysis</h4>"
                    f"<p>Result: {sentiment_result[0]['label']} ({sentiment_result[0]['score']:.1%})</p>"
                    f"<p>Runtime: {sentiment_time:.2f}s</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Please enter some text to analyze.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 16px 0; color: #999; font-size: 0.85rem;'>
            ISOM5240(L1) Deep Learning Business Applications | HKUST |
            Cathay Pacific Customer Review Analyzer
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
