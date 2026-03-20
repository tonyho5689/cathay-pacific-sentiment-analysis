import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import time
import os


# --- Load Models (cached) ---
@st.cache_resource
def load_asr_pipeline():
    """Load Whisper Medium with task=translate (any language audio → English text)."""
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
    This is NOT a core pipeline — it simply translates non-English text to English
    before passing it to Pipeline 2 (Sentiment). For audio input, Whisper's built-in
    task=translate handles translation directly."""
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# --- Sentiment Display Helper ---
SENTIMENT_COLORS = {
    "Very Negative": "#FF4B4B",
    "Negative": "#FF8C00",
    "Neutral": "#FFD700",
    "Positive": "#90EE90",
    "Very Positive": "#00CC00",
}

SENTIMENT_EMOJIS = {
    "Very Negative": "😡",
    "Negative": "😞",
    "Neutral": "😐",
    "Positive": "🙂",
    "Very Positive": "😄",
}


def display_sentiment(result):
    """Display sentiment analysis result with color coding."""
    label = result[0]["label"]
    score = result[0]["score"]

    color = SENTIMENT_COLORS.get(label, "#808080")
    emoji = SENTIMENT_EMOJIS.get(label, "❓")

    st.markdown("### Sentiment Analysis Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentiment", f"{emoji} {label}")
    with col2:
        st.metric("Confidence", f"{score:.1%}")
    with col3:
        st.progress(score, text=f"{score:.1%}")

    st.markdown(
        f'<div style="padding: 20px; border-radius: 10px; '
        f"background-color: {color}20; border-left: 5px solid {color}; "
        f'margin: 10px 0;">'
        f'<h3 style="color: {color}; margin: 0;">{emoji} {label}</h3>'
        f"<p>Confidence: {score:.1%}</p>"
        f"</div>",
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

    # --- Sidebar ---
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/en/thumb/1/17/Cathay_Pacific_logo.svg/300px-Cathay_Pacific_logo.svg.png",
            width=200,
        )
        st.title("About This App")
        st.markdown(
            """
            **Cathay Pacific Customer Review Analyzer** uses deep learning to:

            1. **Transcribe & translate** audio reviews to English text
            2. **Analyze sentiment** of the English text

            **Architecture:**
            ```
            Audio (any lang) → Whisper (translate) → English
            Text (any lang)  → Opus-MT (translate)  → English
                                        ↓
                              Sentiment Analysis → Label
            ```

            **Models Used:**
            - **Pipeline 1 (ASR):** OpenAI Whisper Medium (`task=translate`)
            - **Pipeline 2 (Sentiment):** Fine-tuned DistilBERT (English)
            - **Translation:** Helsinki-NLP/opus-mt-mul-en (for text input)

            **Supported Input Languages:** 99+ (via Whisper & Opus-MT)

            ---
            *ISOM5240 Deep Learning Business Applications*
            *HKUST — Prof. James Kwok*
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
    st.title("✈️ Cathay Pacific Customer Review Analyzer")
    st.markdown(
        """
        > This project aims to perform speech-to-text transcription and sentiment analysis
        > to enhance the understanding of multilingual customer feedback, enabling
        > data-driven service improvement at **Cathay Pacific**.

        Upload an audio review or type your feedback below in **any language**. The app will:
        1. **Translate** your input to English (via Whisper for audio, Opus-MT for text)
        2. **Classify** the sentiment as Very Negative / Negative / Neutral / Positive / Very Positive
        """
    )

    st.markdown("---")

    # Load pipelines
    with st.spinner("Loading models... This may take a moment on first run."):
        asr_pipe = load_asr_pipeline()
        sentiment_pipe = load_sentiment_pipeline()
        translator_tokenizer, translator_model = load_translator()

    st.success("Models loaded successfully!")

    # --- Input Mode Selection ---
    input_mode = st.radio(
        "Choose input method:",
        ["🎤 Upload Audio File", "⌨️ Type Text Directly"],
        horizontal=True,
    )

    # --- Audio Input Mode ---
    if input_mode == "🎤 Upload Audio File":
        st.markdown("### Upload Audio Review")
        st.info(
            "Upload an audio file of a customer review **in any language**. "
            "Whisper will translate it to English automatically. "
            "Supported formats: WAV, MP3, FLAC, M4A, OGG"
        )

        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "m4a", "ogg"],
            help="Upload a customer review audio recording in any language",
        )

        if audio_file is not None:
            # Display audio player
            st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")

            if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
                # --- Pipeline 1: ASR (translate to English) ---
                st.markdown("### Step 1: Speech-to-Text (Translate to English)")
                with st.spinner("Transcribing and translating audio to English..."):
                    start_time = time.time()

                    # Save uploaded file temporarily
                    temp_path = f"temp_audio_{audio_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(audio_file.getbuffer())

                    try:
                        asr_result = asr_pipe(temp_path)
                        english_text = asr_result["text"]
                        asr_time = time.time() - start_time
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                st.success(f"Translation completed in {asr_time:.2f} seconds")
                st.markdown("**English Translation:**")
                st.text_area(
                    "Translation",
                    value=english_text,
                    height=100,
                    disabled=True,
                    label_visibility="collapsed",
                )

                # --- Pipeline 2: Sentiment Analysis ---
                st.markdown("### Step 2: Sentiment Analysis")
                with st.spinner("Analyzing sentiment..."):
                    start_time = time.time()
                    sentiment_result = sentiment_pipe(english_text)
                    sentiment_time = time.time() - start_time

                st.success(f"Sentiment analysis completed in {sentiment_time:.2f} seconds")
                display_sentiment(sentiment_result)

                # --- Summary ---
                st.markdown("---")
                st.markdown("### Pipeline Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pipeline 1 — ASR + Translation (Whisper Medium)**")
                    st.markdown(f"- Runtime: {asr_time:.2f}s")
                    st.markdown(f"- Output: {len(english_text)} characters (English)")
                with col2:
                    st.markdown("**Pipeline 2 — Sentiment Analysis (DistilBERT)**")
                    st.markdown(f"- Runtime: {sentiment_time:.2f}s")
                    st.markdown(
                        f"- Result: {sentiment_result[0]['label']} "
                        f"({sentiment_result[0]['score']:.1%})"
                    )

    # --- Text Input Mode ---
    else:
        st.markdown("### Type Your Review")
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
                    st.markdown("### Step 1: Translation to English")
                    with st.spinner(f"Detected language: **{detected_lang}** — Translating to English..."):
                        start_time = time.time()
                        inputs = translator_tokenizer(text_input, return_tensors="pt", max_length=512, truncation=True)
                        translated = translator_model.generate(**inputs, max_length=512)
                        english_text = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
                        translate_time = time.time() - start_time

                    st.success(f"Translated from **{detected_lang}** to English in {translate_time:.2f}s")
                    st.markdown("**Original:**")
                    st.text(text_input)
                    st.markdown("**English Translation:**")
                    st.text(english_text)
                else:
                    english_text = text_input
                    translate_time = 0

                # --- Sentiment Analysis ---
                st.markdown("### Sentiment Analysis Result" if detected_lang == "en" else "### Step 2: Sentiment Analysis")
                with st.spinner("Analyzing sentiment..."):
                    start_time = time.time()
                    sentiment_result = sentiment_pipe(english_text)
                    sentiment_time = time.time() - start_time

                st.success(
                    f"Sentiment analysis completed in {sentiment_time:.2f} seconds"
                )
                display_sentiment(sentiment_result)

                # --- Summary ---
                st.markdown("---")
                st.markdown("### Pipeline Summary")
                if detected_lang != "en":
                    st.markdown(f"**Translation:** {detected_lang} → English ({translate_time:.2f}s)")
                st.markdown(f"**Sentiment:** {sentiment_result[0]['label']} ({sentiment_result[0]['score']:.1%}) — {sentiment_time:.2f}s")
            else:
                st.warning("Please enter some text to analyze.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "ISOM5240 Deep Learning Business Applications | HKUST | "
        "Cathay Pacific Customer Review Analyzer"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
