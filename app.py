import streamlit as st
from transformers import pipeline
from gtts import gTTS
import librosa
import numpy as np
import time
import os
import io


# --- Load Models (cached) ---
@st.cache_resource
def load_asr_pipeline():
    """Load Whisper Small with task=translate (any language audio -> English text).
    Using whisper-small instead of whisper-medium for Streamlit Cloud memory constraints."""
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
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



# --- Sentiment Display ---
SENTIMENT_CONFIG = {
    "Negative": {"color": "#DC3545", "emoji": "😞", "bg": "#FDE8EA"},
    "Neutral":  {"color": "#F5B041", "emoji": "😐", "bg": "#FEF7E8"},
    "Positive": {"color": "#27AE60", "emoji": "🙂", "bg": "#E8F8EF"},
}


SAMPLE_REVIEWS = {
    "Negative — Flight delay complaint": "I am extremely disappointed with Cathay Pacific. My flight was delayed by six hours and nobody gave us any information. The staff at the gate were rude and unhelpful.",
    "Negative — Lost luggage": "My luggage was lost and it took five days to get it back. Customer service kept transferring me to different departments and nobody took responsibility.",
    "Negative — Poor cabin experience": "The worst flying experience I have ever had. The seat was broken, the entertainment system did not work, and the food was cold and tasteless.",
    "Neutral — Average flight": "The flight was okay overall. Nothing special but nothing terrible either. The seats were average and the food was decent.",
    "Neutral — Standard economy": "It was a standard economy experience. The check-in was smooth but the legroom was a bit tight. The cabin crew did their job. Just an ordinary flight.",
    "Neutral — Mixed experience": "The flight departed on time which was good, but the meal options were limited. Some things were good, some things could be better.",
    "Positive — Business class": "What an amazing experience flying business class with Cathay Pacific! The lounge was beautiful, the food was restaurant quality, and the flat bed seat was so comfortable.",
    "Positive — Great crew": "I want to thank the wonderful cabin crew on my flight. They were attentive, kind, and went out of their way to help with my special meal request. Outstanding service!",
    "Positive — Loyal customer": "Cathay Pacific is my favorite airline. Every time I fly with them the experience is consistently excellent. Clean aircraft, friendly staff, and great entertainment.",
    "Positive — Smooth journey": "Had a wonderful flight from Hong Kong to London. Everything was perfect from check-in to landing. The crew made sure every passenger was comfortable throughout the journey.",
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

1. **Upload** audio in any language
2. **Translate** to English (Whisper)
3. **Classify** sentiment (3 levels)

**Architecture:**
```
Audio (any lang)
      ↓
Whisper (translate) → EN
      ↓
Sentiment → Label
```

**Models:**
- **ASR:** Whisper Small (`translate`)
- **Sentiment:** Fine-tuned DistilBERT

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
        Upload audio <strong>in any language</strong> — the app translates to English and classifies sentiment.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Load pipelines
    with st.spinner("Loading models... This may take a moment on first run."):
        asr_pipe = load_asr_pipeline()
        sentiment_pipe = load_sentiment_pipeline()

    st.success("All models loaded successfully!")

    # --- Generate Sample Audio ---
    with st.expander("🎙️ Generate Sample Audio for Testing"):
        st.markdown("Type your own review or select a preset sample, then generate an audio file to test the analyzer.")

        input_mode = st.radio("Input mode:", ["Write my own", "Choose a preset sample"], horizontal=True)

        if input_mode == "Write my own":
            review_text = st.text_area("Type your airline review:", height=100, placeholder="e.g. The flight was amazing, the crew was very friendly and helpful...")
        else:
            selected = st.selectbox("Choose a sample review:", list(SAMPLE_REVIEWS.keys()))
            review_text = SAMPLE_REVIEWS[selected]
            st.text_area("Review text:", value=review_text, height=80, disabled=True)

        if st.button("🔊 Generate Audio", use_container_width=True):
            if not review_text or not review_text.strip():
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Generating audio..."):
                    tts = gTTS(text=review_text.strip(), lang="en", slow=False)
                    audio_buffer = io.BytesIO()
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)

                st.audio(audio_buffer, format="audio/mp3")
                audio_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download MP3",
                    data=audio_buffer,
                    file_name="sample_review.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                )
                st.markdown("*Download the file above, then upload it below to analyze.*")

    st.markdown("---")

    # --- Audio Input ---
    st.info(
        "Upload one or more audio files of customer reviews **in any language**. "
        "Whisper will translate them to English automatically. "
        "Supported: WAV, MP3, FLAC, M4A, OGG"
    )

    # Track uploader key for clear button
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    audio_files = st.file_uploader(
        "Choose audio file(s)",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        accept_multiple_files=True,
        help="Upload one or more customer review audio recordings in any language",
        key=f"audio_uploader_{st.session_state.uploader_key}",
    )

    if audio_files:
        # Preview uploaded files
        for af in audio_files:
            st.audio(af, format=f"audio/{af.type.split('/')[-1]}")

        col_analyze, col_clear = st.columns([3, 1])
        with col_clear:
            if st.button("🗑️ Clear Files", use_container_width=True):
                st.session_state.uploader_key += 1
                st.rerun()
        button_label = f"🔍 Analyze {len(audio_files)} Review(s)" if len(audio_files) > 1 else "🔍 Analyze Review"
        with col_analyze:
            analyze_clicked = st.button(button_label, type="primary", use_container_width=True)
        if analyze_clicked:
            all_results = []
            progress_bar = st.progress(0, text="Processing audio files...")

            for i, audio_file in enumerate(audio_files):
                progress_bar.progress((i) / len(audio_files), text=f"Processing {audio_file.name} ({i+1}/{len(audio_files)})...")

                # --- Pipeline 1: ASR ---
                temp_path = f"temp_audio_{audio_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                try:
                    start_time = time.time()
                    audio_array, sr = librosa.load(temp_path, sr=16000)
                    asr_result = asr_pipe({"raw": audio_array, "sampling_rate": sr})
                    english_text = asr_result["text"]
                    asr_time = time.time() - start_time
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                # --- Pipeline 2: Sentiment ---
                start_time = time.time()
                sentiment_result = sentiment_pipe(english_text, truncation=True, max_length=512)
                sentiment_time = time.time() - start_time

                label = sentiment_result[0]["label"]
                score = sentiment_result[0]["score"]

                all_results.append({
                    "File": audio_file.name,
                    "Transcription": english_text,
                    "Sentiment": label,
                    "Confidence": f"{score:.1%}",
                    "ASR Time (s)": f"{asr_time:.2f}",
                    "Sentiment Time (s)": f"{sentiment_time:.2f}",
                })

                # --- Per-file result in expander ---
                with st.expander(f"Result: {audio_file.name}", expanded=(len(audio_files) == 1)):
                    st.markdown("**Step 1: Speech-to-Text (Translate to English)**")
                    st.success(f"Translated to English in {asr_time:.2f}s")
                    st.text_area("English Translation", value=english_text, height=80, disabled=True, key=f"trans_{i}")

                    st.markdown("**Step 2: Sentiment Analysis**")
                    display_sentiment(sentiment_result)

                    st.markdown("**Pipeline Summary**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            '<div class="pipeline-card">'
                            "<h4>Pipeline 1 — ASR + Translation</h4>"
                            f"<p>Model: Whisper Small</p>"
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
                            f"<p>Result: {label} ({score:.1%})</p>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

            progress_bar.progress(1.0, text="All files processed!")

            # --- Overall Batch Summary ---
            if len(all_results) > 1:
                st.markdown("---")
                st.markdown("#### Batch Summary")
                import pandas as pd
                st.dataframe(pd.DataFrame(all_results), use_container_width=True, hide_index=True)

    # --- Footer ---
    st.markdown("---")


if __name__ == "__main__":
    main()
