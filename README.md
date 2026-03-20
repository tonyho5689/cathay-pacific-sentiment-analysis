# Cathay Pacific Customer Review Analyzer

**ISOM5240 Deep Learning Business Applications — HKUST**

## Project Objective

> This project aims to perform speech-to-text transcription and sentiment analysis to enhance the understanding of multilingual customer feedback, enabling data-driven service improvement at Cathay Pacific (https://www.cathaypacific.com).

## Architecture

All input (audio or text) is **translated to English first**, then analyzed for sentiment:

```
Audio (any lang) → [Whisper Medium: translate] → English Text → [Sentiment Model] → Label
Text (any lang)  → [Opus-MT: translate]        → English Text → [Sentiment Model] → Label
```

| Pipeline | Task | Model | Fine-Tuned? |
|----------|------|-------|-------------|
| Pipeline 1 (ASR) | Speech → English Text | `openai/whisper-medium` (`task=translate`) | No (pre-trained) |
| Pipeline 2 (Sentiment) | English Text → Sentiment | `tonyho5689/cathay-pacific-sentiment-analysis` | Yes (on English airline reviews) |

> **Note on Translation Utility:** The app also uses `Helsinki-NLP/opus-mt-mul-en` as a **pre-processing utility** for the text input path. It is not a core pipeline — it simply translates non-English text input to English before passing it to Pipeline 2. For audio input, Whisper's built-in `task=translate` handles translation directly. This utility is not fine-tuned or evaluated in the experiments, as it serves only as a convenience layer for the app's text input mode.

## Features

- Multilingual support (99+ languages via Whisper & Opus-MT translation)
- Audio file upload (WAV, MP3, FLAC, M4A, OGG) — auto-translated to English
- Direct text input — auto-detected and translated if non-English
- 5-class sentiment classification (Very Negative → Very Positive)
- Real-time transcription, translation, and analysis
- Color-coded sentiment visualization

## URLs

| Resource | Link |
|----------|------|
| Fine-tuned Model | `https://huggingface.co/tonyho5689/cathay-pacific-sentiment-analysis` |
| Streamlit App | `https://cathay-pacific-sentiment-analysis-jxhdwetuymk7g2ftgq7n7e.streamlit.app` |
| GitHub Repo | `https://github.com/tonyho5689/cathay-pacific-sentiment-analysis` |

## Setup & Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure

```
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
└── notebooks/
    ├── Testing_Experiments.ipynb          # Step 1: Model comparison experiments
    └── Finetuning_Sentiment_Model.ipynb   # Step 2: Fine-tuning best model
```

## Experiment Results

### Pipeline 1: ASR (3 pre-trained models compared)

| Model | Params | English Accuracy (WER) | Chinese Accuracy (CER) | Avg Accuracy | Runtime | Selected? |
|-------|--------|----------------------|----------------------|--------------|---------|-----------|
| `openai/whisper-base` | 74M | 71.5% | 66.4% | 69.0% | 42s | |
| `openai/whisper-small` | 244M | 74.6% | 79.5% | 77.0% | 63s | |
| `openai/whisper-medium` | 769M | 74.7% | 90.1% | **82.4%** | 127s | ✓ |

### Pipeline 2: Sentiment Analysis (3 models fine-tuned & compared)

| Model | Params | Accuracy | F1 Score | Training Time | Inference Time | Selected? |
|-------|--------|----------|----------|---------------|----------------|-----------|
| `tabularisai/multilingual-sentiment-analysis` | 135M | **81.4%** | **0.8135** | 453s | 2.75s | ✓ |
| `xlm-roberta-base` | 278M | 79.4% | 0.7922 | 908s | 4.63s | |
| `bert-base-multilingual-cased` | 178M | 78.6% | 0.7857 | 775s | 4.95s | |

## Dataset

- **Sentiment Fine-tuning:** Skytrax Airline Reviews (Kaggle, ~8K English reviews with 1-10 star ratings → 5 classes)
- **English only** — no multilingual supplement needed since all input is translated to English first
- **ASR Evaluation:** FLEURS (Google) — 50 English + 50 Chinese audio samples
- **Balancing:** Oversampling to equalize all 5 sentiment classes

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Models | HuggingFace Transformers |
| ASR + Translation | OpenAI Whisper Medium (`task=translate`) |
| Text Translation | Helsinki-NLP/opus-mt-mul-en |
| Sentiment | Fine-tuned DistilBERT |
| Fine-Tuning | Google Colab (T4 GPU) |
| App Framework | Streamlit |
| Deployment | Streamlit Cloud |
| Languages | Python |
