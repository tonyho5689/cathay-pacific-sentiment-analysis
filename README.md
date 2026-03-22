# Cathay Pacific Customer Review Analyzer

**ISOM5240 Deep Learning Business Applications — HKUST**

## Project Objective

> This project aims to perform speech-to-text transcription and sentiment analysis to enhance the understanding of multilingual customer feedback, enabling data-driven service improvement at Cathay Pacific (https://www.cathaypacific.com).

## Architecture

All audio input is **translated to English first**, then analyzed for sentiment:

```
Audio (any lang) → [Whisper Small: translate] → English Text → [Sentiment Model] → Label
```

| Pipeline | Task | Model | Fine-Tuned? |
|----------|------|-------|-------------|
| Pipeline 1 (ASR) | Speech → English Text | `openai/whisper-small` (`task=translate`) | No (pre-trained) |
| Pipeline 2 (Sentiment) | English Text → Sentiment | `tonyho5689/cathay-pacific-sentiment-analysis` | Yes (on English airline reviews) |

## Features

- Multilingual support (99+ languages via Whisper translation)
- Audio file upload (WAV, MP3, FLAC, M4A, OGG) — auto-translated to English
- 3-class sentiment classification (Negative / Neutral / Positive)
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
| `tabularisai/multilingual-sentiment-analysis` | 135M | **82.1%** | **0.8222** | 31s | 1.37s | ✓ |
| `xlm-roberta-base` | 278M | 82.7% | 0.8275 | 67s | 1.83s | |
| `bert-base-multilingual-cased` | 178M | 80.6% | 0.8080 | 53s | 1.78s | |

## Dataset

- **Sentiment Fine-tuning:** Skytrax Airline Reviews (Kaggle, ~8K English reviews with 1-10 star ratings → 3 classes)
- **English only** — no multilingual supplement needed since all input is translated to English first
- **ASR Evaluation:** FLEURS (Google) — 50 English + 50 Chinese audio samples
- **Balancing:** Hybrid — undersample large classes + oversample small class to ~2,000 each

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Models | HuggingFace Transformers |
| ASR + Translation | OpenAI Whisper Small (`task=translate`) |
| Sentiment | Fine-tuned DistilBERT |
| Fine-Tuning | Google Colab (A100 GPU) |
| App Framework | Streamlit |
| Deployment | Streamlit Cloud |
| Languages | Python |
