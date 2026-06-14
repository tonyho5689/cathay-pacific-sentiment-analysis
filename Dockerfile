# Hugging Face Space (Docker SDK) running the Streamlit app.
# HF no longer supports a native Streamlit SDK, so we run Streamlit inside a
# container that listens on port 7860 (the port HF Spaces expects).

FROM python:3.11-slim

# System packages: ffmpeg (audio decoding) + libsndfile1 (soundfile backend).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run containers as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python dependencies first for better layer caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (app.py, .streamlit/, etc.).
COPY --chown=user . .

# Streamlit must listen on 0.0.0.0:7860 to be reachable through the HF proxy.
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
