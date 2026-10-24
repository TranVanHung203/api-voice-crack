FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/models/huggingface \
    ZEROTTS_CACHE_DIR=/app/models/zerotts/cache \
    ZEROTTS_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    EDGE_WARMUP=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /app && chown appuser:appuser /app

WORKDIR /app
COPY requirements.txt requirements-hidden-suffix.txt ./
# Avoid downloading CUDA packages for a CPU Space.
RUN pip install --upgrade pip \
    && pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY --chown=appuser:appuser server.py zerotts_backend.py zerotts_suffix.py tts_delivery.py audio_cache_store.py vtts_backend.py vieneu_backend.py gwen_backend.py gwen_worker.py setup_gwen.py requirements-gwen.txt README.md LICENSE ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser voice_packs ./voice_packs
USER appuser

# Gwen needs a different transformers version, so install it in its own venv.
RUN python setup_gwen.py --download-model

# Keep model weights in the image so cold starts do not download them again.
ARG ZEROTTS_REVISION=c2bfbd67dc648cac455077333f7cf5c18a2e3bb4
ENV ZEROTTS_REVISION=${ZEROTTS_REVISION}
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('zeroweight-ai/ZeroTTS', revision=os.environ['ZEROTTS_REVISION'], cache_dir=os.environ['ZEROTTS_CACHE_DIR'])"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
