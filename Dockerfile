FROM python:3.11-slim

WORKDIR /app

# System deps some of the optional packages (pypdf, sentence-transformers'
# tokenizers) need to build/run cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, as its own step -- plain `pip install torch`
# defaults to the full CUDA build (800MB-2GB+), which isn't needed
# since we only ever run tiny custom MLP models on CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY . .

# Hugging Face Spaces expects port 7860 by default; Railway (and most
# other platforms) inject their own $PORT at runtime. Shell-form CMD
# with a fallback handles both: uses $PORT if set, else 7860.
EXPOSE 7860
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}
