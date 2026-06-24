# Dockerfile
# Hybrid Movie Recommendation System — Hugging Face Docker Space
#
# Project structure inside the container:
#   /app/
#   ├── app.py                ← Streamlit UI (entrypoint)
#   ├── requirements.txt
#   └── model/                ← lowercase, matches the repository on disk
#       ├── collab_similarity.npz
#       ├── content_similarity.npz
#       ├── hybrid_recommender_model.pkl
#       └── model.py
#
# Local build & run:
#   docker build -t hybrid-movie-recommender .
#   docker run -p 7860:7860 hybrid-movie-recommender
#
# With optional environment variables:
#   docker run -p 7860:7860 -e TMDB_API_KEY=<key> hybrid-movie-recommender
#   docker run -p 7860:7860 -e DATABASE_URL=<url> hybrid-movie-recommender
#
# Hugging Face Spaces conventions enforced:
#   • App listens on $PORT (defaults to 7860)
#   • Container runs as non-root user (uid 1000)
#   • Dependencies installed explicitly from requirements.txt
#   • Streamlit binds to 0.0.0.0 for external reachability

# --- Base image ---
FROM python:3.11-slim

# --- Environment defaults ---
# PORT can be overridden by the HF Spaces runtime
ENV PORT=7860 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Working directory ---
WORKDIR /app

# --- System dependencies ---
# build-essential : needed for scipy / scikit-learn wheel compilation
# curl            : used by the HEALTHCHECK probe
# git             : sometimes required by pip when pulling VCS-based deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (cached layer) ---
# Copy requirements first so this layer is reused when only source code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source + model artifacts ---
# Linux containers are case-sensitive: folder name must match repo exactly.
COPY app.py   ./
COPY model/   ./model/

# --- Non-root user (HF Spaces best practice) ---
# Create a dedicated user, take ownership of /app, and switch to it.
RUN useradd -m -u 1000 user \
    && chown -R user:user /app
USER user

# --- Streamlit cache locations (must be writable by non-root user) ---
ENV STREAMLIT_HOME=/home/user/.streamlit \
    XDG_CACHE_HOME=/home/user/.cache

# --- Network port ---
EXPOSE 7860

# --- Health check ---
# Streamlit exposes a built-in readiness endpoint at /_stcore/health.
# Uses $PORT so the probe always matches the runtime port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail "http://localhost:${PORT}/_stcore/health" || exit 1

# --- Entrypoint ---
# Shell form so $PORT is expanded at container start.
# --server.address=0.0.0.0 makes the app reachable outside the container.
# --server.headless=true skips Streamlit's first-run prompts.
CMD streamlit run app.py \
        --server.port=${PORT} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false