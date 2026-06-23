# Dockerfile
# Hybrid Movie Recommendation System — containerised Streamlit app
#
# Project structure inside the container:
#   /app/
#   ├── requirements.txt
#   └── src/
#       ├── Model/
#       │   ├── collab_similarity.npz
#       │   ├── content_similarity.npz
#       │   ├── hybrid_recommender_model.pkl
#       │   └── model.py
#       └── streamlit_app.py
#
# Build:
#   docker build -t hybrid-movie-recommender .
#
# Run:
#   docker run -p 8501:8501 hybrid-movie-recommender
#
# Run with TMDB poster support (optional):
#   docker run -p 8501:8501 -e TMDB_API_KEY=your_key_here hybrid-movie-recommender

# --- Base image ---
FROM python:3.13.5-slim

# --- Working directory ---
WORKDIR /app

# --- System dependencies ---
# build-essential: required by scipy / scikit-learn wheel compilation
# curl:            used by the HEALTHCHECK probe
# git:             required by some pip packages that pull from VCS
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies ---
# Copied first so Docker caches this layer when only source code changes
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# --- Application source + model artifacts ---
# Copies src/streamlit_app.py and src/Model/ (model.py + .npz + .pkl)
COPY src/ ./src/

# --- Port ---
EXPOSE 8501

# --- Health check ---
# Streamlit exposes a built-in health endpoint at /_stcore/health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# --- Entrypoint ---
# Launches the Streamlit app on 0.0.0.0:8501 so the container port is reachable
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]