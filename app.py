# app.py
# Hybrid Movie Recommendation System — Streamlit UI
#
# Project structure:
#   HYBRID-MOVIE-RECOMMENDER/
#   ├── model/                              ← lowercase, matches filesystem
#   │   ├── collab_similarity.npz           ← item-item collaborative filtering matrix
#   │   ├── content_similarity.npz          ← content-based similarity matrix
#   │   ├── hybrid_recommender_model.pkl    ← dataframes + lookup dicts
#   │   └── model.py                        ← recommendation engine
#   ├── app.py                              ← this file (container entrypoint)
#   ├── Dockerfile
#   ├── requirements.txt
#   ├── README.md
#   └── .gitattributes
#
# Run locally:
#   streamlit run app.py
#
# Run on Hugging Face Spaces (Docker SDK):
#   Built and launched by the Dockerfile via `streamlit run app.py`
#   on 0.0.0.0:$PORT (defaults to 7860).

import ast
import io
import os
import sys
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# __file__   → app.py (repo root)
# ROOT_DIR   → repo root
# MODEL_DIR  → model/   (lowercase — Linux is case-sensitive)
#
# Insert ROOT_DIR into sys.path so `from model.model import …` resolves
# regardless of the working directory (local dev, Docker, or HF Space).
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parent     # → repo root
MODEL_DIR = ROOT_DIR / "model"                  # → model/  (lowercase!)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Artifact paths (model/model.py loads its own copies via its own BASE_DIR)
# ---------------------------------------------------------------------------
PKL_PATH         = MODEL_DIR / "hybrid_recommender_model.pkl"  # dataframes + lookup dicts
CONTENT_SIM_PATH = MODEL_DIR / "content_similarity.npz"        # content-based sparse matrix
COLLAB_SIM_PATH  = MODEL_DIR / "collab_similarity.npz"         # collaborative filtering sparse matrix

# ---------------------------------------------------------------------------
# Import the recommendation engine from model/model.py
# ---------------------------------------------------------------------------
from model.model import (              # noqa: E402  (import after sys.path patch)
    load_model,
    hybrid_recommend,
    fuzzy_match_movies as engine_fuzzy_match,
    get_movie_genres,
    normalize_scores,
)

TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Read TMDB API key from .env or HF Space secrets (optional — used for poster fetching)
_TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")


# ---------------------------------------------------------------------------
# Poster helpers
# ---------------------------------------------------------------------------
def build_poster_path(poster_path):
    """Normalise a raw poster_path column value into a full TMDB URL."""
    if poster_path is None or (isinstance(poster_path, float) and pd.isna(poster_path)):
        return None
    poster_path = str(poster_path).strip()
    if not poster_path or poster_path.lower() in {"nan", "none", "null"}:
        return None
    if poster_path.startswith("http://") or poster_path.startswith("https://"):
        return poster_path
    if not poster_path.startswith("/"):
        poster_path = f"/{poster_path}"
    return f"{TMDB_POSTER_BASE}{poster_path}"


# Browser-like headers to reduce TMDB request rejections
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_POSTER_CACHE_LIMIT  = 1500   # max in-memory cached posters
_MAX_POSTER_ATTEMPTS = 4      # give up after this many failed fetches per movie

_poster_cache:      dict[str, bytes] = {}
_poster_attempts:   dict[str, int]   = {}
_poster_lock = threading.Lock()
_placeholder_cache: dict[str, bytes] = {}


def _new_session() -> requests.Session:
    """Create an HTTP session with retry logic and browser-like headers."""
    s = requests.Session()
    s.headers.update(_UA)
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=2, connect=2, read=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
    except Exception:
        pass
    return s


def _extract_poster_url(page_html: str) -> str:
    """Scrape og:image or direct media URL from a TMDB movie page."""
    m = re.search(
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        page_html, re.IGNORECASE,
    )
    if m:
        return re.sub(r"/t/p/w\d+/", "/t/p/w500/", m.group(1))
    m = re.search(
        r"https://media\.themoviedb\.org/t/p/w\d+/[^\"<> ]+\.jpg", page_html
    )
    if m:
        return re.sub(r"/t/p/w\d+/", "/t/p/w500/", m.group(0))
    return ""


def _download_poster_bytes(tmdb_id: str) -> bytes | None:
    """
    Fetch poster image bytes for a given TMDB movie ID.

    Strategy:
      1. TMDB REST API  (requires _TMDB_API_KEY)
      2. Scrape the public TMDB movie page as fallback

    All outbound traffic is HTTPS (port 443) — HF Spaces compliant.
    """
    if not tmdb_id:
        return None
    session = _new_session()
    try:
        poster_url = ""

        # --- Strategy 1: official TMDB API ---
        if _TMDB_API_KEY:
            try:
                r = session.get(
                    f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                    params={"api_key": _TMDB_API_KEY},
                    timeout=10,
                )
                if r.ok:
                    pp = r.json().get("poster_path")
                    if pp:
                        poster_url = f"{TMDB_POSTER_BASE}{pp}"
            except Exception:
                pass

        # --- Strategy 2: scrape TMDB movie page ---
        if not poster_url:
            try:
                r = session.get(
                    f"https://www.themoviedb.org/movie/{tmdb_id}", timeout=15
                )
                if r.ok:
                    poster_url = _extract_poster_url(r.text)
            except Exception:
                pass

        if not poster_url:
            return None

        # --- Download the actual image ---
        r = session.get(poster_url, timeout=15)
        if r.ok and r.headers.get("content-type", "").startswith("image/") and r.content:
            return r.content

    except Exception:
        return None
    finally:
        session.close()
    return None


def resolve_poster(tmdb_id: str) -> bytes | None:
    """Return cached poster bytes, or fetch and cache them (thread-safe)."""
    tmdb_id = str(tmdb_id or "")
    if not tmdb_id:
        return None

    with _poster_lock:
        if tmdb_id in _poster_cache:
            return _poster_cache[tmdb_id]
        if _poster_attempts.get(tmdb_id, 0) >= _MAX_POSTER_ATTEMPTS:
            return None  # stop retrying permanently failed IDs

    data = _download_poster_bytes(tmdb_id)

    with _poster_lock:
        if data:
            _poster_cache[tmdb_id] = data
            _poster_attempts.pop(tmdb_id, None)
            # Evict oldest entries when cache is full
            if len(_poster_cache) > _POSTER_CACHE_LIMIT:
                for stale in list(_poster_cache.keys())[: len(_poster_cache) - _POSTER_CACHE_LIMIT]:
                    _poster_cache.pop(stale, None)
        else:
            _poster_attempts[tmdb_id] = _poster_attempts.get(tmdb_id, 0) + 1
    return data


def prefetch_posters(tmdb_ids) -> None:
    """Concurrently warm the poster cache for a batch of TMDB IDs."""
    pending = []
    with _poster_lock:
        for tid in tmdb_ids:
            tid = str(tid or "")
            if not tid or tid in _poster_cache or _poster_attempts.get(tid, 0) >= _MAX_POSTER_ATTEMPTS:
                continue
            pending.append(tid)
    if pending:
        with ThreadPoolExecutor(max_workers=5) as ex:
            list(ex.map(resolve_poster, pending))


def placeholder_poster(title=None) -> bytes:
    """Generate a dark-themed placeholder PNG when a real poster is unavailable."""
    key = (title or "No Poster")[:40]
    if key in _placeholder_cache:
        return _placeholder_cache[key]
    img  = Image.new("RGB", (500, 750), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw.rectangle([28, 32, 472, 718], outline=(99, 102, 241), width=3)
    text = key if len(key) <= 22 else key[:21] + "…"
    draw.text((250, 360), text,                 fill=(229, 231, 235), anchor="mm")
    draw.text((250, 396), "poster unavailable", fill=(148, 163, 184), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    _placeholder_cache[key] = data
    return data


def poster_for(movie) -> bytes:
    """Return poster bytes for a movie dict, falling back to a placeholder."""
    return resolve_poster(movie.get("tmdb_id", "")) or placeholder_poster(movie.get("title"))


def _clean_overview(overview_raw):
    """Normalise overview values that may be stored as lists or stringified lists."""
    if isinstance(overview_raw, list):
        return " ".join(map(str, overview_raw))
    if isinstance(overview_raw, str) and overview_raw.startswith("[") and overview_raw.endswith("]"):
        try:
            parsed = ast.literal_eval(overview_raw)
            if isinstance(parsed, list):
                return " ".join(map(str, parsed))
        except Exception:
            return overview_raw.strip("[]").replace("'", "").replace('"', "")
    return "" if overview_raw is None else str(overview_raw)


# ---------------------------------------------------------------------------
# Engine loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine() -> dict:
    """
    Load all recommender artifacts and build the metadata lookup table.

    Artifacts loaded (all from model/):
      • hybrid_recommender_model.pkl  — new_df1, df1, df2_clean, title2idx,
                                        idx2title (no CSV files needed at runtime)
      • content_similarity.npz        — prebuilt content-based sparse matrix
                                        (loaded inside model/model.py via BASE_DIR)
      • collab_similarity.npz         — prebuilt collaborative filtering sparse
                                        matrix (loaded inside model/model.py via BASE_DIR)

    model/model.py owns the recommendation state; this function additionally
    reads the pickle to extract display metadata (df1) for the Streamlit UI
    (poster URLs, genres, taglines, etc.).
    """
    # Validate artifacts before doing anything else
    for path in (PKL_PATH, CONTENT_SIM_PATH, COLLAB_SIM_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}\n"
                f"Expected location: {MODEL_DIR}"
            )

    # Trigger model/model.py to load its own copies of the matrices + dataframes
    load_model()

    # Load the pickle a second time only to extract display metadata for the UI.
    # model/model.py already holds the recommender state; we just need df1 here.
    import pickle
    with open(PKL_PATH, "rb") as f:
        model_data = pickle.load(f)

    new_df1   = model_data["new_df1"]                          # tagged movie dataframe
    df1       = model_data["df1"]                              # full movie metadata
    df2_clean = model_data.get("df2_clean", pd.DataFrame())    # user ratings (optional)
    title2idx = model_data.get("title2idx", {})                # title → collab matrix index
    idx2title = model_data.get("idx2title", {})                # collab matrix index → title

    # --- Build display metadata lookup (title → display dict) ---
    d = df1.copy()
    d["vote_average"] = pd.to_numeric(d["vote_average"], errors="coerce").fillna(0.0)
    d["vote_count"]   = pd.to_numeric(d["vote_count"],   errors="coerce").fillna(0).astype(int)
    d["runtime"]      = pd.to_numeric(d["runtime"],      errors="coerce").fillna(0.0)
    d["popularity"]   = pd.to_numeric(d["popularity"],   errors="coerce").fillna(0.0)
    d["overview"]     = d["overview"].fillna("")
    d["poster_path"]  = d["poster_path"].fillna("")
    d["release_date"] = d["release_date"].fillna("")
    d["tagline"]      = d["tagline"].fillna("")

    metadata_lookup: dict[str, dict] = {}
    for row in d.itertuples(index=False):
        title = row.original_title
        if not isinstance(title, str) or not title:
            continue
        genres = row.genres if isinstance(row.genres, list) else []
        metadata_lookup[title] = {
            "title":        title,
            "imdb_id":      str(getattr(row, "imdb_id", "") or ""),
            "tmdb_id":      str(getattr(row, "id",      "") or ""),
            "overview":     _clean_overview(row.overview),
            "poster_url":   build_poster_path(row.poster_path),
            "release_date": str(row.release_date),
            "vote_average": float(row.vote_average),
            "vote_count":   int(row.vote_count),
            "tagline":      str(row.tagline),
            "runtime":      float(row.runtime),
            "genres":       genres,
            "popularity":   float(row.popularity),
        }

    return {
        "new_df1":          new_df1,
        "df2_clean":        df2_clean,
        "title2idx":        title2idx,
        "idx2title":        idx2title,
        "metadata_lookup":  metadata_lookup,
        "available_titles": list(metadata_lookup.keys()),
    }


# ---------------------------------------------------------------------------
# Recommendation wrapper
# ---------------------------------------------------------------------------
def get_recommendations(
    eng: dict,
    movies: list[str],
    top_n: int = 10,
    use_genre_filter: bool = True,
) -> tuple[list[dict], list[str]]:
    """
    Thin wrapper around model/model.py's hybrid_recommend().

    Converts (title, score) pairs returned by the engine into the richer
    display dicts that the UI expects, adding poster/metadata fields.
    """
    if not movies:
        return [], []

    # engine_fuzzy_match is imported from model/model.py; it uses the same
    # title index that was built during load_model(), so no duplication occurs.
    matched = engine_fuzzy_match(movies)
    if not matched:
        return [], []

    # hybrid_recommend returns [(title, normalised_score), …]
    raw_results: list[tuple[str, float]] = hybrid_recommend(
        matched,
        top_n=top_n,
        use_genre_filter=use_genre_filter,
        explain=False,
    )

    meta = eng["metadata_lookup"]
    results = []
    for title, score in raw_results:
        if title not in meta:
            continue
        results.append({
            "movie":      meta[title],
            "score":      float(score),
            "match_type": "Hybrid Blend",   # model.py blends both signals internally
        })

    return results, matched


# ---------------------------------------------------------------------------
# Optional database (Postgres via SQLAlchemy)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_db_engine():
    """
    Connect to Postgres if DATABASE_URL is set in the environment.
    Returns None silently when no database is configured
    (local dev, or HF Space without a linked DB).
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
            url.replace("postgres://", "postgresql://", 1),
            pool_pre_ping=True,
            future=True,
        )
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS user_actions ("
                "  id          SERIAL PRIMARY KEY,"
                "  user_id     VARCHAR(128),"
                "  action      VARCHAR(32),"
                "  movie_title VARCHAR(512),"
                "  rating      REAL,"
                "  ts          TIMESTAMPTZ DEFAULT now()"
                ")"
            ))
        return engine
    except Exception:
        return None


def record_action(user_id: str, action: str, movie_title: str, rating=None) -> bool:
    """Persist a user interaction to session state and optionally to Postgres."""
    st.session_state.actions.setdefault(action, {})[movie_title] = {
        "rating": rating,
        "ts":     datetime.now(timezone.utc).isoformat(),
    }
    engine = get_db_engine()
    if not engine:
        return False
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO user_actions (user_id, action, movie_title, rating) "
                     "VALUES (:u, :a, :m, :r)"),
                {"u": user_id, "a": action, "m": movie_title, "r": rating},
            )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("selected",     [])
    ss.setdefault("actions",      {})
    ss.setdefault("user_id",      f"user_{int(time.time())}")
    ss.setdefault("top_n",        10)
    ss.setdefault("genre_filter", True)
    ss.setdefault("sort_by",      "Match Score")


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1400px; }
.reco-logo  { font-family: Georgia, serif; font-size: 1.35rem; font-weight: 800;
              line-height: 1.2; letter-spacing: -.5px; margin-bottom: 4px; }
.reco-sub   { opacity: .65; font-size: .82rem; margin-top: -4px; }
.match-badge{ display:inline-block; background:#4f46e5; color:#fff;
              padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:600; }
.genre-chip { display:inline-block; background:rgba(148,163,184,.18);
              padding:1px 8px; border-radius:6px; font-size:.7rem; margin-right:4px; }
.score-pct  { font-size:1.4rem; font-weight:800; color:#22c55e; }
.tagline    { font-style:italic; opacity:.7; font-size:.82rem; }
.movie-year { opacity:.6; font-weight:400; }
</style>
"""


# ---------------------------------------------------------------------------
# UI renderers
# ---------------------------------------------------------------------------
def render_about(eng: dict) -> None:
    """Landing page shown when no seed movies have been selected yet."""
    n_movies  = len(eng["metadata_lookup"])
    df2       = eng["df2_clean"]
    n_ratings = len(df2) if df2 is not None and not df2.empty else 0
    n_users   = df2["userId"].nunique() if n_ratings > 0 else 0

    st.title("🎬 Hybrid Movie Recommendation System")
    st.markdown(
        "A recommender that blends **content-based similarity** with "
        "**collaborative filtering** to suggest films from a few titles you like — "
        "served end-to-end as a single Streamlit app."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies",       f"{n_movies:,}")
    c2.metric("User ratings", f"{n_ratings:,}")
    c3.metric("Peak RAM",     "< 1 GB")
    c4.metric("Latency",      "~20 ms / query")

    st.subheader("How the model works")
    st.markdown(f"""
- **Hybrid engine** — content similarity (prebuilt `content_similarity.npz`) blended
  with item–item **collaborative filtering** (`collab_similarity.npz`).
- **No CSVs at runtime** — all data is stored inside `hybrid_recommender_model.pkl`;
  the raw dataset files are not needed after training.
- **Memory-efficient** — sparse `.npz` matrices; runs well under **1 GB of RAM**.
- **Smart ranking** — fuzzy title matching (RapidFuzz), quality floor, Jaccard
  genre affinity, and multi-seed coverage boosting.
- **Data** — ~{n_movies:,} movies (TMDB) · ~{n_ratings:,} ratings from {n_users:,}
  users (MovieLens).
    """)
    st.info("⬅ Search for a movie in the sidebar to generate recommendations.")


def render_recommendation(eng: dict, rec: dict, col) -> None:
    """Render a single recommendation card inside the given Streamlit column."""
    movie = rec["movie"]
    with col:
        with st.container(border=True):
            st.image(poster_for(movie), width="stretch")

            year = (movie["release_date"] or "").split("-")[0]
            st.markdown(
                f"**{movie['title']}** <span class='movie-year'>{year}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='score-pct'>{round(rec['score'] * 100)}%</span> "
                f"<span style='opacity:.6'>match</span> &nbsp; "
                f"<span class='match-badge'>{rec['match_type']}</span>",
                unsafe_allow_html=True,
            )

            stars   = "★" * int(round(movie["vote_average"] / 2)) + "☆" * (5 - int(round(movie["vote_average"] / 2)))
            runtime = f" · {int(movie['runtime'])} min" if movie["runtime"] else ""
            st.caption(f"{stars}  ({movie['vote_average']:.1f}){runtime}")

            if movie["genres"]:
                st.markdown(
                    "".join(f"<span class='genre-chip'>{g}</span>" for g in movie["genres"][:3]),
                    unsafe_allow_html=True,
                )
            if movie["tagline"]:
                st.markdown(f"<p class='tagline'>\"{movie['tagline']}\"</p>", unsafe_allow_html=True)
            if movie["overview"]:
                with st.expander("Overview"):
                    st.write(movie["overview"])

            key = re.sub(r"\W+", "_", movie["title"])[:60]
            c1, c2, c3 = st.columns(3)
            uid = st.session_state.user_id

            if c1.button("❤ Fav",    key=f"fav_{key}",   width="stretch"):
                record_action(uid, "favorite",  movie["title"])
                st.toast(f"Added {movie['title']} to favorites")
            if c2.button("➕ Later",  key=f"watch_{key}", width="stretch"):
                record_action(uid, "watchlist", movie["title"])
                st.toast(f"Added {movie['title']} to watchlist")
            if c3.button("✔ Seen",   key=f"seen_{key}",  width="stretch"):
                record_action(uid, "watched",   movie["title"])
                st.toast(f"Marked {movie['title']} as watched")

            rating = st.slider("Your rating", 0, 5, 0, key=f"rate_{key}")
            if rating > 0 and st.session_state.actions.get("rating", {}).get(movie["title"], {}).get("rating") != rating:
                record_action(uid, "rating", movie["title"], rating=rating)
                st.toast(f"Rated {movie['title']} {rating}/5")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # load_engine() calls load_model() internally; both are cached by Streamlit
    with st.spinner("Loading recommendation engine…"):
        eng = load_engine()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            "<p class='reco-logo'>🎬 Hybrid Movie<br>Recommendation System</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='reco-sub'>Content-based + collaborative filtering</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("Add movie seed")
        query = st.text_input(
            "Search a title", placeholder="Toy Story, Inception…",
            label_visibility="collapsed",
        )
        if query.strip():
            ql   = query.lower().strip()
            hits = sorted(
                [m for t, m in eng["metadata_lookup"].items() if ql in t.lower()],
                key=lambda x: x["popularity"],
                reverse=True,
            )
            hits = [m for m in hits if m["title"] not in st.session_state.selected][:8]
            if hits:
                for m in hits:
                    yr = (m["release_date"] or "").split("-")[0]
                    label = f"➕ {m['title']}" + (f" ({yr})" if yr else "")
                    st.button(
                        label, key=f"add_{m['title']}", width="stretch",
                        on_click=lambda t=m["title"]: st.session_state.selected.append(t),
                    )
            else:
                st.caption("No matches found.")

        st.divider()
        st.subheader("Parameters")
        st.session_state.top_n        = st.slider("Quantity", 1, 20, st.session_state.top_n)
        st.session_state.genre_filter = st.checkbox("Strict genre blending", value=st.session_state.genre_filter)

        st.divider()
        st.subheader(f"Seed playlist ({len(st.session_state.selected)})")
        for title in st.session_state.selected:
            c1, c2 = st.columns([5, 1])
            c1.write(title)
            c2.button("✕", key=f"rm_{title}", on_click=lambda t=title: st.session_state.selected.remove(t))
        if st.session_state.selected:
            st.button(
                "Clear seed playlist", width="stretch",
                on_click=lambda: st.session_state.update(selected=[]),
            )

    # --- Main content area ---
    selected = st.session_state.selected
    if not selected:
        render_about(eng)
        return

    t0 = time.time()
    results, matched = get_recommendations(
        eng, selected,
        top_n=st.session_state.top_n,
        use_genre_filter=st.session_state.genre_filter,
    )

    if matched:
        st.caption(
            f"Matched seeds: {', '.join(matched)}  ·  "
            f"computed in {(time.time() - t0) * 1000:.0f} ms"
        )

    if not results:
        st.warning("No recommendations could be generated from the current seeds.")
        return

    head_l, head_r = st.columns([2, 2])
    head_l.markdown("### Recommended Titles")
    st.session_state.sort_by = head_r.radio(
        "Sort by", ["Match Score", "User Rating", "Release Date"],
        horizontal=True, label_visibility="collapsed",
    )

    if   st.session_state.sort_by == "User Rating":   results.sort(key=lambda r: r["movie"]["vote_average"],        reverse=True)
    elif st.session_state.sort_by == "Release Date":  results.sort(key=lambda r: r["movie"]["release_date"] or "",  reverse=True)
    else:                                              results.sort(key=lambda r: r["score"],                        reverse=True)

    with st.spinner("Fetching posters…"):
        prefetch_posters([r["movie"].get("tmdb_id", "") for r in results])

    # Render in a 4-column grid
    for i in range(0, len(results), 4):
        row = st.columns(4)
        for j, rec in enumerate(results[i: i + 4]):
            render_recommendation(eng, rec, row[j])


if __name__ == "__main__":
    main()