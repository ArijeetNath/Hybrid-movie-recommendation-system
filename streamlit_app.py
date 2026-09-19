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
import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

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
)


st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)



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
    (genres, taglines, and other text details).
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
            "overview":     _clean_overview(row.overview),
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
    display dicts that the UI expects, adding display metadata.
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
# Session state initialisation
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("selected",     [])
    ss.setdefault("top_n",        10)
    ss.setdefault("genre_filter", True)


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

    st.title("🎬 Hybrid Movie Recommendation System")
    st.markdown(
        "A hybrid recommender system that blends **content-based similarity** with "
        "**collaborative filtering** to suggest movies based on your favorites."
    )

    c1, c2 = st.columns(2)
    c1.metric("Movies in Database", f"{n_movies:,}")
    c2.metric("Recommendation Speed", "~20 ms")

    st.info("⬅ Search for a movie in the sidebar to generate recommendations.")


def render_recommendation(eng: dict, rec: dict, col) -> None:
    """Render a single recommendation card inside the given Streamlit column."""
    movie = rec["movie"]
    with col:
        with st.container(border=True):
            year = (movie["release_date"] or "").split("-")[0]
            st.subheader(movie["title"], anchor=False)
            st.caption(year or "Release year unavailable")
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.spinner("Loading recommendation engine…"):
        eng = load_engine()

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
            ql = query.lower().strip()
            hits = sorted(
                [m for t, m in eng["metadata_lookup"].items() if ql in t.lower()],
                key=lambda x: x["popularity"], reverse=True,
            )
            hits = [m for m in hits if m["title"] not in st.session_state.selected][:8]
            if hits:
                for movie in hits:
                    year = (movie["release_date"] or "").split("-")[0]
                    label = f"➕ {movie['title']}" + (f" ({year})" if year else "")
                    st.button(
                        label, key=f"add_{movie['title']}",
                        on_click=lambda title=movie["title"]: st.session_state.selected.append(title),
                    )
            else:
                st.caption("No matches found.")

        st.divider()
        st.subheader("Settings")
        st.session_state.top_n = st.slider(
            "Number of recommendations", 5, 20, st.session_state.top_n
        )
        st.divider()
        st.subheader(f"Seed playlist ({len(st.session_state.selected)})")
        for title in st.session_state.selected:
            c1, c2 = st.columns([5, 1])
            c1.write(title)
            c2.button("✕", key=f"rm_{title}",
                      on_click=lambda selected_title=title: st.session_state.selected.remove(selected_title))
        if st.session_state.selected:
            st.button("Clear seed playlist", on_click=lambda: st.session_state.update(selected=[]))

    selected = st.session_state.selected
    if not selected:
        render_about(eng)
        return

    results, matched = get_recommendations(
        eng, selected, top_n=st.session_state.top_n,
        use_genre_filter=st.session_state.genre_filter,
    )
    if matched:
        st.caption(f"Matched seeds: {', '.join(matched)}")
    if not results:
        st.warning("No recommendations could be generated from the current seeds.")
        return

    st.markdown("### Recommended Titles")
    results.sort(key=lambda rec: rec["score"], reverse=True)
    for index in range(0, len(results), 4):
        row = st.columns(4)
        for column, recommendation in enumerate(results[index:index + 4]):
            render_recommendation(eng, recommendation, row[column])


if __name__ == "__main__":
    main()
