# Model/model.py
# Hybrid Movie Recommendation System — core engine
# Loaded by streamlit_app.py (root) via: from Model.model import load_model, hybrid_recommend

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.sparse import load_npz


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
# __file__ resolves to Model/model.py regardless of the caller's location.
# BASE_DIR is always Model/, so all three artifact paths are always correct
# whether this module is run directly or imported from the project root.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent                          # → .../Model/
MODEL_PATH = BASE_DIR / "hybrid_recommender_model.pkl"              # → .../Model/hybrid_recommender_model.pkl
CONTENT_SIMILARITY_PATH = BASE_DIR / "content_similarity.npz"      # → .../Model/content_similarity.npz
COLLAB_SIMILARITY_PATH = BASE_DIR / "collab_similarity.npz"        # → .../Model/collab_similarity.npz

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# STORED_CONTENT_NEIGHBORS must not exceed the TOP_K value used when the
# content_similarity.npz matrix was built (currently capped at 100).
STORED_CONTENT_NEIGHBORS = 100
DEFAULT_CONTENT_CANDIDATES = STORED_CONTENT_NEIGHBORS
DEFAULT_COLLAB_CANDIDATES = 180
FUZZY_MATCH_THRESHOLD = 68

# ---------------------------------------------------------------------------
# Module-level state (populated by load_model)
# ---------------------------------------------------------------------------
new_df1: pd.DataFrame
df1: pd.DataFrame
df2_clean: pd.DataFrame
user2idx: dict
title2idx: dict
idx2title: dict
similarity_content = None
similarity_collab = None

_title_to_content_idx: dict[str, int] = {}
_title_lookup: dict[str, str] = {}      # clean_title → original_title
_movie_meta: dict[str, dict] = {}
_loaded = False


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------
def _load_pickle(path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with path.open("rb") as f:
            return pickle.load(f)


def load_model(force: bool = False) -> bool:
    """
    Load all recommender artifacts from Model/.

    Called once on startup by streamlit_app.py (via st.cache_resource).
    Safe to call multiple times — skips reload unless force=True.
    """
    global _loaded
    global new_df1, df1, df2_clean, user2idx, title2idx, idx2title
    global similarity_content, similarity_collab

    if _loaded and not force:
        return True

    # Validate all three files exist before attempting to load anything.
    missing = [
        p.name
        for p in (MODEL_PATH, CONTENT_SIMILARITY_PATH, COLLAB_SIMILARITY_PATH)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s) in {BASE_DIR}: {', '.join(missing)}"
        )

    # --- Pickle (dataframes + lookup dicts) ---
    model_data = _load_pickle(MODEL_PATH)
    new_df1    = model_data["new_df1"].copy()
    df1        = model_data["df1"].copy()
    df2_clean  = model_data.get("df2_clean", pd.DataFrame()).copy()
    user2idx   = model_data.get("user2idx", {})
    title2idx  = model_data.get("title2idx", {})
    idx2title  = model_data.get("idx2title", {})

    # --- Sparse similarity matrices ---
    similarity_content = load_npz(CONTENT_SIMILARITY_PATH).tocsr()
    similarity_collab  = load_npz(COLLAB_SIMILARITY_PATH).tocsr()

    _build_indexes()
    _loaded = True
    return True


# ---------------------------------------------------------------------------
# Index builders (run once after load_model)
# ---------------------------------------------------------------------------
def _clean_title(title: str) -> str:
    return " ".join(str(title).casefold().split())


def _as_number(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _normalize(series: pd.Series) -> pd.Series:
    series = _as_number(series)
    lo, hi = float(series.min()), float(series.max())
    if math.isclose(lo, hi):
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _build_indexes() -> None:
    """Build fast-lookup dicts and pre-compute per-movie quality scores."""
    global _title_to_content_idx, _title_lookup, _movie_meta

    titles = new_df1["original_title"].astype(str).tolist()
    _title_to_content_idx = {t: i for i, t in enumerate(titles)}
    _title_lookup = {_clean_title(t): t for t in titles}

    meta = df1.copy()
    meta["original_title"] = meta["original_title"].astype(str)

    popularity    = np.log1p(_as_number(meta.get("popularity",    pd.Series(index=meta.index))))
    vote_count    = np.log1p(_as_number(meta.get("vote_count",    pd.Series(index=meta.index))))
    vote_average  = _as_number(meta.get("vote_average",           pd.Series(index=meta.index)))

    meta["_vote_count"] = _as_number(meta.get("vote_count", pd.Series(index=meta.index)))
    meta["_runtime"]    = _as_number(meta.get("runtime",    pd.Series(index=meta.index)))

    min_votes   = float(_as_number(meta.get("vote_count", pd.Series(index=meta.index))).quantile(0.70))
    mean_vote   = float(vote_average.mean())
    # Bayesian-style weighted vote combining per-movie average with global mean
    weighted_vote = (
        (vote_count / (vote_count + min_votes)) * vote_average
        + (min_votes / (vote_count + min_votes)) * mean_vote
    )
    meta["_quality"] = (
        0.50 * _normalize(weighted_vote)
        + 0.30 * _normalize(vote_count)
        + 0.20 * _normalize(popularity)
    )

    _movie_meta = {}
    for _, row in meta.iterrows():
        title  = row["original_title"]
        genres = row.get("genres", [])
        if not isinstance(genres, list):
            genres = []

        release_date = row.get("release_date", "")
        release_year = None
        if isinstance(release_date, str) and len(release_date) >= 4:
            try:
                release_year = int(release_date[:4])
            except ValueError:
                pass

        _movie_meta[title] = {
            "genres":     set(genres),
            "quality":    float(row.get("_quality",  0.0) or 0.0),
            "language":   row.get("original_language", None),
            "vote_count": float(row.get("_vote_count", 0.0) or 0.0),
            "runtime":    float(row.get("_runtime",    0.0) or 0.0),
            "year":       release_year,
        }


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------
def _ensure_loaded() -> None:
    if not _loaded:
        load_model()


def _top_sparse_scores(
    matrix,
    row_index: int,
    top_k: int,
    exclude_index: int | None = None,
) -> list[tuple[int, float]]:
    """Return the top-k (index, score) pairs from a sparse CSR matrix row."""
    row     = matrix.getrow(row_index)
    indices = row.indices
    scores  = row.data

    if scores.size == 0:
        return []
    if exclude_index is not None:
        keep    = indices != exclude_index
        indices = indices[keep]
        scores  = scores[keep]
    if scores.size == 0:
        return []

    limit    = min(top_k, scores.size)
    selected = np.argpartition(scores, -limit)[-limit:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return [(int(indices[i]), float(scores[i])) for i in selected]


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------
def fuzzy_match_movies(
    input_movies: Iterable[str],
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> list[str]:
    """Map raw user-supplied titles to canonical titles in the dataset."""
    _ensure_loaded()
    matches: list[str] = []
    clean_titles = list(_title_lookup.keys())

    for raw in input_movies:
        clean = _clean_title(raw)
        if not clean:
            continue
        # Exact cleaned match first (fast path)
        exact = _title_lookup.get(clean)
        if exact:
            matches.append(exact)
            continue
        # Fuzzy fallback
        hit = process.extractOne(clean, clean_titles, scorer=fuzz.WRatio)
        if hit and hit[1] >= threshold:
            matches.append(_title_lookup[hit[0]])

    return list(dict.fromkeys(matches))  # deduplicate, preserve order


def recommend_content_scores(
    movie_title: str,
    top_k: int = DEFAULT_CONTENT_CANDIDATES,
) -> dict[str, float]:
    """Content-based neighbors from the prebuilt content_similarity.npz matrix."""
    _ensure_loaded()
    idx = _title_to_content_idx.get(movie_title)
    if idx is None:
        return {}
    return {
        new_df1.iloc[ci]["original_title"]: score
        for ci, score in _top_sparse_scores(similarity_content, idx, top_k, exclude_index=idx)
    }


def recommend_collab_scores(
    movie_title: str,
    top_k: int = DEFAULT_COLLAB_CANDIDATES,
) -> dict[str, float]:
    """Collaborative-filtering neighbors from the prebuilt collab_similarity.npz matrix."""
    _ensure_loaded()
    if similarity_collab.shape[0] == 0 or movie_title not in title2idx:
        return {}
    idx = title2idx[movie_title]
    return {
        idx2title[ci]: score
        for ci, score in _top_sparse_scores(similarity_collab, idx, top_k, exclude_index=idx)
        if ci in idx2title
    }


def get_movie_genres(movie_title: str) -> list[str]:
    _ensure_loaded()
    return sorted(_movie_meta.get(movie_title, {}).get("genres", set()))


# ---------------------------------------------------------------------------
# Genre / quality helpers
# ---------------------------------------------------------------------------
def _seed_genres(seed_movies: list[str]) -> set[str]:
    genres: set[str] = set()
    for m in seed_movies:
        genres.update(_movie_meta.get(m, {}).get("genres", set()))
    return genres


def _genre_overlap(candidate: str, seed_movies: list[str]) -> int:
    return len(_movie_meta.get(candidate, {}).get("genres", set()) & _seed_genres(seed_movies))


def _genre_affinity(candidate: str, seed_movies: list[str]) -> float:
    cg = _movie_meta.get(candidate, {}).get("genres", set())
    sg = _seed_genres(seed_movies)
    if not cg or not sg:
        return 0.0
    return len(cg & sg) / len(cg | sg)   # Jaccard similarity


def _candidate_bonus(candidate: str, seed_movies: list[str]) -> float:
    """Small additive bonus for quality, genre affinity, and language match."""
    meta = _movie_meta.get(candidate, {})
    quality_bonus  = 0.18 * float(meta.get("quality", 0.0))
    genre_bonus    = 0.16 * _genre_affinity(candidate, seed_movies)
    seed_languages = {
        _movie_meta.get(m, {}).get("language")
        for m in seed_movies
        if _movie_meta.get(m, {}).get("language")
    }
    language_bonus = 0.04 if meta.get("language") in seed_languages else 0.0
    return quality_bonus + genre_bonus + language_bonus


def _passes_quality_floor(candidate: str) -> bool:
    """Filter out short, obscure, or low-quality entries."""
    meta = _movie_meta.get(candidate, {})
    return (
        float(meta.get("quality",    0.0)) >= 0.08
        and float(meta.get("vote_count", 0.0)) >= 25.0
        and float(meta.get("runtime",    0.0)) >= 45.0
    )


def _collab_weight(movie_title: str) -> float:
    """Alpha blend weight for collaborative filtering (0 if unavailable)."""
    if similarity_collab.shape[0] == 0 or movie_title not in title2idx:
        return 0.0
    return 0.32


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if math.isclose(lo, hi):
        return {t: 1.0 for t in scores}
    return {t: (s - lo) / (hi - lo) for t, s in scores.items()}


# ---------------------------------------------------------------------------
# Main recommendation pipeline
# ---------------------------------------------------------------------------
def hybrid_recommend(
    user_movies: Iterable[str],
    top_n: int = 10,
    use_genre_filter: bool = True,
    explain: bool = True,
) -> list[tuple[str, float]]:
    """
    Blend content + collaborative scores, apply quality/genre filtering,
    and return the top_n recommendations as (title, normalized_score) pairs.
    """
    _ensure_loaded()
    matched = fuzzy_match_movies(user_movies)
    if not matched:
        if explain:
            print("No matching movies found.")
        return []
    if explain:
        print(f"Matched seeds: {', '.join(matched)}")

    raw_scores: dict[str, float] = {}
    source_counts: dict[str, int] = {}

    for seed in matched:
        alpha          = _collab_weight(seed)
        content_scores = recommend_content_scores(seed)
        collab_scores  = recommend_collab_scores(seed) if alpha else {}

        for candidate in set(content_scores) | set(collab_scores):
            if candidate in matched:
                continue
            blended = (
                (1.0 - alpha) * content_scores.get(candidate, 0.0)
                + alpha       * collab_scores.get(candidate,  0.0)
            )
            raw_scores[candidate]    = raw_scores.get(candidate, 0.0) + blended
            source_counts[candidate] = source_counts.get(candidate, 0) + 1

    if not raw_scores:
        return []

    # Boost candidates recommended by multiple seeds and add metadata bonuses
    for candidate in list(raw_scores):
        coverage_bonus = 1.0 + 0.08 * max(source_counts[candidate] - 1, 0)
        raw_scores[candidate] = (
            raw_scores[candidate] * coverage_bonus
            + _candidate_bonus(candidate, matched)
        )

    if use_genre_filter:
        seed_genre_count = len(_seed_genres(matched))
        required_overlap = 2 if seed_genre_count >= 3 else 1
        genre_filtered = {
            title: score
            for title, score in raw_scores.items()
            if (
                _genre_overlap(title, matched) >= required_overlap
                and _passes_quality_floor(title)
            )
        }
        # Only apply the filter if it leaves enough candidates
        if len(genre_filtered) >= top_n:
            raw_scores = genre_filtered

    normalized     = normalize_scores(raw_scores)
    recommendations = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def interactive_mode() -> None:
    print("\nHybrid Movie Recommendation System — interactive mode")
    print("Enter movie titles separated by commas. Type 'quit' to exit.\n")
    while True:
        user_input = input("Movies you like: ").strip()
        if user_input.casefold() in {"quit", "exit"}:
            break
        movies = [m.strip() for m in user_input.split(",") if m.strip()]
        results = hybrid_recommend(movies, top_n=8)
        if not results:
            print("No recommendations found.")
            continue
        print("\nTop Recommendations:")
        for rank, (title, score) in enumerate(results, 1):
            print(f"  {rank}. {title}  (score: {score:.3f})")
        print()


if __name__ == "__main__":
    load_model()
    interactive_mode()