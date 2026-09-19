# Hybrid Movie Recommendation System

A local, poster-free Streamlit movie recommender. Add movies you enjoy and receive ranked suggestions from a hybrid of content-based similarity and collaborative filtering. All recommendation data is bundled with the project; generating recommendations does not require an API key or internet connection.

## Live deployment

Try the complete deployed app on [Hugging Face Spaces](https://huggingface.co/spaces/arijeet-472/Hybrid-movie-recommendation-system).

<!-- Update this section with future deployment links or notes. -->
## Features

- Search a catalog of roughly 30,000 movies.
- Build a seed playlist from one or more favorite movies.
- Combine content similarity with collaborative-filtering signals when available.
- Rank candidates using similarity, genre affinity, language fit, popularity, vote count, and weighted rating.
- Show each recommendation's match score, year, rating, runtime, genres, tagline, and overview.

## Requirements

- Python 3.12 is recommended.
- The model files in `model/` must remain alongside the application.
- Around 100 MB of disk space is required for the bundled model artifacts.

## Run locally

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser. Stop the app with `Ctrl+C` in the terminal.

## How to use it

1. Type a title in **Search a title** in the sidebar.
2. Select one or more matching movies to create a seed playlist.
3. Choose the number of recommendations to show.
4. Review the ranked results. The percentage is a relative match score, not a predicted user rating.

Using several seed movies usually produces more focused results because the engine rewards candidates that match multiple choices.

## How recommendations are calculated

For every selected movie, the app gathers nearby titles from two prebuilt sparse similarity matrices:

- **Content-based similarity** compares movie metadata and tags.
- **Collaborative filtering** uses item-to-item rating behavior when that signal exists.

The engine blends these signals, removes the selected seed movies, boosts candidates supported by multiple seeds, and applies genre and quality filters. Results are normalized to a 0–100% relative match score and returned in descending order.

## Project layout

```text
streamlit_app.py                   Streamlit user interface and display formatting
model/model.py                     Recommendation engine and ranking logic
model/hybrid_recommender_model.pkl Movie metadata and lookup tables
model/content_similarity.npz       Prebuilt content-similarity sparse matrix
model/collab_similarity.npz        Prebuilt collaborative-filtering sparse matrix
requirements.txt                   Runtime Python dependencies
```

## Notes and troubleshooting

- The interface is intentionally poster-free, so it has no TMDB, image-download, or external metadata dependency.
- First startup can take a few seconds while the model artifacts load; later interactions use the cached engine.
- If startup reports a missing artifact, verify that all three files listed under `model/` are present and have not been renamed.
- If port 8501 is occupied, add `--server.port 8502` to the Streamlit command and open the matching port.

## License

No license is currently included. Add one before distributing or reusing the project publicly.
