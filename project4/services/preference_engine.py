import numpy as np
import pandas as pd

def get_random_movies(df: pd.DataFrame, n: int = 2) -> list[dict]:
    """Returns n random movies formatted for the frontend."""
    sampled = df.sample(n=min(n, len(df)))
    movies = []
    for idx, row in sampled.iterrows():
        movies.append({
            "id": int(idx),
            "title": str(row.get("movie_title", "Unknown Title")).strip(),
            "year": int(row.get("title_year", 0)) if pd.notna(row.get("title_year")) else "N/A",
            "genres": str(row.get("genres", "")).replace("|", ", "),
            "director": str(row.get("director_name", "Unknown")),
            "imdb_score": float(row.get("imdb_score", 0.0)),
        })
    return movies

def update_plackett_luce(w: np.ndarray, feature_matrix: np.ndarray, ranked_indices: list[int], lr: float = 0.05) -> np.ndarray:
    """Updates user weight vector w given an ordered list of item indices (Plackett-Luce model)."""
    w = w.copy()
    n = len(ranked_indices)
    for k in range(n - 1):
        rem_indices = ranked_indices[k:]
        X_rem = feature_matrix[rem_indices]  # Shape: (len(rem), num_features)
        
        # Utilities and softmax probabilities over remaining items
        utilities = X_rem @ w
        exp_u = np.exp(utilities - np.max(utilities))  # Numerical stability
        probs = exp_u / np.sum(exp_u)
        
        # Gradient update for top choice at stage k
        x_chosen = feature_matrix[ranked_indices[k]]
        expected_x = np.sum(probs[:, None] * X_rem, axis=0)
        grad = x_chosen - expected_x
        w += lr * grad
        
    return w