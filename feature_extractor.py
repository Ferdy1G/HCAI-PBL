import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

def extract_features(
    df: pd.DataFrame,
    top_n_genres: int = 20,
    top_n_directors: int = 20,
    top_n_actors: int = 30,
    duration_min_cap: float = 60.0,
    duration_max_cap: float = 180.0,
) -> tuple[pd.DataFrame, dict]:
    """Extracts and scales feature vectors for the preference model.
    """
    data = df.copy()

    # Handle Numerical Features (Imputation, Log-Transform, and Capping)
    data["log_gross"] = np.log1p(data["gross"].fillna(data["gross"].median()))
    data["imdb_score_clean"] = data["imdb_score"].fillna(data["imdb_score"].median())
    
    # Cap duration to prevent extreme runtime outliers from distorting the scaling
    duration_filled = data["duration"].fillna(data["duration"].median())
    data["duration_capped"] = np.clip(duration_filled, duration_min_cap, duration_max_cap)

    continuous_cols = ["imdb_score_clean", "log_gross", "duration_capped"]

    scaler = MinMaxScaler()
    scaled_continuous = pd.DataFrame(
        scaler.fit_transform(data[continuous_cols]),
        columns=["num_imdb_score", "num_log_gross", "num_duration"],
        index=data.index,
    )

    # Genre Multi-Hot Encoding (Top N genres)
    genres_series = data["genres"].fillna("").apply(lambda x: [g for g in x.split("|") if g])
    all_genres = [g for sublist in genres_series for g in sublist]
    top_genres = set(pd.Series(all_genres).value_counts().head(top_n_genres).index)

    data["genres_filtered"] = genres_series.apply(
        lambda x: [g for g in x if g in top_genres]
    )

    mlb_genres = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(
        mlb_genres.fit_transform(data["genres_filtered"]),
        columns=[f"genre_{g}" for g in mlb_genres.classes_],
        index=data.index,
    )

    # Director One-Hot Encoding (Top N directors)
    top_directors = set(data["director_name"].value_counts().head(top_n_directors).index)
    data["director_clean"] = data["director_name"].apply(
        lambda x: x if x in top_directors else "Other"
    )
    directors_encoded = pd.get_dummies(
        data["director_clean"], prefix="director", dtype=int
    )
    if "director_Other" in directors_encoded.columns:
        directors_encoded.drop(columns=["director_Other"], inplace=True)

    # Actor Multi-Hot Encoding (Top N actors across all 3 columns)
    actor_cols = ["actor_1_name", "actor_2_name", "actor_3_name"]
    all_actors = data[actor_cols].values.flatten()
    top_actors = set(
        pd.Series([a for a in all_actors if pd.notna(a)])
        .value_counts()
        .head(top_n_actors)
        .index
    )

    data["actors_combined"] = data[actor_cols].apply(
        lambda row: [a for a in row if a in top_actors], axis=1
    )

    mlb_actors = MultiLabelBinarizer()
    actors_encoded = pd.DataFrame(
        mlb_actors.fit_transform(data["actors_combined"]),
        columns=[f"actor_{a}" for a in mlb_actors.classes_],
        index=data.index,
    )

    # Combine all feature sets
    X = pd.concat(
        [scaled_continuous, genres_encoded, directors_encoded, actors_encoded],
        axis=1,
    )

    metadata = {
        "scaler": scaler,
        "top_genres": list(top_genres),
        "top_directors": list(top_directors),
        "top_actors": list(top_actors),
    }

    return X, metadata


# Example execution:
df = pd.read_csv("movie_metadata.csv")
X, meta = extract_features(df)
print(f"Extracted feature matrix shape: {X.shape}")