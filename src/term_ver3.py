def run_term_analysis():
    """Full preprocessing pipeline including encoding, scaling, and saving output."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import RobustScaler, MultiLabelBinarizer
    from pathlib import Path

    input_path = Path("./eda/imdb_movies_processed.csv")
    output_path = Path("outputs/movies_processed_full.csv")

    df = pd.read_csv(input_path)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing values
    df["budget"] = df["budget"].fillna(0)
    df["revenue"] = df["revenue"].fillna(0)

    # Binary target
    df["success"] = np.where(df["revenue"] > df["budget"], 1, 0)

    # Genre multi-hot encoding
    df["genres"] = df["genres"].fillna("").apply(lambda x: x.split(",") if x else [])
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(mlb.fit_transform(df["genres"]), columns=["genre_" + g for g in mlb.classes_])
    df = pd.concat([df, genre_dummies], axis=1).drop(columns=["genres"])

    # Scale budget
    scaler = RobustScaler()
    df["budget_scaled"] = scaler.fit_transform(df[["budget"]])

    # Drop non-numeric columns
    df.drop(columns=["title", "original_title", "release_date", "budget"], inplace=True, errors="ignore")

    df.to_csv(output_path, index=False)
    print(f"[✓] term_ver3.py preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    run_term_analysis()
