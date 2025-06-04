import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import RobustScaler, MultiLabelBinarizer

def run_preprocessing(input_path="data/imdb_movies.csv", output_path="data/imdb_movies_processed.csv") -> pd.DataFrame:
    # ───────────────────────────── 1) 경로 설정
    INPUT_PATH = Path(input_path)
    OUTPUT_PATH = Path(output_path)

    # ───────────────────────────── 2) 열 로드
    USE_COLS = ["names", "score", "genre", "status", "budget_x", "revenue", "date_x"]
    df = pd.read_csv(INPUT_PATH, usecols=USE_COLS, encoding="utf-8")

    # ───────────────────────────── 3) 날짜 → 연도
    df["year"] = pd.to_datetime(df["date_x"], errors="coerce").dt.year
    df.drop(columns="date_x", inplace=True)

    # ───────────────────────────── 4) 수치형 전처리
    NUM_COLS = ["score", "budget_x", "revenue", "year"]

    df.loc[df["budget_x"] < 1e4, "budget_x"] = np.nan
    df.loc[df["revenue"] == 0, "revenue"] = np.nan
    df = df.dropna(subset=["budget_x", "revenue"]).reset_index(drop=True)
    df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())

    for col in NUM_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    scaler = RobustScaler()
    scaled_arr = scaler.fit_transform(df[NUM_COLS])
    scaled_df = pd.DataFrame(scaled_arr, columns=[f"{c}_scaled" for c in NUM_COLS], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    df.drop(columns=["score", "year"], inplace=True)

    # ───────────────────────────── 5) 장르: 멀티-핫 인코딩
    def split_all(genres: str | float) -> list[str]:
        if pd.isna(genres):
            return ["Unknown"]
        return [g.strip() for g in re.split(r'\s*[|,]\s*', genres) if g.strip()]

    genre_lists = df["genre"].apply(split_all)
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(genre_lists),
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index
    )
    df = pd.concat([df.drop(columns="genre"), genre_encoded], axis=1)

    # ───────────────────────────── 6) status 원-핫 인코딩
    df_final = pd.get_dummies(df, columns=["status"], drop_first=True)

    # ───────────────────────────── 7) 저장 및 반환
    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print("preprocessing finished:", df_final.shape)
    print("saved to:", OUTPUT_PATH)

    return df_final
