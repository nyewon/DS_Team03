import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import RobustScaler, MultiLabelBinarizer

# ───────────────────────────── 1) 경로
INPUT_PATH  = Path("C:/dataScience/term.csv")
OUTPUT_PATH = Path("C:/dataScience/movies_processed_full.csv")

# ───────────────────────────── 2) 열 로드
USE_COLS = ["score", "genre", "status",
            "budget_x", "revenue", "date_x"]
df = pd.read_csv(INPUT_PATH, usecols=USE_COLS, encoding="utf-8")

# ───────────────────────────── 3) 날짜 → 연도
df["year"] = pd.to_datetime(df["date_x"], errors="coerce").dt.year
df.drop(columns="date_x", inplace=True)

# ───────────────────────────── 4) 수치형 전처리
NUM_COLS = ["score", "budget_x", "revenue", "year"]

df.loc[df["budget_x"] < 1e4, "budget_x"] = np.nan
df.loc[df["revenue"] == 0, "revenue"] = np.nan
df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())

for col in NUM_COLS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

df[NUM_COLS] = RobustScaler().fit_transform(df[NUM_COLS])

# ───────────────────────────── 5) 장르: **전체** 분리 → 멀티-핫
def split_all(genres: str | float) -> list[str]:
    if pd.isna(genres):
        return ["Unknown"]
    # 파이프(|)·콤마(,) 구분자 모두 처리, 공백 제거
    return [g.strip() for g in re.split(r'\s*[|,]\s*', genres) if g.strip()]

genre_lists = df["genre"].apply(split_all)

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(
    mlb.fit_transform(genre_lists),
    columns=[f"genre_{g}" for g in mlb.classes_],
    index=df.index
)

df = pd.concat([df.drop(columns="genre"), genre_encoded], axis=1)

# ───────────────────────────── 6) status One-Hot
df_final = pd.get_dummies(df, columns=["status"], drop_first=True)

# ───────────────────────────── 7) 저장
df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print("preprocessing finished:", df_final.shape)
print("saved to:", OUTPUT_PATH)
