import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import RobustScaler, MultiLabelBinarizer

# ───────────────────────────── 1) 경로
INPUT_PATH  = Path("imdb_movies.csv")
OUTPUT_PATH = Path("imdb_movies_processed.csv")

# ───────────────────────────── 2) 열 로드
#리포트용으로 영화제목 포함하면 좋을 것 같습니다.
#모덱 학습용에서는 제목 제외하면 될 것 같습니다.ex)df_model  = df_final.drop(columns=["names"])
USE_COLS = ["names","score", "genre", "budget_x",
            "revenue", "date_x", "orig_lang", "country"]
df = pd.read_csv(INPUT_PATH, usecols=USE_COLS, encoding="utf-8")

# ───────────────────────────── 3) 날짜 → 연도
df["year"] = pd.to_datetime(df["date_x"], errors="coerce").dt.year
df.drop(columns="date_x", inplace=True)

# ───────────────────────────── 4) 수치형 전처리
NUM_COLS = ["score", "budget_x", "revenue", "year"]

# 4-① 현실적으로 불가능한 값 → NaN 처리
df.loc[df["budget_x"] < 1e4, "budget_x"] = np.nan           # 예산이 1만 달러 미만이면 누락 처리
df.loc[df["revenue"] == 0, "revenue"] = np.nan              # 수익이 0이면 누락 처리

# 4-② budget_x 또는 revenue 에 NaN 이 있으면 해당 행을 제거
df = df.dropna(subset=["budget_x", "revenue"]).reset_index(drop=True)

# 4-③ 그 외 수치열(score, year 등) 의 NaN 은 중앙값으로 보간
df[NUM_COLS] = df[NUM_COLS].fillna(df[NUM_COLS].median())

# 4-④ IQR 기반 이상치 클리핑
for col in NUM_COLS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 4-⑤ RobustScaler 적용하되, 별도 컬럼(_scaled)으로 보존
scaler = RobustScaler()
scaled_arr = scaler.fit_transform(df[NUM_COLS])
scaled_df = pd.DataFrame(
    scaled_arr,
    columns=[f"{c}_scaled" for c in NUM_COLS],
    index=df.index
)
df = pd.concat([df, scaled_df], axis=1)
df.drop(columns=["score", "year"], inplace=True)#score,year 원본 컬럼 제거
# ───────────────────────────── 5) 장르:멀티-핫
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

# ───────────────────────────── 6) 언어 및 국가: 상위 9개 + 기타(Other) 처리

def group_top_n(series: pd.Series, top_n: int = 9) -> pd.Series:
    top_values = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_values), other='Other')

df["orig_lang"] = group_top_n(df["orig_lang"])
df["country"] = group_top_n(df["country"])

# 원-핫 인코딩
df = pd.get_dummies(df, columns=["orig_lang", "country"], prefix=["lang", "country"], drop_first=True)

# ───────────────────────────── 7) 저장
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print("preprocessing finished:", df.shape)
print("saved to:", OUTPUT_PATH)
