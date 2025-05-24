import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path

# 1) 읽을 파일과 저장 경로 설정
INPUT_PATH  = Path("C:/dataScience/term.csv")          # 업로드된 원본
OUTPUT_PATH = Path("C:/dataScience/movies_processed.csv")  # 결과물

# 2) 메모리 절약을 위해 필요한 열만 선택
USE_COLS = [
    "score", "genre", "status", "orig_lang",
    "budget_x", "revenue", "country", "date_x"
]
df = pd.read_csv(INPUT_PATH, usecols=USE_COLS, encoding="utf-8")

# ──────────────────────────────────────────────────────────────
# 3) 수치형 변수 전처리
NUM_COLS = ["score", "budget_x", "revenue"]

# 3-1) 누락치 정의 & 중앙값 대체
df.loc[df["budget_x"] < 1e4, "budget_x"] = np.nan        # 비정상 예산(1달러 등) → 결측
df.loc[df["revenue"]   == 0   , "revenue"] = np.nan      # 0 수익 → 결측
for col in NUM_COLS:
    df[col].fillna(df[col].median(), inplace=True)

# 3-2) 이상치 IQR-capping
for col in NUM_COLS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    df[col] = df[col].clip(lower, upper)

# 3-3) Robust 스케일링
scaler = RobustScaler()
df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])

# ──────────────────────────────────────────────────────────────
# 4) 날짜 → 연도 추출 후 문자열 범주 변환
df["year"] = pd.to_datetime(df["date_x"]).dt.year.astype("category")
df.drop(columns=["date_x"], inplace=True)

# 5) 원-핫 인코딩 (drop_first=True로 더미 변수 함정 방지)
CAT_COLS = ["genre", "status", "orig_lang", "country", "year"]
df_final = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

# ──────────────────────────────────────────────────────────────
# 6) 저장
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"Preprocessing done! Shape: {df_final.shape}")
print(f"Saved to: {OUTPUT_PATH}")
