import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

# 데이터 로드 및 정리
df = pd.read_csv("../../../term_ver3/imdb_movies_processed.csv")
df = df.drop(columns=["names", "budget_x"])

df = df.drop(columns=["status_ Post Production", "status_ Released"]) # Almost all values are constant => Remove

X = df.drop(columns=["revenue", "revenue_scaled"]) # remove target
y = df["revenue_scaled"]

# LassoCV 학습
alphas_to_test = np.logspace(-3, 2, 20)  # alpha 값 범위
lasso_cv = LassoCV(alphas=alphas_to_test, cv=5, random_state=42)
lasso_cv.fit(X, y)

# 평균 교차검증 에러 추출 (음수로 저장됨 → -붙여서 양수화)
mse_path = -np.mean(lasso_cv.mse_path_, axis=1)

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(lasso_cv.alphas_, mse_path, marker='o')
plt.axvline(lasso_cv.alpha_, color='red', linestyle='--', label=f"Best alpha = {lasso_cv.alpha_:.4f}")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Average CV MSE")
plt.title("LassoCV: Alpha vs Cross-Validated MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()