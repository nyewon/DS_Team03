import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#데이터 불러오기
df = pd.read_csv("../../term_ver3/imdb_movies_processed.csv")

#흥행 여부(수익)를 라벨로 설정
df['is_hit'] = (df['revenue'] > df['budget_x']).astype(int)

#feature 설정
feature_columns = ['budget_x_scaled', 'score_scaled', 'year_scaled'] + [col for col in df.columns if col.startswith('genre_')]
X = df[feature_columns].values
y = df['is_hit'].values

#데이터셋 분리
X_train_roc, X_val_roc, y_train_roc, y_val_roc = train_test_split(X, y, test_size=0.2, random_state=42)

#모델 학습
model_roc = LogisticRegression(max_iter=1000)
model_roc.fit(X_train_roc, y_train_roc)

#positive 클래스 예측
y_scores = model_roc.predict_proba(X_val_roc)[:, 1]

#roc와 auc 계산
fpr, tpr, thresholds = roc_curve(y_val_roc, y_scores)
roc_auc = auc(fpr, tpr)

# 0.1 간격 threshold 표시
target_thresholds = np.arange(0.2, 0.91, 0.1)
markers = []
for t in target_thresholds:
    idx = (np.abs(thresholds - t)).argmin()
    markers.append((fpr[idx], tpr[idx], thresholds[idx]))

# ROC 그래프 그리기
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")

# threshold 점들 표시
for fpr_val, tpr_val, thresh in markers:
    plt.scatter(fpr_val, tpr_val, label=f'Thresh={thresh:.1f}', zorder=5)

plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC Curve and AUC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
