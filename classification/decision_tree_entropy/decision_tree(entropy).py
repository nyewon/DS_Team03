import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#데이터 불러오기
df = pd.read_csv("../../term_ver3/imdb_movies_processed.csv")

#흥행 여부(수익)를 라벨로 설정
df['is_hit'] = (df['revenue'] > df['budget_x']).astype(int)

#feature 설정
feature_columns = ['budget_x_scaled', 'score_scaled', 'year_scaled'] + [col for col in df.columns if col.startswith('genre_')]
X = df[feature_columns].values
y = df['is_hit'].values

#K-fold(K=5)로 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)
avg_conf_matrices = []


fold_cms = []
reports = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Decision Tree 모델 사용
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)

    # 직접 클래스 예측
    y_pred = model.predict(X_val)

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    fold_cms.append(cm)

    report = classification_report(y_val, y_pred, output_dict=True)
    reports.append(report)

    print(f"\n[Fold {fold_idx}]")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    print(f"F1 Score: {report['1']['f1-score']:.4f}")

#confusion matrix값 평균
avg_cm = np.mean(fold_cms, axis=0).astype(int)
avg_conf_matrices.append(avg_cm)

# 평균 성능
print(f"\n-전체 평균 성능 (Entropy)")
print(f"Accuracy: {np.mean([r['accuracy'] for r in reports]):.4f}")
print(f"Precision: {np.mean([r['1']['precision'] for r in reports]):.4f}")
print(f"Recall: {np.mean([r['1']['recall'] for r in reports]):.4f}")
print(f"F1 Score: {np.mean([r['1']['f1-score'] for r in reports]):.4f}")

#confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format='d')
plt.title(f"Decision Tree(Entropy)")
plt.tight_layout()
plt.show()

#decision tree
plt.figure(figsize=(18, 8))
plot_tree(model, feature_names=feature_columns, class_names=['Not Hit', 'Hit'], filled=True, fontsize=6, max_depth=3)
plt.title("Decision Tree(Entropy)")
plt.show()


#feature importance
feature_importance = pd.Series(model.feature_importances_, index=feature_columns)
importances_sorted = feature_importance.sort_values(ascending=True)

plt.figure(figsize=(8,6))
importances_sorted.plot(kind='barh', color='teal')
plt.title("Decision Tree Feature Importance(Entropy)")
plt.tight_layout()
plt.show()