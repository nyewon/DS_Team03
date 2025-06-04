import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def run_logistic(data_path="data/imdb_movies_processed.csv"):
    # 데이터 불러오기
    df = pd.read_csv(data_path)

    #흥행 여부(수익)를 라벨로 설정
    df['is_hit'] = (df['revenue'] > df['budget_x']).astype(int)

    #feature 설정
    feature_columns = ['budget_x_scaled', 'score_scaled', 'year_scaled'] + [col for col in df.columns if col.startswith('genre_')]
    X = df[feature_columns].values
    y = df['is_hit'].values

    #threshold 설정
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    #K-fold(K=5)로 검증
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    avg_conf_matrices = []



    for threshold in thresholds:    
        fold_cms = []
        reports = []

        print(f"\n-----[Threshold {threshold}]-----")
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

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
        print(f"\n-전체 평균 성능")
        print(f"Accuracy: {np.mean([r['accuracy'] for r in reports]):.4f}")
        print(f"Precision: {np.mean([r['1']['precision'] for r in reports]):.4f}")
        print(f"Recall: {np.mean([r['1']['recall'] for r in reports]):.4f}")
        print(f"F1 Score: {np.mean([r['1']['f1-score'] for r in reports]):.4f}")

        print("\n[Model Coefficients]")
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        for name, value in zip(feature_columns, coef):
            direction = "↑" if value > 0 else "↓"
            print(f"{name}: {value:.4f} ({direction} 양성 확률)")
        print(f"Intercept: {intercept:.4f}")

        # ✅ Feature 계수 시각화 추가
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(coef)
        sorted_names = [feature_columns[i] for i in sorted_idx]
        sorted_weights = coef[sorted_idx]
        colors = ["red" if w < 0 else "blue" for w in sorted_weights]

        plt.barh(sorted_names, sorted_weights, color=colors)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.title(f"Feature Coefficients (Threshold = {threshold})")
        plt.xlabel("Coefficient Value")
        plt.tight_layout()
        plt.show()

    #각 threshold에 대한 confusion matrix plot걀과
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes = axes.flatten()

    for i, (thresh, cm) in enumerate(zip(thresholds, avg_conf_matrices)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], values_format='d')
        axes[i].set_title(f"Threshold = {thresh:.1f}")


    plt.tight_layout()
    plt.show()