import pandas as pd
import numpy as np
from itertools import product

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def run_regression(data_path="data/imdb_movies_processed.csv"):
    df = pd.read_csv(data_path)

    # 입력 변수(X)와 목표 변수(y) 정의
    X = df.drop(columns=["names", "revenue", "revenue_scaled", "budget_x"])
    y = df["revenue_scaled"]

    # K-Fold 교차 검증 설정
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 결과 저장 리스트
    results = []

    # ---------- Linear Regression ----------
    model = LinearRegression()
    scores = {'mae': [], 'rmse': [], 'r2': []}
    for train_idx, val_idx in cv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores['mae'].append(mean_absolute_error(y.iloc[val_idx], preds))
        scores['rmse'].append(root_mean_squared_error(y.iloc[val_idx], preds))
        scores['r2'].append(r2_score(y.iloc[val_idx], preds))
    results.append({
        'Model': 'LinearRegression',
        'Params': {},
        'MAE': np.mean(scores['mae']),
        'RMSE': np.mean(scores['rmse']),
        'R2': np.mean(scores['r2']),
    })

    # ---------- Polynomial Regression ----------
    # degree 2와 3에 대해 다항 회귀 모델 평가
    for degree in [2, 3]:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('lr', LinearRegression())
        ])
        scores = {'mae': [], 'rmse': [], 'r2': []}
        for train_idx, val_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores['mae'].append(mean_absolute_error(y.iloc[val_idx], preds))
            scores['rmse'].append(root_mean_squared_error(y.iloc[val_idx], preds))
            scores['r2'].append(r2_score(y.iloc[val_idx], preds))
        results.append({
            'Model': 'PolynomialRegression',
            'Params': {'degree': degree},
            'MAE': np.mean(scores['mae']),
            'RMSE': np.mean(scores['rmse']),
            'R2': np.mean(scores['r2']),
        })

    # ---------- Decision Tree ----------
    # 여러 max_depth와 min_samples_split 조합을 반복적으로 실험
    for max_depth, min_split in product([5, 10, None], [2, 5, 10]):
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_split, random_state=42)
        scores = {'mae': [], 'rmse': [], 'r2': []}
        for train_idx, val_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores['mae'].append(mean_absolute_error(y.iloc[val_idx], preds))
            scores['rmse'].append(root_mean_squared_error(y.iloc[val_idx], preds))
            scores['r2'].append(r2_score(y.iloc[val_idx], preds))
        results.append({
            'Model': 'DecisionTree',
            'Params': {'max_depth': max_depth, 'min_samples_split': min_split},
            'MAE': np.mean(scores['mae']),
            'RMSE': np.mean(scores['rmse']),
            'R2': np.mean(scores['r2']),
        })

    # ---------- Lasso ----------
    # 여러 alpha 값에 대해 Lasso 회귀 평가
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = Lasso(alpha=alpha, max_iter=10000)
        scores = {'mae': [], 'rmse': [], 'r2': []}
        for train_idx, val_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores['mae'].append(mean_absolute_error(y.iloc[val_idx], preds))
            scores['rmse'].append(root_mean_squared_error(y.iloc[val_idx], preds))
            scores['r2'].append(r2_score(y.iloc[val_idx], preds))
        results.append({
            'Model': 'Lasso',
            'Params': {'alpha': alpha},
            'MAE': np.mean(scores['mae']),
            'RMSE': np.mean(scores['rmse']),
            'R2': np.mean(scores['r2']),
        })

    # ---------- 결과 저장 및 상위 5개 모델 출력 ----------
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/regression/regression_result.csv", index=False)

    # RMSE 기준 상위 5개 모델 출력
    top5 = results_df.sort_values(by="RMSE").head(5)
    print("Top 5 Model Combinations:")
    print(top5)