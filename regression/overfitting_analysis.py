import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def run_overfitting_analysis(data_path="data/imdb_movies_processed.csv"):
    # RMSE 계산 함수
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # 점수 계산 함수
    def print_scores(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        return r2, rmse

    # 데이터 로드 및 분할
    df = pd.read_csv(data_path)
    X = df.drop(columns=["names", "revenue", "revenue_scaled", 'budget_x'])
    y = df["revenue_scaled"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 하이퍼파라미터 조합
    param_list = [
        {"max_depth": 10, "min_samples_split": 10},
        {"max_depth": 10, "min_samples_split": 5},
        {"max_depth": 10, "min_samples_split": 2},
        {"max_depth": 5, "min_samples_split": 5},
        {"max_depth": 5, "min_samples_split": 10},
    ]

    results = []

    # 모델 학습 및 평가
    for params in param_list:
        model = DecisionTreeRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_r2, train_rmse = print_scores(y_train, train_pred)
        test_r2, test_rmse = print_scores(y_test, test_pred)
        rmse_gap_pct = (test_rmse - train_rmse) / train_rmse * 100

        results.append({
            "Model & Params": f'Decision Tree: {params}',
            "Train R2": train_r2,
            "Test R2": test_r2,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "RMSE Gap(%)": rmse_gap_pct
        })

    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/regression/overfitting_analysis.csv', index=False)

    # 결과 출력
    print(results_df)