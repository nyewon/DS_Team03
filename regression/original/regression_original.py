import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option('display.max_columns', None)

# 데이터 로드
df = pd.read_csv("../imdb_movies_processed.csv")
df = df.drop(columns=['revenue_scaled', "names", "budget_x"])  # 사용 안할 컬럼 제거

# 평가 결과 저장용 리스트
results = []

# 교차검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 모델별 파라미터 수동 설정
model_configs = [
    {
        "name": "Linear Regression",
        "model_cls": LinearRegression,
        "param_list": [{}]
    },
    {
        "name": "Polynomial Regression",
        "model_cls": LinearRegression,
        "param_list": [{"degree": d} for d in [2, 3]]
    },
    {
        "name": "Lasso Regression",
        "model_cls": Lasso,
        "param_list": [{"alpha": a} for a in [0.01, 0.1, 1, 10]]
    },
    {
        "name": "Decision Tree",
        "model_cls": DecisionTreeRegressor,
        "param_list": [
            {"max_depth": d, "min_samples_split": m}
            for d in [3, 5, 7]
            for m in [2, 5]
        ]
    }
]

# 모델별 교차검증
for config in model_configs:
    for params in config["param_list"]:
        mse_list, mae_list, r2_list = [], [], []

        for train_idx, test_idx in kf.split(df):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # 타겟 컬럼 제거
            X_train = train_df.drop(columns=["revenue"])
            X_test = test_df.drop(columns=["revenue"])

            y_train = train_df["revenue"]
            y_test = test_df["revenue"]

            steps = []

            if config["name"] == "Polynomial Regression":
                steps.append(('poly', PolynomialFeatures(degree=params['degree'])))
                steps.append(('model', config["model_cls"]()))
            else:
                steps.append(('model', config["model_cls"](**params)))

            model = Pipeline(steps)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse_list.append(mean_squared_error(y_test, preds))
            mae_list.append(mean_absolute_error(y_test, preds))
            r2_list.append(r2_score(y_test, preds))

        results.append({
            "Model": config["name"],
            "Params": params,
            "MSE": np.mean(mse_list),
            "MAE": np.mean(mae_list),
            "R2": np.mean(r2_list)
        })

# 결과 정리
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)
results_df.to_csv("regression_results.csv", index=False)
print(results_df.head(5))
