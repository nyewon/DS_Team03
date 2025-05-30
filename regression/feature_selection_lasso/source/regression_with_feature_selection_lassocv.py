import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option('display.max_columns', None)

# 1. Load the data
df = pd.read_csv("../../imdb_movies_processed.csv")
selected_features = ['score_scaled', 'budget_x_scaled', 'genre_Action', 'genre_Adventure',
                     'genre_Animation', 'genre_Comedy', 'genre_Crime', 'genre_Documentary',
                     'genre_Drama', 'genre_Fantasy', 'genre_History', 'genre_Horror',
                     'genre_TV Movie', 'genre_Unknown', 'genre_War', 'lang_ English',
                     'lang_ Spanish, Castilian', 'lang_Other', 'country_FR', 'country_HK',
                     'country_IT', 'country_JP', 'country_Other', 'country_US']
X = df[selected_features]
y = df['revenue']

df_selected = pd.concat([X, y], axis=1)

# 2. Define models and hyperparameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

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
        "name": "Decision Tree",
        "model_cls": DecisionTreeRegressor,
        "param_list": [
            {"max_depth": d, "min_samples_split": m}
            for d in [3, 5, 7]
            for m in [2, 5]
        ]
    }
]

# 3. Train and evaluate models using cross-validation
for config in model_configs:
    for params in config["param_list"]:
        mse_list, mae_list, r2_list = [], [], []

        for train_idx, test_idx in kf.split(df_selected):
            train_df = df_selected.iloc[train_idx]
            test_df = df_selected.iloc[test_idx]

            X_train = train_df.drop(columns=["revenue"])
            y_train = train_df["revenue"]
            X_test = test_df.drop(columns=["revenue"])
            y_test = test_df["revenue"]

            steps = []
            if config["name"] == "Polynomial Regression":
                steps.append(("poly", PolynomialFeatures(degree=params["degree"])))
                steps.append(("model", config["model_cls"]()))
            else:
                steps.append(("model", config["model_cls"](**params)))

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

# 4. Output and save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)
print(results_df.head(5))
results_df.to_csv("../result/regression_results_lasso.csv", index=False)
