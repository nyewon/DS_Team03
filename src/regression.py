def run_regression():
    """Run multiple regression models and save top-performing ones."""
    import pandas as pd
    import numpy as np
    from itertools import product
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import os

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    df = pd.read_csv("src/imdb_movies_processed.csv")
    X = df.drop(columns=["names", "revenue", "revenue_scaled", "budget_x"])
    y = df["revenue_scaled"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # Linear Regression
    model = LinearRegression()
    scores = {'mae': [], 'rmse': [], 'r2': []}
    for train_idx, val_idx in cv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        scores['mae'].append(mean_absolute_error(y.iloc[val_idx], preds))
        scores['rmse'].append(rmse(y.iloc[val_idx], preds))
        scores['r2'].append(r2_score(y.iloc[val_idx], preds))
    results.append({
        'Model': 'LinearRegression',
        'Params': {},
        'MAE': np.mean(scores['mae']),
        'RMSE': np.mean(scores['rmse']),
        'R2': np.mean(scores['r2']),
    })

    # Polynomial Regression
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
            scores['rmse'].append(rmse(y.iloc[val_idx], preds))
            scores['r2'].append(r2_score(y.iloc[val_idx], preds))
        results.append({
            'Model': 'PolynomialRegression',
            'Params': {'degree': degree},
            'MAE': np.mean(scores['mae']),
            'RMSE': np.mean(scores['rmse']),
            'R2': np.mean(scores['r2']),
        })

    # Decision Tree
    for max_depth, min_split in product([5, 10, None], [2, 5, 10]):
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_split, random_state=42)
        scores = {'mae': [], 'rmse': [], 'r2': []}
        for train_idx, val_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores['mae'].append(mean_absolute_error(y.iloc[val_idx], preds))
            scores['rmse'].append(rmse(y.iloc[val_idx], preds))
            scores['r2'].append(r2_score(y.iloc[val_idx], preds))
        results.append({
            'Model': 'DecisionTree',
            'Params': {'max_depth': max_depth, 'min_samples_split': min_split},
            'MAE': np.mean(scores['mae']),
            'RMSE': np.mean(scores['rmse']),
            'R2': np.mean(scores['r2']),
        })

    # Lasso Regression
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = Lasso(alpha=alpha, max_iter=10000)
        scores = {'mae': [], 'rmse': [], 'r2': []}
        for train_idx, val_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores['mae'].append(mean_absolute_error(y.iloc[val_idx], preds))
            scores['rmse'].append(rmse(y.iloc[val_idx], preds))
            scores['r2'].append(r2_score(y.iloc[val_idx], preds))
        results.append({
            'Model': 'Lasso',
            'Params': {'alpha': alpha},
            'MAE': np.mean(scores['mae']),
            'RMSE': np.mean(scores['rmse']),
            'R2': np.mean(scores['r2']),
        })

    # Save results
    os.makedirs("outputs", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/regression_result.csv", index=False)

    print("Top 5 Models by RMSE:")
    print(results_df.sort_values(by="RMSE").head(5))
    print("[✓] regression.py done")

if __name__ == "__main__":
    run_regression()
