def run_feature_importance():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv("./eda/imdb_movies_processed.csv")
    X = df.drop(columns=["revenue"])
    y = df["revenue"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_

    feat_imp = pd.Series(importances, index=X.columns)
    feat_imp.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()
    print("[✓] feature_importance.py")

if __name__ == "__main__":
    run_feature_importance()
