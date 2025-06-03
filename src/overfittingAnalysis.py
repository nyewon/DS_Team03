def run_overfitting_analysis():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv("./eda/imdb_movies_processed.csv")
    X = df[["budget_scaled"]].values
    y = df["revenue"].values

    train_rmse, test_rmse, degrees = [], [], list(range(1, 10))
    for d in degrees:
        poly = PolynomialFeatures(degree=d)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        train_rmse.append(rmse)

    plt.plot(degrees, train_rmse, label="Train RMSE")
    plt.xlabel("Degree")
    plt.ylabel("RMSE")
    plt.title("Overfitting Analysis")
    plt.legend()
    plt.savefig("outputs/overfit_analysis.png")
    plt.close()
    print("[✓] overfitting_analysis.py")

if __name__ == "__main__":
    run_overfitting_analysis()
