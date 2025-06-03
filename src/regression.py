def run_regression():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv("./eda/imdb_movies_processed.csv")
    X = df[["budget_scaled"]]
    y = df["revenue"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Regression RMSE:", rmse)

    plt.scatter(X, y, color="blue", label="Actual")
    plt.plot(X, y_pred, color="red", label="Predicted")
    plt.xlabel("Budget (scaled)")
    plt.ylabel("Revenue")
    plt.title("Linear Regression")
    plt.legend()
    plt.savefig("outputs/regression_plot.png")
    plt.close()

    print("[✓] regression.py 실행 완료")

if __name__ == "__main__":
    run_regression()
