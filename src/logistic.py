def run_logistic():
    """Train logistic regression model and evaluate accuracy."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    df = pd.read_csv("./eda/imdb_movies_processed.csv")
    X = df.drop(columns=["success"])
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig("outputs/logistic_conf_matrix.png")
    plt.close()
    print("[✓] logistic.py done")

if __name__ == "__main__":
    run_logistic()
