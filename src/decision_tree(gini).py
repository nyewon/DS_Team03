def run_decision_tree_gini():
    """Train decision tree using Gini index and visualize it."""
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    df = pd.read_csv("./eda/imdb_movies_processed.csv")
    X = df.drop(columns=["success"])
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = DecisionTreeClassifier(criterion="gini", max_depth=4)
    model.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plot_tree(model, feature_names=X.columns, class_names=True, filled=True)
    plt.title("Decision Tree (Gini)")
    plt.savefig("outputs/decision_tree_gini.png")
    plt.close()
    print("[✓] decision_tree(gini).py done")

if __name__ == "__main__":
    run_decision_tree_gini()
