def run_decision_tree_regressor():
    """Train a Decision Tree Regressor and export a visualized tree."""
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    from graphviz import Source
    import os

    # Load the dataset
    df = pd.read_csv("src/imdb_movies_processed.csv")

    # Define the features for training (exclude unwanted columns)
    selected_features = df.drop(columns=["names", "revenue", "revenue_scaled", "budget_x"]).columns.tolist()
    X = df[selected_features]
    y = df['revenue']

    # Train the Decision Tree Regressor
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    model.fit(X, y)

    # Export tree in DOT format for visualization
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=selected_features,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=3
    )

    graph = Source(dot_data)
    os.makedirs("outputs", exist_ok=True)
    graph.render("outputs/decision_tree_regressor", format="png", cleanup=True)

    print("[✓] decision_tree.py (regressor) done")

if __name__ == "__main__":
    run_decision_tree_regressor()
