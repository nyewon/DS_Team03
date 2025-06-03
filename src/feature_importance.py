def run_feature_importance():
    """Train Decision Tree and plot feature importances."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    import os

    # Load dataset
    df = pd.read_csv("src/imdb_movies_processed.csv")

    # Separate features and target
    X = df.drop(columns=["names", "revenue", "revenue_scaled", "budget_x"])
    y = df["revenue_scaled"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    # Feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    # Plot
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance (Decision Tree)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    print("[✓] feature_importance.py done")

if __name__ == "__main__":
    run_feature_importance()
