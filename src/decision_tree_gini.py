def run_decision_tree_gini():
    """Train Decision Tree (Gini) and visualize performance & importance."""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

    df = pd.read_csv("src/imdb_movies_processed.csv")
    df['is_hit'] = (df['revenue'] > df['budget_x']).astype(int)

    feature_columns = ['budget_x_scaled', 'score_scaled', 'year_scaled'] + [col for col in df.columns if col.startswith('genre_')]
    X = df[feature_columns].values
    y = df['is_hit'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_cms = []
    reports = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = DecisionTreeClassifier(criterion='gini')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        fold_cms.append(cm)

        report = classification_report(y_val, y_pred, output_dict=True)
        reports.append(report)

        print(f"\n[Fold {fold_idx}]")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Precision: {report['1']['precision']:.4f}")
        print(f"Recall: {report['1']['recall']:.4f}")
        print(f"F1 Score: {report['1']['f1-score']:.4f}")

    avg_cm = np.mean(fold_cms, axis=0).astype(int)

    print(f"\n-Overall Average Performance (Gini)")
    print(f"Accuracy: {np.mean([r['accuracy'] for r in reports]):.4f}")
    print(f"Precision: {np.mean([r['1']['precision'] for r in reports]):.4f}")
    print(f"Recall: {np.mean([r['1']['recall'] for r in reports]):.4f}")
    print(f"F1 Score: {np.mean([r['1']['f1-score'] for r in reports]):.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
    disp.plot(values_format='d')
    plt.title("Decision Tree (Gini Impurity)")
    plt.tight_layout()
    plt.savefig("outputs/decision_tree_gini_cm.png")
    plt.close()

    plt.figure(figsize=(18, 8))
    plot_tree(model, feature_names=feature_columns, class_names=['Not Hit', 'Hit'], filled=True, fontsize=6, max_depth=3)
    plt.title("Decision Tree (Gini Impurity)")
    plt.tight_layout()
    plt.savefig("outputs/decision_tree_gini_tree.png")
    plt.close()

    feature_importance = pd.Series(model.feature_importances_, index=feature_columns)
    importances_sorted = feature_importance.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances_sorted.plot(kind='barh', color='teal')
    plt.title("Feature Importance (Decision Tree Gini)")
    plt.tight_layout()
    plt.savefig("outputs/decision_tree_gini_importance.png")
    plt.close()

    print("[✓] decision_tree_gini.py done")

if __name__ == "__main__":
    run_decision_tree_gini()
