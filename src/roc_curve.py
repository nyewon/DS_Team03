def run_roc():
    """Train logistic model and plot ROC curve with threshold markers."""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("src/imdb_movies_processed.csv")

    df['is_hit'] = (df['revenue'] > df['budget_x']).astype(int)

    feature_columns = ['budget_x_scaled', 'score_scaled', 'year_scaled'] + [col for col in df.columns if col.startswith('genre_')]
    X = df[feature_columns].values
    y = df['is_hit'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val, y_scores)
    roc_auc = auc(fpr, tpr)

    target_thresholds = np.arange(0.2, 0.91, 0.1)
    markers = []
    for t in target_thresholds:
        idx = (np.abs(thresholds - t)).argmin()
        markers.append((fpr[idx], tpr[idx], thresholds[idx]))

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")

    for fpr_val, tpr_val, thresh in markers:
        plt.scatter(fpr_val, tpr_val, label=f'Thresh={thresh:.1f}', zorder=5)

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve and AUC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png")
    plt.close()
    print("[✓] roc_curve.py done")

if __name__ == "__main__":
    run_roc()
