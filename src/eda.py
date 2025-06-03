def run_eda():
    """Perform exploratory data analysis and generate visualizations."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv("eda/imdb_movies_processed.csv")
    df["success"] = (df["revenue"] > df["budget_x"]).astype(int)
    df_viz = df[df["genre_Unknown"] == 0].copy()

    sns.set(style="whitegrid")

    # 1. Revenue distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_viz["revenue"], bins=50, kde=True)
    plt.title("Distribution of Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/revenue_distribution.png")
    plt.close()

    # 2. Budget vs Revenue scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_viz, x="budget_x", y="revenue", hue="success", alpha=0.6)
    plt.title("Budget vs. Revenue with Success Indicator")
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig("outputs/budget_vs_revenue.png")
    plt.close()

    # 3. Top 10 genres by success rate
    genre_cols = [col for col in df_viz.columns if col.startswith("genre_") and col != "genre_Unknown"]
    genre_success_rate = {
        genre: df_viz[df_viz[genre] == 1]["success"].mean()
        for genre in genre_cols
    }
    sorted_genres = dict(sorted(genre_success_rate.items(), key=lambda x: x[1], reverse=True)[:10])

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(sorted_genres.values()), y=list(sorted_genres.keys()))
    plt.title("Top 10 Genres by Success Rate")
    plt.xlabel("Success Rate")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig("outputs/top_genres_success_rate.png")
    plt.close()

    # 4. Average revenue by year bins
    df_viz["year_bin"] = pd.qcut(df_viz["year_scaled"], q=10, duplicates='drop')
    year_avg_revenue = df_viz.groupby("year_bin")["revenue"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=year_avg_revenue, x="year_bin", y="revenue")
    plt.title("Average Revenue by Year Bins")
    plt.xlabel("Year (Binned)")
    plt.ylabel("Average Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/revenue_by_year_bins.png")
    plt.close()

    # 5. Boxplot of budget and revenue
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_viz[["budget_x", "revenue"]])
    plt.title("Boxplot of Budget and Revenue")
    plt.tight_layout()
    plt.savefig("outputs/boxplot_budget_revenue.png")
    plt.close()

    # 6. Correlation heatmap among numeric features
    num_cols = [
        "budget_x", "revenue", "score_scaled",
        "budget_x_scaled", "revenue_scaled", "year_scaled"
    ]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap among Numeric Features")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

    print("[✓] eda.py completed")

if __name__ == "__main__":
    run_eda()
