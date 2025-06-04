import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path="data/imdb_movies_processed.csv"):
    # 데이터 불러오기
    df = pd.read_csv(data_path)

    # 흥행 성공 여부 컬럼 생성 (수익 > 예산이면 성공으로 간주)
    df["success"] = (df["revenue"] > df["budget_x"]).astype(int)

    # 시각화에 사용할 데이터프레임 생성 (genre_Unknown이 아닌 경우만 사용)
    df_viz = df[df["genre_Unknown"] == 0].copy()

    # Seaborn 시각화 스타일 설정
    sns.set(style="whitegrid")

    # 1. 수익 분포 히스토그램
    plt.figure(figsize=(8, 5))
    sns.histplot(df_viz["revenue"], bins=50, kde=True)
    plt.title("Distribution of Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 2. 예산 대비 수익 산점도 (성공 여부 색상 표시)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_viz, x="budget_x", y="revenue", hue="success", alpha=0.6)
    plt.title("Budget vs. Revenue with Success Indicator")
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

    # 3. 장르별 성공률 계산 및 상위 10개 시각화
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
    plt.show()

    # 4. 연도 구간별 평균 수익 시각화 (year_scaled를 기준으로 구간화)
    df_viz["year_bin"] = pd.qcut(df_viz["year_scaled"], q=10, duplicates='drop')
    year_avg_revenue = df_viz.groupby("year_bin", observed=False)["revenue"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=year_avg_revenue, x="year_bin", y="revenue")
    plt.title("Average Revenue by Year Bins")
    plt.xlabel("Year (Binned)")
    plt.ylabel("Average Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5. 예산과 수익의 Boxplot (분산과 이상치 시각화)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_viz[["budget_x", "revenue"]])
    plt.title("Boxplot of Budget and Revenue")
    plt.tight_layout()
    plt.show()

    # 6. 수치형 변수들 간의 상관관계 히트맵 (전체 데이터 기준)
    num_cols = ["budget_x", "revenue", "score_scaled", "budget_x_scaled", "revenue_scaled", "year_scaled"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap among Numeric Features (entire dataset)")
    plt.tight_layout()
    plt.show()
