import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def run_feature_importance(data_path="data/imdb_movies_processed.csv"):
    # 데이터 로드
    df = pd.read_csv(data_path)

    # 특성과 타겟 분리
    X = df.drop(columns=["names", "revenue", "revenue_scaled", 'budget_x'])
    y = df["revenue_scaled"]

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 결정 트리 모델 학습
    model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    # 특성 중요도 추출
    importances = model.feature_importances_
    feature_names = X.columns

    # 중요도 시각화
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    # 수평 막대 그래프
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance (Decision Tree: max_depth=5, min_samples_split=10)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
