import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# 1. 데이터 로드
df = pd.read_csv("../../imdb_movies_processed.csv")

# 2. 사용할 피처 정의
selected_features = ['score_scaled', 'budget_x_scaled', 'genre_Action', 'genre_Adventure',
       'genre_Animation', 'genre_Comedy', 'genre_Crime', 'genre_Documentary',
       'genre_Drama', 'genre_Fantasy', 'genre_History', 'genre_Horror',
       'genre_TV Movie', 'genre_Unknown', 'genre_War', 'lang_ English',
       'lang_ Spanish, Castilian', 'lang_Other', 'country_FR', 'country_HK',
       'country_IT', 'country_JP', 'country_Other', 'country_US']
X = df[selected_features]
y = df['revenue']

# 3. 모델 학습
model = DecisionTreeRegressor(max_depth=7, min_samples_split=5, random_state=42)
model.fit(X, y)

# 4. 트리 시각화
from sklearn.tree import export_graphviz
from graphviz import Source

dot_data = export_graphviz(model,
                           out_file=None,
                           feature_names=selected_features,
                           filled=True,
                           rounded=True,
                           special_characters=True,
                           max_depth=4
                           )

# dot_data를 이용해 그래프 생성
graph = Source(dot_data)

graph.render("../plot/decision_tree", format="png", cleanup=True)  # decision_tree.png로 저장됨

