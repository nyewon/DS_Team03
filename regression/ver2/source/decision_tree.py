import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# 1. Load the dataset
df = pd.read_csv("../../../term_ver3/imdb_movies_processed.csv")

# 2. Define the selected features for training
selected_features = ['score_scaled', 'budget_x_scaled', 'genre_Action', 'genre_Adventure',
                     'genre_Animation', 'genre_Crime', 'genre_Documentary', 'genre_Drama',
                     'genre_Family', 'genre_Fantasy', 'genre_History', 'genre_Horror',
                     'genre_Mystery', 'genre_Science Fiction', 'genre_TV Movie',
                     'genre_Thriller', 'genre_War']
X = df[selected_features]
y = df['revenue']

# 3. Train a Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=7, min_samples_split=5, random_state=42)
model.fit(X, y)

# 4. Visualize the decision tree
from sklearn.tree import export_graphviz
from graphviz import Source

# Export the tree structure to DOT format
dot_data = export_graphviz(model,
                           out_file=None,
                           feature_names=selected_features,
                           filled=True,        # Fill node color based on value
                           rounded=True,       # Rounded node corners
                           special_characters=True,
                           max_depth=4         # Limit depth for visualization
                           )

# Create a graph from the DOT data
graph = Source(dot_data)

# Render the graph and save it as a PNG file
graph.render("plot/decision_tree", format="png", cleanup=True)