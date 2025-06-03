import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# 1. Load the dataset
df = pd.read_csv("../../term_ver3/imdb_movies_processed.csv")

# 2. Define the features to use for model training
selected_features = df.drop(columns=["names", "revenue", "revenue_scaled", "budget_x"]).columns.tolist()
X = df[selected_features]
y = df['revenue']

# 3. Train the Decision Tree Regressor model
model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
model.fit(X, y)

# 4. Visualize the trained decision tree
from sklearn.tree import export_graphviz
from graphviz import Source

# Convert the decision tree to DOT format for visualization
dot_data = export_graphviz(model,
                           out_file=None,
                           feature_names=selected_features,  # Feature names for labeling nodes
                           filled=True,                       # Fill node color based on prediction value
                           rounded=True,                      # Round the node boxes
                           special_characters=True,
                           max_depth=3                        # Limit the depth shown in the tree
                           )

# Create a Graphviz graph from the DOT data
graph = Source(dot_data)

# Render the graph as a PNG image and save to 'plot/decision_tree.png'
graph.render("../plot/decision_tree", format="png", cleanup=True)