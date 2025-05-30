import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

# 1. Load and clean the dataset
df = pd.read_csv("../../../term_ver3/imdb_movies_processed.csv")
df = df.drop(columns=["names", "budget_x"])

df = df.drop(columns=["status_ Post Production", "status_ Released"]) # Almost all values are constant => Remove

X = df.drop(columns=["revenue", "revenue_scaled"]) # remove target
y = df["revenue_scaled"]

# 2. Use LassoCV to find the optimal alpha
lasso_cv = LassoCV(cv=5, alphas=np.logspace(-3, 2, 20), random_state=42)
lasso_cv.fit(X, y)

# 3. Print the best alpha
print(f"Best alpha from LassoCV: {lasso_cv.alpha_:.5f}")
lasso_results = pd.Series(lasso_cv.coef_, index=X.columns, name='Lasso Coefficients')
print(lasso_results)

# 4. Get the coefficients
coef = lasso_cv.coef_

# 5. Filter out features with zero coefficients
nonzero_mask = coef != 0
X_nonzero = X.loc[:, nonzero_mask]
print(f"Selected Features: {X_nonzero.columns}")
coef_nonzero = coef[nonzero_mask]

# 6. Sort features by absolute coefficient values
sorted_idx = np.argsort(np.abs(coef_nonzero))[::-1]
sorted_features = X_nonzero.columns[sorted_idx]
sorted_coef = np.abs(coef_nonzero[sorted_idx])

# 7. Visualize non-zero feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[::-1], sorted_coef[::-1], color='royalblue')  # 상위 특성부터 아래로
plt.xlabel("Absolute Coefficient")
plt.title("LassoCV Selected Feature Importance (Zero Coefficients Removed)")
plt.tight_layout()
plt.show()