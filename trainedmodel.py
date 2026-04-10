from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

X = np.array([
    [8, 0, 0],
    [10, 1, 0],
    [12, 2, 0],
    [12, 4, 1],
    [14, 5, 1],
    [16, 4, 1],
    [18, 5, 1],
    [20, 6, 1],
    [9, 1, 0],
    [11, 3, 1],
])

y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")