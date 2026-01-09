import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Create output directories
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/winequality-red.csv", sep=';')

# Features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Save model
joblib.dump(model, "output/model/trained_model.pkl")

# Save metrics
metrics = {
    "MSE": mse,
    "R2_Score": r2
}

with open("output/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
