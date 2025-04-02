import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load data from local CSV
data = pd.read_csv("data/day.csv")
print("First 5 rows of data:")
print(data.head())

# Select features and target
# Features: 'temp', 'windspeed', and 'season' (categorical)
features = ['temp', 'windspeed', 'season']
target = 'cnt'

# One-Hot Encoding for 'season'
# Convert 'season' into dummy variables (drop the first category as baseline)
data_features = pd.get_dummies(data[features], columns=['season'], drop_first=True)

# Define predictor variables (X) and target variable (y)
X = data_features
y = data[target]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Compute evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display metrics
evaluation_metrics = pd.DataFrame({
    "Metric": ["R² Score", "Mean Absolute Error (MAE)", "Root Mean Square Error (RMSE)"],
    "Value": [r2, mae, rmse]
})

print("\nModel Evaluation Metrics:")
print(evaluation_metrics.to_string(index=False))

# Feature Impact Analysis
coefficients = model.coef_
feature_names = X_train.columns

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef_df["Impact"] = np.where(coef_df["Coefficient"] > 0, "Positive", "Negative")

print("\nFeature Impact Analysis:")
print(coef_df.to_string(index=False))
