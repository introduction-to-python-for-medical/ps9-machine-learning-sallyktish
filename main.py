import yaml
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract path and features from the config
path = config['path']
features = config['features']

# Ensure there are exactly 2 features
assert len(features) == 2, "The config file must contain exactly 2 features."

# Load the test data
df_test = pd.read_csv(".tests/test.data")

# Check if the required features and target column exist in the test data
assert all(feature in df_test.columns for feature in features), "Some features are missing in the test data."
assert "status" in df_test.columns, "The 'status' column is missing in the test data."

# Extract input features and target
X = df_test[features]
y = df_test["status"]  # The target column should be categorical

# Load the trained model
model = joblib.load(path)

# Make predictions using the model
prediction = model.predict(X)

# Calculate the accuracy score
score = accuracy_score(y, prediction)

# Check if the accuracy score is above 0.75
assert score > 0.75, f"Model accuracy is below 0.75. Current score: {score}"

print(f"Test passed! Model accuracy: {score}")
