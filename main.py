# Load necessary extensions for autoreloading (if in a Jupyter/Colab environment)
%load_ext autoreload
%autoreload 2

# Download the dataset
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv 

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import yaml

# Load and preprocess the dataset
df = pd.read_csv('/content/parkinsons.csv')
selected_features = ['DFA', 'D2']  # Features to use
target_feature = 'status'

# Scale the selected features
scaler = MinMaxScaler()
df[selected_features] = scaler.fit_transform(df[selected_features])

X = df[selected_features]
y = df[target_feature]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", test_accuracy)

# Save the model
model_path = 'my_model.joblib'
joblib.dump(model, model_path)

# Save the configuration to config.yaml
config = {
    'path': model_path,
    'features': selected_features
}
with open('config.yaml', 'w') as file:
    yaml.dump(config, file)

# Verify model performance on the full dataset (for debugging purposes)
y_full_pred = model.predict(X)
full_accuracy = accuracy_score(y, y_full_pred)
print("Full Dataset Accuracy:", full_accuracy)
