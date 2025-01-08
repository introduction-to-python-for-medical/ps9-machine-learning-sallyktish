# Load necessary extensions for autoreloading (if in a Jupyter/Colab environment)
%load_ext autoreload
%autoreload 2

# Download the dataset
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv 

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess the dataset
df = pd.read_csv('/content/parkinsons.csv')
selected_features = ['DFA', 'D2']
target_feature = 'status'

scaler = MinMaxScaler()
df[selected_features] = scaler.fit_transform(df[selected_features])

X = df[selected_features]
y = df[target_feature]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'my_model.joblib')
