import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parkinsons_df =  pd.read_csv('/content/parkinsons.csv')
parkinsons_df =  parkinsons_df.dropna()
parkinsons_df.head()


input_features = ['DFA', 'PPE']
output_feature = ['status']
X = parkinsons_df[['DFA', 'PPE']]
y = parkinsons_df['status']

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the input features
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


from sklearn.svm import SVC

svc = SVC(kernel='linear', C=2, random_state = 42) 
svc.fit(X_train, y_train)
y_pred = svc.predict(X_val)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
    print("Accuracy is below the required threshold of 0.8. Please adjust the model or features.")

import joblib

joblib.dump(svc, 'par.joblib')
