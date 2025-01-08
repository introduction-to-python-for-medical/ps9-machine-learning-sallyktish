import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
parkinsons_df = pd.read_csv('/content/parkinsons.csv')

# Check if any missing values are present
print(parkinsons_df.isnull().sum())

# Drop missing values
parkinsons_df = parkinsons_df.dropna()

# Check the first few rows to ensure data is loaded correctly
print(parkinsons_df.head())

# Define input and output features
input_features = ['DFA', 'PPE']
output_feature = ['status']

# Ensure the columns exist in the dataframe
if not all(feature in parkinsons_df.columns for feature in input_features):
    print(f"Warning: Some input features are missing from the dataset.")
else:
    X = parkinsons_df[input_features]
    y = parkinsons_df[output_feature]

    from sklearn.preprocessing import MinMaxScaler

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the input features
    X_scaled = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    from sklearn.svm import SVC

    # Initialize and train the model
    svc = SVC(kernel='linear', C=2, random_state=42)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_val)

    from sklearn.metrics import accuracy_score

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {accuracy}")

    if accuracy < 0.8:
        print("Accuracy is below the required threshold of 0.8. Please adjust the model or features.")

    import joblib

    # Save the trained model
    joblib.dump(svc, 'par.joblib')
