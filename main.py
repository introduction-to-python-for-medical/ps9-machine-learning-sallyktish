%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv 
import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()
selected_features = ['DFA', 'D2']
target_feature = 'status'
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_features] = scaler.fit_transform(df[selected_features])
from sklearn.model_selection import train_test_split

X = df[selected_features]
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
accuracy_score(y, model.predict(X))
print (accuracy_score(y, model.predict(X)))
import joblib

joblib.dump(model, 'my_model.joblib')
