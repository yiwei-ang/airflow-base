import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load data
df = pd.read_csv('/tmp/ml/iris.csv')
X = df.drop(columns='target')
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('/tmp/ml/models/', exist_ok=True)
joblib.dump(clf, '/tmp/ml/models/iris_rf_model.joblib')
