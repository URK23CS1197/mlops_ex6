import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import json

data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data (same way as training split)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load('diabetes_model.pkl')

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save metrics
metrics = {'accuracy': accuracy, 'f1_score': f1}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
