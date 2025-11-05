import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data=pd.read_csv('diabetes.csv')
data.describe()

data

data.isnull().sum()

X=data.drop('Outcome',axis=1)
y=data['Outcome']

X_train, X_test, y_train, y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

acc=accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {acc:.4f}")

name="diabetes_model.pkl"
joblib.dump(model, name)

mo=joblib.load('diabetes_model.pkl')

sample = pd.DataFrame({
    "Pregnancies": [0],
    "Glucose": [100],
    "BloodPressure": [120],
    "SkinThickness": [30],
    "Insulin": [0],
    "BMI": [21],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [21]
})

prediction = mo.predict(sample)[0]
print("Diabetic" if prediction == 1 else "Non-Diabetic")
