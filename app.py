from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  
model = joblib.load("diabetes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = [data[feature] for feature in [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]]
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({'prediction': result})
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500

if __name__ == '__main__':
    app.run(debug=True)
