from pathlib import Path
from flask import Flask, render_template, request
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"

FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

app = Flask(__name__)

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route('/')
def home():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = []
        for feature in FEATURES:
            raw_value = request.form.get(feature)
            if raw_value is None or raw_value == '':
                raise ValueError(f"Missing value for {feature}")
            input_values.append(float(raw_value))

        input_df = pd.DataFrame([input_values], columns=FEATURES)
        features_scaled = scaler.transform(input_df)
        prediction = model.predict(features_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return render_template('index.html', prediction=result)
    except (TypeError, ValueError):
        return render_template('index.html', prediction="Invalid Input")


if __name__ == "__main__":
    from os import environ
    app.run(debug=False, host='0.0.0.0', port=int(environ.get("PORT", 5000)))