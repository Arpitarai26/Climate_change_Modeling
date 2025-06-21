from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize app
app = Flask(__name__)

# Load model and scaler
model = joblib.load(os.path.join("model", "model.pkl"))
scaler = joblib.load(os.path.join("model", "scaler.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_data = scaler.transform([features])
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction_text=f"Predicted Temperature Anomaly: {prediction:.3f} °C")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        CO2 = float(request.form['CO2'])
        sea_level = float(request.form['sea_level'])
        solar_radiation = float(request.form['solar_radiation'])

        input_data = np.array([[CO2, sea_level, solar_radiation]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        prediction_text = f"🌡 Predicted Temperature Anomaly: {prediction:.2f} °C"
    except Exception as e:
        prediction_text = f"⚠️ Error: Please enter valid numbers. ({e})"

    return render_template('index.html', prediction_text=prediction_text)
