# 🌍 Climate Change Temperature Anomaly Predictor

This project is a machine learning-based web app that predicts climate temperature anomalies using historical environmental data. Built using Python, Flask, and scikit-learn, it enables users to enter values like CO₂ levels, sea level, and solar radiation to forecast temperature deviation.

---

## 🔍 Project Overview

Climate change is one of the most pressing issues of our time. This project demonstrates how machine learning can model and predict key climate indicators.  
It includes:
- Data preprocessing and feature engineering
- ML model training (Random Forest)
- A Flask web interface for real-time prediction


---

## 📊 Features & Workflow

1. Load historical climate data (CO₂, sea level, solar radiation)
2. Clean and scale the data
3. Train regression models
4. Evaluate using MAE, MSE, R²
5. Deploy a Flask app for user interaction

---

## 🛠 Technologies Used

| Category     | Tools/Frameworks                    |
|--------------|-------------------------------------|
| Programming  | Python                              |
| ML Libraries | scikit-learn, pandas, numpy, joblib |
| Web App      | Flask, HTML (Jinja2 templates)      |
| Deployment   | GitHub, Render (optional)           |

---

## 📁 Project Structure

Climate_change_Modeling/
│
├── app/
│ ├── app.py # Flask server code
│ ├── model/
│ │ ├── model.pkl # Trained model
│ │ └── scaler.pkl # Fitted StandardScaler
│ └── template/
│ └── index.html # Web form template
│
├── data/
│ └── climate_data.csv # Input dataset
│
├── README.md # Project description
├── requirements.txt # Python dependencies
└── .gitignore # Git ignored files

## install dependencies
pip install -r requirements.txt

## train Model 
python train_model.py 


## Run Flask App
python app/app.py
Visit the app at: http://127.0.0.1:5000

 ## 🧠 Model Information
Algorithm: Random Forest Regressor

Inputs: CO₂ (ppm), Sea Level (mm), Solar Radiation (W/m²)

Output: Predicted temperature anomaly in °C

Evaluation Metrics: MAE, MSE, R²


##  Use Cases
Educational tool for climate data modeling

Starter template for advanced forecasting (LSTM, RNN)

Demonstration of ML deployment with Flask

