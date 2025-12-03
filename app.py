# =========================
# 1. Install Required Libraries (Run once)
# =========================
# !pip install numpy pandas matplotlib scikit-learn tensorflow flask flask_cors pyngrok

# =========================
# 2. Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, jsonify, render_template, request
from pyngrok import ngrok
from datetime import datetime
import random
import os

# =========================
# 3. Simulated Data Collection
# =========================
def generate_heart_sample():
    hr = np.random.normal(75, 5)        # Simulated HR
    hrv = np.random.normal(45, 10)      # Simulated HRV
    irregularity = np.random.choice([0,1], p=[0.95,0.05])
    return max(40, min(140, int(hr))), max(10, int(hrv)), irregularity

# Baseline HR for anomaly detection
baseline_samples = [generate_heart_sample()[0] for _ in range(30)]
baseline_mean = np.mean(baseline_samples)
baseline_std = np.std(baseline_samples)
baseline = {"mean": baseline_mean, "std": baseline_std}

# =========================
# 4. Preprocessing Functions
# =========================
def normalize_signal(signal):
    scaler = MinMaxScaler()
    return scaler.fit_transform(signal.reshape(-1,1)).flatten()

def compute_rmssd(rr_intervals):
    diff_rr = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff_rr**2))

# =========================
# 5. Anomaly Detection Functions
# =========================
def detect_hr_anomaly(hr, baseline):
    mean = baseline["mean"]
    std = baseline["std"]
    if hr > mean + 2*std:
        return "high_heart_rate"
    elif hr < mean - 2*std:
        return "low_heart_rate"
    else:
        return "normal"

def detect_irregularity(irregular_flag):
    return "irregular_beat" if irregular_flag == 1 else "normal"

def recommendation(flag):
    messages = {
        "high_heart_rate": "Your heart rate is high. Rest and hydrate.",
        "low_heart_rate": "Your heart rate is low. Seek advice if dizzy.",
        "irregular_beat": "Irregular beat detected. Monitor closely.",
        "normal": "Heart activity looks normal."
    }
    return messages.get(flag, "Normal readings.")

# =========================
# 6. Machine Learning Models
# =========================

# 6A. Random Forest
hr_array = np.array([generate_heart_sample()[0] for _ in range(200)])
labels = np.where(hr_array > baseline["mean"] + 2*baseline["std"], 1, 0)

X_train, X_test, y_train, y_test = train_test_split(hr_array.reshape(-1,1), labels, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, pred))

def rf_anomaly_detector(hr_value):
    return rf.predict([[hr_value]])[0]

# 6B. LSTM Time-Series Prediction
def create_sequences(data, window=20):
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(hr_array)
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1],1))

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(20,1)),
    LSTM(32),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=16, validation_split=0.2)

# =========================
# 7. Flask Web App
# =========================
app = Flask(__name__)
history = []

def get_recommendation_ui(hr):
    flag = detect_hr_anomaly(hr, baseline)
    if flag != "normal":
        return recommendation(flag)
    ml_flag = rf_anomaly_detector(hr)
    if ml_flag == 1:
        return "Anomalous heart rate detected by ML."
    return "Heart rate normal."

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/update", methods=["POST"])
def update():
    data = request.json
    hr = data["heart_rate"]
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {
        "time": timestamp,
        "heart_rate": hr,
        "recommendation": get_recommendation_ui(hr)
    }
    history.append(entry)
    return jsonify({"status":"ok","entry":entry})

@app.route("/data")
def data():
    return jsonify(history[-100:])

# HTML Template
os.makedirs("templates", exist_ok=True)
html_template = """
<!DOCTYPE html>
<html>
<head><title>Heart Dashboard</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script></head>
<body>
<h1>❤️ Heart Dashboard</h1>
<canvas id="hrChart" width="500" height="200"></canvas>
<h3 id="lastHR"></h3><h4 id="recommendation"></h4>
<script>
async function fetchData() {
    const res = await fetch("/data");
    const data = await res.json();
    const labels = data.map(d=>d.time);
    const hr = data.map(d=>d.heart_rate);
    if(data.length>0){
        const last=data[data.length-1];
        document.getElementById("lastHR").innerText="Current HR: "+last.heart_rate;
        document.getElementById("recommendation").innerText="Recommendation: "+last.recommendation;
    }
    hrChart.data.labels=labels; hrChart.data.datasets[0].data=hr; hrChart.update();
}
let ctx=document.getElementById("hrChart").getContext("2d");
let hrChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label:"HR(BPM)",data:[],borderColor:"red",borderWidth:2}]},options:{scales:{y:{min:40,max:150}}}});
setInterval(fetchData,2000);
</script>
</body></html>
"""
with open("templates/index.html","w") as f:
    f.write(html_template)

# =========================
# 8. Real-Time Simulation Loop (Optional, can run in Colab separately)
# =========================
def simulate_real_time_monitoring():
    hr_history_sim = []
    for i in range(50):
        hr, hrv, irr = generate_heart_sample()
        hr_history_sim.append(hr)
        
        hr_flag = detect_hr_anomaly(hr, baseline)
        irr_flag = detect_irregularity(irr)
        final_flag = hr_flag if hr_flag != "normal" else irr_flag
        
        ml_flag = rf_anomaly_detector(hr)
        
        if len(hr_history_sim) >= 20:
            last_seq = np.array(hr_history_sim[-20:]).reshape(1,20,1)
            pred_hr = lstm_model.predict(last_seq, verbose=0)[0][0]
        else:
            pred_hr = None
        
        print(f"Time:{i}s | HR:{hr} | Anomaly:{final_flag} | ML:{ml_flag} | Predicted Next HR:{pred_hr} | Recommendation:{recommendation(final_flag)}")
        time.sleep(0.5)

# =========================
# 9. Run Flask App with ngrok
# =========================
public_url = ngrok.connect(5000)
print("Public URL:", public_url)
app.run()
