import encodings
import encodings.idna
import codecs

print("ENCODINGS PATH:", encodings.__file__, flush=True)

try:
    print("IDNA TEST:", codecs.lookup("idna"), flush=True)
except Exception as e:
    print("IDNA ERROR:", repr(e), flush=True)
import os
import json
import base64
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

from flask import Flask, jsonify, render_template_string

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "lumetech/novalume/1/data"

USERNAME = ""
PASSWORD = ""

CSV_FILE = "frames_summary.csv"
MODEL_FILE = "model.pkl"

# Set True only for the first run to train and save the model.
# After model.pkl is created, change this to False.
TRAIN_MODEL = False

# Isolation Forest settings
CONTAMINATION = 0.02

FEATURES = [
    "vrms_v",
    "irms_a",
    "p_kw",
    "q_kvar",
    "s_kva",
    "pf",
    "thd_pct",
    "h1",
    "h3",
    "h5",
    "h7",
    "h9"
]


model = None

app = Flask(__name__)

result_lock = threading.Lock()

latest_result = {
    "status": "WAITING",
    "device_id": "--",
    "timestamp": "--",
    "frames_processed": 0,
    "score": "--"
}

def train_model():

    print("\nTraining Isolation Forest model...")

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"{CSV_FILE} not found. Place it in the same folder."
        )

    df = pd.read_csv(CSV_FILE)
    X = df[FEATURES].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            random_state=42
        ))
    ])

    pipeline.fit(X)
    joblib.dump(pipeline, MODEL_FILE)

    print("Model trained successfully!")
    print("Model saved as:", MODEL_FILE)

    predictions = pipeline.predict(X)

    normal_count = np.sum(predictions == 1)
    anomaly_count = np.sum(predictions == -1)

    print("Training observations:", len(X))
    print("Normal:", normal_count)
    print("Anomalies flagged:", anomaly_count)

def load_model():

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "model.pkl not found. Set TRAIN_MODEL = True "
            "and run the program once."
        )

    loaded_model = joblib.load(MODEL_FILE)

    print("Model loaded successfully!")

    return loaded_model

def process_payload(payload):

    data = json.loads(payload.decode("utf-8"))

    device_id = data.get("device_id", "Unknown")

    encoding = data.get("enc")

    if encoding != "i16le":
        print("Unsupported encoding:", encoding)
        return []
    v_scale = float(data["v_scale"])
    i_scale = float(data["i_scale"])

    frames = data.get("frames", [])

    processed_frames = []

    for frame in frames:

        # Decode Base64 waveforms
        voltage_bytes = base64.b64decode(frame["v_b64"])
        current_bytes = base64.b64decode(frame["i_b64"])

        # Convert signed int16 little-endian values
        voltage = (
            np.frombuffer(voltage_bytes, dtype="<i2") * v_scale
        )

        current = (
            np.frombuffer(current_bytes, dtype="<i2") * i_scale
        )

        # Check waveform lengths
        if len(voltage) == 0 or len(current) == 0:
            continue

        if len(voltage) != len(current):
            print("Voltage/current sample length mismatch")
            continue

        # Calculate RMS values
        vrms = float(np.sqrt(np.mean(voltage ** 2)))
        irms = float(np.sqrt(np.mean(current ** 2)))

        # Extract features from the frame
        features = {
            "vrms_v": vrms,
            "irms_a": irms,
            "p_kw": float(frame["p_kw"]),
            "q_kvar": float(frame["q_kvar"]),
            "s_kva": float(frame["s_kva"]),
            "pf": float(frame["pf"]),
            "thd_pct": float(frame["thd_pct"]),
            "h1": float(frame["h1"]),
            "h3": float(frame["h3"]),
            "h5": float(frame["h5"]),
            "h7": float(frame["h7"]),
            "h9": float(frame["h9"])
        }

        processed_frames.append({
            "device_id": device_id,
            "timestamp": frame.get("t0_ms", "--"),
            "features": features
        })

    return processed_frames

def detect_anomaly(frames):

    global latest_result

    for frame in frames:

        features = frame["features"]

        # Create DataFrame in the same feature order
        frame_df = pd.DataFrame(
            [[features[f] for f in FEATURES]],
            columns=FEATURES
        )

        # Check for invalid values
        if not np.isfinite(frame_df.to_numpy()).all():
            print("Invalid feature values. Skipping frame.")
            continue

        # Isolation Forest prediction
        prediction = model.predict(frame_df)[0]

        # Decision score
        score = float(model.decision_function(frame_df)[0])

        if prediction == 1:
            status = "NORMAL"
        else:
            status = "ANOMALY"

        # Convert timestamp
        timestamp = frame["timestamp"]

        try:
            timestamp = datetime.fromtimestamp(
                float(timestamp) / 1000
            ).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError, OSError):
            timestamp = str(timestamp)

        # Update dashboard result
        with result_lock:

            latest_result["status"] = status
            latest_result["device_id"] = frame["device_id"]
            latest_result["timestamp"] = timestamp
            latest_result["score"] = round(score, 5)
            latest_result["frames_processed"] += 1

        # Print result in terminal
        print("--------------------------------")
        print("Device:", frame["device_id"])
        print("Timestamp:", timestamp)
        print("Result:", status)
        print("Anomaly score:", round(score, 5))
        print("--------------------------------")

def on_connect(client, userdata, flags, reason_code, properties):

    print("MQTT connection result:", reason_code)

    if getattr(reason_code, "is_failure", False):
        print("MQTT connection failed.")
        return

    print("Connected to MQTT broker!")

    client.subscribe(TOPIC, qos=1)

    print("Subscribed to topic:", TOPIC)


def on_message(client, userdata, msg):

    print("\nNew MQTT message received!")

    try:

        # Decode and process payload
        frames = process_payload(msg.payload)

        if not frames:
            print("No valid frames found.")
            return

        # Predict every frame
        detect_anomaly(frames)

    except Exception as error:

        print("Error processing MQTT message:", error)


def on_disconnect(
    client, userdata, disconnect_flags, reason_code, properties
):

    print("MQTT disconnected:", reason_code)


# =====================================================
# 8. FLASK DASHBOARD
# =====================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection</title>
    <meta http-equiv="refresh" content="2">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f3f4f6;
            text-align: center;
            padding: 40px 15px;
        }
        .dashboard {
            background: white;
            max-width: 450px;
            margin: auto;
            padding: 35px 20px;
            border-radius: 15px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        }
        h1 {
            color: #222;
            font-size: 25px;
        }
        .subtitle {
            color: #777;
        }
        .status {
            font-size: 36px;
            font-weight: bold;
            margin: 30px 0;
        }
        .NORMAL {
            color: #16a34a;
        }
        .ANOMALY {
            color: #dc2626;
        }
        .WAITING {
            color: #d97706;
        }
        .details {
            background: #f8fafc;
            padding: 15px;
            border-radius: 10px;
            text-align: left;
            line-height: 2;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>Anomaly Detection</h1>
        <p class="subtitle">Live MQTT Monitoring Dashboard</p>
        <hr>
        <h3>Latest Result</h3>
        <div class="status {{ status }}">
            {{ status }}
        </div>       
    </div>
</body>
</html>
"""


@app.route("/")
def dashboard():

    with result_lock:
        result = latest_result.copy()

    return render_template_string(HTML, **result)


@app.route("/api/result")
def get_result():

    with result_lock:
        result = latest_result.copy()

    return jsonify(result)

def start_dashboard():

    print("\nStarting dashboard...")
    print("Open http://127.0.0.1:5000 in your browser")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )

def main():

    global model
    if TRAIN_MODEL:
        train_model()

    model = load_model()
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="charger-anomaly-detector"
    )

    if USERNAME and PASSWORD:
        client.username_pw_set(USERNAME, PASSWORD)

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    dashboard_thread = threading.Thread(
        target=start_dashboard,
        daemon=True
    )

    dashboard_thread.start()

    print("\nConnecting to MQTT broker...")

    try:

        client.connect(BROKER, PORT, keepalive=60)
        client.loop_forever()

    except KeyboardInterrupt:

        print("\nProgram stopped.")

    except Exception as error:

        print("Connection error:", error)

    finally:

        client.disconnect()

if __name__ == "__main__":
    main()
else:
    threading.Thread(
        target=main,
        daemon=True
    ).start()
