# import encodings
# import encodings.idna
# import codecs

# print("ENCODINGS PATH:", encodings.__file__, flush=True)

# try:
#     print("IDNA TEST:", codecs.lookup("idna"), flush=True)
# except Exception as e:
#     print("IDNA ERROR:", repr(e), flush=True)
import os
import json
import base64
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt
from flask import Flask, jsonify, render_template_string, Response, stream_with_context

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

result_condition = threading.Condition()
latest_version = 0

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

        global latest_result, latest_version

        with result_condition:

            latest_result["status"] = status
            latest_result["device_id"] = frame["device_id"]
            latest_result["timestamp"] = timestamp
            latest_result["score"] = round(score, 5)
            latest_result["frames_processed"] += 1

            # Notify dashboard that new data has arrived
            latest_version += 1
            result_condition.notify_all()

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


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial;
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
            box-shadow: 0 3px 15px #ddd;
        }

        .status {
            font-size: 36px;
            font-weight: bold;
            margin: 30px;
        }

        .NORMAL { color: green; }
        .ANOMALY { color: red; }
        .WAITING { color: orange; }

        .details {
            background: #f8fafc;
            padding: 15px;
            text-align: left;
            line-height: 2;
            border-radius: 10px;
        }
    </style>
</head>

<body>
    <div class="dashboard">
        <h1>Anomaly Detection</h1>
        <p>Live MQTT Monitoring</p>

        <hr>

        <h3>Latest Result</h3>

        <div id="status" class="status WAITING">
            WAITING
        </div>

        <p id="connection">Connecting...</p>
    </div>

    <script>
        const eventSource = new EventSource("/stream");

        eventSource.onopen = function() {
            document.getElementById("connection").innerText =
                "Live connection active";
        };

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);

            const status = document.getElementById("status");

            status.innerText = data.status;
            status.className = "status " + data.status;

            document.getElementById("device").innerText =
                data.device_id;

            document.getElementById("timestamp").innerText =
                data.timestamp;

            document.getElementById("frames").innerText =
                data.frames_processed;

            document.getElementById("score").innerText =
                data.score;
        };

        eventSource.onerror = function() {
            document.getElementById("connection").innerText =
                "Reconnecting to live stream...";
        };
    </script>
</body>
</html>
"""      
  
@app.route("/")
def dashboard():
    return render_template_string(HTML)


@app.route("/stream")
def stream():

    def event_generator():

        global latest_version

        last_version = -1

        while True:

            with result_condition:

                # Wait until a new prediction arrives
                changed = result_condition.wait_for(
                    lambda: latest_version != last_version,
                    timeout=30
                )

                if changed:

                    data = latest_result.copy()
                    last_version = latest_version

                else:
                    data = None

            if data is not None:

                yield f"data: {json.dumps(data)}\n\n"

            else:

                # Keep the connection alive
                yield ": keep-alive\n\n"

    return Response(
        stream_with_context(event_generator()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/result")
def get_result():

    with result_condition:
        data = latest_result.copy()

    return jsonify(data)

# =====================================================
# START MQTT BACKGROUND SERVICE
# =====================================================

def start_mqtt():

    global model

    try:
        print("Loading ML model...", flush=True)

        model = load_model()

        print("ML model ready!", flush=True)

        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="charger-anomaly-detector"
        )

        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect

        if USERNAME and PASSWORD:
            client.username_pw_set(USERNAME, PASSWORD)

        print("Connecting to MQTT broker...", flush=True)

        client.connect(BROKER, PORT, keepalive=60)

        print("Starting MQTT listener...", flush=True)

        client.loop_forever()

    except Exception as error:

        print("MQTT STARTUP ERROR:", repr(error), flush=True)


# =====================================================
# START APPLICATION
# =====================================================

def start_background_services():

    mqtt_thread = threading.Thread(
        target=start_mqtt,
        daemon=True
    )

    mqtt_thread.start()


if __name__ == "__main__":

    start_background_services()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

else:

    # This executes when Gunicorn imports the Flask app
    start_background_services()






