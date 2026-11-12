
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import uuid
import json

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
SCALOGRAM_DIR = BASE_DIR / "static" / "generated"
MODEL_DIR = BASE_DIR / "model" / "training_results"

MODEL_FILE = MODEL_DIR / "resnet18_ecg_best.pth"
RESULTS_FILE = MODEL_DIR / "training_results.json"
REPORT_FILE = MODEL_DIR / "classification_report.txt"

UPLOAD_DIR.mkdir(exist_ok=True)
SCALOGRAM_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "txt"}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = None
class_names = None


# ==========================================
# 2. LOAD TRAINED RESNET18 MODEL
# ==========================================

def load_model():

    global model, class_names

    if model is not None:
        return

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    try:
        checkpoint = torch.load(
            MODEL_FILE,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_FILE,
            map_location=DEVICE
        )

    class_names = checkpoint["class_names"]

    network = models.resnet18(weights=None)

    network.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(network.fc.in_features, len(class_names))
    )

    network.load_state_dict(checkpoint["model_state_dict"])

    network.to(DEVICE)
    network.eval()

    model = network

    print("ResNet18 model loaded successfully.")
    print("Classes:", class_names)
    print("Device:", DEVICE)


# ==========================================
# 3. READ ECG CSV / TXT FILE
# ==========================================

def read_ecg_file(filepath):

    if filepath.suffix.lower() == ".csv":
        data = pd.read_csv(filepath)
    else:
        data = pd.read_csv(
            filepath,
            sep=None,
            engine="python",
            header=None,
            comment="#"
        )

    if data.empty:
        raise ValueError("The uploaded ECG file is empty.")

    # Detect a time column if available.
    time_column = None

    for column in data.columns:
        if str(column).strip().lower() in {
            "time", "time_s", "time(sec)", "timestamp"
        }:
            time_column = column
            break

    sampling_rate = 100.0

    if time_column is not None:

        time_values = pd.to_numeric(
            data[time_column],
            errors="coerce"
        ).dropna().to_numpy()

        if len(time_values) > 1:

            intervals = np.diff(time_values)
            intervals = intervals[intervals > 0]

            if len(intervals) > 0:
                sampling_rate = 1.0 / np.median(intervals)

    # Prefer a column named ecg_mV or similar.
    preferred_column = None

    for column in data.columns:

        name = str(column).strip().lower()

        if name in {
            "ecg_mv", "ecg", "lead_i", "leadi", "lead_ii", "leadii"
        }:
            preferred_column = column
            break

    if preferred_column is not None:

        signal = pd.to_numeric(
            data[preferred_column],
            errors="coerce"
        ).dropna().to_numpy(dtype=np.float32)

    else:

        numeric_data = data.drop(
            columns=[time_column],
            errors="ignore"
        ).apply(pd.to_numeric, errors="coerce")

        numeric_data = numeric_data.dropna(axis=1, how="all")

        if numeric_data.empty:
            raise ValueError(
                "No numeric ECG signal column was found."
            )

        # Use the first numeric signal column.
        signal = numeric_data.iloc[:, 0].dropna().to_numpy(
            dtype=np.float32
        )

    if len(signal) < 64:
        raise ValueError(
            "ECG signal is too short. Please upload at least 64 samples."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("ECG signal contains invalid values.")

    if sampling_rate <= 0 or not np.isfinite(sampling_rate):
        raise ValueError("Invalid ECG sampling rate.")

    return signal, sampling_rate


# ==========================================
# 4. GENERATE CWT SCALOGRAM
# ==========================================

def generate_scalogram(signal, sampling_rate, output_path):

    # Match the signal normalization used during training.
    signal = (
        signal - np.mean(signal)
    ) / (np.std(signal) + 1e-8)

    scales = np.arange(1, 65)

    coefficients, frequencies = pywt.cwt(
        signal,
        scales,
        "morl",
        sampling_period=1 / sampling_rate
    )

    scalogram = np.abs(coefficients)

    plt.figure(figsize=(5, 4))

    plt.imshow(
        scalogram,
        extent=[
            0,
            len(signal) / sampling_rate,
            scales[-1],
            scales[0]
        ],
        aspect="auto",
        cmap="jet"
    )

    # Do not include the true class label in this image.
    plt.title("ECG Scalogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Wavelet Scale")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ==========================================
# 5. PREDICT ECG CLASS
# ==========================================

def predict_ecg(image_path):

    load_model()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_class = class_names[predicted_index.item()]

    confidence_value = confidence.item() * 100

    return predicted_class, confidence_value


# ==========================================
# 6. TRAINING PERFORMANCE INFORMATION
# ==========================================

def get_training_results():

    results = {}

    if RESULTS_FILE.exists():

        with open(RESULTS_FILE, "r") as file:
            results = json.load(file)

    report = ""

    if REPORT_FILE.exists():

        report = REPORT_FILE.read_text(
            encoding="utf-8"
        )

    return results, report


# ==========================================
# 7. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 8. ECG PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "ecg_file" not in request.files:
        return "Please select an ECG file.", 400

    file = request.files["ecg_file"]

    if file.filename == "":
        return "No file selected.", 400

    extension = Path(file.filename).suffix.lower().replace(".", "")

    if extension not in ALLOWED_EXTENSIONS:
        return "Only CSV and TXT files are supported.", 400

    filename = secure_filename(file.filename)

    unique_id = uuid.uuid4().hex[:10]

    saved_filename = f"{unique_id}_{filename}"

    upload_path = UPLOAD_DIR / saved_filename

    scalogram_filename = f"{unique_id}_scalogram.png"

    scalogram_path = SCALOGRAM_DIR / scalogram_filename

    try:

        # Save uploaded file.
        file.save(upload_path)

        # Read ECG signal.
        signal, sampling_rate = read_ecg_file(upload_path)

        # Generate scalogram.
        generate_scalogram(
            signal,
            sampling_rate,
            scalogram_path
        )

        # Predict using ResNet18.
        predicted_class, confidence = predict_ecg(
            scalogram_path
        )

        # Load model training performance.
        results, report = get_training_results()

        return render_template(
            "result.html",
            filename=filename,
            predicted_class=predicted_class,
            confidence=round(confidence, 2),
            sampling_rate=round(sampling_rate, 2),
            samples=len(signal),
            duration=round(len(signal) / sampling_rate, 2),
            scalogram=url_for(
                "static",
                filename=f"generated/{scalogram_filename}"
            ),
            results=results,
            report=report
        )

    except Exception as error:

        app.logger.exception("ECG processing failed")

        return (
            f"ECG processing failed: {str(error)}"
            "<br><a href='/'>Back to Home</a>",
            500
        )


# ==========================================
# 9. TRAINING GRAPH ROUTE
# ==========================================

@app.route("/training-assets/<path:filename>")
def training_assets(filename):

    return send_from_directory(
        MODEL_DIR,
        filename
    )


# ==========================================
# 10. RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001,
        use_reloader=False
    )
