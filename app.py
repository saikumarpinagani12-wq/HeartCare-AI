
from flask import (
    Flask, render_template, request,
    url_for, send_from_directory
)
from werkzeug.utils import secure_filename

from pathlib import Path
import os
import uuid
import json

import numpy as np


# ==========================================
# 1. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
SCALOGRAM_DIR = BASE_DIR / "static" / "generated"
MODEL_DIR = BASE_DIR / "model" / "training_results"

MODEL_FILE = MODEL_DIR / "resnet18_ecg_best.pth"
RESULTS_FILE = MODEL_DIR / "training_results.json"
REPORT_FILE = MODEL_DIR / "classification_report.txt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SCALOGRAM_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "txt"}

# Render uses CPU.
DEVICE = "cpu"

# Model is loaded only when prediction is requested.
model = None
class_names = None
torch = None


# ==========================================
# 2. CHECK FILE EXTENSION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# 3. LOAD RESNET18 MODEL LAZILY
# ==========================================

def load_model():

    global model, class_names, torch

    if model is not None:
        return

    import torch as torch_module
    import torch.nn as nn
    from torchvision import models

    torch = torch_module

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    print("Loading ResNet18 model...")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    network = models.resnet18(weights=None)

    network.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            network.fc.in_features,
            len(class_names)
        )
    )

    network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    network.eval()

    model = network

    del checkpoint

    print("ResNet18 loaded successfully.")
    print("Classes:", class_names)


# ==========================================
# 4. READ ECG FILE
# ==========================================

def read_ecg_file(filepath):

    import pandas as pd

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
        raise ValueError("Uploaded ECG file is empty.")

    # Detect time column.
    time_column = None

    for column in data.columns:

        if str(column).strip().lower() in {
            "time",
            "time_s",
            "time(sec)",
            "timestamp"
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

                sampling_rate = (
                    1.0 / np.median(intervals)
                )

    # Identify ECG signal column.
    preferred_column = None

    for column in data.columns:

        name = str(column).strip().lower()

        if name in {
            "ecg_mv",
            "ecg",
            "lead_i",
            "leadi",
            "lead_ii",
            "leadii"
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
        ).apply(
            pd.to_numeric,
            errors="coerce"
        )

        numeric_data = numeric_data.dropna(
            axis=1,
            how="all"
        )

        if numeric_data.empty:

            raise ValueError(
                "No numeric ECG signal column found."
            )

        signal = numeric_data.iloc[:, 0].dropna().to_numpy(
            dtype=np.float32
        )

    if len(signal) < 64:

        raise ValueError(
            "ECG signal must contain at least 64 samples."
        )

    if not np.all(np.isfinite(signal)):

        raise ValueError(
            "ECG signal contains invalid values."
        )

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:

        raise ValueError(
            "Invalid ECG sampling rate."
        )

    return signal, sampling_rate


# ==========================================
# 5. GENERATE CWT SCALOGRAM
# ==========================================

def generate_scalogram(
    signal,
    sampling_rate,
    output_path
):

    import pywt

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Normalize signal.
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

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(
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

    ax.set_title("ECG Scalogram")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Wavelet Scale")

    fig.colorbar(
        image,
        ax=ax,
        label="Magnitude"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=100,
        bbox_inches="tight"
    )

    plt.close(fig)

    del coefficients, scalogram, image, fig, ax

    return output_path


# ==========================================
# 6. ECG PREDICTION
# ==========================================

def predict_ecg(image_path):

    load_model()

    from torchvision import transforms
    from PIL import Image

    image_transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image_tensor = image_transform(image).unsqueeze(0)

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = class_names[
        predicted_index.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )

    del image_tensor, outputs, probabilities

    return predicted_class, confidence_value


# ==========================================
# 7. TRAINING RESULTS
# ==========================================

def get_training_results():

    results = {}
    report = ""

    if RESULTS_FILE.exists():

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

    if REPORT_FILE.exists():

        report = REPORT_FILE.read_text(
            encoding="utf-8"
        )

    return results, report


# ==========================================
# 8. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 9. ECG PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "ecg_file" not in request.files:

        return "Please select an ECG file.", 400

    file = request.files["ecg_file"]

    if file.filename == "":

        return "No file selected.", 400

    if not allowed_file(file.filename):

        return (
            "Only CSV and TXT files are supported.",
            400
        )

    filename = secure_filename(file.filename)

    if not filename:

        return "Invalid filename.", 400

    unique_id = uuid.uuid4().hex[:10]

    saved_filename = f"{unique_id}_{filename}"

    upload_path = UPLOAD_DIR / saved_filename

    scalogram_filename = (
        f"{unique_id}_scalogram.png"
    )

    scalogram_path = (
        SCALOGRAM_DIR / scalogram_filename
    )

    try:

        # Save uploaded ECG.
        file.save(upload_path)

        # Read ECG signal.
        signal, sampling_rate = read_ecg_file(
            upload_path
        )

        # Generate CWT scalogram.
        generate_scalogram(
            signal,
            sampling_rate,
            scalogram_path
        )

        # Predict ECG class.
        predicted_class, confidence = predict_ecg(
            scalogram_path
        )

        # Load training performance.
        results, report = get_training_results()

        return render_template(

            "result.html",

            filename=filename,

            predicted_class=predicted_class,

            confidence=round(confidence, 2),

            sampling_rate=round(sampling_rate, 2),

            samples=len(signal),

            duration=round(
                len(signal) / sampling_rate,
                2
            ),

            scalogram=url_for(
                "static",
                filename=f"generated/{scalogram_filename}"
            ),

            results=results,

            report=report

        )

    except Exception as error:

        app.logger.exception(
            "ECG processing failed"
        )

        return (
            f"ECG processing failed: {str(error)}"
            "<br><a href='/'>Back to Home</a>",
            500
        )


# ==========================================
# 10. TRAINING GRAPH ROUTE
# ==========================================

@app.route("/training-assets/<path:filename>")
def training_assets(filename):

    return send_from_directory(
        MODEL_DIR,
        filename
    )


# ==========================================
# 11. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=False,
        use_reloader=False
    )
