
import os
import ast
import numpy as np
import pandas as pd
import wfdb
import pywt
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# =====================================
# 1. DATASET LOCATION
# =====================================

DATASET_PATH = (
    Path.home()
    / "Downloads"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

OUTPUT_PATH = Path("dataset/scalograms")

NORMAL_PATH = OUTPUT_PATH / "Normal"
ABNORMAL_PATH = OUTPUT_PATH / "Abnormal"

NORMAL_PATH.mkdir(parents=True, exist_ok=True)
ABNORMAL_PATH.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. LOAD PTB-XL METADATA
# =====================================

print("Loading PTB-XL dataset...")

metadata = pd.read_csv(DATASET_PATH / "ptbxl_database.csv")
statements = pd.read_csv(
    DATASET_PATH / "scp_statements.csv",
    index_col=0
)

print("Total recordings:", len(metadata))

# =====================================
# 3. IDENTIFY NORMAL / ABNORMAL
# =====================================

def get_label(code_string):

    try:
        codes = ast.literal_eval(code_string)
    except (ValueError, SyntaxError, TypeError):
        return None

    diagnostic_codes = []

    for code in codes:

        if code not in statements.index:
            continue

        diagnostic = statements.loc[code, "diagnostic"]

        if pd.notna(diagnostic) and diagnostic == 1:
            diagnostic_codes.append(code)

    # Exclude ambiguous or unclassified recordings
    if "NORM" in diagnostic_codes:

        if len(diagnostic_codes) == 1:
            return "Normal"

        return None

    if diagnostic_codes:
        return "Abnormal"

    return None


metadata["label"] = metadata["scp_codes"].apply(get_label)

metadata = metadata.dropna(subset=["label"])

print("\nClass distribution:")
print(metadata["label"].value_counts())

# =====================================
# 4. SELECT 10 RECORDINGS
# =====================================

normal_samples = metadata[
    metadata["label"] == "Normal"
].sample(n=500, random_state=42)

abnormal_samples = metadata[
    metadata["label"] == "Abnormal"
].sample(n=500, random_state=42)

samples = pd.concat([
    normal_samples,
    abnormal_samples
])

# =====================================
# 5. GENERATE CWT SCALOGRAM
# =====================================

def generate_scalogram(row):

    ecg_id = row["ecg_id"]
    label = row["label"]

    record_path = DATASET_PATH / row["filename_lr"]

    print(f"\nProcessing ECG ID: {ecg_id} ({label})")

    # Load original ECG signal
    signal, fields = wfdb.rdsamp(str(record_path))

    sampling_rate = fields["fs"]

    # Select Lead I
    ecg_signal = signal[:, 0]

    # Normalize signal
    ecg_signal = (
        ecg_signal - np.mean(ecg_signal)
    ) / (np.std(ecg_signal) + 1e-8)

    # Continuous Wavelet Transform
    scales = np.arange(1, 65)

    coefficients, frequencies = pywt.cwt(
        ecg_signal,
        scales,
        "morl",
        sampling_period=1 / sampling_rate
    )

    scalogram = np.abs(coefficients)

    # Save scalogram image
    output_folder = (
        NORMAL_PATH if label == "Normal"
        else ABNORMAL_PATH
    )

    output_file = output_folder / f"ECG_{ecg_id}.png"

    plt.figure(figsize=(5, 4))

    plt.imshow(
        scalogram,
        extent=[
            0,
            len(ecg_signal) / sampling_rate,
            scales[-1],
            scales[0]
        ],
        aspect="auto",
        cmap="jet"
    )

    plt.title(f"{label} ECG - ID {ecg_id}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Wavelet Scale")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file)


# =====================================
# 6. PROCESS SELECTED RECORDINGS
# =====================================

print("\nGenerating 10 sample scalograms...")

for _, row in samples.iterrows():

    try:
        generate_scalogram(row)

    except Exception as error:
        print(
            f"Error processing ECG {row['ecg_id']}: {error}"
        )

# =====================================
# 7. COMPLETION
# =====================================

print("\n================================")
print("SCALOGRAM GENERATION COMPLETED")
print("================================")

print("Normal folder:", NORMAL_PATH)
print("Abnormal folder:", ABNORMAL_PATH)

print("\nCheck the generated images before training.")
