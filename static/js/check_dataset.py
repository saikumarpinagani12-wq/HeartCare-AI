
import os
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt

# ==========================================
# HEARTCARE AI - PTB-XL DATASET LOADING
# ==========================================

print("\n===================================")
print("       HEARTCARE AI PROJECT")
print("       PTB-XL ECG ANALYSIS")
print("===================================\n")

# Step 1: Dataset location
DATASET_PATH = os.path.expanduser(
    "~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# Step 2: Check dataset folder
if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

print("Dataset folder found successfully!")

# Step 3: Load metadata
csv_path = os.path.join(
    DATASET_PATH,
    "ptbxl_database.csv"
)

df = pd.read_csv(csv_path)

print("\nDATASET INFORMATION")
print("------------------------------")
print("Total ECG recordings:", len(df))
print("Total metadata columns:", len(df.columns))

# Step 4: Display available columns
print("\nAvailable columns:")
print(df.columns.tolist())

# Step 5: Select first ECG recording
recording = df.iloc[0]

record_path = os.path.join(
    DATASET_PATH,
    recording["filename_lr"]
)

print("\nLoading ECG recording...")
print("ECG ID:", recording["ecg_id"])

# Step 6: Load ECG signal
signal, fields = wfdb.rdsamp(record_path)

# Step 7: Display signal details
print("\nECG SIGNAL DETAILS")
print("------------------------------")
print("Signal shape:", signal.shape)
print("Sampling frequency:", fields["fs"], "Hz")
print("Number of leads:", signal.shape[1])
print("Signal duration:", signal.shape[0] / fields["fs"], "seconds")

# Step 8: Check signal validity
if not np.isfinite(signal).all():
    raise ValueError("ECG signal contains invalid values.")

# Step 9: Lead names
lead_names = fields.get(
    "sig_name",
    [f"Lead {i+1}" for i in range(signal.shape[1])]
)

# Step 10: Plot all 12 ECG leads
fig, axes = plt.subplots(
    12, 1,
    figsize=(16, 18),
    sharex=True
)

time = np.arange(signal.shape[0]) / fields["fs"]

for i in range(signal.shape[1]):
    axes[i].plot(time, signal[:, i], linewidth=0.8)
    axes[i].set_ylabel(lead_names[i])
    axes[i].grid(True, alpha=0.3)

axes[0].set_title(
    "Original 12-Lead ECG Signal - PTB-XL",
    fontsize=16,
    fontweight="bold"
)

axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()

# Step 11: Save ECG waveform
output_path = os.path.join(
    os.getcwd(),
    "original_ecg.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nSUCCESS!")
print("Original ECG waveform saved at:")
print(output_path)

# Step 12: Display diagnostic information
print("\nECG DIAGNOSTIC INFORMATION")
print("------------------------------")

if "scp_codes" in df.columns:
    print("Diagnostic codes:", recording["scp_codes"])
else:
    print("Diagnostic codes unavailable.")

# Step 13: Display waveform
plt.savefig("original_ecg.png", dpi=300, bbox_inches="tight")
plt.close()

print("ECG waveform saved successfully!")

print("\n===================================")
print("       ECG ANALYSIS COMPLETED")
print("===================================")
