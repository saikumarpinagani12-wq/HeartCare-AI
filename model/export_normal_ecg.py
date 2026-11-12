
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path

# PTB-XL dataset location
DATASET_PATH = (
    Path.home()
    / "Downloads"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# Genuine Normal ECG: Record ID 3
record_path = DATASET_PATH / "records100" / "00000" / "00003_lr"

# Read original ECG
signal, fields = wfdb.rdsamp(str(record_path))

# Select Lead I
lead_i = signal[:, 0]

# Sampling frequency
sampling_rate = fields["fs"]

# Create time axis
time = np.arange(len(lead_i)) / sampling_rate

# Create CSV
df = pd.DataFrame({
    "time_s": time,
    "lead_i": lead_i
})

# Save file
output_file = Path("Genuine_Normal_ECG_ID_3.csv")
df.to_csv(output_file, index=False)

print("Genuine Normal ECG exported successfully!")
print("Record ID: 3")
print("Sampling rate:", sampling_rate, "Hz")
print("Samples:", len(df))
print("Saved:", output_file.resolve())
