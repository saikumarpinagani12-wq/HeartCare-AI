
import ast
import pandas as pd
import wfdb
from pathlib import Path

ROOT = (
    Path.home() / "Downloads" /
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

metadata = pd.read_csv(ROOT / "ptbxl_database.csv")

# One genuine Normal record and one diagnostic Abnormal record
normal = metadata[
    metadata["scp_codes"].apply(
        lambda x: "NORM" in ast.literal_eval(x)
    )
].iloc[0]

statements = pd.read_csv(ROOT / "scp_statements.csv", index_col=0)

def is_abnormal(value):
    codes = ast.literal_eval(value)
    for code in codes:
        if code in statements.index:
            if statements.loc[code, "diagnostic"] == 1 and code != "NORM":
                return True
    return False

abnormal = metadata[
    metadata["scp_codes"].apply(is_abnormal)
].iloc[0]

for row, label in [(normal, "Normal"), (abnormal, "Abnormal")]:
    signal, fields = wfdb.rdsamp(str(ROOT / row["filename_lr"]))
    df = pd.DataFrame({
        "time_s": range(len(signal)),
        "lead_i": signal[:, 0]
    })
    df["time_s"] = df["time_s"] / fields["fs"]
    df.to_csv(f"Genuine_{label}_ECG.csv", index=False)
    print(f"Created Genuine_{label}_ECG.csv")

print("Export completed.")
