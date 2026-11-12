
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 6
IMAGE_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "scalograms"
RESULTS_DIR = BASE_DIR / "model" / "training_results"

PTBXL_DIR = (
    Path.home()
    / "Downloads"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "resnet18_ecg_best.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type == "cpu":
    torch.set_num_threads(1)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)

# =========================================================
# 2. CHECK DATASET AND PATIENT METADATA
# =========================================================

metadata_path = PTBXL_DIR / "ptbxl_database.csv"

if not metadata_path.exists():
    raise FileNotFoundError(
        f"PTB-XL metadata not found: {metadata_path}"
    )

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

metadata = pd.read_csv(metadata_path)

required_columns = {"ecg_id", "patient_id"}

missing_columns = required_columns - set(metadata.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

metadata["ecg_id"] = pd.to_numeric(
    metadata["ecg_id"],
    errors="coerce"
)

metadata = metadata.dropna(
    subset=["ecg_id", "patient_id"]
).copy()

metadata["ecg_id"] = metadata["ecg_id"].astype(int)

patient_by_ecg = dict(
    zip(
        metadata["ecg_id"],
        metadata["patient_id"]
    )
)

# =========================================================
# 3. LOAD SCALOGRAM IMAGES
# =========================================================

base_dataset = datasets.ImageFolder(
    root=str(DATASET_DIR)
)

class_names = base_dataset.classes

print("\nClasses:", class_names)
print("Total images:", len(base_dataset))

if len(class_names) != 2:
    raise ValueError(
        f"Expected two classes, found: {class_names}"
    )

groups = []
missing_records = []

for image_path, label_index in base_dataset.samples:

    try:
        ecg_id = int(
            Path(image_path).stem.replace("ECG_", "")
        )

    except ValueError:
        raise ValueError(
            f"Unexpected filename: {image_path}"
        )

    patient_id = patient_by_ecg.get(ecg_id)

    if patient_id is None:
        missing_records.append(ecg_id)
    else:
        groups.append(str(patient_id))

if missing_records:
    raise ValueError(
        "Some ECG IDs were not found in PTB-XL metadata. "
        f"Examples: {missing_records[:10]}"
    )

targets = np.asarray(base_dataset.targets)
groups = np.asarray(groups)
indices = np.arange(len(base_dataset))

# =========================================================
# 4. PATIENT-WISE DATASET SPLITTING
# =========================================================

first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_val_positions, test_positions = next(
    first_split.split(
        indices,
        targets,
        groups
    )
)

train_val_indices = indices[train_val_positions]
test_indices = indices[test_positions]

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=SEED
)

train_positions, val_positions = next(
    second_split.split(
        train_val_indices,
        targets[train_val_indices],
        groups[train_val_indices]
    )
)

train_indices = train_val_indices[train_positions]
val_indices = train_val_indices[val_positions]

train_patients = set(groups[train_indices])
val_patients = set(groups[val_indices])
test_patients = set(groups[test_indices])

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

def print_split_summary(name, subset_indices):

    counts = {
        class_names[i]: int(
            np.sum(targets[subset_indices] == i)
        )
        for i in range(len(class_names))
    }

    print(
        f"{name}: {len(subset_indices)} images | "
        f"{len(set(groups[subset_indices]))} patients | "
        f"{counts}"
    )

print("\nPatient-wise dataset split:")

print_split_summary("Training", train_indices)
print_split_summary("Validation", val_indices)
print_split_summary("Testing", test_indices)

for split_name, split_indices in [
    ("training", train_indices),
    ("validation", val_indices),
    ("testing", test_indices)
]:

    if len(np.unique(targets[split_indices])) != len(class_names):

        raise ValueError(
            f"The {split_name} split does not contain both classes. "
            "Try changing SEED or review the dataset."
        )

# =========================================================
# 5. IMAGE TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomRotation(degrees=5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_full = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=train_transform
)

eval_full = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=eval_transform
)

train_dataset = Subset(
    train_full,
    train_indices.tolist()
)

val_dataset = Subset(
    eval_full,
    val_indices.tolist()
)

test_dataset = Subset(
    eval_full,
    test_indices.tolist()
)

# =========================================================
# 6. DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================================================
# 7. PRETRAINED RESNET18 MODEL
# =========================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),

    nn.Linear(
        model.fc.in_features,
        len(class_names)
    )
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    [
        {
            "params": model.layer1.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.layer2.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.layer3.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.layer4.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.conv1.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.bn1.parameters(),
            "lr": 1e-5
        }
    ],
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# =========================================================
# 8. TRAINING FUNCTION
# =========================================================

def run_epoch(loader, training=False):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    ) * 100.0

    return average_loss, accuracy

# =========================================================
# 9. TRAIN WITH EARLY STOPPING
# =========================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
best_epoch = 0

epochs_without_improvement = 0

print("\nStarting patient-wise ResNet18 training...\n")

for epoch in range(MAX_EPOCHS):

    train_loss, train_accuracy = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_accuracy = run_epoch(
        val_loader,
        training=False
    )

    scheduler.step(val_loss)

    history["train_loss"].append(
        float(train_loss)
    )

    history["val_loss"].append(
        float(val_loss)
    )

    history["train_accuracy"].append(
        float(train_accuracy)
    )

    history["val_accuracy"].append(
        float(val_accuracy)
    )

    print(
        f"Epoch [{epoch + 1}/{MAX_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_epoch": best_epoch,
                "validation_accuracy": float(val_accuracy),
                "validation_loss": float(val_loss),
                "image_size": IMAGE_SIZE,
                "architecture": "resnet18"
            },
            MODEL_PATH
        )

        print("Best model saved.")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print(
            "\nEarly stopping: validation loss did not improve."
        )

        break

# =========================================================
# 10. LOAD BEST MODEL
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# =========================================================
# 11. TEST MODEL
# =========================================================

test_loss, _ = run_epoch(
    test_loader,
    training=False
)

all_labels = []
all_predictions = []

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
) * 100.0

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

print("\nIndependent patient-wise test accuracy:",
      round(test_accuracy, 2), "%")

print("\nClassification Report:\n")
print(report)

print("\nConfusion Matrix:\n")
print(cm)

# =========================================================
# 12. SAVE CONFUSION MATRIX
# =========================================================

fig, ax = plt.subplots(figsize=(6, 5))

ax.imshow(
    cm,
    cmap="Blues"
)

ax.set_title(
    "Patient-wise Test Confusion Matrix"
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

ax.set_xticks(
    range(len(class_names))
)

ax.set_yticks(
    range(len(class_names))
)

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

for i in range(len(class_names)):

    for j in range(len(class_names)):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=150
)

plt.close(fig)

# =========================================================
# 13. SAVE ACCURACY GRAPH
# =========================================================

epochs_range = range(
    1,
    len(history["train_accuracy"]) + 1
)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_graph.png",
    dpi=150
)

plt.close()

# =========================================================
# 14. SAVE LOSS GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_graph.png",
    dpi=150
)

plt.close()

# =========================================================
# 15. SAVE CLASSIFICATION REPORT
# =========================================================

with open(
    RESULTS_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

# =========================================================
# 16. SAVE TRAINING RESULTS JSON
# =========================================================

results = {

    "test_accuracy": float(test_accuracy),

    "test_loss": float(test_loss),

    "validation_accuracy": float(
        checkpoint["validation_accuracy"]
    ),

    "validation_loss": float(
        checkpoint["validation_loss"]
    ),

    "best_epoch": int(best_epoch),

    "class_names": class_names,

    "confusion_matrix": cm.tolist(),

    "classification_report": classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    ),

    "training_history": history,

    "dataset_split": {

        "training": int(len(train_indices)),

        "validation": int(len(val_indices)),

        "testing": int(len(test_indices)),

        "training_patients": int(len(train_patients)),

        "validation_patients": int(len(val_patients)),

        "testing_patients": int(len(test_patients))
    },

    "split_method": "GroupShuffleSplit by patient_id",

    "architecture": "ResNet18 pretrained on ImageNet"
}

with open(
    RESULTS_DIR / "training_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

# =========================================================
# 17. FINAL OUTPUT
# =========================================================

print("\n========================================")

print("PATIENT-WISE RESNET18 TRAINING COMPLETED")

print("========================================")

print("Best Epoch:", best_epoch)

print(
    "Best Validation Accuracy:",
    round(checkpoint["validation_accuracy"], 2),
    "%"
)

print(
    "Independent Test Accuracy:",
    round(test_accuracy, 2),
    "%"
)

print("Model saved:", MODEL_PATH)

print("Results saved:", RESULTS_DIR)
