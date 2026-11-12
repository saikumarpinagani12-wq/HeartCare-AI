
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 7

BACKBONE_LR = 0.00001
CLASSIFIER_LR = 0.0003
WEIGHT_DECAY = 0.0005

IMAGE_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "scalograms"

PTBXL_DIR = (
    Path.home()
    / "Downloads"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

DATABASE_FILE = PTBXL_DIR / "ptbxl_database.csv"

RESULT_DIR = BASE_DIR / "training_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = RESULT_DIR / "resnet18_ecg_best.pth"

# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("          HEARTCARE AI - RESNET18")
print("=" * 60)

print("Device:", device)
print("Dataset:", DATASET_DIR)


# ==========================================
# 3. CHECK DATASET
# ==========================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

if not DATABASE_FILE.exists():
    raise FileNotFoundError(
        f"PTB-XL database not found: {DATABASE_FILE}"
    )


# ==========================================
# 4. IMAGE TRANSFORMATIONS
# ==========================================

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# ==========================================
# 5. LOAD DATASET
# ==========================================

train_full = datasets.ImageFolder(
    str(DATASET_DIR),
    transform=train_transform
)

val_full = datasets.ImageFolder(
    str(DATASET_DIR),
    transform=eval_transform
)

test_full = datasets.ImageFolder(
    str(DATASET_DIR),
    transform=eval_transform
)

class_names = train_full.classes

if len(class_names) != 2:
    raise ValueError(
        f"Expected two classes, found {class_names}"
    )

if not (
    train_full.class_to_idx
    == val_full.class_to_idx
    == test_full.class_to_idx
):
    raise ValueError("Class mappings do not match.")

targets = np.array([
    label for _, label in train_full.samples
])

if not (
    train_full.samples
    == val_full.samples
    == test_full.samples
):
    raise ValueError("Dataset file ordering does not match.")

print("\nCLASSES:", class_names)
print("CLASS MAPPING:", train_full.class_to_idx)
print("TOTAL IMAGES:", len(targets))

for name, index in train_full.class_to_idx.items():
    print(
        name,
        ":",
        int(np.sum(targets == index))
    )


# ==========================================
# 6. PATIENT-LEVEL DATA SPLITTING
# ==========================================

print("\nPreparing patient-level split...")

metadata = pd.read_csv(DATABASE_FILE)


def normalize_id(value):

    try:
        return str(int(float(value)))

    except (ValueError, TypeError):
        return str(value).strip()


metadata["ecg_id"] = metadata["ecg_id"].apply(
    normalize_id
)

if metadata["ecg_id"].duplicated().any():
    raise ValueError(
        "Duplicate ECG IDs found in PTB-XL database."
    )

patient_map = dict(zip(
    metadata["ecg_id"],
    metadata["patient_id"]
))

groups = []

for image_path, _ in train_full.samples:

    record_id = normalize_id(
        Path(image_path).stem.replace("ECG_", "")
    )

    if record_id not in patient_map:
        raise ValueError(
            f"ECG ID {record_id} not found in PTB-XL database."
        )

    patient_id = patient_map[record_id]

    if pd.isna(patient_id):
        raise ValueError(
            f"Patient ID missing for ECG {record_id}."
        )

    groups.append(str(patient_id))

groups = np.array(groups)

# First split:
# 80% train + validation
# 20% test

outer_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

outer_splits = list(
    outer_splitter.split(
        np.zeros(len(targets)),
        targets,
        groups
    )
)

selected_outer = None

for candidate_train, candidate_test in outer_splits:

    if (
        len(np.unique(targets[candidate_train])) == 2
        and len(np.unique(targets[candidate_test])) == 2
    ):
        selected_outer = (
            candidate_train,
            candidate_test
        )
        break

if selected_outer is None:
    raise ValueError(
        "Could not create a valid patient-level test split. "
        "Check the number of patients in each class."
    )

trainval_indices, test_indices = selected_outer

# Second split:
# 75% of remaining data for training
# 25% for validation

inner_splitter = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=SEED + 1
)

inner_splits = list(
    inner_splitter.split(
        np.zeros(len(trainval_indices)),
        targets[trainval_indices],
        groups[trainval_indices]
    )
)

selected_inner = None

for candidate_train, candidate_val in inner_splits:

    if (
        len(np.unique(targets[trainval_indices[candidate_train]])) == 2
        and len(np.unique(targets[trainval_indices[candidate_val]])) == 2
    ):
        selected_inner = (
            candidate_train,
            candidate_val
        )
        break

if selected_inner is None:
    raise ValueError(
        "Could not create a valid patient-level validation split."
    )

relative_train, relative_val = selected_inner

train_indices = trainval_indices[relative_train]
val_indices = trainval_indices[relative_val]

# Verify no patient overlap

train_patients = set(groups[train_indices])
val_patients = set(groups[val_indices])
test_patients = set(groups[test_indices])

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

# Verify both classes exist in every split

for name, indices in [
    ("Training", train_indices),
    ("Validation", val_indices),
    ("Testing", test_indices)
]:

    if len(np.unique(targets[indices])) != 2:
        raise ValueError(
            f"Both classes are not present in {name}."
        )

    print(
        name,
        "images:",
        len(indices),
        "| class counts:",
        np.bincount(
            targets[indices],
            minlength=len(class_names)
        ).tolist()
    )

print("\nPATIENT-LEVEL SPLIT SUCCESSFUL")
print("Training patients:", len(train_patients))
print("Validation patients:", len(val_patients))
print("Testing patients:", len(test_patients))
print("No patient overlap detected!")


# ==========================================
# 7. CREATE DATA LOADERS
# ==========================================

train_dataset = Subset(
    train_full,
    train_indices.tolist()
)

val_dataset = Subset(
    val_full,
    val_indices.tolist()
)

test_dataset = Subset(
    test_full,
    test_indices.tolist()
)

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


# ==========================================
# 8. LOAD PRETRAINED RESNET18
# ==========================================

print("\nLoading ResNet18...")

try:

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    print("ImageNet pretrained weights loaded.")

except Exception as error:

    raise RuntimeError(
        "Could not load pretrained ResNet18 weights. "
        "Check your internet connection and try again."
    ) from error

# Replace final classification layer

input_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(input_features, len(class_names))
)

model = model.to(device)


# ==========================================
# 9. CLASS-WEIGHTED LOSS
# ==========================================

train_labels = targets[train_indices]

class_counts = np.bincount(
    train_labels,
    minlength=len(class_names)
)

if np.any(class_counts == 0):
    raise ValueError("Training data is missing a class.")

class_weights = len(train_labels) / (
    len(class_names) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================
# 10. OPTIMIZER AND SCHEDULER
# ==========================================

optimizer = optim.AdamW(
    [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("fc.")
            ],
            "lr": BACKBONE_LR
        },
        {
            "params": model.fc.parameters(),
            "lr": CLASSIFIER_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-7
)


# ==========================================
# 11. TRAINING FUNCTION
# ==========================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total
    )


# ==========================================
# 12. EVALUATION FUNCTION
# ==========================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += (
                loss.item() * images.size(0)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    return (
        total_loss / total,
        100.0 * correct / total,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ==========================================
# 13. TRAINING
# ==========================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
best_epoch = 0
wait = 0

print("\nStarting ResNet18 training...\n")

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc, _, _ = evaluate(val_loader)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        wait = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": train_full.class_to_idx,
                "best_epoch": best_epoch,
                "validation_loss": best_val_loss
            },
            MODEL_FILE
        )

        print("Best model saved.")

    else:

        wait += 1

    if wait >= PATIENCE:

        print("\nEarly stopping activated.")
        break


# ==========================================
# 14. LOAD BEST MODEL
# ==========================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nBest epoch:", checkpoint["best_epoch"])


# ==========================================
# 15. FINAL VALIDATION
# ==========================================

val_loss, val_acc, _, _ = evaluate(val_loader)

print("\nValidation Accuracy:", f"{val_acc:.2f}%")
print("Validation Loss:", round(val_loss, 4))


# ==========================================
# 16. INDEPENDENT TESTING
# ==========================================

test_loss, test_acc, y_true, y_pred = evaluate(
    test_loader
)

print("\n" + "=" * 60)
print("             FINAL TEST RESULTS")
print("=" * 60)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", f"{test_acc:.2f}%")

print("\nCLASSIFICATION REPORT")

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

print(report_text)

with open(
    RESULT_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report_text)


# ==========================================
# 17. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(7, 6))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("HEARTCARE AI - ResNet18 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 18. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title("HEARTCARE AI - ResNet18 Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "accuracy_graph.png",
    dpi=300
)

plt.close()


# ==========================================
# 19. LOSS GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("HEARTCARE AI - ResNet18 Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "loss_graph.png",
    dpi=300
)

plt.close()


# ==========================================
# 20. SAVE RESULTS
# ==========================================

results = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "validation_accuracy": float(val_acc),
    "best_epoch": int(best_epoch),
    "class_names": class_names,
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    ),
    "training_history": history
}

with open(
    RESULT_DIR / "training_results.json",
    "w"
) as file:

    json.dump(results, file, indent=4)


# ==========================================
# 21. COMPLETION
# ==========================================

print("\n" + "=" * 60)
print("          HEARTCARE AI - SUCCESS!")
print("=" * 60)

print("\nValidation Accuracy:", f"{val_acc:.2f}%")
print("Independent Test Accuracy:", f"{test_acc:.2f}%")

print("\nSAVED FILES:")

print("Model:", MODEL_FILE)
print("Accuracy:", RESULT_DIR / "accuracy_graph.png")
print("Loss:", RESULT_DIR / "loss_graph.png")
print("Confusion Matrix:", RESULT_DIR / "confusion_matrix.png")
print("Report:", RESULT_DIR / "classification_report.txt")
print("Results:", RESULT_DIR / "training_results.json")

print("\nTRAINING COMPLETED SUCCESSFULLY!")
