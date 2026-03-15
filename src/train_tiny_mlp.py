"""
train_tiny_mlp.py
Complete training flow: load data -> segment windows -> extract features -> train an 8->8->1 MLP -> evaluate -> export weights.
"""

import os
import glob
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from feature_utils import sliding_windows, extract_features_from_window


# =========================================================
# 1. Config
# =========================================================

@dataclass
class Config:
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    window_size: int = 32             # 32 samples per window
    stride: int = 16                  # Sliding stride
    test_size: float = 0.2
    random_seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    hidden_dim: int = 8               # 8 -> 8 -> 1
    export_dir: str = os.path.join(os.path.dirname(__file__), "..", "export")


cfg = Config()
np.random.seed(cfg.random_seed)
torch.manual_seed(cfg.random_seed)


# =========================================================
# 2. Load a single sequence file
# =========================================================

def infer_label_from_filename(filepath: str) -> int:
    """Return 0 if the filename contains 'normal'; otherwise return 1."""
    name = os.path.basename(filepath).lower()
    if "normal" in name:
        return 0
    return 1


def load_signal_file(filepath: str):
    """
    Returns:
        current: np.ndarray shape [N]
        label: int
    """
    df = pd.read_csv(filepath)

    if "current" not in df.columns:
        raise ValueError(f"{filepath} is missing the 'current' column")

    current = df["current"].to_numpy(dtype=np.float32)

    if "label" in df.columns:
        label = int(df["label"].mode()[0])
    else:
        label = infer_label_from_filename(filepath)

    return current, label


# =========================================================
# 3. Build the feature dataset
# =========================================================

def build_feature_dataset(data_dir: str, window_size: int, stride: int):
    """
    Build the dataset from all CSV files under data_dir:
        X: [num_samples, 8]
        y: [num_samples]
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {data_dir}")

    all_features = []
    all_labels = []

    for filepath in csv_files:
        signal, file_label = load_signal_file(filepath)
        windows = sliding_windows(signal, window_size, stride)

        for w in windows:
            feat = extract_features_from_window(w)
            all_features.append(feat)
            all_labels.append(file_label)

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.float32)

    return X, y


# =========================================================
# 4. PyTorch Dataset
# =========================================================

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# 5. Tiny MLP: 8 -> 8 -> 1
# =========================================================

class TinyMLP(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Logits; training uses BCEWithLogitsLoss.


# =========================================================
# 6. Training
# =========================================================

def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float().cpu().numpy().flatten()

                val_preds.extend(preds.tolist())
                val_targets.extend(y_batch.numpy().flatten().tolist())

        val_f1 = f1_score(val_targets, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {np.mean(train_losses):.4f} "
            f"Val F1: {val_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================================================
# 7. Evaluation
# =========================================================

def evaluate_model(model, X_test, y_test, device):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    preds = (probs >= 0.5).astype(np.int32)
    y_true = y_test.astype(np.int32)

    print("\n=== Test Metrics ===")
    print("Accuracy:", accuracy_score(y_true, preds))
    print("F1 Score:", f1_score(y_true, preds))
    print("\nClassification Report:")
    print(classification_report(y_true, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, preds))

    return probs, preds


# =========================================================
# 8. Export the floating-point model and scaler for FPGA deployment and later fixed-point conversion
# =========================================================

def export_model_and_scaler(model, scaler, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    state = model.state_dict()

    export_data = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fc1_weight": state["fc1.weight"].cpu().numpy().tolist(),  # [8, 8]
        "fc1_bias": state["fc1.bias"].cpu().numpy().tolist(),      # [8]
        "fc2_weight": state["fc2.weight"].cpu().numpy().tolist(),  # [1, 8]
        "fc2_bias": state["fc2.bias"].cpu().numpy().tolist()       # [1]
    }

    json_path = os.path.join(export_dir, "tiny_mlp_export.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nModel and scaler exported to: {json_path}")


# =========================================================
# 9. Main flow
# =========================================================

def main():
    # 1) Build the feature dataset
    X, y = build_feature_dataset(
        data_dir=cfg.data_dir,
        window_size=cfg.window_size,
        stride=cfg.stride
    )

    print("Feature matrix shape:", X.shape)
    print("Label shape:", y.shape)
    print("Class distribution:", np.bincount(y.astype(np.int32)))

    # 2) Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y
    )

    # 3) Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split off a small validation set from the training data.
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.2,
        random_state=cfg.random_seed,
        stratify=y_train
    )

    # 4) DataLoader
    train_dataset = FeatureDataset(X_train_final, y_train_final)
    val_dataset = FeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 5) Build and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMLP(input_dim=8, hidden_dim=cfg.hidden_dim).to(device)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        device=device
    )

    # 6) Evaluate on the test set
    probs, preds = evaluate_model(model, X_test_scaled, y_test, device)

    # 7) Export the model and scaler
    export_model_and_scaler(model, scaler, cfg.export_dir)


if __name__ == "__main__":
    main()
