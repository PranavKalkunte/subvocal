"""Training pipeline for sEMG command classifiers.

Supports:
1. RandomForest (Standard ML baseline using TD10 aggregated features).
2. 1D CNN (Deep learning model operating on raw time-series segments).
3. GRU (Bidirectional temporal RNN operating on raw time-series segments).
"""

import os
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from emg_core import config
from emg_core.dsp.features import extract_features
from emg_core.dsp.filters import preprocess_multichannel
from emg_core.ml.model_io import save_model


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Model Architectures
# ══════════════════════════════════════════════════════════════════════════════

class EMG1DCNN(nn.Module):
    """1D CNN for sEMG segment classification.

    Applies convolutions over the temporal dimension of raw multichannel sEMG.
    """
    def __init__(self, num_channels: int, num_classes: int, segment_length: int = 150):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, C, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EMGGRU(nn.Module):
    """Bidirectional GRU for temporal tracking of sEMG gestures."""
    def __init__(self, num_channels: int, num_classes: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, C, T) -> Transpose to (B, T, C) for GRU
        x = x.transpose(1, 2)
        out, _ = self.gru(x)  # (B, T, hidden_size * 2)
        out = torch.mean(out, dim=1)  # Temporal pooling
        return self.fc(out)


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(user_id: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load the calibration dataset for a user.

    Returns:
        segments: list of raw 2D arrays, each (segment_length, num_channels)
        labels: list of label strings corresponding to each segment
    """
    data_path = os.path.join(config.DATA_DIR, f"{user_id}_calib.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No calibration data for user '{user_id}' at {data_path}")

    data = np.load(data_path, allow_pickle=True)
    segments = [np.array(s, dtype=np.float64) for s in data["segments"]]
    labels = list(data["labels"])
    return segments, labels


def preprocess_segments(
    segments: List[np.ndarray],
    fs: float,
    low: float,
    high: float
) -> np.ndarray:
    """Preprocess raw segments and stack into a 3D numpy array of shape (N, C, T)."""
    processed_list = []
    for seg in segments:
        # Preprocess: shape remains (segment_length, num_channels)
        p_seg = preprocess_multichannel(seg, fs=fs, low=low, high=high)
        # Transpose to (num_channels, segment_length)
        processed_list.append(p_seg.T)
    return np.array(processed_list)


# ══════════════════════════════════════════════════════════════════════════════
# Training Routines
# ══════════════════════════════════════════════════════════════════════════════

def train_model(user_id: str, model_type: str = "rf", test_size: float = 0.2) -> Dict[str, Any]:
    """Train a sEMG model for a user and save it to disk.

    Args:
        user_id: The ID of the user.
        model_type: "rf" (RandomForest), "cnn" (1D CNN), or "gru" (GRU).
        test_size: Split ratio for testing/validation.

    Returns:
        A dictionary containing training metrics and results.
    """
    raw_segs, labels_raw = load_dataset(user_id)
    unique_labels = [str(l) for l in sorted(set(labels_raw))]
    num_classes = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in labels_raw], dtype=np.int64)

    # 1. Preprocess
    X_pre = preprocess_segments(
        raw_segs,
        fs=config.SAMPLE_RATE,
        low=config.BANDPASS_LOW,
        high=config.BANDPASS_HIGH
    )  # Shape: (N, C, T)

    # Split indices for consistency across types
    indices = np.arange(len(raw_segs))
    try:
        idx_train, idx_test, y_train, y_test = train_test_split(
            indices, y, test_size=test_size, stratify=y, random_state=42
        )
    except ValueError:
        # Handle cases where stratification is impossible
        idx_train, idx_test, y_train, y_test = train_test_split(
            indices, y, test_size=test_size, random_state=42
        )

    if model_type == "rf":
        # RandomForest uses 1D TD10 feature vectors
        # Extract features for all preprocessed segments
        X_feats = []
        for i in range(len(raw_segs)):
            # extract_features expects (T, C)
            feat = extract_features(X_pre[i].T, sample_rate=config.SAMPLE_RATE)
            X_feats.append(feat)
        X_feats = np.array(X_feats)

        X_train_f, X_test_f = X_feats[idx_train], X_feats[idx_test]

        # Build pipeline
        steps = [('scaler', StandardScaler())]
        if num_classes > 2:
            lda_comp = min(config.LDA_COMPONENTS, num_classes - 1)
            steps.append(('lda', LinearDiscriminantAnalysis(n_components=lda_comp)))

        steps.append(('clf', RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            random_state=42,
            n_jobs=-1
        )))
        pipeline = Pipeline(steps)
        pipeline.fit(X_train_f, y_train)

        # Predict
        y_pred = pipeline.predict(X_test_f)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

        # Save model data
        model_data = {
            "model": pipeline,
            "labels": unique_labels,
            "model_type": "rf"
        }
        save_model(model_data, user_id, model_type="rf")

    else:
        # Deep learning models (CNN / GRU)
        X_train_p = X_pre[idx_train].astype(np.float32)
        X_test_p = X_pre[idx_test].astype(np.float32)

        # Z-score normalization along channels
        mean = X_train_p.mean(axis=(0, 2), keepdims=True)
        std = X_train_p.std(axis=(0, 2), keepdims=True) + 1e-8

        X_train_p = (X_train_p - mean) / std
        X_test_p = (X_test_p - mean) / std

        # Setup PyTorch model on CPU to avoid MPS pooling compatibility issues
        device = torch.device("cpu")
        segment_len = X_pre.shape[2]
        num_channels = X_pre.shape[1]

        if model_type == "cnn":
            model = EMG1DCNN(num_channels, num_classes, segment_length=segment_len)
        elif model_type == "gru":
            model = EMGGRU(num_channels, num_classes)
        else:
            raise ValueError(f"Unknown model type '{model_type}'")

        model = model.to(device)

        # Datasets
        train_ds = TensorDataset(torch.tensor(X_train_p), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Optimization loop
        epochs = 40
        model.train()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(X_test_p).to(device)
            outputs = model(test_tensor)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

        # Save model data
        model_data = {
            "state_dict": model.state_dict(),
            "labels": unique_labels,
            "mean": mean,
            "std": std,
            "model_type": model_type,
            "num_channels": num_channels,
            "segment_length": segment_len
        }
        save_model(model_data, user_id, model_type=model_type)

    # Compute per-class accuracy
    per_class_acc = {}
    for i, label in enumerate(unique_labels):
        mask = y_test == i
        if mask.sum() > 0:
            per_class_acc[label] = float(accuracy_score(y_test[mask], y_pred[mask]))
        else:
            per_class_acc[label] = 0.0

    return {
        "accuracy": float(acc),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "labels": unique_labels,
        "num_samples": len(raw_segs),
    }
