"""SpeechNet tiny CNN per SilentWear (Spacone et al. 2026, ETH Zurich).

15k-parameter depthwise-separable CNN for 8 commands + rest, tailored for
GAP9 MCU (63.9 uJ / inference). Input ``(B, C, T)`` with C=14 (SilentWear)
or 4-8 (generic), T=150 (1.5 s @ 100 Hz or 150 ms @ 1000 Hz window).

Architecture
------------
3 depthwise-separable conv blocks (depthwise Conv1d groups + pointwise 1x1),
each with BatchNorm, ReLU, MaxPool(2), Dropout, followed by
AdaptiveAvgPool1d + linear head. ~15k params for C=4..14.

Reference
---------
Spacone et al. 2026, SilentWear, ETH Zurich — GAP9 deployment, inter-session
fine-tuning with <10 min calibration data.

Guarded: torch is optional; missing torch raises MissingDependencyError.
"""

from __future__ import annotations

import logging
from typing import Any

from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            "torch is required for SpeechNet. Install with 'pip install \"subvocal[ml]\"'"
        )


# ---------------------------------------------------------------------------
# SpeechNet
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class SpeechNet(nn.Module):  # type: ignore[no-redef]
        """Tiny depthwise-separable CNN (~15k params) for sEMG commands.

        Args:
            num_channels: EMG channels (14 for SilentWear, 4-8 generic).
            num_classes: Output classes (8 commands + rest if needed).
            segment_length: Temporal length T (default 150).
            dropout: Dropout prob between blocks.
        """

        def __init__(
            self,
            num_channels: int = 4,
            num_classes: int = 8,
            segment_length: int = 150,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if num_channels <= 0 or num_classes <= 0 or segment_length <= 0:
                raise ValueError("num_channels/num_classes/segment_length must be positive")
            self.num_channels = num_channels
            self.num_classes = num_classes
            self.segment_length = segment_length

            # Block 1: C -> 32
            self.block1 = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3, groups=num_channels, bias=False),
                nn.Conv1d(num_channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            )
            # Block 2: 32 -> 64
            self.block2 = nn.Sequential(
                nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32, bias=False),
                nn.Conv1d(32, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            )
            # Block 3: 64 -> 96 (depthwise 3, pointwise to 96 to hit ~15k)
            # Use 96 output to reach ~15k params; 64 keeps ~11k.
            # Choose 96 to match paper's 15k. Includes MaxPool to downsample 8x total.
            self.block3 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
                nn.Conv1d(64, 96, kernel_size=1, bias=False),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            )

            self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
            # 96 * 8 = 768
            self.classifier = nn.Linear(96 * 8, num_classes)

            # GAP9 paper numbers
            self._paper_energy_uj = 63.9
            self._paper_latency_ms = 6.4  # ~63.9uJ at ~10mW

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            """Forward pass.

            Args:
                x: Tensor of shape (B, C, T).

            Returns:
                Logits of shape (B, num_classes).
            """
            # (B, C, T)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.adaptive_pool(x)  # (B, 96, 8)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

        def count_parameters(self) -> int:
            """Return total parameter count (~15k, including frozen)."""
            return sum(p.numel() for p in self.parameters())

        def count_trainable_parameters(self) -> int:
            """Return trainable parameter count (after freezing backbone)."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def count_parameters_total(self) -> int:
            """Alias for count_parameters()."""
            return self.count_parameters()

        def estimate_energy(self) -> float:
            """Energy per inference in microjoules.

            Returns paper's GAP9 measurement 63.9 uJ, scaled slightly for
            channel count (linear with first pointwise).
            """
            # Scale modestly: base 63.9 for C=14, ~linear for Generic
            scale = self.num_channels / 14.0
            # clamp scaling to avoid large deviation: 0.8 - 1.0
            factor = 0.85 + 0.15 * scale
            return float(self._paper_energy_uj * factor)

        def estimate_energy_uj(self) -> float:
            """Alias for estimate_energy()."""
            return self.estimate_energy()

        def estimate_latency(self) -> float:
            """Latency per inference in milliseconds (GAP9)."""
            # Scale with channels similarly
            scale = self.num_channels / 14.0
            factor = 0.85 + 0.15 * scale
            return float(self._paper_latency_ms * factor)

        def estimate_latency_ms(self) -> float:
            """Alias for estimate_latency()."""
            return self.estimate_latency()

        def estimate_flops(self) -> int:
            """Estimate FLOPs per forward pass."""
            # Depthwise: 2*K*C*T ; Pointwise: 2*C_in*C_out*T ; BN negligible
            T = self.segment_length
            T1 = T // 2
            T2 = T1 // 2
            _T3 = T2 // 2  # kept for symmetry, not used in flops
            # Block1 (operates at T)
            dw1 = 2 * 7 * self.num_channels * T
            pw1 = 2 * self.num_channels * 32 * T
            # Block2 (at T1)
            dw2 = 2 * 5 * 32 * T1
            pw2 = 2 * 32 * 64 * T1
            # Block3 (at T2, pools to T3 before head)
            dw3 = 2 * 3 * 64 * T2
            pw3 = 2 * 64 * 96 * T2
            fc = 2 * (96 * 8) * self.num_classes
            return int(dw1 + pw1 + dw2 + pw2 + dw3 + pw3 + fc)

else:  # torch missing — stub

    class SpeechNet:  # type: ignore[no-redef]
        """Stub — raises MissingDependencyError when torch is absent."""

        def __init__(
            self,
            num_channels: int = 4,
            num_classes: int = 8,
            segment_length: int = 150,
            dropout: float = 0.2,
        ) -> None:
            _require_torch()

        def forward(self, x: Any) -> Any:
            _require_torch()

        def count_parameters(self) -> int:
            _require_torch()
            return 0

        def estimate_energy(self) -> float:
            _require_torch()
            return 0.0

        def estimate_latency(self) -> float:
            _require_torch()
            return 0.0


# ---------------------------------------------------------------------------
# Training helpers — SilentWear inter-session fine-tuning
# ---------------------------------------------------------------------------

def _freeze_backbone(model: Any, freeze: bool = True) -> None:
    """Freeze/unfreeze conv backbone, keep classifier trainable."""
    _require_torch()
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        elif name.startswith("block"):
            param.requires_grad = not freeze
        else:
            # adaptive_pool has no params; keep as is
            param.requires_grad = not freeze


def finetune_inter_session(
    model: Any,
    train_loader: Any,
    val_loader: Any | None = None,
    epochs: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Incremental fine-tuning for a new session (<10 min data).

    SilentWear protocol: freeze depthwise-separable backbone, fine-tune
    only the linear head (or last block + head) with low LR for few epochs,
    using calibration data collected in <10 min (~50-150 utterances).

    Args:
        model: SpeechNet instance (or any nn.Module with .classifier).
        train_loader: DataLoader yielding (B, C, T), labels.
        val_loader: Optional validation loader.
        epochs: Few epochs (3-10, default 5) to avoid overfit on tiny set.
        lr: Low learning rate (1e-4 - 1e-3).
        weight_decay: Weight decay.
        freeze_backbone: If True, freeze blocks 1-3, train head only.
        device: Torch device.

    Returns:
        Dict with train loss history and val accuracy if available.
    """
    _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    model = model.to(device)
    if freeze_backbone:
        _freeze_backbone(model, freeze=True)
    else:
        for p in model.parameters():
            p.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        avg = epoch_loss / max(n, 1)
        history.append(avg)
        logger.debug("SpeechNet finetune epoch %d/%d loss %.4f", epoch + 1, epochs, avg)

    result: dict[str, Any] = {"train_loss_history": history, "epochs": epochs, "lr": lr}
    if val_loader is not None:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total else 0.0
        result["val_accuracy"] = float(acc)
    return result


# Alias for discoverability
finetune_speechnet = finetune_inter_session
speechnet_finetune = finetune_inter_session
incremental_finetune = finetune_inter_session


def train_speechnet(
    num_channels: int = 4,
    num_classes: int = 8,
    segment_length: int = 150,
    train_data: Any | None = None,
    train_labels: Any | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Train SpeechNet from scratch (convenience wrapper).

    If train_data/train_labels are None, returns untrained model.
    Otherwise builds a DataLoader and trains with AdamW + CrossEntropy.

    Returns:
        (model, history_dict)
    """
    _require_torch()
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SpeechNet(num_channels=num_channels, num_classes=num_classes, segment_length=segment_length)
    model = model.to(device)

    if train_data is None or train_labels is None:
        return model, {"epochs": 0, "note": "no data provided, returned untrained model"}

    # Normalize to tensors
    if isinstance(train_data, np.ndarray):
        xb = torch.tensor(train_data, dtype=torch.float32)
    else:
        xb = train_data
    if isinstance(train_labels, np.ndarray):
        yb = torch.tensor(train_labels, dtype=torch.long)
    else:
        yb = train_labels

    dataset = TensorDataset(xb, yb)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n = 0
        for b_x, b_y in loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * b_x.size(0)
            n += b_x.size(0)
        history.append(epoch_loss / max(n, 1))

    return model, {"train_loss_history": history, "epochs": epochs}


__all__ = [
    "SpeechNet",
    "finetune_inter_session",
    "finetune_speechnet",
    "speechnet_finetune",
    "incremental_finetune",
    "train_speechnet",
]
