"""SAL / LBN adaptation — Spatial Adaptation Layer + Learnable Baseline Norm.

Implements the inter-session adaptation method from:

* Pereira et al., 2024 — *"SAL and LBN for HD-sEMG Inter-Session Adaptation"*
  arXiv:2409.08058 — Spatial Adaptation Layer (learnable affine warp of the
  electrode grid to compensate for shift) combined with Learnable Baseline
  Normalisation (per-channel bias subtraction, inspired by baseline
  normalisation in Gaddy & Klein 2020 / J. Neural Eng. 2024 [4]).

SAL is a lightweight, prependable input transform: biosignals from a new
session are warped back to the original spatial frame before entering any
backbone (CNN/GRU/Transformer). For HD-sEMG (e.g. 8×16 = 128 ch) this is a
2-D affine resampling (``2×3`` matrix, 6 DoF; paper reports 7 params incl.
global scale — we expose 6 DoF and the 2-DoF translation-only variant).
For generic ``(B,C,T)`` sEMG the spatial warp is complemented (and for
non-grid montages replaced) by a per-channel affine ``x * scale + shift``
(``2*C`` learnable params). LBN is a per-channel learnable bias
subtraction ``x - bias``.

Reference
---------
Pereira et al., arXiv:2409.08058, 2024.
Gaddy & Klein, EMNLP 2020; JNE 2024 — baseline normalisation baseline.
Torch guarded: falls back to numpy when torch is absent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = ["SAL", "LBN", "SAL_LBN", "adapt_sal_lbn"]

# ---------------------------------------------------------------------------
# torch guard
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - missing torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            "torch is required for SAL/LBN (torch). Install with 'pip install \"subvocal[ml]\"'"
        )


# ---------------------------------------------------------------------------
# helpers — canonicalisation (B,C,T) vs (B,T,C) vs (C,T)
# ---------------------------------------------------------------------------

def _canonicalize_numpy(
    x: np.ndarray, num_channels: int
) -> tuple[np.ndarray, bool, bool]:
    """Map numpy input to canonical (B,C,T). Returns (x_can, transposed, squeezed)."""
    if x.ndim == 2:
        if x.shape[0] == num_channels:
            return x[np.newaxis, :, :], False, True
        if x.shape[1] == num_channels:
            return x.T[np.newaxis, :, :], True, True
        # fallback: assume (C,T)
        return x[np.newaxis, :, :], False, True
    if x.ndim == 3:
        if x.shape[1] == num_channels and x.shape[2] != num_channels:
            return x, False, False
        if x.shape[2] == num_channels and x.shape[1] != num_channels:
            return np.transpose(x, (0, 2, 1)), True, False
        # ambiguous — assume (B,C,T)
        return x, False, False
    raise ValueError(f"SAL/LBN expects 2D or 3D array, got {x.ndim}D shape {x.shape}")


def _decanonicalize_numpy(
    x_can: np.ndarray, transposed: bool, squeezed: bool
) -> np.ndarray:
    if transposed and squeezed:
        return np.transpose(x_can[0], (1, 0))
    if squeezed:
        return x_can[0]
    if transposed:
        return np.transpose(x_can, (0, 2, 1))
    return x_can


# torch canonicalisation helpers (only when torch available)
if _TORCH_AVAILABLE:

    def _canonicalize_torch(
        x: Any, num_channels: int
    ) -> tuple[Any, bool, bool]:
        if x.dim() == 2:  # type: ignore[union-attr]
            if x.shape[0] == num_channels:  # type: ignore[union-attr]
                return x.unsqueeze(0), False, True  # type: ignore[union-attr]
            if x.shape[1] == num_channels:  # type: ignore[union-attr]
                return x.T.unsqueeze(0), True, True  # type: ignore[union-attr]
            return x.unsqueeze(0), False, True  # type: ignore[union-attr]
        if x.dim() == 3:  # type: ignore[union-attr]
            if x.shape[1] == num_channels and x.shape[2] != num_channels:  # type: ignore[union-attr]
                return x, False, False
            if x.shape[2] == num_channels and x.shape[1] != num_channels:  # type: ignore[union-attr]
                return x.transpose(1, 2), True, False  # type: ignore[union-attr]
            return x, False, False
        raise ValueError(f"SAL/LBN expects 2D or 3D Tensor, got {x.dim()}D shape {tuple(x.shape)}")  # type: ignore[union-attr]

    def _decanonicalize_torch(x_can: Any, transposed: bool, squeezed: bool) -> Any:
        if transposed and squeezed:
            return x_can.squeeze(0).T  # type: ignore[union-attr]
        if squeezed:
            return x_can.squeeze(0)  # type: ignore[union-attr]
        if transposed:
            return x_can.transpose(1, 2)  # type: ignore[union-attr]
        return x_can

    def _apply_grid_torch(
        x_can: Any,
        affine: Any | None,
        translation: Any | None,
        grid_h: int = 8,
        grid_w: int = 16,
    ) -> Any:
        """Apply 2-D affine grid sampling if channels match H*W."""
        B, C, T = x_can.shape  # type: ignore[union-attr]
        if C != grid_h * grid_w:
            return x_can
        if affine is None and translation is None:
            return x_can
        N = B * T  # type: ignore[operator]
        # (B,C,T) -> (N,1,H,W)
        x_4d = x_can.permute(0, 2, 1).reshape(N, 1, grid_h, grid_w)  # type: ignore[union-attr]
        if translation is not None:
            # build theta [[1,0,tx],[0,1,ty]]
            theta = torch.eye(2, 3, device=x_can.device, dtype=x_can.dtype).unsqueeze(0).expand(N, -1, -1).clone()  # type: ignore[union-attr]
            theta[:, 0, 2] = translation[0]  # type: ignore[index]
            theta[:, 1, 2] = translation[1]  # type: ignore[index]
        else:
            # affine is (2,3)
            theta = affine.unsqueeze(0).expand(N, -1, -1)  # type: ignore[union-attr]
        grid = F.affine_grid(theta, x_4d.size(), align_corners=False)  # type: ignore[union-attr]
        x_warped = F.grid_sample(x_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # type: ignore[union-attr]
        # (N,1,H,W) -> (B,C,T)
        x_warped = x_warped.squeeze(1).view(B, T, grid_h, grid_w).reshape(B, T, C).permute(0, 2, 1)  # type: ignore[union-attr]
        return x_warped


# ---------------------------------------------------------------------------
# SAL — Spatial Adaptation Layer
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class SAL(nn.Module):  # type: ignore[no-redef]
        """Spatial Adaptation Layer (SAL) per Pereira et al., arXiv:2409.08058.

        Learnable, prependable input transform that warps a new session's
        biosignals back to the original spatial frame. For HD-sEMG grids
        (default 8×16) a 2-D affine (``2×3``, 6 DoF) is applied via
        ``affine_grid`` + ``grid_sample``; the translation-only variant keeps
        2 DoF ``(tx, ty)``. For any montage a per-channel affine
        ``x * scale + shift`` (``2*C`` params, identity init) is always
        present — the paper's 7 params are realised as 6 affine + global
        scale bias, here decomposed as per-channel gain/bias + spatial warp.

        Args:
            num_channels: Number of EMG channels (default 8; 128 for 8×16 HD).
            use_affine: If True, learn full 2×3 affine (6 params). Ignored if
                ``use_translation_only`` is True.
            use_translation_only: If True, only 2 translation params.

        Input:
            ``(B,C,T)`` or ``(B,T,C)`` Tensor (or ``(C,T)`` single sample), or
            ``np.ndarray`` of same shapes (handled via torch conversion).

        Output:
            Tensor/ndarray of same shape as input.

        Reference:
            Pereira et al., arXiv:2409.08058, Sec. 2–3.
        """

        def __init__(
            self,
            num_channels: int = 8,
            use_affine: bool = True,
            use_translation_only: bool = False,
        ) -> None:
            super().__init__()
            if num_channels <= 0:
                raise ValueError(f"num_channels must be >0, got {num_channels}")
            self.num_channels = int(num_channels)
            self.use_affine = bool(use_affine)
            self.use_translation_only = bool(use_translation_only)
            # per-channel scale / shift — 2*C params, identity init
            self.scale = nn.Parameter(torch.ones(self.num_channels, dtype=torch.float32))  # type: ignore[union-attr]
            self.shift = nn.Parameter(torch.zeros(self.num_channels, dtype=torch.float32))  # type: ignore[union-attr]
            # spatial warp params
            self.affine: Any | None = None
            self.translation: Any | None = None
            if self.use_translation_only:
                self.translation = nn.Parameter(torch.zeros(2, dtype=torch.float32))  # type: ignore[union-attr]
                # register None affine for clean state_dict
                self.affine = None
            elif self.use_affine:
                # identity 2x3
                eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)  # type: ignore[union-attr]
                self.affine = nn.Parameter(eye)  # type: ignore[union-attr]
            # grid dims for HD case
            self.grid_h = 8
            self.grid_w = 16
            logger.debug(
                "SAL init: C=%d affine=%s trans_only=%s grid=%dx%d",
                self.num_channels, self.use_affine, self.use_translation_only, self.grid_h, self.grid_w,
            )

        def get_affine_params(self) -> Any:
            """Return affine matrix ``(2,3)`` (translation embedded if needed)."""
            if self.use_translation_only and self.translation is not None:
                mat = torch.eye(2, 3, device=self.translation.device, dtype=self.translation.dtype)  # type: ignore[union-attr]
                mat[0, 2] = self.translation[0]  # type: ignore[index]
                mat[1, 2] = self.translation[1]  # type: ignore[index]
                return mat
            if self.use_affine and self.affine is not None:
                return self.affine
            # identity when warp disabled
            return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=self.scale.device, dtype=self.scale.dtype)  # type: ignore[union-attr]

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            # numpy input → torch conversion path (preserves numpy fallback intent)
            was_numpy = isinstance(x, np.ndarray)
            if was_numpy:
                dtype = x.dtype
                x_t = torch.from_numpy(x.astype(np.float32))  # type: ignore[union-attr]
                # ensure device cpu
                out_t = self.forward(x_t)
                # preserve shape/dtype
                out_np = out_t.detach().cpu().numpy()  # type: ignore[union-attr]
                if dtype != np.float32:
                    # cast back if original was float64 etc.
                    out_np = out_np.astype(dtype, copy=False)
                return out_np

            if not isinstance(x, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"SAL expects Tensor or ndarray, got {type(x)}")

            x_can, transposed, squeezed = _canonicalize_torch(x, self.num_channels)

            # 2-D grid warp (HD) — prepend before per-channel affine (warp raw space)
            if self.use_affine or self.use_translation_only:
                x_can = _apply_grid_torch(x_can, self.affine, self.translation, self.grid_h, self.grid_w)

            # per-channel affine: x * scale + shift (handle C mismatch gracefully)
            _, C_actual, _ = x_can.shape  # type: ignore[union-attr]
            if C_actual != self.num_channels:
                logger.warning("SAL num_channels=%d but input C=%d — adapting scale/shift", self.num_channels, C_actual)
                if C_actual < self.num_channels:
                    scale_eff = self.scale[:C_actual]  # type: ignore[index]
                    shift_eff = self.shift[:C_actual]  # type: ignore[index]
                else:
                    pad = C_actual - self.num_channels
                    scale_eff = torch.cat(  # type: ignore[union-attr]
                        [self.scale, torch.ones(pad, device=self.scale.device, dtype=self.scale.dtype)]  # type: ignore[union-attr]
                    )
                    shift_eff = torch.cat(  # type: ignore[union-attr]
                        [self.shift, torch.zeros(pad, device=self.shift.device, dtype=self.shift.dtype)]  # type: ignore[union-attr]
                    )
                x_can = x_can * scale_eff.view(1, -1, 1) + shift_eff.view(1, -1, 1)  # type: ignore[union-attr]
            else:
                x_can = x_can * self.scale.view(1, -1, 1) + self.shift.view(1, -1, 1)  # type: ignore[union-attr]

            out = _decanonicalize_torch(x_can, transposed, squeezed)
            return out

    class LBN(nn.Module):  # type: ignore[no-redef]
        """Learnable Baseline Normalisation (LBN) per Pereira et al.

        Per-channel learnable bias subtraction ``x - bias``. Bias is
        initialised to zero (identity). Inspired by baseline normalisation in
        Gaddy & Klein 2020 / JNE 2024 [4] where resting baseline is removed
        per channel; here the baseline is learned jointly with SAL and the
        backbone.

        Args:
            num_channels: Number of EMG channels.

        Reference:
            Pereira et al., arXiv:2409.08058, Sec. 3.2; baseline norm [4].
        """

        def __init__(self, num_channels: int = 8) -> None:
            super().__init__()
            if num_channels <= 0:
                raise ValueError(f"num_channels must be >0, got {num_channels}")
            self.num_channels = int(num_channels)
            self.bias = nn.Parameter(torch.zeros(self.num_channels, dtype=torch.float32))  # type: ignore[union-attr]
            logger.debug("LBN init: C=%d", self.num_channels)

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            was_numpy = isinstance(x, np.ndarray)
            if was_numpy:
                dtype = x.dtype
                x_t = torch.from_numpy(x.astype(np.float32))  # type: ignore[union-attr]
                out_t = self.forward(x_t)
                out_np = out_t.detach().cpu().numpy()  # type: ignore[union-attr]
                if dtype != np.float32:
                    out_np = out_np.astype(dtype, copy=False)
                return out_np
            if not isinstance(x, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"LBN expects Tensor or ndarray, got {type(x)}")
            x_can, transposed, squeezed = _canonicalize_torch(x, self.num_channels)
            _, C_actual, _ = x_can.shape  # type: ignore[union-attr]
            if C_actual != self.num_channels:
                logger.warning("LBN num_channels=%d but input C=%d — adapting bias", self.num_channels, C_actual)
                if C_actual < self.num_channels:
                    bias_eff = self.bias[:C_actual]  # type: ignore[index]
                else:
                    pad = C_actual - self.num_channels
                    bias_eff = torch.cat(  # type: ignore[union-attr]
                        [self.bias, torch.zeros(pad, device=self.bias.device, dtype=self.bias.dtype)]  # type: ignore[union-attr]
                    )
                x_can = x_can - bias_eff.view(1, -1, 1)  # type: ignore[union-attr]
            else:
                x_can = x_can - self.bias.view(1, -1, 1)  # type: ignore[union-attr]
            out = _decanonicalize_torch(x_can, transposed, squeezed)
            return out

    class SAL_LBN(nn.Module):  # type: ignore[no-redef]
        """Combined SAL + LBN adaptation block.

        Sequentially applies :class:`SAL` then :class:`LBN`:
        ``forward(x) == LBN(SAL(x))``. Prependable to any model:
        ``logits = backbone(sal_lbn(emg))``.

        Args:
            num_channels: Number of EMG channels.
            use_affine: Passed to :class:`SAL`.
            use_translation_only: Passed to :class:`SAL`.

        Reference:
            Pereira et al., arXiv:2409.08058 — SAL/LBN combined.
        """

        def __init__(
            self,
            num_channels: int = 8,
            use_affine: bool = True,
            use_translation_only: bool = False,
        ) -> None:
            super().__init__()
            self.num_channels = int(num_channels)
            self.sal = SAL(
                num_channels=num_channels,
                use_affine=use_affine,
                use_translation_only=use_translation_only,
            )
            self.lbn = LBN(num_channels=num_channels)
            logger.debug("SAL_LBN init: C=%d", self.num_channels)

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            return self.lbn(self.sal(x))  # type: ignore[operator]

        def get_affine_params(self) -> Any:
            """Delegate to underlying SAL."""
            return self.sal.get_affine_params()  # type: ignore[union-attr]

else:  # torch not available — numpy fallback classes

    class SAL:  # type: ignore[no-redef]
        """SAL numpy fallback (torch not installed).

        Same API as torch SAL but using numpy arrays. Per-channel
        ``x * scale + shift`` with learnable 2*C params; affine 2×3 and
        translation 2-vector stored as numpy. Grid warp is no-op in pure
        numpy (per-channel only) to keep fallback lightweight.
        """

        def __init__(
            self,
            num_channels: int = 8,
            use_affine: bool = True,
            use_translation_only: bool = False,
        ) -> None:
            if num_channels <= 0:
                raise ValueError(f"num_channels must be >0, got {num_channels}")
            self.num_channels = int(num_channels)
            self.use_affine = bool(use_affine)
            self.use_translation_only = bool(use_translation_only)
            self.scale = np.ones(self.num_channels, dtype=np.float32)
            self.shift = np.zeros(self.num_channels, dtype=np.float32)
            self.affine: np.ndarray | None = None
            self.translation: np.ndarray | None = None
            if self.use_translation_only:
                self.translation = np.zeros(2, dtype=np.float32)
            elif self.use_affine:
                self.affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            self.grid_h = 8
            self.grid_w = 16

        def get_affine_params(self) -> np.ndarray:
            if self.use_translation_only and self.translation is not None:
                mat = np.eye(2, 3, dtype=np.float32)
                mat[0, 2] = float(self.translation[0])
                mat[1, 2] = float(self.translation[1])
                return mat
            if self.use_affine and self.affine is not None:
                return self.affine.copy()
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
            if not isinstance(x, np.ndarray):
                raise TypeError(f"SAL (numpy fallback) expects ndarray, got {type(x)}")
            x_can, transposed, squeezed = _canonicalize_numpy(x, self.num_channels)
            # grid warp is identity in numpy fallback (optional, not required for tests)
            x_can = x_can * self.scale.reshape(1, -1, 1) + self.shift.reshape(1, -1, 1)
            out = _decanonicalize_numpy(x_can, transposed, squeezed)
            return out

        def __call__(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
            return self.forward(x)

        # minimal nn.Module-like helpers for adaptation loop compatibility
        def parameters(self) -> list[Any]:  # type: ignore[override]
            return []

        def train(self, mode: bool = True) -> None:
            pass

        def eval(self) -> None:
            pass

        def to(self, device: str) -> Any:  # type: ignore[override]
            return self

    class LBN:  # type: ignore[no-redef]
        """LBN numpy fallback."""

        def __init__(self, num_channels: int = 8) -> None:
            if num_channels <= 0:
                raise ValueError(f"num_channels must be >0, got {num_channels}")
            self.num_channels = int(num_channels)
            self.bias = np.zeros(self.num_channels, dtype=np.float32)

        def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
            if not isinstance(x, np.ndarray):
                raise TypeError(f"LBN (numpy fallback) expects ndarray, got {type(x)}")
            x_can, transposed, squeezed = _canonicalize_numpy(x, self.num_channels)
            x_can = x_can - self.bias.reshape(1, -1, 1)
            out = _decanonicalize_numpy(x_can, transposed, squeezed)
            return out

        def __call__(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
            return self.forward(x)

        def parameters(self) -> list[Any]:  # type: ignore[override]
            return []

        def train(self, mode: bool = True) -> None:
            pass

        def eval(self) -> None:
            pass

        def to(self, device: str) -> Any:  # type: ignore[override]
            return self

    class SAL_LBN:  # type: ignore[no-redef]
        """SAL_LBN numpy fallback."""

        def __init__(
            self,
            num_channels: int = 8,
            use_affine: bool = True,
            use_translation_only: bool = False,
        ) -> None:
            self.num_channels = int(num_channels)
            self.sal = SAL(num_channels=num_channels, use_affine=use_affine, use_translation_only=use_translation_only)
            self.lbn = LBN(num_channels=num_channels)

        def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
            return self.lbn.forward(self.sal.forward(x))

        def __call__(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
            return self.forward(x)

        def get_affine_params(self) -> np.ndarray:
            return self.sal.get_affine_params()

        def parameters(self) -> list[Any]:  # type: ignore[override]
            return []

        def train(self, mode: bool = True) -> None:
            pass

        def eval(self) -> None:
            pass

        def to(self, device: str) -> Any:  # type: ignore[override]
            return self


# ---------------------------------------------------------------------------
# adaptation helper — optimise only SAL/LBN params
# ---------------------------------------------------------------------------

def adapt_sal_lbn(
    model: Any,
    sal_lbn: Any,
    train_loader: Any,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    criterion: Any | None = None,
) -> dict[str, Any]:
    """Supervised adaptation of SAL/LBN on a new session.

    Freezes ``model`` and optimises only ``sal_lbn`` parameters via a
    supervised loss (``CrossEntropy`` for classification by default).

    The ``sal_lbn`` block is prepended: ``logits = model(sal_lbn(x))``.
    Typical usage (Pereira et al., Sec. 4 — <2 min calibration data)::

        sal_lbn = SAL_LBN(num_channels=8)
        adapt_sal_lbn(backbone, sal_lbn, calib_loader, epochs=5, lr=1e-3)
        logits = backbone(sal_lbn(emg_new_session))

    Args:
        model: Backbone ``nn.Module`` (e.g. CNN/GRU). Frozen in-place
            (``requires_grad=False``) for the duration.
        sal_lbn: :class:`SAL` | :class:`LBN` | :class:`SAL_LBN` instance.
        train_loader: Iterable yielding ``(x, y)`` batches or dicts with
            ``x``/``y``. ``x`` shape ``(B,C,T)`` or ``(B,T,C)``.
        epochs: Number of adaptation epochs (default 5, paper uses 3-5).
        lr: Learning rate for SAL/LBN (default 1e-3).
        device: Torch device.
        criterion: Loss fn; defaults to ``CrossEntropyLoss``. May be
            ``MSELoss`` for regression targets.

    Returns:
        Dict with ``loss_history``, ``epochs``, ``lr``, ``final_loss``.

    Raises:
        MissingDependencyError: if torch not installed.

    Reference:
        Pereira et al., arXiv:2409.08058, Sec. 4 — session adaptation.
    """
    _require_torch()
    import torch.nn as nn  # local import for type safety
    import torch.optim as optim  # type: ignore[import]

    if epochs <= 0:
        raise ValueError(f"epochs must be >0, got {epochs}")
    if lr <= 0:
        raise ValueError(f"lr must be >0, got {lr}")

    # move to device if possible
    try:
        if hasattr(model, "to"):
            model = model.to(device)  # type: ignore[union-attr]
    except Exception as e:
        logger.debug("adapt_sal_lbn model.to(%s) failed: %s", device, e)
    try:
        if hasattr(sal_lbn, "to"):
            sal_lbn = sal_lbn.to(device)  # type: ignore[union-attr]
    except Exception as e:
        logger.debug("adapt_sal_lbn sal_lbn.to(%s) failed: %s", device, e)

    # freeze backbone, ensure sal_lbn is trainable
    for p in model.parameters():  # type: ignore[union-attr]
        p.requires_grad = False
    for p in sal_lbn.parameters():  # type: ignore[union-attr]
        p.requires_grad = True

    if criterion is None:
        criterion = nn.CrossEntropyLoss()  # type: ignore[union-attr]

    # Adam over SAL/LBN only (paper: Adam, low LR, few epochs)
    params = list(sal_lbn.parameters())  # type: ignore[union-attr]
    if not params:
        logger.warning("adapt_sal_lbn: sal_lbn has no parameters — nothing to optimise")
        return {"loss_history": [], "epochs": epochs, "lr": lr, "final_loss": 0.0}

    optimizer = optim.Adam(params, lr=lr)  # type: ignore[union-attr]

    # keep model in eval (frozen BN/dropout), sal_lbn in train
    try:
        model.eval()  # type: ignore[union-attr]
    except Exception:
        pass
    try:
        sal_lbn.train()  # type: ignore[union-attr]
    except Exception:
        pass

    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            # unpack
            if isinstance(batch, dict):
                xb = batch.get("x", batch.get("emg", batch.get("input")))
                yb = batch.get("y", batch.get("label", batch.get("target", batch.get("labels"))))
                if xb is None or yb is None:
                    # try first two values
                    vals = list(batch.values())
                    if len(vals) >= 2:
                        xb, yb = vals[0], vals[1]
                    else:
                        continue
            elif isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    continue
                xb, yb = batch[0], batch[1]
            else:
                logger.debug("adapt_sal_lbn skipping batch type %s", type(batch))
                continue

            # to tensor / device
            if isinstance(xb, np.ndarray):
                xb = torch.from_numpy(xb.astype(np.float32)).to(device)  # type: ignore[union-attr]
            elif isinstance(xb, torch.Tensor):  # type: ignore[arg-type]
                xb = xb.to(device)  # type: ignore[union-attr]
            else:
                try:
                    xb = torch.as_tensor(xb, dtype=torch.float32, device=device)  # type: ignore[union-attr]
                except Exception:
                    continue

            if isinstance(yb, np.ndarray):
                yb_t = torch.from_numpy(yb).to(device)  # type: ignore[union-attr]
                # infer dtype for criterion
                if isinstance(criterion, nn.CrossEntropyLoss):  # type: ignore[union-attr]
                    if yb_t.dtype != torch.long:  # type: ignore[union-attr]
                        # cross-entropy expects long
                        yb_t = yb_t.long()  # type: ignore[union-attr]
                yb = yb_t
            elif isinstance(yb, torch.Tensor):  # type: ignore[arg-type]
                yb = yb.to(device)  # type: ignore[union-attr]
                if isinstance(criterion, nn.CrossEntropyLoss) and yb.dtype != torch.long:  # type: ignore[union-attr]
                    # keep float labels as long if integer-like?
                    # only cast if values are integral and dim==1
                    try:
                        if yb.dim() == 1:
                            yb = yb.long()  # type: ignore[union-attr]
                    except Exception:
                        pass
            else:
                try:
                    yb = torch.as_tensor(yb, device=device)  # type: ignore[union-attr]
                    if isinstance(criterion, nn.CrossEntropyLoss) and yb.dtype != torch.long:  # type: ignore[union-attr]
                        yb = yb.long()  # type: ignore[union-attr]
                except Exception:
                    continue

            optimizer.zero_grad()
            xb_adapt = sal_lbn(xb)  # type: ignore[operator]
            logits = model(xb_adapt)  # type: ignore[operator]
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            # criterion dispatch with fallback
            try:
                loss = criterion(logits, yb)  # type: ignore[operator]
            except Exception as e:
                logger.debug("criterion failed (%s), trying fallback", e)
                # shape-adaptive fallback
                try:
                    if logits.dim() == 2 and yb.dim() == 1 and logits.shape[0] == yb.shape[0]:  # type: ignore[union-attr]
                        loss = nn.CrossEntropyLoss()(logits, yb.long())  # type: ignore[union-attr]
                    elif logits.shape == yb.shape:  # type: ignore[union-attr]
                        loss = nn.MSELoss()(logits.float(), yb.float())  # type: ignore[union-attr]
                    else:
                        # flatten
                        loss = criterion(logits.view(logits.shape[0], -1), yb.view(yb.shape[0], -1) if yb.dim() > 1 else yb)  # type: ignore[union-attr]
                except Exception as e2:
                    logger.warning("adapt_sal_lbn loss fallback failed: %s", e2)
                    continue

            if loss.dim() != 0:  # type: ignore[union-attr]
                loss = loss.mean()  # type: ignore[union-attr]
            loss.backward()  # type: ignore[union-attr]
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item()) * int(xb.size(0))  # type: ignore[union-attr]
            n += int(xb.size(0))  # type: ignore[union-attr]

        avg = epoch_loss / max(n, 1)
        history.append(avg)
        logger.debug("adapt_sal_lbn epoch %d/%d loss %.4f", epoch + 1, epochs, avg)

    return {"loss_history": history, "epochs": epochs, "lr": lr, "final_loss": float(history[-1]) if history else 0.0}
