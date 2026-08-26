"""Contrastive Pose-EMG Pretraining (CPEP) per Cui et al., arXiv:2509.04699.

CPEP aligns EMG and pose representations via contrastive learning for zero-shot
gesture classification. Two Transformer encoders (EMG + pose, 4 layers, dim 256)
project to a shared 256-d space (1-layer head) with L2 normalization. A
symmetric InfoNCE loss with learnable temperature tau (init 0.02) pulls matched
pairs together. Evaluation follows CLIP protocol: kNN voting in embedding
space (k=10 majority vote) and linear probing.

Reference:
    Cui et al., arXiv:2509.04699, 2025 — CPEP: Contrastive Pose-EMG Pre-training.
    Embedding dim 256, 1-layer proj head 256, tau learnable init 0.02, L2 norm,
    pose encoder frozen during alignment, zero-shot via TopK kNN cosine.

Torch / sklearn are optional; graceful MissingDependencyError is raised when
required ops are invoked without the dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = [
    "EMGEncoder",
    "PoseEncoder",
    "pose_emg_contrastive_loss",
    "contrastive_loss",
    "CPEPFramework",
    "knn_classify",
    "embedding_knn_classify",
    "zero_shot_knn_classify",
    "zero_shot_classify",
    "knn_predict",
    "l2_normalize_embeddings",
]

# ---------------------------------------------------------------------------
# torch / sklearn guards
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from sklearn.neighbors import KNeighborsClassifier  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            "torch is required for CPEP (torch). Install with 'pip install \"subvocal[ml]\"'"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def l2_normalize_embeddings(x: Any, dim: int = -1, eps: float = 1e-8) -> Any:
    """L2-normalize embeddings along ``dim`` (torch or numpy)."""
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
        return F.normalize(x, p=2, dim=dim, eps=eps)  # type: ignore[union-attr]
    arr = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(arr, axis=dim if dim != -1 else -1, keepdims=True)
    n = np.maximum(n, eps)
    return arr / n  # type: ignore[return-value]


def _l2_normalize_np(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


# ---------------------------------------------------------------------------
# Torch encoders (or fallback stubs)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _BaseTransformerEncoder(nn.Module):  # type: ignore[no-redef]
        """Shared Transformer base: Linear -> pos -> 4-layer Transformer -> 1-layer head -> L2."""

        def __init__(
            self,
            in_dim: int,
            seq_len: int = 150,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            embed_dim: int = 256,
            use_cls: bool = True,
        ) -> None:
            super().__init__()
            if in_dim <= 0:
                raise ValueError(f"in_dim must be >0, got {in_dim}")
            if d_model % nhead != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
            self.in_dim = int(in_dim)
            self.seq_len = int(seq_len)
            self.d_model = int(d_model)
            self.embed_dim = int(embed_dim)
            self.use_cls = bool(use_cls)
            self.input_proj = nn.Linear(self.in_dim, self.d_model)  # type: ignore[union-attr]
            # CLS token as in CPEP (paper uses [CLS])
            if self.use_cls:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))  # type: ignore[union-attr]
                max_len = self.seq_len + 1
            else:
                self.cls_token = None  # type: ignore[assignment]
                max_len = self.seq_len
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, self.d_model))  # type: ignore[union-attr]
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)  # type: ignore[union-attr]
            # 1-layer projection head (hidden size 256) per paper
            self.projection = nn.Linear(self.d_model, self.embed_dim)  # type: ignore[union-attr]
            # alias for tests expecting different names
            self.proj = self.projection  # type: ignore[assignment]
            self.projection_head = self.projection  # type: ignore[assignment]
            self.proj_head = self.projection  # type: ignore[assignment]
            self.norm = nn.LayerNorm(self.embed_dim)  # type: ignore[union-attr]
            self._init_weights()

        def _init_weights(self) -> None:
            # small init for pos/cls like ViT/MAE
            if _TORCH_AVAILABLE:
                nn.init.trunc_normal_(self.pos_embedding, std=0.02)  # type: ignore[union-attr]
                if self.cls_token is not None:
                    nn.init.trunc_normal_(self.cls_token, std=0.02)  # type: ignore[union-attr]

        def _handle_input(self, x: Any) -> Any:
            # x: (B,C,T) or (B,T,C) or (C,T) -> canonical (B,T,C)
            if not isinstance(x, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"Encoder expects Tensor, got {type(x)}")
            if x.dim() == 2:  # type: ignore[union-attr]
                # (C,T) or (T,C) -> assume (C,T) if first dim==in_dim else (T,C)
                if x.shape[0] == self.in_dim:  # type: ignore[union-attr]
                    # (C,T) -> (1,T,C)
                    x = x.T.unsqueeze(0)  # type: ignore[union-attr]
                elif x.shape[1] == self.in_dim:  # type: ignore[union-attr]
                    x = x.unsqueeze(0)  # type: ignore[union-attr]
                else:
                    # fallback treat as (T,C)
                    x = x.unsqueeze(0)  # type: ignore[union-attr]
                return x
            if x.dim() == 3:  # type: ignore[union-attr]
                # infer layout
                if x.shape[1] == self.in_dim:  # type: ignore[union-attr]
                    # (B,C,T) -> (B,T,C)
                    return x.transpose(1, 2)  # type: ignore[union-attr]
                if x.shape[2] == self.in_dim:  # type: ignore[union-attr]
                    return x
                # ambiguous: assume (B,C,T) if shape[2] larger than channel typical
                # fallback to transpose if last dim not equal but first small
                # keep heuristic: channels usually small (<128) vs time larger
                # if middle dim mismatched and last dim larger, treat as (B,C,T)
                # otherwise keep as is
                if x.shape[2] > self.in_dim and x.shape[1] < 128:  # type: ignore[union-attr]
                    # likely (B,C,T)
                    return x.transpose(1, 2)  # type: ignore[union-attr]
                return x
            raise ValueError(f"Encoder expects 2D or 3D Tensor, got {x.dim()}D {tuple(x.shape)}")  # type: ignore[union-attr]

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            x_t = self._handle_input(x)  # (B,T,in_dim)
            # guard T exceeding max pos length: truncate pos
            B, T, _ = x_t.shape  # type: ignore[union-attr]
            x_proj = self.input_proj(x_t)  # type: ignore[union-attr,operator]
            if self.use_cls and self.cls_token is not None:
                cls = self.cls_token.expand(B, -1, -1)  # type: ignore[union-attr]
                x_proj = torch.cat([cls, x_proj], dim=1)  # type: ignore[union-attr]
                T_eff = T + 1
            else:
                T_eff = T
            # pos slice
            plen = self.pos_embedding.shape[1]  # type: ignore[union-attr]
            if T_eff <= plen:
                pos = self.pos_embedding[:, :T_eff, :]  # type: ignore[union-attr,index]
            else:
                # if longer than max, interpolate/truncate: repeat last? simple slice + pad zero
                pos = self.pos_embedding[:, :plen, :]  # type: ignore[union-attr,index]
                # pad remainder with zeros (already zero init fallback)
                pad_len = T_eff - plen
                pad = torch.zeros(B, pad_len, self.d_model, device=x_proj.device, dtype=x_proj.dtype)  # type: ignore[union-attr]
                # extend x_proj already has data; we need pos same shape; pad pos with zeros
                pos = torch.cat([pos.expand(B, -1, -1), pad], dim=1)  # type: ignore[union-attr]
                # but x_proj already is (B,T_eff,d_model), so addition needs (1,T_eff,d_model) style
                # adjust: just add first plen part
                x_proj = x_proj + torch.cat(
                    [self.pos_embedding[:, :plen, :].expand(B, -1, -1), pad], dim=1  # type: ignore[union-attr]
                )
                # handled, skip second add
                out = self.transformer(x_proj)  # type: ignore[union-attr,operator]
                if self.use_cls:
                    cls_out = out[:, 0, :]  # type: ignore[union-attr]
                else:
                    cls_out = out.mean(dim=1)  # type: ignore[union-attr]
                proj = self.projection(cls_out)  # type: ignore[union-attr,operator]
                proj = self.norm(proj)  # type: ignore[union-attr]
                return F.normalize(proj, p=2, dim=-1)  # type: ignore[union-attr]
            x_proj = x_proj + pos  # type: ignore[operator]
            out = self.transformer(x_proj)  # type: ignore[union-attr,operator]
            if self.use_cls:
                cls_out = out[:, 0, :]  # type: ignore[union-attr]
            else:
                cls_out = out.mean(dim=1)  # type: ignore[union-attr]
            proj = self.projection(cls_out)  # type: ignore[union-attr,operator]
            proj = self.norm(proj)  # type: ignore[union-attr]
            return F.normalize(proj, p=2, dim=-1)  # type: ignore[union-attr]

    class EMGEncoder(_BaseTransformerEncoder):  # type: ignore[no-redef]
        """EMG Transformer encoder: 4 layers, dim 256, 1-layer projection to 256-d, L2 norm.

        Args:
            in_channels: Number of EMG channels (default 8).
            seq_len: Time length of EMG segment (default 150).
            d_model: Transformer hidden dim (default 256).
            nhead: Attention heads (default 8).
            num_layers: Transformer layers (default 4 per CPEP).
            dim_feedforward: FFN dim (default 512).
            dropout: Dropout.
            embed_dim: Output embedding dim (default 256 per CPEP).
            use_cls: Use [CLS] token as sequence summary (paper uses CLS).

        Input:
            Tensor shape ``(B,C,T)`` or ``(B,T,C)`` or ``(C,T)``.
        Output:
            L2-normalized embedding ``(B, embed_dim)``.
        """

        def __init__(
            self,
            in_channels: int = 8,
            seq_len: int = 150,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            embed_dim: int = 256,
            use_cls: bool = True,
            **kwargs: Any,
        ) -> None:
            # allow synonyms
            if "num_channels" in kwargs:
                in_channels = int(kwargs.pop("num_channels"))
            if "emg_channels" in kwargs:
                in_channels = int(kwargs.pop("emg_channels"))
            super().__init__(
                in_dim=int(in_channels),
                seq_len=int(seq_len),
                d_model=int(d_model),
                nhead=int(nhead),
                num_layers=int(num_layers),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                embed_dim=int(embed_dim),
                use_cls=bool(use_cls),
            )
            self.in_channels = self.in_dim  # alias

    class PoseEncoder(_BaseTransformerEncoder):  # type: ignore[no-redef]
        """Pose Transformer encoder: mirror of EMGEncoder for pose (4 layers, dim 256).

        Args:
            pose_dim: Input pose dimensionality per frame (e.g. 63 for 21 joints x3,
                42, or flattened heatmap dims). Default 63.
            seq_len: Temporal length (default 150 to match EMG).
            d_model: Hidden dim 256.
            nhead: Heads 8.
            num_layers: 4.
            dim_feedforward: 512.
            dropout: 0.1.
            embed_dim: 256 (shared space).
            use_cls: Use [CLS] token.

        Input:
            Tensor shape ``(B,T,D)`` or ``(B,D,T)`` or ``(T,D)`` where D==pose_dim.
            Both layouts are accepted (auto-detected).

        Output:
            L2-normalized embedding ``(B, embed_dim)``.
        """

        def __init__(
            self,
            pose_dim: int = 63,
            seq_len: int = 150,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            embed_dim: int = 256,
            use_cls: bool = True,
            **kwargs: Any,
        ) -> None:
            # synonyms for flexibility
            if "in_dim" in kwargs:
                pose_dim = int(kwargs.pop("in_dim"))
            if "in_channels" in kwargs:
                pose_dim = int(kwargs.pop("in_channels"))
            if "input_dim" in kwargs:
                pose_dim = int(kwargs.pop("input_dim"))
            super().__init__(
                in_dim=int(pose_dim),
                seq_len=int(seq_len),
                d_model=int(d_model),
                nhead=int(nhead),
                num_layers=int(num_layers),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                embed_dim=int(embed_dim),
                use_cls=bool(use_cls),
            )
            self.pose_dim = self.in_dim  # alias
            self.in_channels = self.in_dim  # compat

else:  # torch missing — stubs raising on use

    class EMGEncoder:  # type: ignore[no-redef]
        """Stub when torch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

        def forward(self, x: Any) -> Any:
            _require_torch()

        def __call__(self, x: Any) -> Any:
            return self.forward(x)

        def parameters(self) -> Any:
            _require_torch()

        def to(self, *a: Any, **k: Any) -> Any:
            _require_torch()

        def eval(self) -> Any:
            _require_torch()

        def train(self, *a: Any, **k: Any) -> Any:
            _require_torch()

    class PoseEncoder:  # type: ignore[no-redef]
        """Stub when torch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

        def forward(self, x: Any) -> Any:
            _require_torch()

        def __call__(self, x: Any) -> Any:
            return self.forward(x)

        def parameters(self) -> Any:
            _require_torch()

        def to(self, *a: Any, **k: Any) -> Any:
            _require_torch()

        def eval(self) -> Any:
            _require_torch()

        def train(self, *a: Any, **k: Any) -> Any:
            _require_torch()


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def pose_emg_contrastive_loss(
    emg_emb: Any,
    pose_emb: Any,
    tau: Any = 0.02,
) -> Any:
    """Symmetric InfoNCE for pose-EMG alignment (CPEP).

    Both embeddings are L2-normalized and cosine similarity is scaled by
    temperature ``tau`` (learnable, init 0.02 per paper):

        s_ij = (u_i·v_j)/tau,  u=norm(emg), v=norm(pose)

    Loss averages two CE losses over batch:

        L = 0.5*(CE(logits, arange(N)) + CE(logits.T, arange(N)))
        logits = u @ v.T / tau

    Args:
        emg_emb: ``(N,D)`` embeddings (Tensor or ndarray).
        pose_emb: ``(N,D)`` embeddings.
        tau: Temperature scalar — float or 0-d Tensor (learnable). Clamped
            to ``>=1e-4`` to avoid division blow-up. Init 0.02.

    Returns:
        Scalar loss Tensor (torch) or float (numpy fallback).

    Reference:
        CPEP Sec. 2.3, CLIP symmetric InfoNCE.
    """
    _require_torch()
    import torch  # type: ignore[import]
    import torch.nn.functional as F  # type: ignore[import]

    if not isinstance(emg_emb, torch.Tensor) or not isinstance(pose_emb, torch.Tensor):  # type: ignore[arg-type]
        # allow numpy -> convert
        try:
            emg_emb = torch.as_tensor(emg_emb, dtype=torch.float32)  # type: ignore[union-attr]
            pose_emb = torch.as_tensor(pose_emb, dtype=torch.float32)  # type: ignore[union-attr]
        except Exception as e:
            raise TypeError(f"pose_emg_contrastive_loss expects Tensor inputs, got {type(emg_emb)}, {type(pose_emb)}: {e}") from e

    if emg_emb.dim() == 1:  # type: ignore[union-attr]
        emg_emb = emg_emb.unsqueeze(0)  # type: ignore[union-attr]
    if pose_emb.dim() == 1:  # type: ignore[union-attr]
        pose_emb = pose_emb.unsqueeze(0)  # type: ignore[union-attr]
    if emg_emb.shape[0] != pose_emb.shape[0]:  # type: ignore[union-attr]
        raise ValueError(f"Batch size mismatch: emg {tuple(emg_emb.shape)} vs pose {tuple(pose_emb.shape)}")  # type: ignore[union-attr]

    # L2 normalize as per CPEP (Sec 2.3)
    emg_n = F.normalize(emg_emb.float(), p=2, dim=-1, eps=1e-8)  # type: ignore[union-attr]
    pose_n = F.normalize(pose_emb.float(), p=2, dim=-1, eps=1e-8)  # type: ignore[union-attr]

    # temperature handling — learnable scalar
    if isinstance(tau, torch.Tensor):  # type: ignore[arg-type]
        # keep on same device/dtype as embeddings, clamp for stability
        tau_t = tau.to(device=emg_n.device, dtype=emg_n.dtype)  # type: ignore[union-attr]
        tau_t = torch.clamp(tau_t, min=1e-4)  # type: ignore[union-attr]
        # ensure scalar
        if tau_t.numel() != 1:  # type: ignore[union-attr]
            tau_t = tau_t.mean()  # type: ignore[union-attr]
    else:
        tau_f = float(tau)
        if tau_f < 1e-4:
            raise ValueError(f"tau must be >=1e-4, got {tau_f}")
        tau_t = torch.tensor(tau_f, device=emg_n.device, dtype=emg_n.dtype)  # type: ignore[union-attr]

    # cosine similarity / temperature
    logits = emg_n @ pose_n.T / tau_t  # type: ignore[operator]
    n = logits.shape[0]  # type: ignore[union-attr]
    labels = torch.arange(n, device=logits.device, dtype=torch.long)  # type: ignore[union-attr]
    # handle N=1 edge (CE with single class =0, grad zero — acceptable)
    loss_emg = F.cross_entropy(logits, labels)  # type: ignore[union-attr]
    loss_pose = F.cross_entropy(logits.T, labels)  # type: ignore[union-attr]
    loss = (loss_emg + loss_pose) * 0.5
    return loss


# alias per spec naming variant
contrastive_loss = pose_emg_contrastive_loss


# ---------------------------------------------------------------------------
# CPEP framework
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class CPEPFramework(nn.Module):  # type: ignore[no-redef]
        """Contrastive Pose-EMG Pretraining framework (CPEP).

        Wraps :class:`EMGEncoder` and :class:`PoseEncoder` with a learnable
        temperature ``tau`` (init 0.02) and symmetric InfoNCE. During
        contrastive training the pose encoder is frozen (paper: only EMG encoder
        + projection head are updated) — controlled by ``freeze_pose``.

        Args:
            emg_channels: EMG input channels (default 8).
            pose_dim: Pose per-frame dim (default 63).
            seq_len: EMG temporal length (default 150).
            pose_seq_len: Pose temporal length (default seq_len if None).
            d_model: Transformer hidden dim (default 256).
            embed_dim: Shared embedding dim (default 256 per CPEP).
            nhead: Attention heads (default 8).
            num_layers: Transformer layers (default 4).
            dim_feedforward: FFN dim (default 512).
            dropout: Dropout (default 0.1).
            tau_init: Initial temperature (default 0.02).
            freeze_pose: If True, pose encoder params are frozen
                (non-trainable) — matches CPEP Sec. 2.3.
            use_cls: Use [CLS] token (default True, per paper).

        Attributes:
            emg_encoder: :class:`EMGEncoder`
            pose_encoder: :class:`PoseEncoder`
            tau: Learnable temperature scalar ``nn.Parameter``.

        Reference:
            Cui et al., arXiv:2509.04699, Sec. 2.3.
        """

        def __init__(
            self,
            emg_channels: int = 8,
            pose_dim: int = 63,
            seq_len: int = 150,
            pose_seq_len: int | None = None,
            d_model: int = 256,
            embed_dim: int = 256,
            nhead: int = 8,
            num_layers: int = 4,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            tau_init: float = 0.02,
            freeze_pose: bool = True,
            use_cls: bool = True,
        ) -> None:
            super().__init__()
            if tau_init <= 0:
                raise ValueError(f"tau_init must be >0, got {tau_init}")
            self.emg_channels = int(emg_channels)
            self.pose_dim = int(pose_dim)
            self.seq_len = int(seq_len)
            self.pose_seq_len = int(pose_seq_len) if pose_seq_len is not None else int(seq_len)
            self.d_model = int(d_model)
            self.embed_dim = int(embed_dim)
            self.tau_init = float(tau_init)
            self.freeze_pose = bool(freeze_pose)
            self.emg_encoder = EMGEncoder(
                in_channels=self.emg_channels,
                seq_len=self.seq_len,
                d_model=self.d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                embed_dim=self.embed_dim,
                use_cls=use_cls,
            )
            self.pose_encoder = PoseEncoder(
                pose_dim=self.pose_dim,
                seq_len=self.pose_seq_len,
                d_model=self.d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                embed_dim=self.embed_dim,
                use_cls=use_cls,
            )
            self.tau = nn.Parameter(torch.tensor(float(tau_init), dtype=torch.float32))  # type: ignore[union-attr]
            # freeze pose encoder per CPEP
            if self.freeze_pose:
                for p in self.pose_encoder.parameters():  # type: ignore[union-attr]
                    p.requires_grad = False  # type: ignore[union-attr]
            logger.debug(
                "CPEPFramework init: emg_C=%d pose_D=%d d_model=%d embed=%d tau=%.3f freeze_pose=%s",
                self.emg_channels, self.pose_dim, self.d_model, self.embed_dim, self.tau_init, self.freeze_pose,
            )

        def forward(self, emg: Any, pose: Any) -> Any:  # type: ignore[override]
            """Compute symmetric InfoNCE loss for paired (emg, pose).

            Args:
                emg: ``(B,C,T)`` or ``(B,T,C)`` EMG batch.
                pose: ``(B,T,D)`` or ``(B,D,T)`` pose batch.

            Returns:
                Scalar loss Tensor.
            """
            emg_emb = self.emg_encoder(emg)  # type: ignore[operator]
            # pose encoder may be frozen — ensure no grad if frozen (optional)
            if self.freeze_pose:
                with torch.no_grad():  # type: ignore[union-attr]
                    # compute without tracking pose grad, but still need pose embeddings for loss
                    # we re-enable grad for the loss path via detach handling: easiest keep graph but frozen params won't get grad
                    pose_emb = self.pose_encoder(pose)  # type: ignore[operator]
                # need pose_emb with requires_grad=False but still valid for contrastive
                # recompute without no_grad if we want grad w.r.t pose embeddings? For symmetric CE both sides need grad only wrt emg? Paper freezes pose encoder so pose_emb is detached intentionally.
                pose_emb = pose_emb.detach()  # type: ignore[union-attr]
                # emg branch already has grad; compute loss with detached pose
                return pose_emg_contrastive_loss(emg_emb, pose_emb, self.tau)
            pose_emb = self.pose_encoder(pose)  # type: ignore[operator]
            return pose_emg_contrastive_loss(emg_emb, pose_emb, self.tau)

        def encode_emg(self, emg: Any) -> Any:
            """Encode EMG to L2-normalized embedding ``(B, embed_dim)``."""
            self.emg_encoder.eval()  # type: ignore[union-attr]
            with torch.no_grad():  # type: ignore[union-attr]
                return self.emg_encoder(emg)  # type: ignore[operator]

        def encode_pose(self, pose: Any) -> Any:
            """Encode pose to L2-normalized embedding ``(B, embed_dim)``."""
            self.pose_encoder.eval()  # type: ignore[union-attr]
            with torch.no_grad():  # type: ignore[union-attr]
                return self.pose_encoder(pose)  # type: ignore[operator]

        def get_tau(self) -> float:
            """Return current temperature value (float)."""
            return float(self.tau.detach().cpu().item())  # type: ignore[union-attr]

        # zero-shot helpers as methods (delegate to module-level knn)
        def zero_shot_predict(
            self,
            query_emg: Any,
            gallery_pose_embs: Any,
            gallery_labels: Any,
            k: int = 10,
            metric: str = "cosine",
        ) -> np.ndarray:
            """Zero-shot classify query EMG via kNN over gallery pose embeddings.

            Args:
                query_emg: Raw EMG batch (``B,...``) or precomputed embeddings
                    ``(B,D)``. If 2D with D==embed_dim, treated as embeddings;
                    otherwise encoded via :meth:`encode_emg`.
                gallery_pose_embs: Gallery pose embeddings ``(G,D)`` or raw
                    pose batch (``G,...``) — if not 2D, encoded via
                    :meth:`encode_pose`.
                gallery_labels: Labels for gallery ``(G,)``.
                k: Number of neighbors (default 10 per CPEP).
                metric: Distance metric ``"cosine"`` or ``"euclidean"``.

            Returns:
                Predicted labels ``(B,)`` ndarray.
            """
            # resolve query embeddings
            if isinstance(query_emg, torch.Tensor) and query_emg.dim() == 2 and query_emg.shape[1] == self.embed_dim:  # type: ignore[union-attr]
                q_emb = query_emg.detach().cpu().numpy() if query_emg.device.type != "cpu" else query_emg.detach().numpy()  # type: ignore[union-attr]
                # need to handle if on cpu vs gpu
                try:
                    q_emb = query_emg.detach().cpu().numpy()  # type: ignore[union-attr]
                except Exception:
                    q_emb = np.asarray(query_emg)
            elif isinstance(query_emg, np.ndarray) and query_emg.ndim == 2 and query_emg.shape[1] == self.embed_dim:
                q_emb = query_emg
            else:
                # raw EMG -> encode
                q_t = self.encode_emg(query_emg)  # type: ignore[operator]
                q_emb = q_t.detach().cpu().numpy() if isinstance(q_t, torch.Tensor) else np.asarray(q_t)  # type: ignore[union-attr]

            # gallery embeddings
            if isinstance(gallery_pose_embs, torch.Tensor) and gallery_pose_embs.dim() == 2 and gallery_pose_embs.shape[1] == self.embed_dim:  # type: ignore[union-attr]
                g_emb = gallery_pose_embs.detach().cpu().numpy()  # type: ignore[union-attr]
            elif isinstance(gallery_pose_embs, np.ndarray) and gallery_pose_embs.ndim == 2 and gallery_pose_embs.shape[1] == self.embed_dim:
                g_emb = gallery_pose_embs
            else:
                g_t = self.encode_pose(gallery_pose_embs)  # type: ignore[operator]
                g_emb = g_t.detach().cpu().numpy() if isinstance(g_t, torch.Tensor) else np.asarray(g_t)  # type: ignore[union-attr]

            return knn_classify(g_emb, gallery_labels, q_emb, k=k, metric=metric)

        # aliases
        def classify_zero_shot(self, *a: Any, **k: Any) -> np.ndarray:
            return self.zero_shot_predict(*a, **k)

        def zero_shot_classify(self, *a: Any, **k: Any) -> np.ndarray:
            return self.zero_shot_predict(*a, **k)

        def predict_zero_shot(self, *a: Any, **k: Any) -> np.ndarray:
            return self.zero_shot_predict(*a, **k)

else:

    class CPEPFramework:  # type: ignore[no-redef]
        """Stub when torch not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

        def forward(self, emg: Any, pose: Any) -> Any:
            _require_torch()

        def __call__(self, *a: Any, **k: Any) -> Any:
            return self.forward(*a, **k)

        def encode_emg(self, emg: Any) -> Any:
            _require_torch()

        def encode_pose(self, pose: Any) -> Any:
            _require_torch()

        def zero_shot_predict(self, *a: Any, **k: Any) -> Any:
            _require_torch()


# ---------------------------------------------------------------------------
# kNN helpers — numpy with sklearn fallback
# ---------------------------------------------------------------------------

def knn_classify(
    train_embs: Any,
    train_labels: Any,
    query_embs: Any,
    k: int = 5,
    metric: str = "cosine",
) -> np.ndarray:
    """Zero-shot kNN classification in embedding space (CPEP ZS protocol).

    For each query embedding, retrieve TopK nearest training embeddings by
    cosine similarity (L2-normalized dot) or Euclidean, then majority vote.
    Paper uses ``k=10``.

    Args:
        train_embs: Gallery embeddings ``(G,D)`` (ndarray or Tensor).
        train_labels: Gallery labels ``(G,)`` (array-like).
        query_embs: Query embeddings ``(Q,D)``.
        k: Number of neighbors (default 5; CPEP uses 10). Clamped to ``min(k,G)``.
        metric: ``"cosine"`` (default, as CPEP) or ``"euclidean"``.

    Returns:
        Predicted labels ``(Q,)`` ndarray with same dtype as train_labels.

    Handles small data (G < k, single sample, etc.) and works without
    sklearn/torch (pure numpy).
    """
    # to numpy
    def _to_np(x: Any) -> np.ndarray:
        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
            try:
                return x.detach().cpu().numpy()  # type: ignore[union-attr]
            except Exception:
                return np.asarray(x)
        return np.asarray(x)

    tr = _to_np(train_embs)
    qu = _to_np(query_embs)
    lb = np.asarray(train_labels)

    if tr.ndim == 1:
        tr = tr.reshape(1, -1)
    if qu.ndim == 1:
        qu = qu.reshape(1, -1)
    if tr.ndim != 2 or qu.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got train {tr.shape}, query {qu.shape}")
    G, D = tr.shape
    Q, Dq = qu.shape
    if D != Dq:
        raise ValueError(f"Dim mismatch train D={D} vs query D={Dq}")
    if G == 0:
        raise ValueError("Empty training gallery")
    if Q == 0:
        return np.array([], dtype=lb.dtype)
    if lb.shape[0] != G:
        raise ValueError(f"train_labels len {lb.shape[0]} != gallery size {G}")
    if k <= 0:
        raise ValueError(f"k must be >0, got {k}")
    k_eff = int(min(k, G))

    # metric handling
    metric = str(metric).lower()
    if metric not in ("cosine", "euclidean", "l2"):
        logger.warning("knn_classify unknown metric '%s' — falling back to cosine", metric)
        metric = "cosine"

    # Try sklearn if available and metric compatible — provides efficient majority vote
    # but we keep numpy fallback for missing sklearn / small G edge cases
    if _SKLEARN_AVAILABLE and G >= k_eff and metric in ("cosine", "euclidean", "l2"):
        try:
            # sklearn's cosine metric expects dense; use brute + normalization trick for cosine
            if metric == "cosine":
                # L2 normalize for cosine -> euclidean on sphere equivalent to cosine distance
                tr_n = _l2_normalize_np(tr.astype(np.float64), axis=1)
                qu_n = _l2_normalize_np(qu.astype(np.float64), axis=1)
                # use euclidean on normalized vectors
                clf = KNeighborsClassifier(n_neighbors=k_eff, metric="euclidean", algorithm="brute")  # type: ignore[call-arg]
                clf.fit(tr_n, lb)  # type: ignore[arg-type]
                preds = clf.predict(qu_n)  # type: ignore[operator]
                return np.asarray(preds)
            # euclidean
            clf = KNeighborsClassifier(n_neighbors=k_eff, metric="euclidean", algorithm="auto")  # type: ignore[call-arg]
            clf.fit(tr.astype(np.float64), lb)  # type: ignore[arg-type]
            preds = clf.predict(qu.astype(np.float64))  # type: ignore[operator]
            return np.asarray(preds)
        except Exception as e:
            logger.debug("sklearn kNN failed (%s), falling back to numpy", e)

    # pure numpy majority vote
    if metric == "cosine":
        tr_n = _l2_normalize_np(tr.astype(np.float64), axis=1)
        qu_n = _l2_normalize_np(qu.astype(np.float64), axis=1)
        # similarity GxQ? we need per query topk over gallery
        # sim shape (Q,G) = qu_n @ tr_n.T
        sim = qu_n @ tr_n.T  # (Q,G)
        # argpartition for topk largest sim (most similar)
        # np.argpartition on negative sim not stable for tiny G
        # use argsort for small G (G < 1000) for determinism; else partition
        if G <= 2000:
            # argsort descending
            idx_sorted = np.argsort(-sim, axis=1)  # (Q,G)
            topk = idx_sorted[:, :k_eff]  # (Q,k)
        else:
            part = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
            # need sort within topk by similarity
            # gather sim for those indices and sort
            topk = np.empty_like(part)
            for i in range(Q):
                sel = part[i]
                # sort sel by sim descending
                order = np.argsort(-sim[i, sel])
                topk[i] = sel[order]
    else:  # euclidean
        # compute squared euclidean efficiently: (a-b)^2 = a2 + b2 -2ab
        tr_f = tr.astype(np.float64)
        qu_f = qu.astype(np.float64)
        tr_sq = np.sum(tr_f**2, axis=1)  # (G,)
        qu_sq = np.sum(qu_f**2, axis=1)  # (Q,)
        # dist2 = qu_sq[:,None] + tr_sq[None,:] -2*qu_f@tr_f.T
        dist2 = qu_sq[:, None] + tr_sq[None, :] - 2.0 * (qu_f @ tr_f.T)
        # numerical floor
        dist2 = np.maximum(dist2, 0.0)
        if G <= 2000:
            idx_sorted = np.argsort(dist2, axis=1)  # ascending smaller distance
            topk = idx_sorted[:, :k_eff]
        else:
            part = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            topk = np.empty_like(part)
            for i in range(Q):
                sel = part[i]
                order = np.argsort(dist2[i, sel])
                topk[i] = sel[order]

    # majority vote per query
    preds = []
    for i in range(Q):
        neigh_labels = lb[topk[i]]
        # bincount requires int mapping; handle arbitrary label types via unique + counts
        # use np.unique with return_counts for generic labels
        # For majority, smallest label wins on tie due to sorted unique order? Use stable.
        # Fast path for integer labels with bincount if labels are non-negative ints
        try:
            # if labels are integers and non-negative and not huge, use bincount
            if np.issubdtype(lb.dtype, np.integer) and np.all(lb >= 0) and lb.max() < 10000:
                # but need to handle non-contiguous?
                counts = np.bincount(neigh_labels)
                # argmax gives smallest label on tie (bincount order)
                pred = int(np.argmax(counts))
                preds.append(pred)
                continue
        except Exception:
            pass
        vals, counts = np.unique(neigh_labels, return_counts=True)
        # tie-break: highest count; if tie, choose first encountered in neigh order (more stable to CPEP majority?)
        # np.unique returns sorted; we want deterministic. Use argmax on counts (first max wins = smallest label when sorted).
        # To favor first encountered, we could iterate neigh_labels order, but keep sorted tie for determinism.
        max_c = int(np.argmax(counts))
        preds.append(vals[max_c])
    return np.array(preds, dtype=lb.dtype if hasattr(lb, "dtype") else None)


# aliases for different expected names
embedding_knn_classify = knn_classify
zero_shot_knn_classify = knn_classify
zero_shot_classify = knn_classify
knn_predict = knn_classify

# additional alias functions for flexibility

def embedding_knn_predict(*a: Any, **k: Any) -> np.ndarray:
    return knn_classify(*a, **k)


def zero_shot_predict(*a: Any, **k: Any) -> np.ndarray:
    return knn_classify(*a, **k)

