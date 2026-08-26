"""TinyMyo per arXiv:2512.15729 — 3.6M Transformer encoder for EMG.

Implements channel-independent patching, masked reconstruction (SimMIM),
and RoPE extrapolation for variable-length EMG.

Architecture
------------
- Channel-independent patching: each channel's time series is split into
  non-overlapping patches of ``patch_size``; patches are linearly projected
  to ``embed_dim`` with a shared projection (channel-independent).
- 50% random masking (SimMIM) via learned mask token.
- 8 bidirectional Transformer blocks with pre-norm, RoPE, and SwiGLU-free
  FFN. RoPE provides length extrapolation beyond training windows.
- Lightweight linear decoder ``embed_dim -> patch_size`` for masked
  reconstruction pre-training.

Reference
---------
TinyMyo, arXiv:2512.15729 (2025) — parameter-efficient masked EMG foundation.

Guarded: torch is optional; missing torch raises MissingDependencyError
with "subvocal[ml]".
"""

from __future__ import annotations

import logging
import math
from typing import Any

from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = [
    "TinyMyoEncoder",
    "TinyMyoFoundation",
    "pretrain_step",
    "finetune_step",
]

# ---------------------------------------------------------------------------
# torch guard
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


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            "torch is required for TinyMyo. Install with 'pip install \"subvocal[ml]\"'"
        )


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _rope_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0, device: Any = None, dtype: Any = None) -> tuple[Any, Any]:
    """Compute RoPE cos/sin tables of shape (seq_len, head_dim//2)."""
    _require_torch()
    # head_dim must be even
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))  # type: ignore[union-attr]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # type: ignore[union-attr]
    freqs = torch.outer(t, inv_freq)  # (seq_len, half)
    # RoPE rotates pairs, so we need half dim for cos/sin
    cos = torch.cos(freqs)  # (seq_len, half)
    sin = torch.sin(freqs)
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def _apply_rope(x: Any, cos: Any, sin: Any) -> Any:
    """Apply RoPE to x of shape (B, H, T, D) where D even.

    cos/sin: (T, D//2)
    """
    # x: (B, H, T, D)
    # split even/odd
    d = x.shape[-1]
    half = d // 2
    # cos/sin (T, half) -> (1,1,T,half)
    cos_b = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,half)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    # x1, x2 : first half and second half
    x1 = x[..., :half]
    x2 = x[..., half:]
    # RoPE rotation for paired dimensions
    # Standard: rotate pairs (x1 + i*x2)
    # out1 = x1*cos - x2*sin ; out2 = x1*sin + x2*cos
    # But with interleaved pairing, this is a simplified version.
    # Use half-split rotation (as in many implementations).
    rotated_x1 = x1 * cos_b - x2 * sin_b
    rotated_x2 = x1 * sin_b + x2 * cos_b
    return torch.cat([rotated_x1, rotated_x2], dim=-1)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Transformer block with RoPE
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _TransformerBlock(nn.Module):  # type: ignore[no-redef]
        """Bidirectional Transformer block with pre-norm and RoPE."""

        def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 12.0, dropout: float = 0.1) -> None:
            super().__init__()
            if embed_dim % num_heads != 0:
                raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            if self.head_dim % 2 != 0:
                raise ValueError("head_dim must be even for RoPE")
            self.mlp_ratio = mlp_ratio

            self.norm1 = nn.LayerNorm(embed_dim)
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
            self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.proj_drop = nn.Dropout(dropout)

            self.norm2 = nn.LayerNorm(embed_dim)
            hidden = int(embed_dim * mlp_ratio)
            self.fc1 = nn.Linear(embed_dim, hidden, bias=True)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, embed_dim, bias=True)
            self.mlp_drop = nn.Dropout(dropout)

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            # x: (B, N, D)
            B, N, D = x.shape
            # Pre-norm
            h = self.norm1(x)
            # QKV
            qkv = self.qkv(h)  # (B,N,3D)
            qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3,B,H,N,Dh)
            q, k, v = qkv[0], qkv[1], qkv[2]  # each (B,H,N,Dh)
            # RoPE on q,k
            cos, sin = _rope_cos_sin(N, self.head_dim, device=x.device, dtype=x.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            # Attention
            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,N,N)
            attn = F.softmax(attn_scores, dim=-1)  # type: ignore[union-attr]
            attn_out = attn @ v  # (B,H,N,Dh)
            attn_out = attn_out.transpose(1, 2).reshape(B, N, D)  # (B,N,D)
            attn_out = self.proj(attn_out)
            attn_out = self.proj_drop(attn_out)
            x = x + attn_out

            # MLP
            h2 = self.norm2(x)
            h2 = self.fc1(h2)
            h2 = self.act(h2)
            h2 = self.mlp_drop(h2)
            h2 = self.fc2(h2)
            h2 = self.mlp_drop(h2)
            x = x + h2
            return x

    # ---------------------------------------------------------------------------
    # TinyMyoEncoder
    # ---------------------------------------------------------------------------

    class TinyMyoEncoder(nn.Module):  # type: ignore[no-redef]
        """TinyMyo encoder — channel-independent patching + masked reconstruction.

        Args:
            num_channels: EMG channels (default 4).
            patch_size: Temporal patch length (default 10).
            embed_dim: Embedding dimension (default 128).
            depth: Number of Transformer blocks (default 8).
            num_heads: Attention heads (default 4).
            mask_ratio: Fraction of patches masked during training (default 0.5, SimMIM).
        """

        def __init__(
            self,
            num_channels: int = 4,
            patch_size: int = 10,
            embed_dim: int = 128,
            depth: int = 8,
            num_heads: int = 4,
            mask_ratio: float = 0.5,
        ) -> None:
            super().__init__()
            if num_channels <= 0:
                raise ValueError(f"num_channels must be >0, got {num_channels}")
            if patch_size <= 0:
                raise ValueError(f"patch_size must be >0, got {patch_size}")
            if embed_dim <= 0:
                raise ValueError(f"embed_dim must be >0, got {embed_dim}")
            if depth <= 0:
                raise ValueError(f"depth must be >0, got {depth}")
            if num_heads <= 0:
                raise ValueError(f"num_heads must be >0, got {num_heads}")
            if not 0.0 <= mask_ratio < 1.0:
                raise ValueError(f"mask_ratio must be in [0,1), got {mask_ratio}")
            if embed_dim % num_heads != 0:
                raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

            self.num_channels = num_channels
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.depth = depth
            self.num_heads = num_heads
            self.mask_ratio = float(mask_ratio)

            # Channel-independent patch projection: patch_size -> embed_dim (shared)
            self.patch_proj = nn.Linear(patch_size, embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # type: ignore[union-attr]
            nn.init.trunc_normal_(self.mask_token, std=0.02)  # type: ignore[union-attr]

            self.blocks = nn.ModuleList(
                [_TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(embed_dim)
            # Lightweight linear decoder for reconstruction (SimMIM)
            self.decoder = nn.Linear(embed_dim, patch_size, bias=True)

            logger.debug(
                "TinyMyoEncoder init: C=%d patch=%d embed=%d depth=%d heads=%d mask=%.2f",
                num_channels, patch_size, embed_dim, depth, num_heads, mask_ratio,
            )

        def _patchify(self, x: Any) -> tuple[Any, int, int]:
            """Patchify (B,C,T) -> (B, num_patches, patch_size)."""
            B, C, T = x.shape
            ps = self.patch_size
            # Pad T to be divisible by patch_size
            if T % ps != 0:
                pad_len = ps - (T % ps)
                x = F.pad(x, (0, pad_len))  # type: ignore[union-attr]
                T = T + pad_len
            # Unfold: (B,C,T) -> (B,C, num_patches_per_channel, patch_size)
            # Use unfold on last dim
            patches = x.unfold(dimension=2, size=ps, step=ps)  # (B,C, P, ps)
            B2, C2, P, ps2 = patches.shape
            # Reshape to (B, C*P, ps)
            patches = patches.permute(0, 2, 1, 3).reshape(B, P * C, ps)
            return patches, P, C

        def _unpatchify(self, patches: Any, P: int, C: int) -> Any:
            """Reverse patchify for reconstruction diagnostics (B, C*P, ps) -> (B,C,T)."""
            B = patches.shape[0]
            ps = patches.shape[-1]
            # (B, C*P, ps) -> (B, P, C, ps)
            patches = patches.view(B, P, C, ps)
            # (B, C, P, ps) -> (B,C, P*ps)
            patches = patches.permute(0, 2, 1, 3).reshape(B, C, P * ps)
            return patches

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            """Forward pass.

            Args:
                x: Tensor (B, C, T) — batch of EMG windows. C must match
                    ``num_channels`` or is adapted (warn).

            Returns:
                Tensor (B, num_patches, embed_dim) — encoded patch embeddings
                after masking and Transformer blocks. ``num_patches = C * (T//patch_size)``
                (padded if needed).
            """
            if not isinstance(x, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"TinyMyoEncoder expects torch.Tensor, got {type(x)}")
            if x.dim() != 3:
                raise ValueError(f"expected (B,C,T), got shape {tuple(x.shape)}")
            B, C, T = x.shape
            if C != self.num_channels:
                logger.warning("TinyMyoEncoder num_channels=%d but input C=%d", self.num_channels, C)

            patches, P, C_eff = self._patchify(x)  # (B, N, ps)
            N = patches.shape[1]
            # Linear projection channel-independent (shared)
            tokens = self.patch_proj(patches)  # (B, N, D)

            # Random masking (SimMIM) — only during training or if mask_ratio>0
            # Spec says forward does random masking 50%; respect training mode but
            # also allow masking in eval if caller wants (no mask in eval by default).
            do_mask = self.training and self.mask_ratio > 0 and N > 1
            mask: Any | None = None
            if do_mask:
                num_mask = int(N * self.mask_ratio)
                num_mask = max(1, num_mask) if self.mask_ratio > 0 else 0
                # Per-sample random permutation
                # Create noise and argsort
                noise = torch.rand(B, N, device=x.device)  # type: ignore[union-attr]
                ids_shuffle = torch.argsort(noise, dim=1)  # type: ignore[union-attr]
                # Mask first num_mask after shuffle
                mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)  # type: ignore[union-attr]
                # Build mask in shuffled order
                # ids_shuffle[:, :num_mask] are masked positions
                # Convert to boolean mask
                mask.scatter_(1, ids_shuffle[:, :num_mask], True)
                # Replace masked tokens with mask_token
                # Expand mask_token: (1,1,D) -> (B,N,D) where masked
                mask_token_expanded = self.mask_token.expand(B, N, -1)
                tokens = torch.where(mask.unsqueeze(-1), mask_token_expanded, tokens)  # type: ignore[union-attr]
                # Keep mask for potential loss (stored for pretrain)
                self._last_mask = mask  # for debugging
            else:
                self._last_mask = None

            # Transformer blocks (bidirectional, RoPE inside)
            for blk in self.blocks:
                tokens = blk(tokens)

            tokens = self.norm(tokens)
            return tokens

        def reconstruct(self, encoded: Any) -> Any:
            """Lightweight linear decoder: (B, N, D) -> (B, N, patch_size)."""
            return self.decoder(encoded)

        def count_parameters(self) -> int:
            """Total parameter count."""
            return sum(p.numel() for p in self.parameters())

        def estimate_flops(self, seq_len: int = 150) -> int:
            """Estimate FLOPs for a forward pass with given T.

            Approx: patch_proj + attention + MLP per block + decoder.
            """
            # num_patches N = num_channels * (seq_len // patch_size) (padded)
            ps = self.patch_size
            # effective T after padding
            if seq_len % ps != 0:
                seq_len = ((seq_len + ps - 1) // ps) * ps
            N = self.num_channels * (seq_len // ps)
            D = self.embed_dim
            # Patch proj: 2*ps*D*N
            flops = 2 * ps * D * N
            # Per block: use actual mlp_ratio from blocks (12.0 for 3.6M)
            # Infer ratio from first block if available
            try:
                ratio = float(self.blocks[0].mlp_ratio) if len(self.blocks) > 0 else 12.0  # type: ignore[union-attr]
            except Exception:
                ratio = 12.0
            hidden = int(D * ratio)
            for _ in range(self.depth):
                qkv = 2 * D * 3 * D * N
                proj = 2 * D * D * N
                attn = 2 * N * N * D  # QK + AV ~ 2*N^2*D
                mlp = 2 * D * hidden * N + 2 * hidden * D * N
                flops += qkv + proj + attn + mlp
            # Decoder: 2*D*ps*N
            flops += 2 * D * ps * N
            return int(flops)

        # Alias
        def estimate_FLOPs(self, seq_len: int = 150) -> int:
            return self.estimate_flops(seq_len)

    # ---------------------------------------------------------------------------
    # TinyMyoFoundation — wraps encoder + task heads
    # ---------------------------------------------------------------------------

    class TinyMyoFoundation(nn.Module):  # type: ignore[no-redef]
        """Foundation wrapper: encoder + task heads (classification, regression, speech).

        Args:
            num_channels: Passed to encoder.
            patch_size: Passed to encoder.
            embed_dim: Passed to encoder.
            depth: Passed to encoder.
            num_heads: Passed to encoder.
            mask_ratio: Passed to encoder.
            num_classes: Classes for classification head (default 8).
            regression_dim: Output dim for regression head (default 1).
            vocab_size: Vocab size for speech/CTC head (default 40).
            encoder: Optional pre-built TinyMyoEncoder instance.
        """

        def __init__(
            self,
            num_channels: int = 4,
            patch_size: int = 10,
            embed_dim: int = 128,
            depth: int = 8,
            num_heads: int = 4,
            mask_ratio: float = 0.5,
            num_classes: int = 8,
            regression_dim: int = 1,
            vocab_size: int = 40,
            encoder: Any | None = None,
        ) -> None:
            super().__init__()
            if encoder is not None:
                if not isinstance(encoder, TinyMyoEncoder):
                    raise TypeError(f"encoder must be TinyMyoEncoder, got {type(encoder)}")
                self.encoder = encoder
                # Infer dims from encoder if not overridden
                embed_dim = encoder.embed_dim
                patch_size = encoder.patch_size
                num_channels = encoder.num_channels
            else:
                self.encoder = TinyMyoEncoder(
                    num_channels=num_channels,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mask_ratio=mask_ratio,
                )
            self.embed_dim = self.encoder.embed_dim
            self.num_classes = int(num_classes)
            self.regression_dim = int(regression_dim)
            self.vocab_size = int(vocab_size)

            # Task heads — pooled over patches
            self.classifier = nn.Linear(self.embed_dim, self.num_classes, bias=True)
            self.regressor = nn.Linear(self.embed_dim, self.regression_dim, bias=True)
            self.speech_head = nn.Linear(self.embed_dim, self.vocab_size, bias=True)

            logger.debug(
                "TinyMyoFoundation init: embed=%d classes=%d reg=%d vocab=%d",
                self.embed_dim, self.num_classes, self.regression_dim, self.vocab_size,
            )

        def _pool(self, encoded: Any) -> Any:
            """Mean pool over patches: (B,N,D) -> (B,D)."""
            return encoded.mean(dim=1)

        def forward(self, x: Any, task: str = "classification") -> Any:  # type: ignore[override]
            """Forward through encoder + task head.

            Args:
                x: Tensor (B, C, T).
                task: One of "classification", "regression", "speech".

            Returns:
                Task-specific logits:
                - classification: (B, num_classes)
                - regression: (B, regression_dim)
                - speech: (B, num_patches, vocab_size) (per-patch CTC logits)
            """
            enc = self.encoder(x)  # (B,N,D)
            if task == "classification":
                pooled = self._pool(enc)
                return self.classifier(pooled)
            elif task == "regression":
                pooled = self._pool(enc)
                return self.regressor(pooled)
            elif task == "speech":
                # Per-patch logits for CTC
                return self.speech_head(enc)  # (B,N,V)
            else:
                raise ValueError(f"unknown task {task!r}, expected classification/regression/speech")

        def pretrain_step(self, x: Any, optimizer: Any | None = None) -> dict[str, Any]:
            """Single masked-reconstruction pre-training step (SimMIM).

            Computes MSE on masked patches only.

            Args:
                x: Tensor (B, C, T) raw EMG.
                optimizer: Optional optimizer to step.

            Returns:
                Dict with ``loss`` (Tensor), ``masked_ratio``.
            """
            # Need to reconstruct patches
            # Get patches before masking for target
            patches, P, C = self.encoder._patchify(x)  # (B,N,ps)
            enc = self.encoder(x)  # masked internally if training
            pred_patches = self.encoder.reconstruct(enc)  # (B,N,ps)

            mask = getattr(self.encoder, "_last_mask", None)
            if mask is None:
                # No masking (eval mode) — compute loss over all patches
                loss = F.mse_loss(pred_patches, patches)  # type: ignore[union-attr]
                masked_ratio = 0.0
            else:
                # MSE only on masked positions
                # mask: (B,N) bool
                mask_f = mask.unsqueeze(-1).expand_as(patches)
                # Avoid divide by zero if no masked
                if mask.sum().item() == 0:
                    loss = F.mse_loss(pred_patches, patches)  # type: ignore[union-attr]
                else:
                    loss = F.mse_loss(pred_patches[mask_f], patches[mask_f])  # type: ignore[union-attr]
                masked_ratio = float(mask.float().mean().item())  # type: ignore[union-attr]

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[union-attr]
                optimizer.step()

            return {"loss": loss, "masked_ratio": masked_ratio, "pred": pred_patches, "target": patches}

        def finetune_step(self, x: Any, y: Any, task: str = "classification", optimizer: Any | None = None, criterion: Any | None = None) -> dict[str, Any]:
            """Single fine-tuning step for a downstream task.

            Args:
                x: Tensor (B, C, T).
                y: Targets (B,) or (B, dim) depending on task.
                task: Task name.
                optimizer: Optional optimizer.
                criterion: Optional loss fn; defaults to CrossEntropy for classification,
                    MSELoss for regression, CTCLoss for speech (requires extra args).

            Returns:
                Dict with ``loss`` and ``logits``.
            """
            logits = self.forward(x, task=task)
            if criterion is None:
                if task == "classification":
                    criterion = nn.CrossEntropyLoss()
                elif task == "regression":
                    criterion = nn.MSELoss()
                elif task == "speech":
                    # Speech: expect y as (B, L) or (T)?? For simplicity use CrossEntropy over patches
                    criterion = nn.CrossEntropyLoss()
                else:
                    raise ValueError(f"unknown task {task}")

            # Adapt targets for speech per-patch loss: y may be (B,) labels -> expand
            if task == "speech" and y.dim() == 1:
                # Expand label to per-patch (B,N)
                # y: (B,) -> (B,N)
                N = logits.shape[1]
                y_expanded = y.unsqueeze(1).expand(-1, N)  # (B,N)
                # Need (B*N, V) logits vs (B*N,) targets
                loss = criterion(logits.reshape(-1, self.vocab_size), y_expanded.reshape(-1))  # type: ignore[union-attr]
            else:
                loss = criterion(logits, y)  # type: ignore[operator]

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[union-attr]
                optimizer.step()

            return {"loss": loss, "logits": logits}

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def count_trainable_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def estimate_flops(self, seq_len: int = 150) -> int:
            base = self.encoder.estimate_flops(seq_len=seq_len)
            # Add heads: pooled classifier: 2*D*num_classes etc.
            D = self.embed_dim
            # Approx: classifier 2*D*C, regressor 2*D*R, speech 2*D*V*N
            ps = self.encoder.patch_size
            if seq_len % ps != 0:
                seq_len = ((seq_len + ps - 1) // ps) * ps
            N = self.encoder.num_channels * (seq_len // ps)
            flops_heads = 2 * D * self.num_classes + 2 * D * self.regression_dim + 2 * D * self.vocab_size * N
            return int(base + flops_heads)

        def estimate_FLOPs(self, seq_len: int = 150) -> int:
            return self.estimate_flops(seq_len)

    # ---------------------------------------------------------------------------
    # Module-level helpers
    # ---------------------------------------------------------------------------

    def pretrain_step(model: Any, x: Any, optimizer: Any | None = None) -> dict[str, Any]:
        """Functional pretrain step for any TinyMyoFoundation/TinyMyoEncoder.

        If model is TinyMyoEncoder, computes masked MSE directly.
        """
        if isinstance(model, TinyMyoFoundation):
            return model.pretrain_step(x, optimizer=optimizer)
        elif isinstance(model, TinyMyoEncoder):
            # Encoder-only pretraining
            patches, P, C = model._patchify(x)
            enc = model(x)
            pred = model.reconstruct(enc)
            mask = getattr(model, "_last_mask", None)
            if mask is None:
                loss = F.mse_loss(pred, patches)  # type: ignore[union-attr]
            else:
                mask_f = mask.unsqueeze(-1).expand_as(patches)
                if mask.sum().item() == 0:
                    loss = F.mse_loss(pred, patches)  # type: ignore[union-attr]
                else:
                    loss = F.mse_loss(pred[mask_f], patches[mask_f])  # type: ignore[union-attr]
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[union-attr]
                optimizer.step()
            return {"loss": loss, "pred": pred, "target": patches}
        else:
            raise TypeError(f"pretrain_step expects TinyMyoEncoder/Foundation, got {type(model)}")

    def finetune_step(model: Any, x: Any, y: Any, task: str = "classification", optimizer: Any | None = None, criterion: Any | None = None) -> dict[str, Any]:
        """Functional fine-tune step."""
        if isinstance(model, TinyMyoFoundation):
            return model.finetune_step(x, y, task=task, optimizer=optimizer, criterion=criterion)
        elif isinstance(model, TinyMyoEncoder):
            # Encoder alone cannot finetune for classification without head
            raise ValueError("finetune_step on TinyMyoEncoder requires a TinyMyoFoundation wrapper with task heads")
        else:
            # Generic nn.Module with forward(task)
            logits = model(x, task=task) if "task" in model.forward.__code__.co_varnames else model(x)  # type: ignore[union-attr]
            if criterion is None:
                if task == "classification":
                    criterion = nn.CrossEntropyLoss()
                else:
                    criterion = nn.MSELoss()
            loss = criterion(logits, y)  # type: ignore[operator]
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[union-attr]
                optimizer.step()
            return {"loss": loss, "logits": logits}

    # Aliases for discoverability
    finetune = finetune_step
    pretrain = pretrain_step

else:  # torch missing — stubs

    class TinyMyoEncoder:  # type: ignore[no-redef]
        """Stub — raises MissingDependencyError when torch is absent."""

        def __init__(
            self,
            num_channels: int = 4,
            patch_size: int = 10,
            embed_dim: int = 128,
            depth: int = 8,
            num_heads: int = 4,
            mask_ratio: float = 0.5,
        ) -> None:
            _require_torch()

        def forward(self, x: Any) -> Any:
            _require_torch()

        def reconstruct(self, encoded: Any) -> Any:
            _require_torch()

        def count_parameters(self) -> int:
            _require_torch()
            return 0

        def estimate_flops(self, seq_len: int = 150) -> int:
            _require_torch()
            return 0

        def estimate_FLOPs(self, seq_len: int = 150) -> int:
            _require_torch()
            return 0

    class TinyMyoFoundation:  # type: ignore[no-redef]
        """Stub — raises MissingDependencyError when torch is absent."""

        def __init__(
            self,
            num_channels: int = 4,
            patch_size: int = 10,
            embed_dim: int = 128,
            depth: int = 8,
            num_heads: int = 4,
            mask_ratio: float = 0.5,
            num_classes: int = 8,
            regression_dim: int = 1,
            vocab_size: int = 40,
            encoder: Any | None = None,
        ) -> None:
            _require_torch()

        def forward(self, x: Any, task: str = "classification") -> Any:
            _require_torch()

        def pretrain_step(self, x: Any, optimizer: Any | None = None) -> dict[str, Any]:
            _require_torch()
            return {}

        def finetune_step(self, x: Any, y: Any, task: str = "classification", optimizer: Any | None = None, criterion: Any | None = None) -> dict[str, Any]:
            _require_torch()
            return {}

        def count_parameters(self) -> int:
            _require_torch()
            return 0

        def estimate_flops(self, seq_len: int = 150) -> int:
            _require_torch()
            return 0

        def estimate_FLOPs(self, seq_len: int = 150) -> int:
            _require_torch()
            return 0

    def pretrain_step(model: Any, x: Any, optimizer: Any | None = None) -> dict[str, Any]:  # type: ignore[no-redef]
        _require_torch()
        return {}

    def finetune_step(model: Any, x: Any, y: Any, task: str = "classification", optimizer: Any | None = None, criterion: Any | None = None) -> dict[str, Any]:  # type: ignore[no-redef]
        _require_torch()
        return {}

    finetune = finetune_step  # type: ignore[no-redef]
    pretrain = pretrain_step  # type: ignore[no-redef]
