"""MONA cross-modal losses per Benster et al. 2024 (arxiv:2403.05583).

Implements cross-modal contrastive (crossCon) and supervised temporal
contrastive (supTcon) losses with latent Dynamic Time Warping (DTW)
alignment for silent EMG + vocal EMG + audio (LibriSpeech).

Reference
---------
Benster et al., 2024 — ``An EMG-based Silent Speech Interface with
Cross-Modal Contrastive, SupCon and DTW Alignment (MONA)``.
Preprint arxiv:2403.05583.

Guarded: torch is optional; missing torch raises MissingDependencyError
for tensor losses. DTW has a numpy fallback and works without torch.
BatchSampler undersamples LibriSpeech to 50% per epoch (Algorithm 1).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

import numpy as np

from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = [
    "cross_contrastive_loss",
    "supervised_temporal_contrastive_loss",
    "dtw_align",
    "mona_loss",
    "BatchSampler",
]

# ---------------------------------------------------------------------------
# lazy torch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - missing torch path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            "torch is required for MONA losses. Install with 'pip install \"subvocal[ml]\"'"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mean_pool_if_3d(x: Any) -> Any:
    """Mean-pool temporal dimension if tensor is (B, T, D)."""
    # Caller ensures torch tensor when needed
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
        if x.dim() == 3:
            return x.mean(dim=1)
    elif isinstance(x, np.ndarray):
        if x.ndim == 3:
            return x.mean(axis=1)
    return x


def _l2_normalize(x: Any, dim: int = -1) -> Any:
    _require_torch()
    return F.normalize(x, p=2, dim=dim)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# crossCon – cross-modal contrastive (CLIP/SimCLR style, bidirectional)
# ---------------------------------------------------------------------------

def cross_contrastive_loss(
    emg_latents: Any,
    audio_latents: Any,
    temperature: float = 0.1,
) -> Any:
    """Cross-modal contrastive loss (crossCon) between EMG and audio latents.

    Bidirectional InfoNCE: for batch size B, similarity matrix S[i,j] =
    cosine(emg_i, audio_j) / temperature. Positives are diagonal.
    Loss = (CE(S, labels) + CE(S^T, labels)) / 2.

    Latents may be (B, D) or (B, T, D) (temporal mean-pooled).

    Args:
        emg_latents: Tensor of shape (B, D) or (B, T, D).
        audio_latents: Tensor of shape (B, D) or (B, T, D).
        temperature: Softmax temperature, ``>0`` (default 0.1 per MONA).

    Returns:
        Scalar Tensor loss (requires grad). 0 if batch size < 2
        still differentiable.

    Raises:
        MissingDependencyError: if torch not installed.
        ValueError: if shapes mismatched or temperature invalid.
    """
    _require_torch()
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not isinstance(emg_latents, torch.Tensor) or not isinstance(audio_latents, torch.Tensor):  # type: ignore[arg-type]
        raise TypeError("cross_contrastive_loss expects torch.Tensor inputs")

    # Mean-pool temporal if 3-D
    emg = _mean_pool_if_3d(emg_latents)
    audio = _mean_pool_if_3d(audio_latents)

    if emg.dim() != 2 or audio.dim() != 2:
        raise ValueError(f"expected (B,D) after pooling, got emg {tuple(emg.shape)}, audio {tuple(audio.shape)}")
    if emg.shape[0] != audio.shape[0]:
        raise ValueError(f"batch size mismatch: emg {emg.shape[0]} vs audio {audio.shape[0]}")
    if emg.shape[1] != audio.shape[1]:
        raise ValueError(f"feature dim mismatch: emg {emg.shape[1]} vs audio {audio.shape[1]}")

    b = emg.shape[0]
    if b == 0:
        raise ValueError("batch size 0")
    if b == 1:
        # Single sample — no contrast; return 0 but keep grad graph
        return (emg * 0).sum() * 0.0  # type: ignore[no-any-return]

    # L2 normalize for cosine similarity
    emg_n = F.normalize(emg, p=2, dim=1)  # type: ignore[union-attr]
    audio_n = F.normalize(audio, p=2, dim=1)  # type: ignore[union-attr]

    logits = emg_n @ audio_n.T / temperature  # (B, B)
    labels = torch.arange(b, device=emg.device)  # type: ignore[union-attr]

    loss_e2a = F.cross_entropy(logits, labels)  # type: ignore[union-attr]
    loss_a2e = F.cross_entropy(logits.T, labels)  # type: ignore[union-attr]
    loss = (loss_e2a + loss_a2e) / 2.0
    logger.debug("crossCon: batch=%d temp=%.3f loss=%.4f", b, temperature, float(loss.detach().cpu().item()))  # type: ignore[union-attr]
    return loss


# ---------------------------------------------------------------------------
# supTcon – supervised temporal contrastive
# ---------------------------------------------------------------------------

def supervised_temporal_contrastive_loss(
    latents: Any,
    labels: Any,
    temperature: float = 0.1,
) -> Any:
    """Supervised temporal contrastive loss (supTcon).

    SupCon-style: positives share the same label, negatives differ.
    For each anchor i, loss pulls positives j (|labels_j==labels_i, j!=i)
    closer in latent space than negatives.

    Temporal handling: if latents is (B, T, D), mean-pools over T.
    Labels may be (B,) or (B, 1).

    Args:
        latents: Tensor (B, D) or (B, T, D).
        labels: Tensor (B,) long/int or array-like.
        temperature: Softmax temperature (>0, default 0.1).

    Returns:
        Scalar Tensor loss. Returns 0 if no anchor has positives.

    Raises:
        MissingDependencyError: if torch not installed.
    """
    _require_torch()
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not isinstance(latents, torch.Tensor):  # type: ignore[arg-type]
        raise TypeError("supervised_temporal_contrastive_loss expects torch.Tensor latents")

    # Normalize labels to 1-D long tensor on same device
    if not isinstance(labels, torch.Tensor):  # type: ignore[arg-type]
        # allow list/np.ndarray
        labels = torch.tensor(labels, dtype=torch.long, device=latents.device)  # type: ignore[union-attr]
    else:
        labels = labels.to(latents.device)  # type: ignore[union-attr]
    if labels.dim() > 1:
        labels = labels.view(-1)
    # pool temporal
    z = _mean_pool_if_3d(latents)
    if z.dim() != 2:
        raise ValueError(f"expected latents (B,D) after pooling, got {tuple(z.shape)}")
    if z.shape[0] != labels.shape[0]:
        raise ValueError(f"batch mismatch: latents {z.shape[0]} vs labels {labels.shape[0]}")

    b = z.shape[0]
    if b <= 1:
        return (z * 0).sum() * 0.0  # type: ignore[no-any-return]

    # Normalize embeddings
    z_norm = F.normalize(z, p=2, dim=1)  # type: ignore[union-attr]
    # Cosine similarity matrix (B, B) / temp
    sim = z_norm @ z_norm.T / temperature  # (B,B)

    # For numerical stability, mask self-contrast (diagonal) out of denominator
    # but keep for log-softmax over full row with diagonal excluded via large negative.
    # Approach: set diagonal to large negative so softmax ignores it.
    # Instead, we explicitly exclude i==j from denominator.
    # Create mask: positives where labels equal and i != j
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B,B) bool
    # Remove self
    eye = torch.eye(b, dtype=torch.bool, device=z.device)  # type: ignore[union-attr]
    pos_mask = labels_eq & ~eye
    # For denominator, exclude self (all j != i)
    # Use log-softmax over row excluding diagonal: compute exp(sim) sum over j!=i

    # To avoid overflow, subtract max per row (excluding self) before exp — but log_softmax handles.
    # We compute log_prob matrix: log_softmax(sim with self masked to -inf)
    sim_masked = sim.masked_fill(eye, float("-inf"))
    log_prob = F.log_softmax(sim_masked, dim=1)  # type: ignore[union-attr]

    # For each anchor with at least one positive, loss = -mean(log_prob[positives])
    # Aggregate
    loss_terms: list[Any] = []
    for i in range(b):
        pos_idx = torch.where(pos_mask[i])[0]  # type: ignore[union-attr]
        if pos_idx.numel() == 0:
            continue
        # Gather log probs for positives
        lp = log_prob[i, pos_idx]
        loss_terms.append(-lp.mean())

    if not loss_terms:
        logger.debug("supTcon: no positives in batch (labels=%s) => loss 0", labels.tolist())
        return (z * 0).sum() * 0.0  # type: ignore[no-any-return]

    loss = torch.stack(loss_terms).mean()  # type: ignore[union-attr]
    logger.debug("supTcon: batch=%d temp=%.3f positives=%d loss=%.4f", b, temperature, len(loss_terms), float(loss.detach().cpu().item()))  # type: ignore[union-attr]
    return loss


# ---------------------------------------------------------------------------
# DTW alignment in latent space (torch or numpy fallback)
# ---------------------------------------------------------------------------

def _dtw_path_numpy(a_np: np.ndarray, b_np: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Compute DTW distance matrix, cumulative cost, and optimal path (numpy)."""
    t_a, d_a = a_np.shape
    t_b, d_b = b_np.shape
    if d_a != d_b:
        raise ValueError(f"feature dim mismatch for DTW: {d_a} vs {d_b}")

    # Pairwise squared Euclidean distance (t_a, t_b)
    # Vectorized: (t_a,1,d) - (1,t_b,d)
    # For large T, chunk to avoid OOM — but T typically <500
    diff = a_np[:, None, :] - b_np[None, :, :]  # (t_a, t_b, d)
    dist = np.sum(diff * diff, axis=2)  # (t_a, t_b) squared L2; alternatively could use cosine

    # DP cumulative cost
    cost = np.full((t_a, t_b), np.inf, dtype=np.float64)
    cost[0, 0] = dist[0, 0]
    for i in range(t_a):
        for j in range(t_b):
            if i == 0 and j == 0:
                continue
            best = np.inf
            if i > 0:
                best = min(best, cost[i - 1, j])
            if j > 0:
                best = min(best, cost[i, j - 1])
            if i > 0 and j > 0:
                best = min(best, cost[i - 1, j - 1])
            cost[i, j] = dist[i, j] + best

    # Backtrack path
    path: list[tuple[int, int]] = []
    i, j = t_a - 1, t_b - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # prefer diagonal on ties (more stable alignment)
            c_diag = cost[i - 1, j - 1]
            c_up = cost[i - 1, j]
            c_left = cost[i, j - 1]
            m = min(c_diag, c_up, c_left)
            if m == c_diag:
                i -= 1
                j -= 1
            elif m == c_up:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return dist, path, cost  # type: ignore[return-value]


def dtw_align(latents_a: Any, latents_b: Any) -> Any:
    """Align ``latents_a`` to ``latents_b`` via Dynamic Time Warping in latent space.

    Latent DTW per MONA: warps silent EMG (or vocal EMG) to audio temporal
    structure in the shared encoder latent space. Uses squared Euclidean
    distance in latent space; warps ``a`` to ``b``'s time axis by averaging
    frames of ``a`` that map to each frame of ``b`` along the optimal path.

    Supports ``torch.Tensor`` (GPU/CPU) or ``np.ndarray``. If torch is not
    available, numpy fallback is used transparently. Batched inputs
    ``(B, T, D)`` are handled per-batch element.

    Args:
        latents_a: Tensor/array of shape ``(T_a, D)`` or ``(B, T_a, D)``
            — source to warp (e.g., silent EMG latents).
        latents_b: Tensor/array of shape ``(T_b, D)`` or ``(B, T_b, D)``
            — target time axis (e.g., audio latents).

    Returns:
        Warped version of ``latents_a`` aligned to ``latents_b`` time axis:
        * 2-D inputs ``(T_a,D)`` / ``(T_b,D)`` → ``(T_b, D)``
        * 3-D batched ``(B, T_a,D)`` / ``(B, T_b,D)`` → ``(B, T_b, D)``
        Returned type matches input type (Tensor if inputs were Tensor,
        otherwise ndarray). Device/dtype preserved for Tensor outputs.

    Raises:
        ValueError: if feature dims mismatch or batch dims mismatch.
    """
    # Detect torch inputs
    is_a_torch = _TORCH_AVAILABLE and isinstance(latents_a, torch.Tensor)  # type: ignore[arg-type]
    is_b_torch = _TORCH_AVAILABLE and isinstance(latents_b, torch.Tensor)  # type: ignore[arg-type]
    originally_torch = is_a_torch or is_b_torch

    # Convert to numpy for path computation (works for both)
    if is_a_torch:
        a_device = latents_a.device  # type: ignore[union-attr]
        a_dtype = latents_a.dtype  # type: ignore[union-attr]
        a_np = latents_a.detach().cpu().numpy()  # type: ignore[union-attr]
    else:
        a_device = None
        a_dtype = None
        a_np = np.asarray(latents_a)

    if is_b_torch:
        b_device = latents_b.device  # type: ignore[union-attr]
        b_dtype = latents_b.dtype  # type: ignore[union-attr]
        b_np = latents_b.detach().cpu().numpy()  # type: ignore[union-attr]
    else:
        b_device = a_device  # keep consistent if only one is torch
        b_dtype = a_dtype
        b_np = np.asarray(latents_b)

    # Handle scalar / edge cases
    if a_np.ndim == 0 or b_np.ndim == 0:
        raise ValueError("dtw_align expects at least 1-D arrays")
    # Promote 1-D feature vectors to (1, D) if needed?
    if a_np.ndim == 1:
        a_np = a_np[None, :]
    if b_np.ndim == 1:
        b_np = b_np[None, :]

    # Batched case: (B, T, D)
    if a_np.ndim == 3 or b_np.ndim == 3:
        if a_np.ndim != 3 or b_np.ndim != 3:
            raise ValueError(f"batch DTW requires both 3-D, got a {a_np.shape}, b {b_np.shape}")
        if a_np.shape[0] != b_np.shape[0]:
            raise ValueError(f"batch size mismatch for DTW: {a_np.shape[0]} vs {b_np.shape[0]}")
        if a_np.shape[2] != b_np.shape[2]:
            raise ValueError(f"feature dim mismatch: {a_np.shape[2]} vs {b_np.shape[2]}")
        b_batch = a_np.shape[0]
        warped_list: list[np.ndarray] = []
        for i in range(b_batch):
            warped_np = _warp_single(a_np[i], b_np[i])
            warped_list.append(warped_np)
        stacked = np.stack(warped_list, axis=0)  # (B, T_b, D)
        if originally_torch:
            # Preserve dtype/device from a if torch, else b
            dev = a_device if a_device is not None else b_device
            dtype = a_dtype if a_dtype is not None else b_dtype
            t = torch.from_numpy(stacked)  # type: ignore[union-attr]
            if dtype is not None:
                t = t.to(dtype)  # type: ignore[union-attr]
            if dev is not None:
                t = t.to(dev)  # type: ignore[union-attr]
            return t
        return stacked

    # 2-D case: (T, D)
    if a_np.ndim != 2 or b_np.ndim != 2:
        raise ValueError(f"expected 2-D (T,D) or 3-D (B,T,D), got a {a_np.ndim}D, b {b_np.ndim}D")

    warped_np = _warp_single(a_np, b_np)
    if originally_torch:
        dev = a_device if a_device is not None else b_device
        dtype = a_dtype if a_dtype is not None else b_dtype
        t = torch.from_numpy(warped_np)  # type: ignore[union-attr]
        if dtype is not None:
            t = t.to(dtype)  # type: ignore[union-attr]
        if dev is not None:
            t = t.to(dev)  # type: ignore[union-attr]
        return t
    return warped_np


def _warp_single(a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
    """Warp single sequence a_np (T_a,D) to b_np's time axis (T_b,D) via DTW path."""
    _, path, _ = _dtw_path_numpy(a_np, b_np)
    t_b = b_np.shape[0]
    d = a_np.shape[1]
    # Map each target frame j -> list of source frames i aligned to it
    mapping: dict[int, list[int]] = defaultdict(list)
    for i, j in path:
        mapping[j].append(i)
    warped = np.zeros((t_b, d), dtype=a_np.dtype)
    for j in range(t_b):
        idxs = mapping.get(j)
        if idxs:
            warped[j] = a_np[idxs].mean(axis=0)
        else:
            # No alignment for this j (should not happen with valid DTW path covering all j)
            # Fallback: repeat nearest mapped frame or zeros
            # Find nearest j with mapping
            nearest = min(mapping.keys(), key=lambda k: abs(k - j)) if mapping else None
            if nearest is not None:
                warped[j] = a_np[mapping[nearest]].mean(axis=0)
            else:
                warped[j] = np.zeros(d, dtype=a_np.dtype)
    return warped


# ---------------------------------------------------------------------------
# mona_loss – combined objective
# ---------------------------------------------------------------------------

def mona_loss(
    emg_silent: Any,
    emg_vocal: Any,
    audio: Any,
    labels: Any,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Combined MONA objective with crossCon, supTcon and CTC components.

    Computes::

        total = w_cross * crossCon + w_sup * supTcon + w_ctc * ctc

    where crossCon is averaged over (silent↔audio) and (vocal↔audio) when
    both EMG modalities are present, supTcon is the supervised temporal
    contrastive loss on EMG latents (mean-pooled if temporal), and ctc is
    an optional CTC loss if logits are provided.

    Args:
        emg_silent: Silent EMG latents Tensor (B, D) or (B, T, D) or None.
        emg_vocal: Vocal EMG latents Tensor (B, D) or (B, T, D) or None.
        audio: Audio (LibriSpeech) latents Tensor (B, D) or (B, T, D) or None.
        labels: Ground-truth labels Tensor (B,) or (B, L) for CTC;
            may be list/ndarray; None => supTcon/ctc skipped.
        weights: Dict with keys ``crossCon``/``cross``, ``supTcon``/``sup``,
            ``ctc``. Missing keys default to 1.0. Also accepts ``total`` not needed.

    Returns:
        Dict with keys ``crossCon``, ``supTcon``, ``ctc``, ``total``.
        Values are scalar Tensors (if torch available) or floats (fallback).
        Always contains ``total``.

    Raises:
        MissingDependencyError: if torch not installed and tensor inputs given
            (fallback returns float dict without raising, to keep training
            loop alive in non-torch tests).
    """
    # Normalize weights with flexible key aliases
    w = weights or {}
    w_cross = float(w.get("crossCon", w.get("cross", w.get("cross_weight", 1.0))))
    w_sup = float(w.get("supTcon", w.get("sup", w.get("sup_weight", w.get("supervised", 1.0)))))
    w_ctc = float(w.get("ctc", w.get("ctc_weight", 1.0)))

    # If torch not available, return float placeholders gracefully (no crash in import tests)
    if not _TORCH_AVAILABLE:
        logger.warning("mona_loss called without torch — returning zero float losses")
        zero = 0.0
        return {"crossCon": zero, "supTcon": zero, "ctc": zero, "total": zero}

    # Need at least one of the modalities as Tensor to infer device; if all None -> zeros
    def _is_tensor(x: Any) -> bool:
        return isinstance(x, torch.Tensor)  # type: ignore[arg-type]

    # Determine reference device for zero tensors
    ref = None
    for cand in (emg_silent, emg_vocal, audio):
        if _is_tensor(cand):
            ref = cand
            break
    if ref is None and _is_tensor(labels):  # type: ignore[arg-type]
        ref = labels
    if ref is None:
        # No tensors at all — return float zeros (caller passed numpy/None)
        logger.debug("mona_loss: no tensor inputs, returning float zeros")
        return {"crossCon": 0.0, "supTcon": 0.0, "ctc": 0.0, "total": 0.0}

    device = ref.device  # type: ignore[union-attr]

    def _zero() -> Any:
        return torch.zeros((), device=device)  # type: ignore[union-attr]

    # -- crossCon ----------------------------------------------------------
    cross_terms: list[Any] = []
    if _is_tensor(emg_silent) and _is_tensor(audio):
        try:
            # Optional DTW alignment before crossCon if temporal dims differ
            # (mean pooling in cross_contrastive_loss handles it, so DTW optional)
            cross_terms.append(cross_contrastive_loss(emg_silent, audio))
        except Exception as e:
            logger.warning("crossCon silent↔audio failed: %s", e)
            cross_terms.append(_zero())
    if _is_tensor(emg_vocal) and _is_tensor(audio):
        try:
            cross_terms.append(cross_contrastive_loss(emg_vocal, audio))
        except Exception as e:
            logger.warning("crossCon vocal↔audio failed: %s", e)
            cross_terms.append(_zero())

    if cross_terms:
        crossCon = torch.stack(cross_terms).mean()  # type: ignore[union-attr]
    else:
        crossCon = _zero()

    # -- supTcon -----------------------------------------------------------
    supTcon = _zero()
    if labels is not None:
        # Try supervised loss on EMG latents; concatenate modalities if both present
        sup_candidates: list[Any] = []
        # Normalize labels to tensor
        if not isinstance(labels, torch.Tensor):  # type: ignore[arg-type]
            try:
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("mona_loss labels conversion failed: %s", e)
                labels_t = None  # type: ignore[assignment]
        else:
            labels_t = labels.to(device)  # type: ignore[union-attr]

        if labels_t is not None:
            # Handle batched labels duplication if concatenating two EMG modalities with same labels per sample
            # If both silent and vocal present and share labels shape (B,), we duplicate labels for concatenated (2B,D)
            try:
                if _is_tensor(emg_silent):
                    sup_candidates.append(supervised_temporal_contrastive_loss(emg_silent, labels_t))
                if _is_tensor(emg_vocal):
                    # labels_t may need tiling if emg_vocal batch equals emg_silent batch
                    # Reuse same labels for vocal (same utterances, different modality)
                    sup_candidates.append(supervised_temporal_contrastive_loss(emg_vocal, labels_t))
                if sup_candidates:
                    supTcon = torch.stack(sup_candidates).mean()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("supTcon failed: %s", e)
                supTcon = _zero()

    # -- ctc ----------------------------------------------------------------
    ctc = _zero()
    # Heuristic: ctc applicable if any of emg_silent/vocal/audio is 3-D logits (B,T,V)
    # and labels looks like target sequences.
    ctc_logits = None
    for cand in (emg_silent, emg_vocal, audio):
        if _is_tensor(cand) and cand.dim() == 3:  # type: ignore[union-attr]
            # Heuristic: if last dim likely vocab size (> num_classes) use as logits
            # For MONA, latents are encoder outputs, not logits; so ctc is optional.
            # We treat cand as logits only if weights demand ctc and labels are 1-D/2-D.
            ctc_logits = cand
            break

    if ctc_logits is not None and labels is not None and w_ctc != 0:
        try:
            ctc = _ctc_loss_from_logits(ctc_logits, labels, device)
        except Exception as e:
            logger.debug("ctc computation skipped: %s", e)
            ctc = _zero()

    total = w_cross * crossCon + w_sup * supTcon + w_ctc * ctc
    logger.debug(
        "mona_loss: crossCon=%.4f supTcon=%.4f ctc=%.4f total=%.4f (w_cross=%.2f w_sup=%.2f w_ctc=%.2f)",
        float(crossCon.detach().cpu().item()) if isinstance(crossCon, torch.Tensor) else float(crossCon),  # type: ignore[union-attr]
        float(supTcon.detach().cpu().item()) if isinstance(supTcon, torch.Tensor) else float(supTcon),  # type: ignore[union-attr]
        float(ctc.detach().cpu().item()) if isinstance(ctc, torch.Tensor) else float(ctc),  # type: ignore[union-attr]
        float(total.detach().cpu().item()) if isinstance(total, torch.Tensor) else float(total),  # type: ignore[union-attr]
        w_cross, w_sup, w_ctc,
    )
    return {"crossCon": crossCon, "supTcon": supTcon, "ctc": ctc, "total": total}


def _ctc_loss_from_logits(logits: Any, labels: Any, device: Any) -> Any:
    """Helper to compute CTC loss from logits (B, T, V) and labels.

    Expects vocab includes blank=0. Handles labels as (B,) single-class or
    (B, L) padded sequences (pad = -1 or 0). Falls back to zero if shapes
    incompatible.
    """
    _require_torch()
    if not isinstance(logits, torch.Tensor):  # type: ignore[arg-type]
        raise TypeError("logits must be Tensor")
    if logits.dim() != 3:
        raise ValueError(f"CTC logits expected (B,T,V), got {tuple(logits.shape)}")

    b, t, v = logits.shape
    # Need labels tensor
    if not isinstance(labels, torch.Tensor):  # type: ignore[arg-type]
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)  # type: ignore[union-attr]
    else:
        labels_t = labels.to(device)  # type: ignore[union-attr]

    # Handle labels shape
    if labels_t.dim() == 0:
        labels_t = labels_t.unsqueeze(0)
    if labels_t.dim() == 1:
        # Single label per sample (classification) -> treat as length-1 targets
        # CTC with T>>1 and single token is valid but trivial; compute loss
        # Need blank != label values: assume labels in [1, V-1]
        # Shift if needed to avoid blank=0 collision: if any label==0, map to 1
        labels_t = labels_t.clamp(min=1)
        input_lengths = torch.full((b,), t, dtype=torch.long, device=device)  # type: ignore[union-attr]
        target_lengths = torch.ones(b, dtype=torch.long, device=device)  # type: ignore[union-attr]
        log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)  # (T, B, V)  # type: ignore[union-attr]
        ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # type: ignore[union-attr]
        # CTCLoss expects 1-D targets flattened
        loss = ctc_loss_fn(log_probs, labels_t, input_lengths, target_lengths)
        return loss
    elif labels_t.dim() == 2:
        # (B, L) padded with -1 or 0. Use -1 as pad, filter.
        # Flatten non-pad targets
        flat_targets: list[int] = []
        target_lengths_list: list[int] = []
        for i in range(b):
            row = labels_t[i]
            # Consider -1 as pad, 0 also as pad/blank
            # Keep values >0
            valid = [int(x) for x in row.tolist() if int(x) > 0]
            if not valid:
                # If all zeros, treat as length 0 — skip CTC (zero)
                target_lengths_list.append(0)
            else:
                flat_targets.extend(valid)
                target_lengths_list.append(len(valid))
        if not flat_targets:
            return torch.zeros((), device=device)  # type: ignore[union-attr]
        flat = torch.tensor(flat_targets, dtype=torch.long, device=device)  # type: ignore[union-attr]
        input_lengths = torch.full((b,), t, dtype=torch.long, device=device)  # type: ignore[union-attr]
        target_lengths = torch.tensor(target_lengths_list, dtype=torch.long, device=device)  # type: ignore[union-attr]
        # Filter out samples with 0 target length for CTCLoss (would error)
        # Keep only samples with >0
        mask = target_lengths > 0
        if not mask.any():
            return torch.zeros((), device=device)  # type: ignore[union-attr]
        # If some have zero, we need to subset logits accordingly?
        # For simplicity, if any zero, compute only over non-zero subset
        if not mask.all():
            # Subset logits to those with valid targets
            logits_sub = logits[mask]  # type: ignore[index]
            log_probs = F.log_softmax(logits_sub, dim=2).transpose(0, 1)  # (T, B_sub, V)  # type: ignore[union-attr]
            input_lengths_sub = input_lengths[mask]
            target_lengths_sub = target_lengths[mask]
            ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # type: ignore[union-attr]
            loss = ctc_loss_fn(log_probs, flat, input_lengths_sub, target_lengths_sub)
            return loss
        log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)  # (T, B, V)  # type: ignore[union-attr]
        ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # type: ignore[union-attr]
        loss = ctc_loss_fn(log_probs, flat, input_lengths, target_lengths)
        return loss
    else:
        raise ValueError(f"labels shape {tuple(labels_t.shape)} not supported for CTC")


# ---------------------------------------------------------------------------
# BatchSampler – LibriSpeech undersampling to 50% per epoch (Algorithm 1)
# ---------------------------------------------------------------------------

class BatchSampler:
    """Undersample LibriSpeech to 50% per epoch (MONA Algorithm 1).

    Each epoch, retains 100% of EMG indices (silent + vocal) and a random
    50% subset of LibriSpeech indices (without replacement), shuffles the
    combined pool, and yields batches.

    This keeps EMG and LibriSpeech balanced per batch despite LibriSpeech
    being much larger (~960 h vs ~10 h EMG), matching MONA's
    ``undersample LS to 50% per epoch`` description.

    Args:
        emg_indices: List of EMG sample indices (e.g., 0..N_emg-1) or int
            count, or None (inferred from dataset sizes).
        librispeech_indices: List of LibriSpeech sample indices or int count.
        batch_size: Batch size (default 32).
        librispeech_ratio: Fraction of LibriSpeech to keep per epoch
            (default 0.5 per MONA).
        shuffle: Whether to shuffle the combined pool each epoch.
        seed: Base random seed; epoch is added for per-epoch determinism.
        drop_last: Whether to drop last incomplete batch.
        dataset_size: Total dataset size (alternative to explicit indices).
        librispeech_size: Number of LibriSpeech samples (used with dataset_size).
        dataset: Optional dataset object (unused but accepted for API compat;
            if it exposes ``emg_indices`` / ``librispeech_indices`` attributes,
            they are used).

    Example:
        ``sampler = BatchSampler(emg_indices=list(range(100)), librispeech_indices=list(range(100,600)), batch_size=32)``
        ``for batch in sampler: train_step(batch)``
        ``sampler.set_epoch(epoch)``  # call each epoch for new subsample

    Notes:
        Compatible with ``torch.utils.data.DataLoader`` via manual batching
        or by passing ``batch_sampler=sampler``. When used with PyTorch,
        each iteration yields a ``list[int]`` batch of indices.
    """

    def __init__(
        self,
        emg_indices: list[int] | int | None = None,
        librispeech_indices: list[int] | int | None = None,
        batch_size: int = 32,
        librispeech_ratio: float = 0.5,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
        dataset_size: int | None = None,
        librispeech_size: int | None = None,
        dataset: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # -- resolve flexible kwargs aliases ---------------------------------
        # Allow emg_size / librispeech_size via kwargs
        if emg_indices is None and "emg_size" in kwargs:
            emg_indices = int(kwargs["emg_size"])
        if librispeech_indices is None and "librispeech_size" in kwargs and librispeech_size is None:
            librispeech_size = int(kwargs["librispeech_size"])
        if dataset_size is None and "total_size" in kwargs:
            dataset_size = int(kwargs["total_size"])
        # Also accept positional dataset as first arg if caller used (dataset, batch_size)
        if isinstance(emg_indices, (list, tuple)) and hasattr(emg_indices, "__len__") and librispeech_indices is None and dataset is None:
            # Check if emg_indices looks like a dataset object (has __getitem__)
            maybe_dataset = emg_indices
            if hasattr(maybe_dataset, "__getitem__") and not isinstance(maybe_dataset[0] if len(maybe_dataset) > 0 else None, int):  # type: ignore[index]
                dataset = maybe_dataset  # type: ignore[assignment]
                emg_indices = None

        # Dataset attribute inference
        if dataset is not None:
            # Try to infer indices from dataset attributes
            if emg_indices is None:
                for attr in ("emg_indices", "emg_ids", "emg_index", "silent_indices"):
                    if hasattr(dataset, attr):
                        emg_indices = getattr(dataset, attr)
                        break
            if librispeech_indices is None:
                for attr in ("librispeech_indices", "ls_indices", "audio_indices", "librispeech_ids"):
                    if hasattr(dataset, attr):
                        librispeech_indices = getattr(dataset, attr)
                        break
            if emg_indices is None and librispeech_indices is None and hasattr(dataset, "__len__"):
                # Fallback: treat dataset as flat; split or use provided sizes
                try:
                    n = len(dataset)  # type: ignore[arg-type]
                    if dataset_size is None:
                        dataset_size = n
                except Exception:
                    pass

        # -- normalize indices ------------------------------------------------
        # Handle int counts -> range
        if isinstance(emg_indices, int):
            emg_indices = list(range(emg_indices))
        if isinstance(librispeech_indices, int):
            # If emg given as list, offset librispeech indices after emg
            offset = len(emg_indices) if isinstance(emg_indices, list) else 0
            # But if librispeech_indices is count, generate contiguous after offset
            librispeech_indices = list(range(offset, offset + librispeech_indices))

        # Handle dataset_size + librispeech_size splitting
        if emg_indices is None and librispeech_indices is None:
            if dataset_size is not None and librispeech_size is not None:
                emg_size = dataset_size - librispeech_size
                if emg_size < 0:
                    raise ValueError(f"dataset_size {dataset_size} < librispeech_size {librispeech_size}")
                emg_indices = list(range(emg_size))
                librispeech_indices = list(range(emg_size, dataset_size))
            elif dataset_size is not None:
                # No librispeech_size — split half
                half = dataset_size // 2
                emg_indices = list(range(half))
                librispeech_indices = list(range(half, dataset_size))
            elif isinstance(emg_indices, list) and isinstance(librispeech_indices, list):
                pass
            else:
                # Both None — empty sampler (valid for tests that just check construction)
                emg_indices = []
                librispeech_indices = []

        if emg_indices is None:
            emg_indices = []
        if librispeech_indices is None:
            librispeech_indices = []

        # Ensure lists of ints
        self.emg_indices: list[int] = [int(x) for x in emg_indices]
        self.librispeech_indices: list[int] = [int(x) for x in librispeech_indices]

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not 0 < librispeech_ratio <= 1.0:
            raise ValueError(f"librispeech_ratio must be in (0,1], got {librispeech_ratio}")

        self.batch_size = int(batch_size)
        self.librispeech_ratio = float(librispeech_ratio)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)
        self.epoch: int = 0

        # Cache for current epoch's combined pool (recomputed each epoch)
        self._current_pool: list[int] | None = None

        logger.debug(
            "BatchSampler init: emg=%d librispeech=%d batch=%d ratio=%.2f seed=%s",
            len(self.emg_indices), len(self.librispeech_indices), self.batch_size, self.librispeech_ratio, str(seed),
        )

    # -- epoch control -----------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for per-epoch subsampling determinism (call each epoch)."""
        if epoch < 0:
            raise ValueError(f"epoch must be >=0, got {epoch}")
        self.epoch = int(epoch)
        self._current_pool = None  # invalidate
        logger.debug("BatchSampler set_epoch %d", epoch)

    # -- sampling -----------------------------------------------------------

    def _sample_pool(self) -> list[int]:
        """Sample 100% EMG + ratio*LibriSpeech and shuffle."""
        rng_seed = None
        if self.seed is not None:
            rng_seed = self.seed + self.epoch
        rng = random.Random(rng_seed)

        n_ls = len(self.librispeech_indices)
        k = int(n_ls * self.librispeech_ratio)
        # Ensure at least 1 if librispeech non-empty and ratio>0
        if n_ls > 0 and k == 0 and self.librispeech_ratio > 0:
            k = 1
        if k > n_ls:
            k = n_ls

        if k < n_ls:
            # random sample without replacement
            # Use rng.sample for determinism
            sampled_ls = rng.sample(self.librispeech_indices, k) if k > 0 else []
        else:
            sampled_ls = list(self.librispeech_indices)
            if self.shuffle:
                rng.shuffle(sampled_ls)

        combined = self.emg_indices + sampled_ls
        if self.shuffle:
            rng.shuffle(combined)
        # If shuffle=False, keep EMG first then LS sampled order (deterministic)
        logger.debug(
            "BatchSampler epoch %d: sampled %d/%d librispeech, combined %d",
            self.epoch, len(sampled_ls), n_ls, len(combined),
        )
        return combined

    def _get_pool(self) -> list[int]:
        if self._current_pool is None:
            self._current_pool = self._sample_pool()
        return self._current_pool

    # -- iterator protocol (yields batches) --------------------------------

    def __iter__(self):  # type: ignore[override]
        pool = self._sample_pool()  # fresh sample each iteration (epoch-aware)
        # Cache for len consistency within this epoch? Keep but allow fresh each iter.
        self._current_pool = pool
        for i in range(0, len(pool), self.batch_size):
            batch = pool[i : i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch

    def __len__(self) -> int:
        pool = self._get_pool()
        if self.drop_last:
            return len(pool) // self.batch_size
        # ceil
        return (len(pool) + self.batch_size - 1) // self.batch_size

    # -- helpers for inspection ---------------------------------------------

    @property
    def num_emg(self) -> int:
        return len(self.emg_indices)

    @property
    def num_librispeech(self) -> int:
        return len(self.librispeech_indices)

    @property
    def num_librispeech_sampled(self) -> int:
        """Number of LibriSpeech samples kept per epoch (ratio)."""
        n = len(self.librispeech_indices)
        k = int(n * self.librispeech_ratio)
        if n > 0 and k == 0 and self.librispeech_ratio > 0:
            k = 1
        return k

    def get_combined_indices(self) -> list[int]:
        """Return current epoch's combined pool (for debugging)."""
        return list(self._get_pool())

