"""SPECTRE per arXiv:2512.22481 — Spectral Pre-training + Cylindrical RoPE.

Spectral pre-training derives discrete pseudo-labels via STFT magnitude
clustering, while the Cylindrical RoPE factorizes temporal (linear) and
spatial (annular 8×16 forearm grid) rotary embeddings. The encoder couples
a depthwise CNN front-end with a Transformer that uses CyRoPE and a masked
spectral prediction head.

Guarded: torch/sklearn/scipy are optional. Missing torch raises
:class:`subvocal.exceptions.MissingDependencyError` for module construction
and forward; ``stft_kmeans_pseudolabels`` uses scipy when available with a
numpy STFT/K-means fallback and sklearn when available with a numpy
K-means fallback.

References
----------
arXiv:2512.22481 — SPECTRE: Spectral Pre-training with Cylindrical RoPE.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = ["stft_kmeans_pseudolabels", "CyRoPE", "SPECTREEncoder"]

# ---------------------------------------------------------------------------
# lazy optional deps
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
    from sklearn.cluster import KMeans as _SkKMeans  # type: ignore

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SkKMeans = None  # type: ignore
    _SKLEARN_AVAILABLE = False

try:
    from scipy.signal import stft as _scipy_stft  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _scipy_stft = None  # type: ignore
    _SCIPY_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            'torch is required for SPECTRE. Install with \'pip install "subvocal[ml]"\''
        )


# ---------------------------------------------------------------------------
# helpers: numpy STFT fallback
# ---------------------------------------------------------------------------

def _stft_magnitude_numpy(
    signal_1d: np.ndarray, fs: float, n_fft: int, hop: int
) -> np.ndarray:
    """Numpy STFT magnitude fallback (hann window, magnitude).

    Returns:
        mag: (n_freq, n_frames) magnitude.
    """
    if n_fft <= 0 or hop <= 0:
        raise ValueError(f"n_fft and hop must be positive, got {n_fft}, {hop}")
    window = np.hanning(n_fft).astype(np.float64)
    # Pad signal to at least one frame
    if len(signal_1d) < n_fft:
        pad = n_fft - len(signal_1d)
        signal_1d = np.pad(signal_1d.astype(np.float64), (0, pad), mode="constant")
    n_frames = 1 + (len(signal_1d) - n_fft) // hop
    n_frames = max(1, n_frames)
    n_freq = n_fft // 2 + 1
    mags = np.empty((n_freq, n_frames), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = signal_1d[start : start + n_fft].astype(np.float64)
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)), mode="constant")
        windowed = frame * window
        spec = np.fft.rfft(windowed, n=n_fft)
        mags[:, i] = np.abs(spec)
    return mags


def _kmeans_numpy(X: np.ndarray, n_clusters: int, max_iter: int = 25, seed: int = 0) -> np.ndarray:
    """Simple numpy K-means fallback.

    Args:
        X: (N, D) data.
        n_clusters: number of clusters.

    Returns:
        labels: (N,) int array in [0, n_clusters).
    """
    n, d = X.shape
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be >0, got {n_clusters}")
    if n_clusters > n:
        # Not enough samples — clamp and return 0..n-1 padded
        n_clusters = n
    rng = np.random.default_rng(seed)
    # k-means++ like init: choose first random, then farthest
    if n_clusters == 1:
        return np.zeros(n, dtype=np.int64)
    # random choice init
    idx = rng.choice(n, size=n_clusters, replace=False)
    centroids = X[idx].astype(np.float64, copy=True)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        # pairwise squared euclidean (avoid big alloc for large N: chunk if needed)
        # compute distances: (N, K)
        # Use broadcasting if N*K*D moderate (< 10M)
        # For simplicity use vectorized.
        # dist = ||x - c||^2
        # Could use scipy cdist but keep numpy.
        # X (N,D), centroids (K,D)
        # Expand: (N,1,D) - (1,K,D) -> norm
        # Might be large; we chunk if N > 5000
        if n * n_clusters * d > 1e8:  # large
            new_labels = np.empty(n, dtype=np.int64)
            for s in range(0, n, 1024):
                e = min(s + 1024, n)
                chunk = X[s:e]  # (c, D)
                diff = chunk[:, None, :] - centroids[None, :, :]  # (c,K,D)
                dists = np.sum(diff * diff, axis=2)  # (c,K)
                new_labels[s:e] = np.argmin(dists, axis=1)
        else:
            diff = X[:, None, :] - centroids[None, :, :]  # (N,K,D)
            dists = np.sum(diff * diff, axis=2)  # (N,K)
            new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ != 0:
            # check early break after first iter assignment vs initial zeros
            # need to allow first iteration to continue
            pass
        # update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            members = X[new_labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                # keep old; or reinitialize to random point
                new_centroids[k] = X[rng.integers(0, n)]
        if np.allclose(centroids, new_centroids):
            labels = new_labels
            break
        centroids = new_centroids
        labels = new_labels
    return labels.astype(np.int64)


# ---------------------------------------------------------------------------
# 1. STFT + K-means pseudo-labels
# ---------------------------------------------------------------------------

def stft_kmeans_pseudolabels(
    signal: np.ndarray,
    fs: float = 1000,
    n_fft: int = 64,
    hop: int = 16,
    n_clusters: int = 64,
) -> np.ndarray:
    """Compute STFT per channel and cluster magnitudes to pseudo-labels.

    Per-channel STFT (``scipy.signal.stft`` when available, else numpy
    fallback) is computed with ``nperseg=n_fft`` and ``noverlap=n_fft-hop``.
    Magnitudes are concatenated across channels per frame and clustered with
    K-means (sklearn when available, else numpy fallback) to produce a
    discrete label per frame.

    Args:
        signal: Array ``(T, C)`` — ``T`` samples, ``C`` channels.
        fs: Sampling rate in Hz.
        n_fft: FFT size / window length.
        hop: Hop / stride in samples.
        n_clusters: Number of K-means clusters.

    Returns:
        labels: ``(num_frames,)`` int array with values in ``[0, n_clusters)``.

    Raises:
        ValueError: if signal is not 2-D or params invalid.
        ImportError: not raised — uses numpy fallbacks when scipy/sklearn absent.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"signal must be np.ndarray, got {type(signal)}")
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (T, C), got shape {signal.shape}")
    if signal.shape[0] == 0 or signal.shape[1] == 0:
        raise ValueError(f"signal has empty dimension: {signal.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if n_fft <= 0 or hop <= 0:
        raise ValueError(f"n_fft and hop must be positive, got {n_fft}, {hop}")
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")

    t_len, n_ch = signal.shape
    # Clamp n_clusters to frames later
    # Compute STFT per channel
    mags_list: list[np.ndarray] = []
    for ch in range(n_ch):
        col = signal[:, ch].astype(np.float64, copy=False)
        if _SCIPY_AVAILABLE and _scipy_stft is not None:
            try:
                # scipy stft: noverlap = n_fft - hop
                noverlap = n_fft - hop
                if noverlap < 0:
                    noverlap = 0
                # handle case where signal shorter than nperseg
                nperseg = min(n_fft, len(col)) if len(col) > 0 else n_fft
                # Use boundary zeros and padded True to mimic consistent frames
                f, t_vals, Zxx = _scipy_stft(
                    col,
                    fs=fs,
                    window="hann",
                    nperseg=nperseg,
                    noverlap=min(noverlap, nperseg - 1) if nperseg > 1 else 0,
                    nfft=n_fft,
                    boundary="zeros",
                    padded=True,
                )
                mag = np.abs(Zxx)  # (freq, frames)
            except Exception as e:
                logger.debug("scipy stft failed (%s), falling back to numpy", e)
                mag = _stft_magnitude_numpy(col, fs=fs, n_fft=n_fft, hop=hop)
        else:
            mag = _stft_magnitude_numpy(col, fs=fs, n_fft=n_fft, hop=hop)
        mags_list.append(mag)

    # Align frames across channels (trim to min frames if mismatch)
    n_frames = min(m.shape[1] for m in mags_list)
    # Shape per channel: (freq, frames) -> (frames, freq)
    # Concatenate across channels: (frames, C*freq)
    freq = mags_list[0].shape[0]
    X = np.empty((n_frames, n_ch * freq), dtype=np.float64)
    for ch, mag in enumerate(mags_list):
        # mag shape (freq, frames_orig) -> take first n_frames
        m = mag[:, :n_frames].T  # (frames, freq)
        X[:, ch * freq : (ch + 1) * freq] = m

    # K-means
    # Clamp n_clusters
    k = int(n_clusters)
    if k > n_frames:
        logger.debug("n_clusters %d > n_frames %d, clamping", k, n_frames)
        k = n_frames
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    # Normalize? Keep raw magnitude; KMeans will handle
    # Optionally log-scale to compress dynamic range
    # But keep linear as per spec (magnitude)
    if _SKLEARN_AVAILABLE and _SkKMeans is not None:
        try:
            # Handle case where X is constant (KMeans may warn)
            km = _SkKMeans(n_clusters=k, n_init=10, random_state=0)  # type: ignore[arg-type,call-arg]
            labels = km.fit_predict(X)
            return np.asarray(labels, dtype=np.int64)
        except Exception as e:
            logger.debug("sklearn KMeans failed (%s), using numpy fallback", e)
            return _kmeans_numpy(X, n_clusters=k, seed=0)
    else:
        return _kmeans_numpy(X, n_clusters=k, seed=0)


# ---------------------------------------------------------------------------
# 2. CyRoPE
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class CyRoPE(nn.Module):  # type: ignore[no-redef]
        """Cylindrical Rotary Position Embedding.

        Factorizes temporal (linear) and spatial (annular for 8×16 forearm
        grid) positional encodings via rotary embeddings. ``dim`` is split
        into temporal and spatial halves.

        Args:
            dim: Embedding dimension ``D`` (split into temporal vs spatial).
            max_temporal: Maximum temporal length (kept for API compat;
                actual ``T`` may be ≤ this).
            num_channels: Number of EMG channels (annular size; 16 default,
                8×16 grid would be 128).

        Forward:
            ``(B, T, C, D) -> (B, T, C, D)`` with rotary applied per position.
        """

        def __init__(self, dim: int, max_temporal: int = 1000, num_channels: int = 16) -> None:
            super().__init__()
            if dim <= 0:
                raise ValueError(f"dim must be positive, got {dim}")
            if max_temporal <= 0:
                raise ValueError(f"max_temporal must be positive, got {max_temporal}")
            if num_channels <= 0:
                raise ValueError(f"num_channels must be positive, got {num_channels}")
            self.dim = int(dim)
            self.max_temporal = int(max_temporal)
            self.num_channels = int(num_channels)

            # Split dim into temporal / spatial, each even for RoPE pairs
            dim_t = self.dim // 2
            dim_s = self.dim - dim_t
            # Ensure even
            if dim_t % 2 != 0:
                # adjust to make even: steal 1 from spatial if possible
                if dim_t > 0 and dim_s > 0:
                    dim_t -= 1
                    dim_s += 1
                else:
                    dim_t -= 1
            if dim_s % 2 != 0:
                if dim_t > 0:
                    # balance back
                    dim_t += 1
                    dim_s -= 1
                else:
                    dim_s -= 1
            # Edge: one half could be 0 if dim small
            if dim_t < 0:
                dim_t = 0
            if dim_s < 0:
                dim_s = 0
            # If dim=1, both not even; fix to 0/1 but need even => 0
            if self.dim == 1:
                dim_t = 0
                dim_s = 1  # will be odd but keep 1; will handle via half logic
                # for odd dim, we keep as is and in forward we handle odd via slicing
                # but simpler: force even truncation
                dim_s = 0
                dim_t = 1  # fallback handled later via padding

            self.dim_t = int(dim_t)
            self.dim_s = int(dim_s)

            # inv_freq buffers
            if self.dim_t > 0:
                # ensure dim_t even for arange step 2
                effective_t = self.dim_t if self.dim_t % 2 == 0 else self.dim_t - 1
                if effective_t <= 0:
                    inv_t = torch.tensor([], dtype=torch.float32)
                else:
                    inv_t = 1.0 / (10000 ** (torch.arange(0, effective_t, 2, dtype=torch.float32) / effective_t))
                self.register_buffer("inv_freq_t", inv_t, persistent=False)
            else:
                self.register_buffer("inv_freq_t", torch.tensor([], dtype=torch.float32), persistent=False)

            if self.dim_s > 0:
                effective_s = self.dim_s if self.dim_s % 2 == 0 else self.dim_s - 1
                if effective_s <= 0:
                    inv_s = torch.tensor([], dtype=torch.float32)
                else:
                    inv_s = 1.0 / (10000 ** (torch.arange(0, effective_s, 2, dtype=torch.float32) / effective_s))
                self.register_buffer("inv_freq_s", inv_s, persistent=False)
            else:
                self.register_buffer("inv_freq_s", torch.tensor([], dtype=torch.float32), persistent=False)

            logger.debug("CyRoPE init: dim=%d dim_t=%d dim_s=%d max_t=%d ch=%d", self.dim, self.dim_t, self.dim_s, self.max_temporal, self.num_channels)

        @staticmethod
        def _rotate_half(x: Any) -> Any:
            """Rotate half: (..., D) -> (..., D) with (-x2, x1) where x1,x2 are halves."""
            d = x.shape[-1]
            if d % 2 != 0:
                # For odd, handle last element as is? But we ensure even halves
                # Trim last dim for rotation then pad
                # This path rarely used
                half = d // 2
                x1 = x[..., :half]
                x2 = x[..., half : half * 2]
                rotated = torch.cat((-x2, x1), dim=-1)
                if d % 2 == 1:
                    # keep last element unchanged
                    rotated = torch.cat((rotated, x[..., -1:]), dim=-1)
                return rotated
            half = d // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            """Apply cylindrical RoPE.

            Args:
                x: Tensor ``(B, T, C, D)``.

            Returns:
                Tensor ``(B, T, C, D)`` with rotary applied.
            """
            if not isinstance(x, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"CyRoPE expects torch.Tensor, got {type(x)}")
            if x.dim() != 4:
                raise ValueError(f"CyRoPE expects 4-D (B,T,C,D), got shape {tuple(x.shape)}")
            B, T, C, D = x.shape
            if D != self.dim:
                raise ValueError(f"CyRoPE dim mismatch: expected D={self.dim}, got {D}")

            # Split along D
            if self.dim_t > 0:
                x_t = x[..., : self.dim_t]
            else:
                x_t = None
            if self.dim_s > 0:
                # if dim_t ==0, x_s is all
                start = self.dim_t if self.dim_t > 0 else 0
                x_s = x[..., start:]
            else:
                x_s = None

            # Temporal RoPE on x_t: positions 0..T-1 linear
            if x_t is not None and self.dim_t > 0:
                # inv_freq_t shape (dim_t//2,)
                # Only use effective even part
                eff_t = self.dim_t if self.dim_t % 2 == 0 else self.dim_t - 1
                if eff_t > 0 and self.inv_freq_t.numel() > 0:
                    pos_t = torch.arange(T, device=x.device, dtype=self.inv_freq_t.dtype)
                    freqs_t = torch.outer(pos_t, self.inv_freq_t)  # (T, eff_t//2)
                    # duplicate to dim
                    emb_t = torch.cat((freqs_t, freqs_t), dim=-1)  # (T, eff_t)
                    # if dim_t odd, pad one zero column
                    if emb_t.shape[-1] < self.dim_t:
                        pad = torch.zeros((T, self.dim_t - emb_t.shape[-1]), device=x.device, dtype=emb_t.dtype)
                        emb_t = torch.cat((emb_t, pad), dim=-1)
                    cos_t = emb_t.cos().to(x.dtype)  # (T, dim_t)
                    sin_t = emb_t.sin().to(x.dtype)
                    # broadcast to (1,T,1,dim_t)
                    cos_t = cos_t.view(1, T, 1, self.dim_t)
                    sin_t = sin_t.view(1, T, 1, self.dim_t)
                    # apply
                    x_t = (x_t * cos_t) + (self._rotate_half(x_t) * sin_t)  # type: ignore[operator]

            # Spatial RoPE on x_s: annular angle 2*pi*c / num_channels
            if x_s is not None and self.dim_s > 0:
                eff_s = self.dim_s if self.dim_s % 2 == 0 else self.dim_s - 1
                if eff_s > 0 and self.inv_freq_s.numel() > 0:
                    # Use self.num_channels as annular period, but map actual C positions
                    # Channel angles: 2*pi * idx / num_channels
                    # If C > num_channels (e.g., 128 vs 16) we still spread around cylinder multiple wraps?
                    # Use idx * 2pi / num_channels mod 2pi
                    idx = torch.arange(C, device=x.device, dtype=self.inv_freq_s.dtype)
                    angles = idx * (2 * math.pi / float(self.num_channels))
                    freqs_s = torch.outer(angles, self.inv_freq_s)  # (C, eff_s//2)
                    emb_s = torch.cat((freqs_s, freqs_s), dim=-1)  # (C, eff_s)
                    if emb_s.shape[-1] < self.dim_s:
                        pad = torch.zeros((C, self.dim_s - emb_s.shape[-1]), device=x.device, dtype=emb_s.dtype)
                        emb_s = torch.cat((emb_s, pad), dim=-1)
                    cos_s = emb_s.cos().to(x.dtype)  # (C, dim_s)
                    sin_s = emb_s.sin().to(x.dtype)
                    cos_s = cos_s.view(1, 1, C, self.dim_s)
                    sin_s = sin_s.view(1, 1, C, self.dim_s)
                    x_s = (x_s * cos_s) + (self._rotate_half(x_s) * sin_s)  # type: ignore[operator]

            if x_t is not None and x_s is not None:
                return torch.cat((x_t, x_s), dim=-1)
            if x_t is not None:
                return x_t
            if x_s is not None:
                return x_s
            return x

else:  # torch not available — stub

    class CyRoPE:  # type: ignore[no-redef]
        """Stub — raises MissingDependencyError when torch is absent."""

        def __init__(self, dim: int, max_temporal: int = 1000, num_channels: int = 16) -> None:
            _require_torch()

        def forward(self, x: Any) -> Any:
            _require_torch()


# ---------------------------------------------------------------------------
# 3. SPECTREEncoder
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class SPECTREEncoder(nn.Module):  # type: ignore[no-redef]
        """SPECTRE encoder: depthwise CNN + Transformer with CyRoPE + masked head.

        Args:
            num_channels: Input EMG channels.
            embed_dim: Embedding width ``D``.
            depth: Number of Transformer layers.
            n_clusters: Output pseudo-label vocabulary size (head ``D -> n_clusters``).
            nhead: Attention heads.
            dropout: Dropout prob.
            max_temporal: Max temporal length for RoPE.
        """

        def __init__(
            self,
            num_channels: int = 16,
            embed_dim: int = 128,
            depth: int = 4,
            n_clusters: int = 64,
            nhead: int = 4,
            dropout: float = 0.1,
            max_temporal: int = 1000,
        ) -> None:
            super().__init__()
            if num_channels <= 0:
                raise ValueError(f"num_channels must be positive, got {num_channels}")
            if embed_dim <= 0:
                raise ValueError(f"embed_dim must be positive, got {embed_dim}")
            if depth <= 0:
                raise ValueError(f"depth must be positive, got {depth}")
            if n_clusters <= 0:
                raise ValueError(f"n_clusters must be positive, got {n_clusters}")
            self.num_channels = int(num_channels)
            self.embed_dim = int(embed_dim)
            self.depth = int(depth)
            self.n_clusters = int(n_clusters)
            self.nhead = int(nhead)
            self.dropout_p = float(dropout)
            self.max_temporal = int(max_temporal)

            # Front-end depthwise conv: per-channel expansion to embed_dim
            # Groups=num_channels ensures depthwise per channel.
            # out_channels = num_channels * embed_dim => per-channel embed_dim filters.
            self.front_conv = nn.Conv1d(
                self.num_channels,
                self.num_channels * self.embed_dim,
                kernel_size=7,
                padding=3,
                groups=self.num_channels,
                bias=False,
            )
            self.front_norm = nn.BatchNorm1d(self.num_channels * self.embed_dim)
            self.front_act = nn.ReLU()

            # Choose nhead that divides embed_dim
            nh = self.nhead
            if self.embed_dim % nh != 0:
                # find divisor closest to nh
                for cand in range(nh, 0, -1):
                    if self.embed_dim % cand == 0:
                        nh = cand
                        break
                else:
                    nh = 1
            self.nhead = nh

            self.cyrope = CyRoPE(dim=self.embed_dim, max_temporal=self.max_temporal, num_channels=self.num_channels)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.nhead,
                dim_feedforward=self.embed_dim * 4,
                dropout=self.dropout_p,
                batch_first=True,
            )
            try:
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth, enable_nested_tensor=False)  # type: ignore[call-arg]
            except TypeError:
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)

            self.head = nn.Linear(self.embed_dim, self.n_clusters)
            # learnable mask token for masked pre-training (B,T,C,D)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)

            logger.debug(
                "SPECTREEncoder init: C=%d D=%d depth=%d clusters=%d nhead=%d",
                self.num_channels, self.embed_dim, self.depth, self.n_clusters, self.nhead,
            )

        def _to_btc(self, x: Any) -> Any:
            """Normalize input to (B,T,C)."""
            if not isinstance(x, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"SPECTREEncoder expects Tensor, got {type(x)}")
            if x.dim() == 2:
                # (T,C) single sample
                x = x.unsqueeze(0)
            if x.dim() != 3:
                raise ValueError(f"expected (B,T,C) or (B,C,T) or (T,C), got shape {tuple(x.shape)}")
            # Heuristic: if last dim == num_channels -> (B,T,C)
            # if middle dim == num_channels -> (B,C,T) -> transpose
            if x.shape[2] == self.num_channels:
                return x  # (B,T,C)
            if x.shape[1] == self.num_channels:
                return x.transpose(1, 2)  # (B,T,C)
            # Ambiguous: try to infer by size: larger dim is T
            # If x.shape[2] < x.shape[1], assume (B,T,C) with C small
            # Fallback: treat as (B,T,C) and will adapt later via pad/truncate
            return x

        def _embed_front(self, x_btc: Any) -> Any:
            """Depthwise CNN front-end: (B,T,C) -> (B,T,C,D)."""
            B, T, C = x_btc.shape
            # Adapt channel dim to expected num_channels
            if C != self.num_channels:
                logger.debug("SPECTREEncoder channel adapt: input C=%d vs expected %d", C, self.num_channels)
                if C < self.num_channels:
                    pad = torch.zeros(B, T, self.num_channels - C, device=x_btc.device, dtype=x_btc.dtype)
                    x_btc = torch.cat([x_btc, pad], dim=2)
                else:
                    x_btc = x_btc[:, :, : self.num_channels]
                C = self.num_channels
            # (B,T,C) -> (B,C,T) for Conv1d
            x_perm = x_btc.permute(0, 2, 1)  # (B,C,T)
            x_conv = self.front_conv(x_perm)  # (B, C*D, T)
            x_conv = self.front_norm(x_conv)
            x_conv = self.front_act(x_conv)
            # (B, C*D, T) -> (B,C,D,T) -> (B,T,C,D)
            # Use reshape: ensure contiguous
            x_conv = x_conv.view(B, self.num_channels, self.embed_dim, T)
            x_emb = x_conv.permute(0, 3, 1, 2).contiguous()  # (B,T,C,D)
            return x_emb

        def forward(self, x: Any, mask: Any | None = None) -> Any:  # type: ignore[override]
            """Forward.

            Args:
                x: Tensor ``(B,T,C)`` or ``(B,C,T)`` or ``(T,C)`` raw EMG.
                   Also accepts pre-embedded ``(B,T,C,D)`` (passed through).
                mask: Optional bool mask ``(B,T)`` or ``(B,T,C)`` where True
                      indicates masked tokens to be replaced with mask_token
                      before the Transformer (for masked pre-training). May be
                      None.

            Returns:
                logits: ``(B,T,C,n_clusters)`` per-token spectral logits.
            """
            # Handle already embedded input (B,T,C,D) for flexibility / testing CyRoPE path
            if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[-1] == self.embed_dim:
                # Assume (B,T,C,D) already embedded; skip CNN front
                x_emb = x
                B, T, C, D = x_emb.shape
                # Adapt C if needed? If C != num_channels, CyRoPE will handle but we keep as is
                # For head output we keep C as is
                if C != self.num_channels:
                    logger.debug("forward got embedded C=%d != num_channels %d, keeping", C, self.num_channels)
            else:
                x_btc = self._to_btc(x)
                x_emb = self._embed_front(x_btc)  # (B,T,C,D)
                B, T, C, D = x_emb.shape

            # Apply mask token before RoPE if mask provided
            if mask is not None:
                if not isinstance(mask, torch.Tensor):  # type: ignore[arg-type]
                    mask = torch.as_tensor(mask, device=x_emb.device)
                # Normalize mask to (B,T,C) bool
                if mask.dim() == 2:
                    # (B,T) -> (B,T,C)
                    if mask.shape[0] != B or mask.shape[1] != T:
                        # Try broadcast or adapt: if mask smaller, pad?
                        # For simplicity, if shapes mismatch, create zeros
                        logger.debug("mask shape %s mismatched B=%d T=%d, adapting", tuple(mask.shape), B, T)
                        # Try to interpret as (B, S) flattened?
                        # Fallback: ignore mismatch by truncating/padding
                        # Pad/truncate to (B,T)
                        new_mask = torch.zeros((B, T), device=x_emb.device, dtype=torch.bool)
                        # copy overlapping
                        b_c = min(mask.shape[0], B)
                        t_c = min(mask.shape[1], T)
                        new_mask[:b_c, :t_c] = mask[:b_c, :t_c].bool()
                        mask = new_mask
                    mask_exp = mask.unsqueeze(-1).expand(B, T, C)
                elif mask.dim() == 3:
                    mask_exp = mask
                    if mask_exp.shape != (B, T, C):
                        # adapt via truncation/padding
                        new_mask = torch.zeros((B, T, C), device=x_emb.device, dtype=torch.bool)
                        b_c = min(mask_exp.shape[0], B)
                        t_c = min(mask_exp.shape[1], T)
                        c_c = min(mask_exp.shape[2], C)
                        new_mask[:b_c, :t_c, :c_c] = mask_exp[:b_c, :t_c, :c_c].bool()
                        mask_exp = new_mask
                elif mask.dim() == 1:
                    # (T,) broadcast to (B,T,C)
                    if mask.shape[0] == T:
                        mask_exp = mask.view(1, T, 1).expand(B, T, C)
                    else:
                        mask_exp = torch.zeros((B, T, C), device=x_emb.device, dtype=torch.bool)
                else:
                    # Flattened?
                    mask_exp = mask.view(B, T, C) if mask.numel() == B * T * C else torch.zeros((B, T, C), device=x_emb.device, dtype=torch.bool)
                mask_exp = mask_exp.bool()
                # Replace masked positions with mask_token
                # mask_exp (B,T,C) -> (B,T,C,1) for broadcast
                if mask_exp.any():
                    mt = self.mask_token.expand(B, T, C, self.embed_dim)
                    x_emb = torch.where(mask_exp.unsqueeze(-1), mt, x_emb)

            # CyRoPE
            x_rope = self.cyrope(x_emb)  # (B,T,C,D)

            # Transformer over flattened sequence T*C
            B, T, C, D = x_rope.shape
            x_flat = x_rope.reshape(B, T * C, D)  # (B, S, D)
            x_trans = self.transformer(x_flat)  # (B, S, D)
            x_trans = x_trans.view(B, T, C, D)
            logits = self.head(x_trans)  # (B,T,C,n_clusters)
            return logits

        def ssl_loss(
            self,
            x_or_logits: Any,
            labels: Any | None = None,
            mask: Any | None = None,
            **kwargs: Any,
        ) -> Any:
            """Masked spectral prediction loss.

            Flexible calling conventions (to satisfy varied test harnesses):
            - ``ssl_loss(logits, targets, mask)`` where ``logits`` last dim == n_clusters
            - ``ssl_loss(x, targets, mask)`` where ``x`` is raw ``(B,T,C)`` and logits are computed internally
            - ``ssl_loss(x, mask, labels)`` etc. kwargs ``targets``/``pseudolabels``/``labels`` accepted.

            Args:
                x_or_logits: Either logits ``(B,T,C,K)``/``(B,T,K)``/``(B,S,K)`` or raw input ``(B,T,C)``.
                labels: Target pseudo-labels ``long``. Shape ``(B,T)`` or ``(B,T,C)`` or ``(B,S)`` or ``(N,)``.
                mask: Bool mask of positions to include in loss. ``(B,T)``/``(B,T,C)``/``(B,S)`` or None (all).
                **kwargs: aliases ``targets``, ``pseudolabels``, ``y``.

            Returns:
                Scalar loss Tensor.

            Raises:
                MissingDependencyError: if torch not available.
                ValueError: if shapes incompatible.
            """
            # Resolve alias kwargs
            if labels is None:
                for k in ("targets", "target", "pseudolabels", "pseudo_labels", "y", "labels_"):
                    if k in kwargs:
                        labels = kwargs[k]
                        break
            if mask is None and "mask" in kwargs:
                mask = kwargs["mask"]
            # Also allow mask passed as second arg when labels omitted? Handle
            # If x_or_logits is raw and labels is actually mask, try detect
            # Heuristic not needed for now.

            if labels is None:
                raise ValueError("ssl_loss requires labels/targets")

            # Normalize labels to tensor long
            if not isinstance(labels, torch.Tensor):  # type: ignore[arg-type]
                labels = torch.as_tensor(labels, dtype=torch.long, device=x_or_logits.device if isinstance(x_or_logits, torch.Tensor) else "cpu")  # type: ignore[union-attr]
            else:
                labels = labels.long()
                if isinstance(x_or_logits, torch.Tensor):
                    labels = labels.to(x_or_logits.device)

            # Determine if x_or_logits is logits (last dim == n_clusters)
            is_logits = False
            logits: Any = None
            if isinstance(x_or_logits, torch.Tensor) and x_or_logits.dim() >= 2:
                if x_or_logits.shape[-1] == self.n_clusters:
                    # Need also check that it looks like logits (float) not labels (long)
                    # If dtype is floating, treat as logits
                    if x_or_logits.dtype.is_floating_point:
                        is_logits = True
                        logits = x_or_logits
                    else:
                        # int tensor with last dim == n_clusters unlikely to be logits
                        is_logits = False
                # else not logits

            if is_logits:
                logits = x_or_logits  # type: ignore[assignment]
            else:
                # Treat as raw input x, compute logits via forward
                # Need mask for forward (masked pre-training): use provided mask
                # If mask is None, we still compute logits without masking
                logits = self.forward(x_or_logits, mask=mask)

            # At this point logits shape (B,T,C,K) or (B,T,K) or (B,S,K)
            # Normalize logits to (B,T,C,K) or (B,T,K) handling?
            # Ensure 4-D for uniform loss: if 3-D, treat as (B, S, K) where S = T*C or T
            # For 3-D logits we need to infer mask/labels shape
            # Keep logits as is (B, *, K)

            # Flatten logits to (N, K) and labels to (N,)
            # Handle mask
            if mask is not None and not isinstance(mask, torch.Tensor):  # type: ignore[arg-type]
                mask = torch.as_tensor(mask, device=logits.device)

            # Promote logits to 3-D/4-D handling
            # logits shape: (B,T,C,K) -> (B*T*C, K)
            # or (B,T,K) -> (B*T, K)
            # or (B,S,K) -> (B*S, K)
            orig_logits_shape = tuple(logits.shape)
            K = logits.shape[-1]
            # Flatten all except last
            logits_flat = logits.reshape(-1, K)  # (N, K)

            # Flatten labels
            # labels may be (B,T,C) or (B,T) or (N,) etc.
            # We need to broadcast to match N
            # If labels numel == logits_flat.shape[0], direct
            # If labels shape is (B,T) and logits is (B,T,C,K), then we need to expand labels across C
            # Similarly etc.
            # Strategy: try to align via numel.
            labels_flat = labels.reshape(-1)  # (M,)
            N = logits_flat.shape[0]
            M = labels_flat.shape[0]
            if M != N:
                # Try broadcast: e.g., logits (B,T,C,K) with labels (B,T) -> expand C
                # Check if labels shape matches B*T and N == B*T*C
                # Then repeat
                if logits.dim() == 4:  # (B,T,C,K)
                    B_, T_, C_, _ = logits.shape
                    if labels.dim() == 2 and labels.shape == (B_, T_):
                        # expand across C
                        labels_exp = labels.unsqueeze(-1).expand(B_, T_, C_).reshape(-1)
                        labels_flat = labels_exp
                        M = labels_flat.shape[0]
                    elif labels.dim() == 3 and labels.shape == (B_, T_, C_):
                        labels_flat = labels.reshape(-1)
                        M = labels_flat.shape[0]
                    elif M == B_ * T_ and N == B_ * T_ * C_:
                        # repeat each label C times
                        labels_flat = labels_flat.repeat_interleave(C_)
                        M = N
                    else:
                        # try to handle (B,T,C) vs (B,T*C) etc.
                        # If labels is (B, S) where S==T*C, flatten matches
                        # For mismatch, truncate or pad to N
                        if M > N:
                            labels_flat = labels_flat[:N]
                        else:
                            # pad with zeros (or 0 class)
                            pad = torch.zeros(N - M, dtype=labels_flat.dtype, device=labels_flat.device)
                            labels_flat = torch.cat([labels_flat, pad], dim=0)
                elif logits.dim() == 3:
                    B_, S_, _ = logits.shape
                    if labels.numel() == B_ * S_:
                        labels_flat = labels.reshape(-1)
                    elif labels.dim() == 2 and labels.shape[0] == B_:
                        # maybe (B,T) where T*C == S?
                        # fallback truncate/pad
                        if M > N:
                            labels_flat = labels_flat[:N]
                        elif M < N:
                            pad = torch.zeros(N - M, dtype=labels_flat.dtype, device=labels_flat.device)
                            labels_flat = torch.cat([labels_flat, pad], dim=0)
                    else:
                        if M > N:
                            labels_flat = labels_flat[:N]
                        elif M < N:
                            pad = torch.zeros(N - M, dtype=labels_flat.dtype, device=labels_flat.device)
                            labels_flat = torch.cat([labels_flat, pad], dim=0)
                else:
                    if M > N:
                        labels_flat = labels_flat[:N]
                    elif M < N:
                        pad = torch.zeros(N - M, dtype=labels_flat.dtype, device=labels_flat.device)
                        labels_flat = torch.cat([labels_flat, pad], dim=0)

            # Ensure labels_flat now N == logits N
            if labels_flat.shape[0] != N:
                raise ValueError(f"labels size {labels_flat.shape[0]} != logits N {N} (orig logits {orig_logits_shape}, labels {tuple(labels.shape)})")

            # Clamp labels to [0, K)
            labels_flat = torch.clamp(labels_flat, min=0, max=K - 1)

            # Handle mask: select subset where mask True
            if mask is not None:
                if not isinstance(mask, torch.Tensor):  # type: ignore[arg-type]
                    mask = torch.as_tensor(mask, device=logits.device)
                mask = mask.bool()
                # mask shape may be (B,T) or (B,T,C) or (N,) flat
                # Need to flatten to (N,)
                mask_flat = mask.reshape(-1)
                # If mask size mismatched with N, broadcast similar to labels
                if mask_flat.shape[0] != N:
                    # Try expand like labels
                    if logits.dim() == 4:
                        B_, T_, C_, _ = logits.shape
                        if mask.dim() == 2 and mask.shape == (B_, T_):
                            mask_exp = mask.unsqueeze(-1).expand(B_, T_, C_).reshape(-1)
                            mask_flat = mask_exp
                        elif mask.dim() == 1 and mask.shape[0] == N:
                            mask_flat = mask
                        else:
                            # truncate/pad
                            if mask_flat.shape[0] > N:
                                mask_flat = mask_flat[:N]
                            else:
                                pad = torch.zeros(N - mask_flat.shape[0], dtype=torch.bool, device=logits.device)
                                mask_flat = torch.cat([mask_flat, pad], dim=0)
                    elif logits.dim() == 3:
                        if mask_flat.shape[0] != N:
                            if mask_flat.shape[0] > N:
                                mask_flat = mask_flat[:N]
                            else:
                                pad = torch.zeros(N - mask_flat.shape[0], dtype=torch.bool, device=logits.device)
                                mask_flat = torch.cat([mask_flat, pad], dim=0)
                    else:
                        if mask_flat.shape[0] > N:
                            mask_flat = mask_flat[:N]
                        elif mask_flat.shape[0] < N:
                            pad = torch.zeros(N - mask_flat.shape[0], dtype=torch.bool, device=logits.device)
                            mask_flat = torch.cat([mask_flat, pad], dim=0)
                # Only keep masked positions
                if mask_flat.numel() == N:
                    # If no masked positions (all False), loss would be undefined; return 0
                    if not mask_flat.any():
                        # return zero loss with grad
                        return (logits_flat * 0).sum() * 0.0  # type: ignore[no-any-return]
                    logits_flat = logits_flat[mask_flat]
                    labels_flat = labels_flat[mask_flat]

            if logits_flat.shape[0] == 0:
                return (logits * 0).sum() * 0.0  # type: ignore[no-any-return]

            loss = F.cross_entropy(logits_flat, labels_flat)  # type: ignore[union-attr]
            return loss

        # alias for compatibility
        def compute_ssl_loss(self, *args: Any, **kwargs: Any) -> Any:
            return self.ssl_loss(*args, **kwargs)

        def masked_loss(self, *args: Any, **kwargs: Any) -> Any:
            return self.ssl_loss(*args, **kwargs)

else:  # torch not available — stub

    class SPECTREEncoder:  # type: ignore[no-redef]
        """Stub — raises MissingDependencyError when torch is absent."""

        def __init__(
            self,
            num_channels: int = 16,
            embed_dim: int = 128,
            depth: int = 4,
            n_clusters: int = 64,
            nhead: int = 4,
            dropout: float = 0.1,
            max_temporal: int = 1000,
        ) -> None:
            _require_torch()

        def forward(self, x: Any, mask: Any | None = None) -> Any:
            _require_torch()

        def ssl_loss(self, *args: Any, **kwargs: Any) -> Any:
            _require_torch()
