"""SPD matrix representation for sEMG.

Implements symmetric positive-definite (SPD) covariance representation
and Riemannian tangent-space mapping per:

* Gowda & Miller, "SPD Manifold Learning for Silent Speech sEMG",
  Findings of ACL 2026.
* Gowda et al., "Riemannian Geometry for sEMG Decoding",
  J. Neural Eng. 2024.
* Gowda 2025/2026 — SPD-GRU CTC extensions (companion :mod:`spd_gru`).

The SPD manifold captures second-order channel covariation (muscle
synergies) that is invariant to first-order drift and more stable than
raw amplitude across sessions (J Neural Eng 2024, Sec. 3.2). Mapping via
the affine-invariant logarithm (``logm``) projects SPD points to the
tangent space at identity, where Euclidean GRU layers operate
(ACL 2026 Findings, Fig. 2).

All functions operate on :mod:`numpy` arrays; batch and single-matrix
cases are supported. Singular matrices are regularised with ``eps*I``.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    from scipy.linalg import eigh as _scipy_eigh
except ImportError:  # pragma: no cover - fallback if scipy missing
    from numpy.linalg import eigh as _scipy_eigh  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "compute_covariance_matrix",
    "compute_spd_matrix",
    "compute_spd_timevarying",
    "spd_logm",
    "spd_flatten_upper",
    "spd_riemannian_features",
    "get_riemannian_features",
    "extract_spd_features",
    "get_riemannian_features_timevarying",
    "extract_riemannian_features",
    "get_spd_features",
    "spd_features",
]


# ---------------------------------------------------------------------------
# covariance / SPD
# ---------------------------------------------------------------------------

def compute_covariance_matrix(segment: np.ndarray) -> np.ndarray:
    """Compute channel covariance ``(C, C)`` for a segment.

    Args:
        segment: Array ``(T, C)`` — ``T`` time samples, ``C`` channels.
            Convention follows :mod:`subvocal.emg_core.dsp.filters`
            (``(num_samples, num_channels)``). A ``(C, T)`` array with
            ``C < T`` is **not** transposed automatically; pass ``(T, C)``.

    Returns:
        Covariance matrix ``(C, C)``, symmetric positive semi-definite.
        Shape is ``(C, C)`` where ``C = segment.shape[1]``.

    References:
        Gowda & Miller, ACL 2026 Findings, Sec. 3.1 (sample covariance on
        mV-normalised sEMG windows); J Neural Eng 2024, Eq. 1.
    """
    if not isinstance(segment, np.ndarray):
        raise TypeError(f"segment must be np.ndarray, got {type(segment)}")
    if segment.ndim != 2:
        raise ValueError(f"segment must be 2-D (T, C), got shape {segment.shape}")
    if segment.shape[0] == 0 or segment.shape[1] == 0:
        raise ValueError(f"segment has empty dimension: {segment.shape}")

    seg = segment.astype(np.float64, copy=False)
    n_samples = seg.shape[0]
    # centre per channel
    mean = np.mean(seg, axis=0, keepdims=True)
    x = seg - mean

    if n_samples > 1:
        # unbiased estimator (N-1) per J Neural Eng 2024
        cov = (x.T @ x) / float(n_samples - 1)
    else:
        cov = (x.T @ x) / float(max(n_samples, 1))

    # enforce symmetry (numerical drift)
    cov = (cov + cov.T) * 0.5
    # ensure float64 and correct shape (C,C)
    cov = np.asarray(cov, dtype=np.float64)
    logger.debug("compute_covariance_matrix: T=%d C=%d cov_shape=%s", n_samples, seg.shape[1], cov.shape)
    return cov


def compute_spd_matrix(segment: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute SPD regularised covariance ``(C, C)``.

    Adds ``eps * I`` to the sample covariance so the result is strictly
    symmetric positive-definite even when the segment is rank-deficient
    (e.g., constant channel or ``T < C``).

    Args:
        segment: Array ``(T, C)``.
        eps: Ridge regulariser added to the diagonal (default ``1e-6``).
            Must be non-negative; ``1e-6`` follows ACL 2026 (``1e-6*I``).

    Returns:
        SPD matrix ``(C, C)``.

    References:
        Gowda & Miller, ACL 2026 Findings, Sec. 3.2 (``C + eps I``);
        J Neural Eng 2024, Eq. 2.
    """
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    cov = compute_covariance_matrix(segment)
    c = cov.shape[0]
    spd = cov + eps * np.eye(c, dtype=np.float64)
    # re-symmetrise after addition (no-op but ensures exact symmetry)
    spd = (spd + spd.T) * 0.5
    logger.debug("compute_spd_matrix: eps=%.1e spd_shape=%s", eps, spd.shape)
    return spd


def compute_spd_timevarying(
    segment: np.ndarray,
    fs: float = 250,
    window_ms: int = 50,
    step_ms: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """Sliding-window SPD sequence ``(num_windows, C, C)``.

    Splits *segment* into overlapping windows of length ``window_ms``
    with hop ``step_ms`` and computes an SPD matrix per window via
    :func:`compute_spd_matrix`. Overlap (``step < window``) yields
    dense temporal sampling for the downstream SPD-GRU (ACL 2026, Sec. 4).

    Args:
        segment: Array ``(T, C)``.
        fs: Sampling rate in Hz.
        window_ms: Window length in ms (default 50 ms → 13 samples @250 Hz
            rounded; actual ``window_samples = round(fs*window_ms/1000)``).
        step_ms: Hop / shift in ms (default 20 ms).
        eps: SPD regulariser per window.

    Returns:
        Array ``(num_windows, C, C)``. If ``T < window_samples`` the
        segment is zero-padded to produce a single window (consistent with
        :func:`subvocal.emg_core.dsp.handcrafted.extract_handcrafted_timevarying`).

    References:
        Gowda & Miller, ACL 2026 Findings, Sec. 4.1 (50 ms / 20 ms sliding
        SPD); J Neural Eng 2024, Sec. 3.3 (overlap windowing).
    """
    if not isinstance(segment, np.ndarray):
        raise TypeError(f"segment must be np.ndarray, got {type(segment)}")
    if segment.ndim != 2:
        raise ValueError(f"segment must be 2-D (T, C), got shape {segment.shape}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError(f"window_ms and step_ms must be positive, got {window_ms}, {step_ms}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    n_samples, n_channels = segment.shape
    window_samples = max(1, int(round(fs * window_ms / 1000.0)))
    step_samples = max(1, int(round(fs * step_ms / 1000.0)))

    logger.debug(
        "compute_spd_timevarying: T=%d C=%d fs=%.1f window=%dms (%d samp) step=%dms (%d samp) eps=%.1e",
        n_samples, n_channels, fs, window_ms, window_samples, step_ms, step_samples, eps,
    )

    if n_samples < window_samples:
        # pad to at least one window (zero-pad preserves covariance structure with eps)
        pad_len = window_samples - n_samples
        logger.debug("segment shorter than window (%d < %d) – padding %d samples", n_samples, window_samples, pad_len)
        padded = np.pad(segment, ((0, pad_len), (0, 0)), mode="constant")
        spd = compute_spd_matrix(padded, eps=eps)
        return spd[np.newaxis, :, :]

    num_windows = 1 + (n_samples - window_samples) // step_samples
    num_windows = max(1, num_windows)

    out = np.empty((num_windows, n_channels, n_channels), dtype=np.float64)
    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        window = segment[start:end, :]
        out[i] = compute_spd_matrix(window, eps=eps)

    logger.debug("compute_spd_timevarying: num_windows=%d out_shape=%s", num_windows, out.shape)
    return out


# ---------------------------------------------------------------------------
# Riemannian log map + flatten
# ---------------------------------------------------------------------------

def spd_logm(spd: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Matrix logarithm via eigendecomposition (``V log(D) V^T``).

    Uses :func:`scipy.linalg.eigh` which exploits symmetry/sparsity of
    SPD matrices and avoids the ``O(C^3)`` Schur bottleneck of
    :func:`scipy.linalg.logm` (ACL 2026, App. B: “sparse spectral domain”).

    Eigenvalues are clamped to ``eps`` before ``log`` for numerical
    stability on rank-deficient inputs (also covers explicit
    ``eps*I`` regularisation).

    Args:
        spd: SPD matrix ``(C, C)`` or stack ``(..., C, C)`` (e.g.,
            ``(num_windows, C, C)`` or ``(B, T, C, C)``).
        eps: Minimum eigenvalue before log (default ``1e-6``).

    Returns:
        Matrix logarithm(s) with same shape as *spd*.

    References:
        Gowda & Miller, ACL 2026 Findings, Eq. 4; J Neural Eng 2024,
        Eq. 5 (affine-invariant log map at identity).
    """
    if not isinstance(spd, np.ndarray):
        raise TypeError(f"spd must be np.ndarray, got {type(spd)}")
    if spd.ndim < 2 or spd.shape[-1] != spd.shape[-2]:
        raise ValueError(f"spd must be (..., C, C), got shape {spd.shape}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    c = spd.shape[-1]
    orig_shape = spd.shape

    # reshape to (-1, C, C) for uniform loop; vectorise not needed for C<=8
    flat = spd.reshape(-1, c, c).astype(np.float64, copy=False)
    out = np.empty_like(flat, dtype=np.float64)

    for i in range(flat.shape[0]):
        mat = flat[i]
        # ensure symmetry before eigh (numerical noise)
        mat = (mat + mat.T) * 0.5
        # add tiny ridge if needed to guarantee SPD for eigh stability
        # (clamping eigenvalues later also handles it)
        try:
            w, v = _scipy_eigh(mat)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("eigh failed (%s) – adding eps*I and retrying", e)
            mat = mat + eps * np.eye(c, dtype=np.float64)
            w, v = _scipy_eigh(mat)  # type: ignore[arg-type]

        # clamp eigenvalues in sparse spectral domain
        w = np.maximum(w, eps)
        log_w = np.log(w)
        # V diag(log_w) V^T ; scale columns of V by log_w
        # v columns correspond to w (scipy eigh ascending)
        vt = v.T
        # (v * log_w) @ v.T  – broadcast: v * log_w scales columns
        # Equivalent to v @ diag(log_w) @ v.T
        scaled = v * log_w  # shape (C, C) * (C,) -> last axis broadcast = column scaling? Need to verify: numpy broadcasting of (C,C)*(C,) scales columns? Actually (C,C)*(C,) broadcasts last axis, scaling columns if C last dim is columns. For eigh, columns are eigenvectors; so v * log_w scales columns correctly.
        # To be explicit we use: v * log_w[np.newaxis, :] or v * log_w
        # Both scale columns in numpy. Keep explicit:
        # scaled = v * log_w[np.newaxis, :] equivalent to v*log_w but we above used v*log_w which already broadcasts on last axis? Let's keep precise:
        # We'll recompute to avoid ambiguity:
        scaled = v * log_w[np.newaxis, :] if v.shape[1] == log_w.shape[0] else v * log_w
        # Actually v shape (C,C), log_w shape (C,) -> v * log_w broadcasts log_w over rows, scaling columns – correct.

        # Use manual: scaled = v @ np.diag(log_w) would be same but slower.
        # Then logm = scaled @ vt
        # To avoid confusion, compute via explicit diag for correctness, but scaled method is faster.
        # We'll compute via scaled @ vt
        logm = scaled @ vt
        # re-symmetrise (removes tiny asymmetry from eig reconstruction)
        logm = (logm + logm.T) * 0.5
        out[i] = logm

    result = out.reshape(orig_shape)
    # ensure float64
    result = result.astype(np.float64, copy=False)
    logger.debug("spd_logm: in_shape=%s out_shape=%s eps=%.1e", orig_shape, result.shape, eps)
    return result


def spd_flatten_upper(spd: np.ndarray) -> np.ndarray:
    """Flatten symmetric matrix to upper-triangular vector.

    For ``(C, C)`` returns ``(K,)`` where ``K = C*(C+1)//2``;
    for ``(..., C, C)`` returns ``(..., K)``. Order is row-major
    ``np.triu_indices(C)`` (ACL 2026, Sec. 4.2: “upper-tri vectorisation”).

    Off-diagonal terms are **not** scaled by ``sqrt(2)``; the plain
    upper-tri is used so that :func:`spd_logm` outputs can be consumed
    directly by a linear projection. If the norm-preserving variant is
    needed, multiply off-diagonal entries by ``sqrt(2)`` after calling.

    Args:
        spd: Symmetric matrix ``(C, C)`` or stack ``(..., C, C)``. Input
            is typically a tangent-space matrix from :func:`spd_logm`.

    Returns:
        Vectorised array.

    References:
        Gowda & Miller, ACL 2026 Findings, Sec. 4.2.
    """
    if not isinstance(spd, np.ndarray):
        raise TypeError(f"spd must be np.ndarray, got {type(spd)}")
    if spd.ndim < 2 or spd.shape[-1] != spd.shape[-2]:
        raise ValueError(f"spd must be (..., C, C), got shape {spd.shape}")

    c = spd.shape[-1]
    triu_rows, triu_cols = np.triu_indices(c)
    k = triu_rows.size  # C*(C+1)//2

    if spd.ndim == 2:
        return spd[triu_rows, triu_cols].astype(np.float64, copy=False)

    # ND case: (..., C, C) -> (..., K)
    orig_shape = spd.shape
    # reshape leading dims to -1 for iteration-free gather
    # Use advanced indexing: we need to gather per matrix.
    # Reshape to (N, C, C) where N = prod(leading dims)
    leading = int(np.prod(orig_shape[:-2])) if spd.ndim > 2 else 1
    flat = spd.reshape(leading, c, c)
    out = np.empty((leading, k), dtype=np.float64)
    for i in range(leading):
        out[i] = flat[i][triu_rows, triu_cols]
    # restore leading shape + K
    new_shape = orig_shape[:-2] + (k,)
    result = out.reshape(new_shape)
    logger.debug("spd_flatten_upper: C=%d K=%d in_shape=%s out_shape=%s", c, k, orig_shape, result.shape)
    return result


# ---------------------------------------------------------------------------
# Riemannian feature helpers
# ---------------------------------------------------------------------------

def spd_riemannian_features(spd: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Tangent-space Riemannian features: ``flatten_upper(logm(spd))``.

    Convenience wrapper combining :func:`spd_logm` and
    :func:`spd_flatten_upper`. For a stack ``(..., C, C)`` returns
    ``(..., K)`` with ``K = C*(C+1)//2`` (e.g., ``C=4 → K=10``).

    Args:
        spd: SPD matrix or stack ``(..., C, C)``.
        eps: Eigenvalue clamp for the log map.

    Returns:
        Riemannian feature vector(s) ``(..., K)`` in the tangent space
        at identity.

    References:
        Gowda & Miller, ACL 2026 Findings, Sec. 4.2, Fig. 2;
        J Neural Eng 2024, Sec. 3.4.
    """
    logm = spd_logm(spd, eps=eps)
    return spd_flatten_upper(logm)


def get_riemannian_features(segment: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Riemannian features directly from a raw segment.

    Computes ``spd_riemannian_features(compute_spd_matrix(segment))``.

    Args:
        segment: Array ``(T, C)``.
        eps: Regulariser for SPD and log map.

    Returns:
        Vector ``(K,)`` with ``K = C*(C+1)//2``.

    References:
        ACL 2026 Findings, Sec. 4.2; J Neural Eng 2024, Sec. 3.4.
    """
    spd = compute_spd_matrix(segment, eps=eps)
    return spd_riemannian_features(spd, eps=eps)


def extract_spd_features(segment: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Alias for :func:`get_riemannian_features`."""
    return get_riemannian_features(segment, eps=eps)


def get_riemannian_features_timevarying(
    segment: np.ndarray,
    fs: float = 250,
    window_ms: int = 50,
    step_ms: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """Time-varying Riemannian features ``(num_windows, K)``.

    Computes sliding SPD stack via :func:`compute_spd_timevarying` then
    maps each SPD to tangent space.

    Args:
        segment: Array ``(T, C)``.
        fs: Sampling rate.
        window_ms: Window length in ms.
        step_ms: Hop in ms.
        eps: SPD regulariser / eigenvalue clamp.

    Returns:
        Array ``(num_windows, K)``.

    References:
        Gowda & Miller, ACL 2026 Findings, Sec. 4.1–4.2.
    """
    spd_seq = compute_spd_timevarying(segment, fs=fs, window_ms=window_ms, step_ms=step_ms, eps=eps)
    return spd_riemannian_features(spd_seq, eps=eps)


# alias for helper naming variants expected by downstream code
extract_riemannian_features = get_riemannian_features
get_spd_features = get_riemannian_features
spd_features = get_riemannian_features
get_riemannian_feature = get_riemannian_features
riemannian_features = get_riemannian_features
extract_riemannian_features_timevarying = get_riemannian_features_timevarying
extract_spd_features_timevarying = get_riemannian_features_timevarying
