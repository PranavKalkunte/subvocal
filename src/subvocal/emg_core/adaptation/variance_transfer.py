"""Inter-Subject Variance Transfer per Yoneda et al., EMBC 2024 (Bayesian GCM).

Share variance (precision) not mean: per-class mean is subject-specific,
precision/covariance is shared across subjects. Bayesian update with
source posterior -> target. Uses Normal-Wishart conjugate prior:

    p(mu_c | Lambda_c) = N(m0, (beta0 Lambda_c)^-1)
    p(Lambda_c) = Wishart(W0, nu0)

Pre-training aggregates source subjects to get posterior for shared Lambda_c.
Transfer scales uncertainty via w_s coefficient (larger w_s = less uncertainty,
more transfer). Target calibration (e.g. 1 trial) updates posterior with
small data, regularized for singularity.

Reference:
    Yoneda & Furui, EMBC 2024 / arXiv:2505.15381 — Inter-Subject Variance
    Transfer Learning for EMG Pattern Classification Based on Bayesian Inference.
    Prior hyperparams: m0=0, beta0=1, nu0=D+1, W0=I.

Numpy (+ optional scipy) only; handles small data via diagonal regularization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "GaussianClassificationModel",
    "VarianceTransferGCM",
    "pretrain_variance_transfer",
    "transfer_to_target",
    "predict",
]

# ---------------------------------------------------------------------------
# optional scipy
# ---------------------------------------------------------------------------

try:
    import scipy.linalg as _sla  # type: ignore[import]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _sla = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

_EPS = 1e-6
_REG = 1e-6


def _ensure_2d(x: np.ndarray, D: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        if D is not None and arr.shape[0] != D:
            # single sample with D features is 1D — reshape to (1,D)
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _inv_reg(mat: np.ndarray, eps: float = _REG) -> np.ndarray:
    """Regularized inverse / pseudo-inverse."""
    n = mat.shape[0]
    reg = mat + eps * np.eye(n, dtype=mat.dtype)
    try:
        return np.linalg.inv(reg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(reg)


def _slogdet_prec(prec: np.ndarray, eps: float = 1e-9) -> float:
    """Log determinant of precision (regularized)."""
    n = prec.shape[0]
    reg = prec + eps * np.eye(n, dtype=prec.dtype)
    try:
        sign, ld = np.linalg.slogdet(reg)
        if sign > 0 and np.isfinite(ld):
            return float(ld)
    except Exception:
        pass
    # fallback via eigenvalues
    try:
        w = np.linalg.eigvalsh(reg)
        w = np.clip(w, 1e-12, None)
        return float(np.sum(np.log(w)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Gaussian Classification Model
# ---------------------------------------------------------------------------

@dataclass
class GaussianClassificationModel:
    """Per-class Gaussian classifier with shared-variance semantics.

    Attributes:
        means: ``(C,D)`` per-class means (subject-specific after transfer).
        precisions: ``(C,D,D)`` per-class precision matrices (shared variance
            origin, after Bayesian update).
        covariances: ``(C,D,D)`` derived covariances (inv precisions).
        classes: Unique label values ``(C,)`` (original dtype preserved).
        priors: Class priors ``(C,)`` summing to 1 (uniform if None).
        n_classes: Number of classes.
        n_features: Feature dimension D.
        class_to_idx: Mapping label -> index.

    Prediction is QDA/LDA-like via Gaussian log-likelihood with precision form:

        log p(x|k) = -0.5*(x-mu_k)^T Lambda_k (x-mu_k)
                     +0.5*log|Lambda_k| -0.5*D*log(2pi)

    plus log prior.

    Reference: Yoneda et al., Sec. II-B GCM.
    """

    means: np.ndarray
    precisions: np.ndarray
    classes: np.ndarray
    priors: np.ndarray | None = None
    covariances: np.ndarray | None = None
    n_classes: int = field(init=False)
    n_features: int = field(init=False)
    class_to_idx: dict[Any, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float64)
        self.precisions = np.asarray(self.precisions, dtype=np.float64)
        self.classes = np.asarray(self.classes)
        if self.means.ndim != 2:
            raise ValueError(f"means must be 2D (C,D), got {self.means.shape}")
        if self.precisions.ndim != 3:
            raise ValueError(f"precisions must be 3D (C,D,D), got {self.precisions.shape}")
        C, D = self.means.shape
        C2, D2, D3 = self.precisions.shape
        if C != C2 or D != D2 or D != D3:
            raise ValueError(f"Shape mismatch means {self.means.shape} vs precisions {self.precisions.shape}")
        if self.classes.shape[0] != C:
            raise ValueError(f"classes len {self.classes.shape[0]} != C {C}")
        self.n_classes = int(C)
        self.n_features = int(D)
        if self.priors is None:
            self.priors = np.full(self.n_classes, 1.0 / max(self.n_classes, 1), dtype=np.float64)
        else:
            self.priors = np.asarray(self.priors, dtype=np.float64)
            if self.priors.shape[0] != self.n_classes:
                raise ValueError(f"priors len {self.priors.shape[0]} != C {C}")
            s = float(self.priors.sum())
            if s <= 0:
                self.priors = np.full(self.n_classes, 1.0 / self.n_classes)
            elif not np.isclose(s, 1.0):
                self.priors = self.priors / s
        if self.covariances is None:
            covs = []
            for k in range(self.n_classes):
                covs.append(_inv_reg(self.precisions[k]))
            self.covariances = np.stack(covs) if covs else np.empty((0, D, D))
        else:
            self.covariances = np.asarray(self.covariances, dtype=np.float64)
        # mapping with proper handling for numpy scalars
        self.class_to_idx = {}
        for i, c in enumerate(self.classes):
            # use Python scalar if possible for dict key stability
            try:
                key = c.item() if hasattr(c, "item") else c
            except Exception:
                key = c
            self.class_to_idx[key] = i
            # also keep original value as key if different
            if c not in self.class_to_idx:
                self.class_to_idx[c] = i  # type: ignore[index]
        # compat aliases for shared variance access
        self.shared_precision = self.precisions  # alias: shared across subjects
        self.shared_variance = self.covariances  # alias
        self.class_means = self.means  # alias
        self.covariances_ = self.covariances  # sklearn-like
        self.precisions_ = self.precisions

    @property
    def shared_covariance(self) -> np.ndarray:
        return self.covariances  # type: ignore[return-value]

    @property
    def variances(self) -> np.ndarray:
        return self.covariances  # type: ignore[return-value]

    def _log_pdf(self, x: np.ndarray, k: int) -> float:
        mu = self.means[k]
        prec = self.precisions[k]
        diff = x - mu
        # mahal = diff^T prec diff
        try:
            mahal = float(diff @ (prec @ diff))
        except Exception:
            mahal = float(np.dot(diff, np.dot(prec, diff)))
        logdet = _slogdet_prec(prec)
        # constant -0.5*D*log(2pi)
        const = -0.5 * self.n_features * np.log(2.0 * np.pi)
        return float(-0.5 * mahal + 0.5 * logdet + const)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for ``X`` shape ``(N,D)`` or ``(D,)``.

        Returns:
            Array ``(N,)`` with same dtype as ``classes``.
        """
        arr = np.asarray(X)
        # handle empty
        if arr.size == 0:
            return np.array([], dtype=self.classes.dtype)
        # ensure 2D
        if arr.ndim == 1:
            # single sample?
            if arr.shape[0] == self.n_features:
                arr = arr.reshape(1, -1)
            else:
                # ambiguous: treat as single row if not matching D but could be (N,) with 1 feature
                arr = arr.reshape(1, -1)
                if arr.shape[1] != self.n_features:
                    if arr.shape[1] < self.n_features:
                        # pad
                        pad = self.n_features - arr.shape[1]
                        arr = np.concatenate([arr, np.zeros((1, pad))], axis=1)
                    else:
                        arr = arr[:, : self.n_features]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[1] != self.n_features:
                if arr.shape[1] < self.n_features:
                    pad = self.n_features - arr.shape[1]
                    arr = np.concatenate([arr, np.zeros((arr.shape[0], pad))], axis=1)
                else:
                    arr = arr[:, : self.n_features]
        else:
            # 2D check dims
            if arr.shape[1] != self.n_features:
                # maybe transposed? if shape[0]==D and shape[1] small, transpose?
                if arr.shape[0] == self.n_features and arr.shape[1] != self.n_features:
                    # keep as is but pad/truncate rows? ambiguous, fallback to transpose only if second dim matches expected samples?
                    pass
                # handle mismatch by padding/truncating columns
                if arr.shape[1] < self.n_features:
                    pad = self.n_features - arr.shape[1]
                    arr = np.concatenate([arr, np.zeros((arr.shape[0], pad))], axis=1)
                elif arr.shape[1] > self.n_features:
                    arr = arr[:, : self.n_features]

        n = arr.shape[0]
        out = np.empty(n, dtype=self.classes.dtype)
        # precompute log priors
        log_priors = np.log(np.maximum(self.priors, 1e-12))
        for i in range(n):
            xi = arr[i]
            if xi.shape[0] != self.n_features:
                if xi.shape[0] < self.n_features:
                    tmp = np.zeros(self.n_features, dtype=np.float64)
                    tmp[: xi.shape[0]] = xi
                    xi = tmp
                else:
                    xi = xi[: self.n_features]
            best = -np.inf
            best_k = 0
            for k in range(self.n_classes):
                lp = self._log_pdf(xi, k) + float(log_priors[k])
                if lp > best:
                    best = lp
                    best_k = k
            out[i] = self.classes[best_k]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior class probabilities ``(N,C)`` via softmax over log joint."""
        arr = _ensure_2d(np.asarray(X), self.n_features)
        if arr.shape[1] != self.n_features:
            if arr.shape[1] < self.n_features:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], self.n_features - arr.shape[1]))], axis=1)
            else:
                arr = arr[:, : self.n_features]
        n = arr.shape[0]
        log_priors = np.log(np.maximum(self.priors, 1e-12))
        probs = np.zeros((n, self.n_classes), dtype=np.float64)
        for i in range(n):
            xi = arr[i]
            lps = np.array([self._log_pdf(xi, k) + float(log_priors[k]) for k in range(self.n_classes)], dtype=np.float64)
            # softmax
            m = float(np.max(lps))
            exps = np.exp(lps - m)
            s = float(np.sum(exps))
            probs[i] = exps / max(s, 1e-12)
        return probs


# ---------------------------------------------------------------------------
# Pretrain: estimate shared variance posterior from source subjects
# ---------------------------------------------------------------------------

def pretrain_variance_transfer(
    source_data: list[np.ndarray],
    source_labels: list[np.ndarray],
) -> dict[str, Any]:
    """Pre-train shared variance posterior from multiple source subjects.

    Pools EMG feature vectors per class across subjects to estimate
    per-class Wishart posterior for shared precision. Means remain
    subject-specific -> not pooled beyond class scatter.

    Args:
        source_data: List of feature arrays per source subject.
            Each element shape ``(N_s, D)`` (2D) or ``(N_s,)`` for D=1.
            Can also be 1D per trial — stacked internally.
        source_labels: List of label arrays per source subject, each
            shape ``(N_s,)`` with same length as corresponding data.

    Returns:
        Posterior dict with keys:
            ``classes`` (np.ndarray ``C``), ``n_classes``, ``n_features``,
            ``W`` (``C,D,D`` Wishart scale), ``nu`` (``C``), ``beta`` (``C``),
            ``m`` (``C,D`` posterior mean estimates), plus prior hyperparams
            ``beta0``, ``m0``, ``nu0``, ``W0`` for downstream transfer.
            Suitable to pass as ``posterior`` to :func:`transfer_to_target`.

    Reference:
        Yoneda et al., Sec. II-C1 Pre-training with source subjects.
        Prior hyperparams fixed: m0=0, beta0=1, nu0=D+1, W0=I.
    """
    if source_data is None or source_labels is None:
        raise ValueError("source_data and source_labels must not be None")
    if len(source_data) == 0 or len(source_labels) == 0:
        raise ValueError("source_data and source_labels must be non-empty")
    if len(source_data) != len(source_labels):
        raise ValueError(f"Length mismatch: {len(source_data)} vs {len(source_labels)}")

    # Collect pooled data
    pooled_data_parts: list[np.ndarray] = []
    pooled_label_parts: list[np.ndarray] = []
    D_infer: int | None = None
    for idx, (d, l) in enumerate(zip(source_data, source_labels, strict=False)):
        d_arr = np.asarray(d, dtype=np.float64)
        l_arr = np.asarray(l)
        if d_arr.size == 0 or l_arr.size == 0:
            logger.debug("pretrain source subject %d empty — skipping", idx)
            continue
        if d_arr.ndim == 1:
            d_arr = d_arr.reshape(-1, 1)
        elif d_arr.ndim > 2:
            d_arr = d_arr.reshape(d_arr.shape[0], -1)
        # now 2D (N,D)
        if d_arr.shape[0] != l_arr.shape[0]:
            raise ValueError(
                f"Subject {idx}: data rows {d_arr.shape[0]} != labels {l_arr.shape[0]} "
                f"(data shape {d_arr.shape}, labels shape {l_arr.shape})"
            )
        if D_infer is None:
            D_infer = int(d_arr.shape[1])
        elif d_arr.shape[1] != D_infer:
            # allow mismatch by padding/truncating to first D? Better raise with handling
            # attempt to make consistent: pad smaller to D_infer
            cur_D = int(d_arr.shape[1])
            if cur_D < D_infer:
                pad = D_infer - cur_D
                d_arr = np.concatenate([d_arr, np.zeros((d_arr.shape[0], pad), dtype=np.float64)], axis=1)
            else:
                # truncate larger to D_infer (keep first D)
                d_arr = d_arr[:, :D_infer]
                logger.warning("Subject %d D=%d != expected %d — truncating", idx, cur_D, D_infer)
        pooled_data_parts.append(d_arr)
        pooled_label_parts.append(l_arr)

    if not pooled_data_parts:
        raise ValueError("No valid source data after filtering")

    all_data = np.vstack(pooled_data_parts)  # (N_total,D)
    all_labels = np.concatenate(pooled_label_parts)
    D = int(all_data.shape[1])
    classes = np.unique(all_labels)
    # sorted unique already
    n_classes = int(classes.shape[0])
    if n_classes == 0:
        raise ValueError("No classes found in source labels")

    # priors per paper
    beta0 = 1.0
    m0 = np.zeros(D, dtype=np.float64)
    nu0 = float(D + 1)
    W0 = np.eye(D, dtype=np.float64)
    # W0 inverse
    try:
        W0_inv = np.linalg.inv(W0 + 1e-12 * np.eye(D))
    except np.linalg.LinAlgError:
        W0_inv = np.linalg.pinv(W0)

    W_list: list[np.ndarray] = []
    nu_list: list[float] = []
    beta_list: list[float] = []
    m_list: list[np.ndarray] = []

    for c in classes:
        mask = all_labels == c
        Xc = all_data[mask]
        Nc = int(Xc.shape[0]) if Xc.size else 0
        if Nc == 0:
            x_bar = m0.copy()
            S = np.zeros((D, D), dtype=np.float64)
        elif Nc == 1:
            x_bar = Xc[0].astype(np.float64)
            S = np.zeros((D, D), dtype=np.float64)
        else:
            x_bar = np.mean(Xc, axis=0).astype(np.float64)
            diff = Xc - x_bar
            S = (diff.T @ diff).astype(np.float64)
        beta_n = float(beta0 + Nc)
        # posterior mean
        if Nc > 0:
            m_n = (beta0 * m0 + Nc * x_bar) / beta_n
        else:
            m_n = m0.copy()
        # Wishart posterior scale
        beta_term = np.zeros((D, D), dtype=np.float64)
        if Nc > 0:
            outer = np.outer(x_bar - m0, x_bar - m0)
            beta_term = (beta0 * Nc / beta_n) * outer
        W_n_inv = W0_inv + S + beta_term
        # regularize to keep invertible (handles small N < D singular S)
        W_n_inv_reg = W_n_inv + _REG * np.eye(D)
        try:
            W_n = np.linalg.inv(W_n_inv_reg)
        except np.linalg.LinAlgError:
            W_n = np.linalg.pinv(W_n_inv_reg)
        # symmetrize
        W_n = (W_n + W_n.T) * 0.5
        nu_n = float(nu0 + Nc)
        W_list.append(W_n)
        nu_list.append(nu_n)
        beta_list.append(beta_n)
        m_list.append(m_n)

    posterior: dict[str, Any] = {
        "classes": classes,
        "n_classes": n_classes,
        "n_features": D,
        "W": np.stack(W_list) if W_list else np.empty((0, D, D)),
        "nu": np.array(nu_list, dtype=np.float64),
        "beta": np.array(beta_list, dtype=np.float64),
        "m": np.stack(m_list) if m_list else np.empty((0, D)),
        "beta0": float(beta0),
        "m0": m0,
        "nu0": float(nu0),
        "W0": W0,
    }
    logger.debug("pretrain_variance_transfer: pooled N=%d D=%d C=%d", all_data.shape[0], D, n_classes)
    return posterior


# ---------------------------------------------------------------------------
# Transfer to target
# ---------------------------------------------------------------------------

def transfer_to_target(
    posterior: dict[str, Any],
    target_calib_data: np.ndarray,
    target_calib_labels: np.ndarray,
    w_s: float = 1.0,
) -> GaussianClassificationModel:
    """Transfer shared variance to target subject with limited calibration.

    Bayesian update using scaled source posterior as prior for shared
    precision, plus target calibration data (e.g. 1 trial). Only variance
    (precision) is transferred; means are re-estimated from target data
    (subject-specific).

    The ``w_s`` coefficient controls uncertainty of the source posterior:
    larger ``w_s`` = less uncertainty = more influence. Scaling keeps
    expected precision ``E[Lambda]=nu*W`` constant while increasing
    concentration ``nu`` with ``w_s`` (variance of Wishart decreases).

        nu_scaled = max(D+1, w_s * nu_src)
        W_scaled  = W_src * (nu_src / nu_scaled)   # keep mean fixed

    For each class, target posterior:

        beta_post = beta0 + N_t
        m_post    = (beta0*m0 + N_t*x_bar_t)/beta_post
        W_post^-1 = W_scaled^-1 + S_t + (beta0*N_t/beta_post)*(x_bar_t-m0)(x_bar_t-m0)^T
        nu_post   = nu_scaled + N_t
        precision = nu_post * W_post

    Handles small data (N_t=0 or 1) with regularization; unseen classes get
    posterior from scaled prior alone.

    Args:
        posterior: Dict from :func:`pretrain_variance_transfer`.
        target_calib_data: Calibration features ``(N_t, D)`` or ``(N_t, )``.
            Single-trial calibration typical (e.g. few samples per class).
        target_calib_labels: Labels ``(N_t,)`` matching data rows.
        w_s: Weight coefficient ``>0`` controlling transferred amount.
            ``1.0`` default (no scaling). Larger -> stronger transfer.

    Returns:
        :class:`GaussianClassificationModel` fitted to target (subject-specific
        means + variance-transferred precisions).

    Reference:
        Yoneda et al., Sec. II-C2 Transfer learning to target, Eq. 13-22.
    """
    if posterior is None or not isinstance(posterior, dict):
        raise ValueError("posterior must be dict from pretrain_variance_transfer")
    if "classes" not in posterior or "W" not in posterior or "nu" not in posterior:
        raise ValueError("posterior dict missing required keys 'classes','W','nu'")
    if w_s is None or not np.isfinite(float(w_s)) or float(w_s) <= 0:
        raise ValueError(f"w_s must be >0 finite, got {w_s}")
    w_s = float(w_s)

    calib_data = np.asarray(target_calib_data, dtype=np.float64)
    calib_labels = np.asarray(target_calib_labels)
    if calib_data.size == 0 or calib_labels.size == 0:
        raise ValueError("target_calib_data and labels must be non-empty")

    # normalize calib_data to 2D (N,D)
    if calib_data.ndim == 1:
        calib_data = calib_data.reshape(-1, 1)
    elif calib_data.ndim > 2:
        calib_data = calib_data.reshape(calib_data.shape[0], -1)
    if calib_data.shape[0] != calib_labels.shape[0]:
        raise ValueError(f"Calibration rows {calib_data.shape[0]} != labels {calib_labels.shape[0]}")

    D_post = int(posterior["n_features"])
    D_calib = int(calib_data.shape[1])
    # handle D mismatch: pad/truncate calib to D_post
    if D_calib != D_post:
        if D_calib < D_post:
            pad = D_post - D_calib
            calib_data = np.concatenate(
                [calib_data, np.zeros((calib_data.shape[0], pad), dtype=np.float64)], axis=1
            )
            logger.warning("transfer: calib D=%d < posterior D=%d — zero-padded", D_calib, D_post)
            D_calib = D_post
        else:
            calib_data = calib_data[:, :D_post]
            logger.warning("transfer: calib D=%d > posterior D=%d — truncated", D_calib, D_post)
            D_calib = D_post
    D = D_post
    classes: np.ndarray = np.asarray(posterior["classes"])
    n_classes: int = int(posterior["n_classes"])
    W_src: np.ndarray = np.asarray(posterior["W"], dtype=np.float64)
    nu_src: np.ndarray = np.asarray(posterior["nu"], dtype=np.float64)
    # prior hyperparams
    beta0: float = float(posterior.get("beta0", 1.0))
    m0: np.ndarray = np.asarray(posterior.get("m0", np.zeros(D, dtype=np.float64)), dtype=np.float64)
    if m0.shape[0] != D:
        m0 = np.zeros(D, dtype=np.float64)

    if W_src.shape[0] != n_classes or nu_src.shape[0] != n_classes:
        raise ValueError(f"Posterior W/nu length {W_src.shape[0]}/{nu_src.shape[0]} != n_classes {n_classes}")

    means: list[np.ndarray] = []
    precisions: list[np.ndarray] = []
    covariances: list[np.ndarray] = []

    for idx, c in enumerate(classes):
        W_c_src = W_src[idx]
        nu_c_src = float(nu_src[idx])

        # scale uncertainty via w_s
        # nu_scaled = clamp(w_s * nu_src, >= D+1)
        nu_scaled = float(max(D + 1.0, w_s * nu_c_src))
        # keep expectation nu*W constant -> W_scaled = W_src * nu_src / nu_scaled
        # handles clipping case where nu_scaled != w_s*nu_src (keeps mean approximately constant)
        if w_s != 1.0:
            # avoid division by zero
            scale_factor = nu_c_src / nu_scaled if nu_scaled != 0 else 1.0
            W_scaled = W_c_src * scale_factor
            # symmetrize
            W_scaled = (W_scaled + W_scaled.T) * 0.5
        else:
            W_scaled = W_c_src

        # target stats for class c
        # mask handles generic label dtypes (int/str)
        # Use vectorized equality; for object arrays works
        try:
            mask = calib_labels == c
            # need to handle case where c is numpy scalar vs python scalar with dtype mismatch?
            # For some dtypes equality may be elementwise with broadcasting? ensure 1D bool
            if mask.ndim != 1:
                mask = np.asarray([x == c for x in calib_labels])
        except Exception:
            mask = np.array([x == c for x in calib_labels], dtype=bool)

        if np.any(mask):
            Xc = calib_data[mask]
            Nt = int(Xc.shape[0])
        else:
            Xc = np.empty((0, D), dtype=np.float64)
            Nt = 0

        # compute posterior for this class
        if Nt == 0:
            # No calibration data for this class — posterior = scaled prior alone
            # mean stays at prior m0 (subject-specific mean unknown -> 0)
            # Alternatively could use m from source? But per paper mean not transferred -> use m0
            m_post = m0.copy()
            # W_post = W_scaled, nu_post = nu_scaled
            # Derive via formula with S=0, beta_term=0: W_post^-1 = W_scaled^-1
            # So W_post = W_scaled; nu_post = nu_scaled
            W_post = W_scaled
            nu_post = nu_scaled
            beta_post = beta0
        elif Nt == 1:
            x_bar = Xc[0].astype(np.float64)
            S_t = np.zeros((D, D), dtype=np.float64)
            beta_post = float(beta0 + Nt)
            # subject-specific mean: use MLE (no shrinkage) for 1-trial calibration
            # paper: means are subject-specific, so prior should be vague for target
            m_post = x_bar.copy()
            beta_term = (beta0 * Nt / beta_post) * np.outer(x_bar - m0, x_bar - m0)
            try:
                W_scaled_inv = _inv_reg(W_scaled, eps=1e-12)  # want true inverse without strong reg? Use small
                # but our _inv_reg adds eps; for Wishart we want precise inv
                # use direct inv with small diag
                W_scaled_inv = np.linalg.inv(W_scaled + 1e-12 * np.eye(D))
            except np.linalg.LinAlgError:
                W_scaled_inv = np.linalg.pinv(W_scaled)
            W_post_inv = W_scaled_inv + S_t + beta_term
            W_post_inv_reg = W_post_inv + _REG * np.eye(D)
            try:
                W_post = np.linalg.inv(W_post_inv_reg)
            except np.linalg.LinAlgError:
                W_post = np.linalg.pinv(W_post_inv_reg)
            W_post = (W_post + W_post.T) * 0.5
            nu_post = float(nu_scaled + Nt)
        else:
            x_bar = np.mean(Xc, axis=0).astype(np.float64)
            diff = Xc - x_bar
            S_t = (diff.T @ diff).astype(np.float64)
            beta_post = float(beta0 + Nt)
            # use empirical mean (MLE) for subject-specific means to avoid shrinkage with small N
            # for larger N, shrinkage is negligible; keep MLE for consistency with 1-trial spec
            m_post = x_bar.copy()
            beta_term = (beta0 * Nt / beta_post) * np.outer(x_bar - m0, x_bar - m0)
            try:
                W_scaled_inv = np.linalg.inv(W_scaled + 1e-12 * np.eye(D))
            except np.linalg.LinAlgError:
                W_scaled_inv = np.linalg.pinv(W_scaled)
            W_post_inv = W_scaled_inv + S_t + beta_term
            W_post_inv_reg = W_post_inv + _REG * np.eye(D)
            try:
                W_post = np.linalg.inv(W_post_inv_reg)
            except np.linalg.LinAlgError:
                W_post = np.linalg.pinv(W_post_inv_reg)
            W_post = (W_post + W_post.T) * 0.5
            nu_post = float(nu_scaled + Nt)

        # expected precision = nu_post * W_post
        # Ensure PD by regularizing W_post if needed
        precision = nu_post * W_post
        # symmetrize
        precision = (precision + precision.T) * 0.5
        # add tiny diagonal to ensure positive definite for prediction stability
        # Check eigenvalues; add reg if negative
        try:
            # quick check: try cholesky
            if _SCIPY_AVAILABLE:
                # use scipy to test PD via cholesky?
                pass
            # eigenvalue floor
            w = np.linalg.eigvalsh(precision)
            if np.any(w <= 1e-9):
                precision = precision + (1e-6 - float(np.min(w)) + 1e-9) * np.eye(D)
        except Exception:
            precision = precision + 1e-6 * np.eye(D)

        # covariance for completeness
        cov = _inv_reg(precision, eps=_REG)
        cov = (cov + cov.T) * 0.5

        means.append(m_post.astype(np.float64))
        precisions.append(precision.astype(np.float64))
        covariances.append(cov.astype(np.float64))

    means_arr = np.stack(means) if means else np.empty((0, D), dtype=np.float64)
    precisions_arr = np.stack(precisions) if precisions else np.empty((0, D, D), dtype=np.float64)
    covariances_arr = np.stack(covariances) if covariances else np.empty((0, D, D), dtype=np.float64)

    # priors: empirical from calibration with Laplace smoothing (avoid zero)
    priors = np.zeros(n_classes, dtype=np.float64)
    for idx, c in enumerate(classes):
        try:
            mask = calib_labels == c
            if mask.ndim != 1:
                mask = np.array([x == c for x in calib_labels], dtype=bool)
            cnt = int(np.sum(mask))
        except Exception:
            cnt = int(sum(1 for x in calib_labels if x == c))
        priors[idx] = float(cnt)
    if priors.sum() > 0:
        # Laplace smoothing alpha=1
        priors = (priors + 1.0) / (priors.sum() + n_classes)
    else:
        priors = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)

    model = GaussianClassificationModel(
        means=means_arr,
        precisions=precisions_arr,
        covariances=covariances_arr,
        classes=classes,
        priors=priors,
    )
    logger.debug(
        "transfer_to_target: w_s=%.3f calib N=%d D=%d -> model C=%d",
        w_s, calib_data.shape[0], D, n_classes,
    )
    return model


# ---------------------------------------------------------------------------
# predict wrapper
# ---------------------------------------------------------------------------

def predict(model: GaussianClassificationModel, test_data: np.ndarray) -> np.ndarray:
    """Predict labels for test_data using a :class:`GaussianClassificationModel`.

    Args:
        model: Fitted :class:`GaussianClassificationModel`.
        test_data: Features ``(N,D)`` or ``(D,)`` single sample.
            Can be list of arrays; converted via ``np.asarray``.

    Returns:
        Predicted labels ``(N,)`` ndarray.
    """
    if not isinstance(model, GaussianClassificationModel):
        raise TypeError(f"model must be GaussianClassificationModel, got {type(model)}")
    arr = np.asarray(test_data)
    # handle case test_data is list of arrays? Already converted
    # empty check
    if arr.size == 0:
        return np.array([], dtype=model.classes.dtype)
    # if arr is object array containing arrays (jagged), try to stack
    if arr.dtype == object and arr.ndim == 1:
        # list of vectors with maybe varying? attempt to stack
        try:
            arr = np.vstack([np.asarray(x, dtype=np.float64).reshape(1, -1) for x in arr])  # type: ignore[union-attr]
        except Exception:
            arr = np.asarray(list(test_data), dtype=np.float64)
    return model.predict(arr)


# ---------------------------------------------------------------------------
# High-level wrapper class
# ---------------------------------------------------------------------------

class VarianceTransferGCM:
    """End-to-end variance-transfer GCM wrapper (Yoneda et al.).

    Convenience class that holds the source posterior and target model,
    delegating to :func:`pretrain_variance_transfer`,
    :func:`transfer_to_target`, and :func:`predict`.

    Example:
        vt = VarianceTransferGCM()
        post = vt.pretrain(source_data, source_labels)
        model = vt.transfer(post, calib_data, calib_labels, w_s=1.0)
        preds = vt.predict(model, test_data)  # or vt.predict(test_data)

        # or functional:
        post = pretrain_variance_transfer(src_data, src_labels)
        model = transfer_to_target(post, calib_data, calib_labels, w_s=1.0)
        preds = predict(model, test_data)

    Attributes:
        posterior: Last computed posterior dict (or None).
        model: Last fitted :class:`GaussianClassificationModel` (or None).

    Reference:
        Yoneda et al., EMBC 2024 / arXiv:2505.15381.
    """

    def __init__(self) -> None:
        self.posterior: dict[str, Any] | None = None
        self.model: GaussianClassificationModel | None = None

    def pretrain(
        self,
        source_data: list[np.ndarray],
        source_labels: list[np.ndarray],
    ) -> dict[str, Any]:
        """Pre-train on source subjects (delegates to :func:`pretrain_variance_transfer`)."""
        self.posterior = pretrain_variance_transfer(source_data, source_labels)
        return self.posterior

    def transfer(
        self,
        posterior: dict[str, Any] | None,
        target_calib_data: np.ndarray,
        target_calib_labels: np.ndarray,
        w_s: float = 1.0,
    ) -> GaussianClassificationModel:
        """Transfer to target (delegates to :func:`transfer_to_target`)."""
        if posterior is None:
            posterior = self.posterior
        if posterior is None:
            raise ValueError("No posterior available — call pretrain first or provide posterior")
        self.model = transfer_to_target(posterior, target_calib_data, target_calib_labels, w_s=w_s)
        return self.model

    # alias names for spec compatibility
    def adapt(
        self,
        target_calib_data: np.ndarray,
        target_calib_labels: np.ndarray,
        w_s: float = 1.0,
        posterior: dict[str, Any] | None = None,
    ) -> GaussianClassificationModel:
        return self.transfer(posterior, target_calib_data, target_calib_labels, w_s=w_s)

    def fit_target(
        self,
        posterior: dict[str, Any],
        target_calib_data: np.ndarray,
        target_calib_labels: np.ndarray,
        w_s: float = 1.0,
    ) -> GaussianClassificationModel:
        return self.transfer(posterior, target_calib_data, target_calib_labels, w_s=w_s)

    def predict(self, test_data: np.ndarray, model: GaussianClassificationModel | None = None) -> np.ndarray:
        """Predict with stored or provided model."""
        m = model if model is not None else self.model
        if m is None:
            raise ValueError("No model available — call transfer/adapt first or provide model")
        return predict(m, test_data)

    # Legacy alias matching spec: transfer_to_target naming
    def transfer_to_target(
        self,
        posterior: dict[str, Any],
        target_calib_data: np.ndarray,
        target_calib_labels: np.ndarray,
        w_s: float = 1.0,
    ) -> GaussianClassificationModel:
        return self.transfer(posterior, target_calib_data, target_calib_labels, w_s=w_s)

    # alias for spec function name inside class
    def pretrain_variance_transfer(
        self,
        source_data: list[np.ndarray],
        source_labels: list[np.ndarray],
    ) -> dict[str, Any]:
        return self.pretrain(source_data, source_labels)

    def fit(self, *a: Any, **k: Any) -> GaussianClassificationModel:
        return self.transfer(*a, **k)

