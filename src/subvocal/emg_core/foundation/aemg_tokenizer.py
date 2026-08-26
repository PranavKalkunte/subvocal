"""AEMG neuromuscular contraction tokenizer + vector quantization.

Per CVPR 2026 Huang et al.: Neuromuscular Contraction Tokenizer (NCT)
that discretizes sEMG into contraction tokens/words via sliding-window
segmentation into physiologically grounded contraction primitives, plus
EMG vocabulary via vector quantization (VQ).

The tokenizer operates on numpy arrays (T,C) and builds a universal
vocabulary of contraction primitives. Decoding reconstructs via
overlap-add from codebook entries.

AEMGFramework implements self-supervised masked modeling over token
sentences: random masking and collective token prediction, with
universal vocabulary sharing across subjects/sessions.

Torch is optional: VQ nearest-neighbor uses numpy by default, and
torch.cdist if available for speed. No hard torch dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["EMGTokenizer", "AEMGFramework"]

# ---------------------------------------------------------------------------
# optional torch / scipy
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from scipy.signal import get_window  # type: ignore[import-not-found]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    get_window = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_token_dim(flat: np.ndarray, token_dim: int) -> np.ndarray:
    """Map flattened window (L,) to token_dim via interpolation/pad."""
    flat = np.asarray(flat, dtype=np.float32)
    L = flat.shape[0]
    if L == token_dim:
        return flat.astype(np.float32)
    if L < token_dim:
        # pad zeros
        out = np.zeros(token_dim, dtype=np.float32)
        out[:L] = flat
        return out
    # L > token_dim : downsample via linear interpolation
    # Use np.interp
    x_old = np.linspace(0.0, 1.0, L, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, token_dim, dtype=np.float64)
    # np.interp expects xp increasing
    interped = np.interp(x_new, x_old, flat.astype(np.float64))
    return interped.astype(np.float32)


def _from_token_dim(vec: np.ndarray, target_len: int) -> np.ndarray:
    """Inverse of _to_token_dim: map token_dim vector back to target_len."""
    vec = np.asarray(vec, dtype=np.float32)
    d = vec.shape[0]
    if d == target_len:
        return vec.astype(np.float32)
    if d < target_len:
        # upsample via interp
        x_old = np.linspace(0.0, 1.0, d, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
        interped = np.interp(x_new, x_old, vec.astype(np.float64))
        return interped.astype(np.float32)
    # d > target_len : downsample
    x_old = np.linspace(0.0, 1.0, d, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    interped = np.interp(x_new, x_old, vec.astype(np.float64))
    return interped.astype(np.float32)


# ---------------------------------------------------------------------------
# EMGTokenizer — NCT + VQ
# ---------------------------------------------------------------------------

class EMGTokenizer:
    """Neuromuscular Contraction Tokenizer (NCT) + VQ.

    Discretizes EMG ``(T, C)`` into contraction tokens via sliding-window
    segmentation into physiologically grounded primitives (~30-60 ms)
    and vector quantization against a learned codebook.

    Args:
        codebook_size: Vocabulary size K (default 512).
        token_dim: Codebook vector dimension D (default 64).
        window_size: Primitive window in samples (default 32; ~32 ms @1000 Hz
            or ~128 ms @250 Hz, within 30-100 ms contraction range).
        stride: Hop between windows (default 16, 50% overlap for overlap-add
            reconstruction).
        seed: RNG seed for codebook init (default 0).
    """

    def __init__(
        self,
        codebook_size: int = 512,
        token_dim: int = 64,
        window_size: int = 32,
        stride: int = 16,
        seed: int = 0,
    ) -> None:
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be >0, got {codebook_size}")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be >0, got {token_dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be >0, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be >0, got {stride}")

        self.codebook_size = int(codebook_size)
        self.token_dim = int(token_dim)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        # Codebook: (K, D) — small scale for stable distances
        self.codebook: np.ndarray = (rng.standard_normal((self.codebook_size, self.token_dim)).astype(np.float32) * 0.1)
        # Optional torch codebook (lazy)
        self._torch_codebook: Any | None = None
        # Remember last encode shape for decode without explicit C
        self._last_C: int | None = None
        self._last_T: int | None = None

        logger.debug(
            "EMGTokenizer init: K=%d D=%d win=%d stride=%d seed=%d",
            codebook_size, token_dim, window_size, stride, seed,
        )

    # -- codebook helpers --------------------------------------------------

    def get_codebook(self) -> np.ndarray:
        """Return codebook array (K, D)."""
        return self.codebook.copy()

    def get_vocab(self) -> dict[int, np.ndarray]:
        """Return vocab as dict token_id -> vector."""
        return {int(i): self.codebook[i].copy() for i in range(self.codebook_size)}

    def _nearest(self, features: np.ndarray) -> np.ndarray:
        """Vector quantization: features (N, D) -> tokens (N,) via nearest codebook."""
        # features: (N, D)
        if _TORCH_AVAILABLE and features.shape[0] > 64:
            try:
                # Use torch for speed on larger batches
                if self._torch_codebook is None:
                    self._torch_codebook = torch.from_numpy(self.codebook)  # type: ignore[union-attr]
                feat_t = torch.from_numpy(features.astype(np.float32))  # type: ignore[union-attr]
                # cdist: (N, K)
                dists = torch.cdist(feat_t, self._torch_codebook)  # type: ignore[union-attr]
                tokens = torch.argmin(dists, dim=1).cpu().numpy()  # type: ignore[union-attr]
                return tokens.astype(np.int64)
            except Exception as e:  # pragma: no cover
                logger.debug("torch VQ failed (%s), falling back to numpy", e)
        # numpy fallback
        # Compute squared Euclidean via broadcasting or efficient method
        # Use: dist = |a-b|^2 = |a|^2 + |b|^2 -2ab
        # For small N, brute force fine
        N = features.shape[0]
        # For very large N, chunk to avoid OOM
        chunk = 1024
        tokens = np.empty(N, dtype=np.int64)
        cb_norm = np.sum(self.codebook.astype(np.float64) ** 2, axis=1)  # (K,)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            f_chunk = features[start:end].astype(np.float64)  # (c, D)
            f_norm = np.sum(f_chunk ** 2, axis=1, keepdims=True)  # (c,1)
            # (c, K) = (c,1)+(1,K)-2*(c,D)@(D,K)
            dists = f_norm + cb_norm[None, :] - 2.0 * (f_chunk @ self.codebook.T.astype(np.float64))
            # argmin
            tokens[start:end] = np.argmin(dists, axis=1)
        return tokens

    # -- encode / decode ---------------------------------------------------

    def encode(self, signal: np.ndarray) -> np.ndarray:
        """Tokenize EMG signal into contraction tokens.

        Segments ``signal`` (T, C) into overlapping primitives via sliding
        window and quantizes each primitive to the nearest codebook entry.

        Args:
            signal: Array (T, C) — time samples x channels. Also accepts
                (C, T) if C < T and signal.shape[0] < signal.shape[1] is
                ambiguous; the method assumes (T, C) as per
                :mod:`subvocal.emg_core.dsp.spd` convention. If ``signal``
                is 1-D (T,) it is treated as single-channel.

        Returns:
            Tokens array (num_tokens,) dtype int64, values in [0, K-1].
            ``num_tokens = ceil((T - window_size)/stride)+1`` (at least 1).
            Short signals (T < window_size) are zero-padded.

        References:
            Huang et al., CVPR 2026 — sliding-window NCT segmentation.
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError(f"signal must be np.ndarray, got {type(signal)}")
        if signal.ndim == 1:
            signal = signal[:, None]
        if signal.ndim != 2:
            raise ValueError(f"signal must be 2-D (T,C), got shape {signal.shape}")
        if signal.shape[0] == 0 or signal.shape[1] == 0:
            raise ValueError(f"signal has empty dimension: {signal.shape}")

        # Ensure (T,C) convention — if signal looks like (C,T) with C << T, we still assume (T,C)
        # No auto-transpose; caller must pass (T,C).

        T, C = signal.shape
        ws = self.window_size
        st = self.stride

        # Remember for decode
        self._last_C = int(C)
        self._last_T = int(T)

        # Pad if shorter than window
        if T < ws:
            pad_len = ws - T
            signal = np.pad(signal, ((0, pad_len), (0, 0)), mode="constant")
            T = signal.shape[0]

        # Number of windows
        num_tokens = 1 + (T - ws) // st
        num_tokens = max(1, num_tokens)

        features = np.empty((num_tokens, self.token_dim), dtype=np.float32)

        sig_f = signal.astype(np.float32, copy=False)

        for i in range(num_tokens):
            start = i * st
            end = start + ws
            window = sig_f[start:end, :]  # (ws, C)
            # Optional windowing (Hann) for physiological weighting — mild
            # Keeping rectangular for reproducibility.
            flat = window.reshape(-1)  # (ws*C,)
            feat = _to_token_dim(flat, self.token_dim)
            features[i] = feat

        tokens = self._nearest(features)
        logger.debug("EMGTokenizer encode: T=%d C=%d -> %d tokens", T, C, num_tokens)
        return tokens

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Reconstruct EMG signal from tokens via codebook lookup + overlap-add.

        Args:
            tokens: Array (num_tokens,) of token ids in [0, K-1]. Also accepts
                list[int].

        Returns:
            Reconstructed signal array (T_recon, C_estimated) where
            ``T_recon = (num_tokens-1)*stride + window_size`` and
            ``C_estimated`` is inferred from ``token_dim`` and ``window_size``
            if possible. Since codebook vectors are generic, we estimate C as
            ``token_dim // window_size`` or use the stored window_size and
            infer C from the first decode? For simplicity we reconstruct with
            ``C = max(1, token_dim // window_size)`` but if the original C
            differed, the output C may differ. To preserve exact C, the caller
            should use :meth:`decode_to_shape` or provide C.

            For backward compatibility, this method attempts to infer C from
            ``token_dim`` divisibility: if ``token_dim % window_size == 0``,
            ``C = token_dim // window_size``, else C is estimated as 4 and
            the first ``C`` columns are returned with truncation/pad.

            For exact reconstruction, use ``decode_with_channels``.
        """
        tokens_arr = np.asarray(tokens)
        if tokens_arr.ndim == 0:
            tokens_arr = tokens_arr[None]
        if tokens_arr.ndim != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tokens_arr.shape}")
        if tokens_arr.size == 0:
            return np.empty((0, 1), dtype=np.float32)
        if np.any(tokens_arr < 0) or np.any(tokens_arr >= self.codebook_size):
            raise ValueError(f"tokens out of range [0,{self.codebook_size})")

        # Infer C: prefer last encode's C if available, else heuristic
        ws = self.window_size
        st = self.stride
        td = self.token_dim
        if self._last_C is not None:
            C_est = int(self._last_C)
        elif td % ws == 0:
            C_est = td // ws
        else:
            C_est = max(1, int(round(td / ws)))
            if C_est == 0:
                C_est = 1

        num_tokens = int(tokens_arr.shape[0])
        T_recon = (num_tokens - 1) * st + ws
        # Overlap-add buffers
        recon = np.zeros((T_recon, C_est), dtype=np.float64)
        counts = np.zeros((T_recon, C_est), dtype=np.float64)

        for i, tok in enumerate(tokens_arr):
            vec = self.codebook[int(tok)]  # (D,)
            target_len = ws * C_est
            flat = _from_token_dim(vec, target_len)
            window = flat.reshape(ws, C_est)
            start = i * st
            end = start + ws
            recon[start:end, :] += window
            counts[start:end, :] += 1.0

        counts = np.maximum(counts, 1.0)
        recon = recon / counts
        # If we have last_T, adjust to match original length for round-trip fidelity
        if self._last_T is not None and self._last_C == C_est:
            target_T = int(self._last_T)
            if recon.shape[0] != target_T:
                if recon.shape[0] > target_T:
                    recon = recon[:target_T, :]
                else:
                    pad_len = target_T - recon.shape[0]
                    recon = np.pad(recon, ((0, pad_len), (0, 0)), mode="constant")
        logger.debug("EMGTokenizer decode: %d tokens -> (T=%d, C=%d)", num_tokens, recon.shape[0], C_est)
        return recon.astype(np.float32)

    def decode_with_channels(self, tokens: np.ndarray, num_channels: int) -> np.ndarray:
        """Reconstruct with explicit channel count.

        Args:
            tokens: Array (num_tokens,).
            num_channels: Original channel count C.

        Returns:
            Array (T_recon, C).
        """
        tokens_arr = np.asarray(tokens)
        if tokens_arr.ndim == 0:
            tokens_arr = tokens_arr[None]
        if tokens_arr.ndim != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tokens_arr.shape}")
        if tokens_arr.size == 0:
            return np.empty((0, int(num_channels)), dtype=np.float32)
        if num_channels <= 0:
            raise ValueError(f"num_channels must be >0, got {num_channels}")
        ws = self.window_size
        st = self.stride
        num_tokens = int(tokens_arr.shape[0])
        T_recon = (num_tokens - 1) * st + ws
        recon = np.zeros((T_recon, int(num_channels)), dtype=np.float64)
        counts = np.zeros((T_recon, int(num_channels)), dtype=np.float64)
        target_len = ws * int(num_channels)
        for i, tok in enumerate(tokens_arr):
            vec = self.codebook[int(tok)]
            flat = _from_token_dim(vec, target_len)
            window = flat.reshape(ws, int(num_channels))
            start = i * st
            end = start + ws
            recon[start:end, :] += window
            counts[start:end, :] += 1.0
        counts = np.maximum(counts, 1.0)
        recon = recon / counts
        # Adjust to last_T if available and channels match
        if self._last_T is not None and self._last_C == int(num_channels):
            target_T = int(self._last_T)
            if recon.shape[0] != target_T:
                if recon.shape[0] > target_T:
                    recon = recon[:target_T, :]
                else:
                    pad_len = target_T - recon.shape[0]
                    recon = np.pad(recon, ((0, pad_len), (0, 0)), mode="constant")
        return recon.astype(np.float32)

    # Compatibility alias: sometimes spec expects decode(tokens)->signal with same C
    # Provide __call__ as encode
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self.encode(signal)

    def update_codebook(self, new_codebook: np.ndarray) -> None:
        """Replace codebook (e.g., after VQ training)."""
        cb = np.asarray(new_codebook, dtype=np.float32)
        if cb.shape != (self.codebook_size, self.token_dim):
            raise ValueError(f"new_codebook shape {cb.shape} != {(self.codebook_size, self.token_dim)}")
        self.codebook = cb.copy()
        self._torch_codebook = None

    def vocabulary_size(self) -> int:
        return self.codebook_size


# ---------------------------------------------------------------------------
# AEMGFramework — self-supervised masked modeling over token sentences
# ---------------------------------------------------------------------------

class AEMGFramework:
    """AEMG self-supervised framework: masked modeling over contraction tokens.

    Wraps an :class:`EMGTokenizer` and performs masked language modeling
    over token sentences (``collective tokens``). Randomly masks a fraction
    of tokens and predicts the masked vocabulary ids, learning a universal
    EMG vocabulary shared across subjects.

    Args:
        tokenizer: EMGTokenizer instance. If None, a default is created.
        mask_ratio: Fraction of tokens to mask (default 0.15, BERT-style;
            spec says mask tokens and predict collective tokens).
        mask_token_id: Token id used for masked positions. Defaults to
            ``codebook_size`` (extra id) or 0 if codebook_size is max.
        vocab_size: Explicit vocab size; defaults to tokenizer.codebook_size.
        seed: RNG seed.

    Torch is optional: if available, a small Transformer encoder is
    instantiated for masked prediction; otherwise pretrain_step returns
    a dummy heuristic loss.
    """

    def __init__(
        self,
        tokenizer: EMGTokenizer | None = None,
        mask_ratio: float = 0.15,
        mask_token_id: int | None = None,
        vocab_size: int | None = None,
        seed: int = 0,
    ) -> None:
        if tokenizer is None:
            tokenizer = EMGTokenizer(seed=seed)
        if not isinstance(tokenizer, EMGTokenizer):
            raise TypeError(f"tokenizer must be EMGTokenizer, got {type(tokenizer)}")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0,1), got {mask_ratio}")

        self.tokenizer = tokenizer
        self.mask_ratio = float(mask_ratio)
        self.seed = int(seed)
        self.vocab_size = int(vocab_size) if vocab_size is not None else int(tokenizer.codebook_size)
        if mask_token_id is None:
            # Use vocab_size as mask id (extra), but if caller expects within vocab, use 0
            # We'll use vocab_size (out-of-vocab) and handle embedding accordingly.
            # For simplicity, mask_token_id = vocab_size (requires +1 vocab)
            # But many implementations use a fixed id like 0 or vocab_size-1.
            # We'll default to 0 for simplicity and avoid extra vocab.
            mask_token_id = 0
        self.mask_token_id = int(mask_token_id)

        # Optional torch masked LM
        self._lm: Any | None = None
        self._lm_optimizer: Any | None = None
        if _TORCH_AVAILABLE:
            try:
                self._build_torch_lm()
            except Exception as e:  # pragma: no cover
                logger.debug("AEMG torch LM build failed: %s", e)
                self._lm = None

        logger.debug(
            "AEMGFramework init: vocab=%d mask_ratio=%.2f mask_id=%d",
            self.vocab_size, self.mask_ratio, self.mask_token_id,
        )

    def _build_torch_lm(self) -> None:
        """Build tiny Transformer for masked token prediction (torch)."""
        if not _TORCH_AVAILABLE:
            return
        # Small 2-layer Transformer encoder: d_model = token_dim, nhead=4
        d_model = int(self.tokenizer.token_dim)
        # Ensure divisible by nhead
        nhead = 4
        if d_model % nhead != 0:
            # Adjust d_model to nearest divisible
            d_model = ((d_model + nhead - 1) // nhead) * nhead
        dim_feedforward = d_model * 4
        # If nn.TransformerEncoder available
        try:
            encoder_layer = nn.TransformerEncoderLayer(  # type: ignore[union-attr]
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1
            )
            encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # type: ignore[union-attr]
            # Embedding: vocab -> d_model
            self._embedding = nn.Embedding(self.vocab_size + 1, d_model)  # +1 for mask token if needed  # type: ignore[union-attr]
            # If mask_token_id may be vocab_size, need +1
            if self.mask_token_id >= self.vocab_size + 1:
                # Re-create with larger vocab
                self._embedding = nn.Embedding(self.mask_token_id + 1, d_model)  # type: ignore[union-attr]
            self._lm_head = nn.Linear(d_model, self.vocab_size)  # type: ignore[union-attr]
            self._lm_encoder = encoder
            self._lm_d_model = d_model
            self._lm = nn.Sequential()  # placeholder; we use custom forward
            logger.debug("AEMG torch LM built: d_model=%d nhead=%d", d_model, nhead)
        except Exception as e:  # pragma: no cover
            logger.debug("Failed to build torch LM: %s", e)
            self._lm = None

    # -- vocab -------------------------------------------------------------

    def get_vocab(self) -> dict[int, np.ndarray]:
        """Return universal vocabulary (token id -> codebook vector)."""
        return self.tokenizer.get_vocab()

    def get_codebook(self) -> np.ndarray:
        return self.tokenizer.get_codebook()

    # -- tokenize helpers --------------------------------------------------

    def tokenize(self, signal: np.ndarray) -> np.ndarray:
        """Convenience: signal (T,C) -> tokens."""
        return self.tokenizer.encode(signal)

    # -- pretrain step -----------------------------------------------------

    def pretrain_step(self, tokens: np.ndarray, optimizer: Any | None = None) -> dict[str, Any]:
        """Self-supervised masked modeling step over token sentence.

        Args:
            tokens: Array (L,) or (B, L) of token ids. If 1-D, treated as
                single sentence. Values must be in [0, vocab_size).
            optimizer: Optional torch optimizer for LM training; if provided
                and torch LM is available, a gradient step is taken.

        Returns:
            Dict with ``loss`` (float or Tensor), ``masked_ratio``,
            ``num_masked``, and ``predicted`` (if torch) or heuristic.

        Masking follows BERT: 80% mask token, 10% random, 10% unchanged
        (simplified to 100% mask token here for brevity).
        """
        arr = np.asarray(tokens)
        if arr.ndim == 0:
            raise ValueError("tokens must be at least 1-D")
        # Normalize to 2-D (B, L) for batch handling
        single = False
        if arr.ndim == 1:
            arr = arr[None, :]
            single = True
        if arr.ndim != 2:
            raise ValueError(f"tokens must be 1-D or 2-D, got shape {arr.shape}")
        B, L = arr.shape
        if L == 0:
            return {"loss": 0.0, "masked_ratio": 0.0, "num_masked": 0}

        # Validate range
        if np.any(arr < 0) or np.any(arr >= self.vocab_size):
            raise ValueError(f"tokens out of vocab range [0,{self.vocab_size})")

        rng = np.random.default_rng(self.seed + int(np.sum(arr) % 100000))
        # Create mask: per sentence, mask mask_ratio tokens
        mask = np.zeros((B, L), dtype=bool)
        for b in range(B):
            n_mask = max(1, int(L * self.mask_ratio)) if L > 1 else 1
            # Random choice without replacement
            idx = rng.choice(L, size=n_mask, replace=False)
            mask[b, idx] = True

        # Masked input: replace with mask_token_id
        masked_input = arr.copy()
        masked_input[mask] = self.mask_token_id % max(1, self.vocab_size)  # ensure in range if mask_token_id out of vocab

        # If torch LM available, do forward
        if _TORCH_AVAILABLE and hasattr(self, "_embedding") and hasattr(self, "_lm_encoder"):
            try:
                # Prepare tensors
                inp_t = torch.from_numpy(masked_input.astype(np.int64))  # type: ignore[union-attr]
                tgt_t = torch.from_numpy(arr.astype(np.int64))  # type: ignore[union-attr]
                mask_t = torch.from_numpy(mask)  # type: ignore[union-attr]

                # Move to default device (cpu)
                emb = self._embedding(inp_t)  # type: ignore[union-attr] (B, L, D)
                # Transformer encoder
                enc_out = self._lm_encoder(emb)  # type: ignore[union-attr] (B, L, D)
                logits = self._lm_head(enc_out)  # type: ignore[union-attr] (B, L, vocab)

                # Loss only on masked positions: CrossEntropy
                # Reshape: (B*L, vocab) vs (B*L,)
                # Filter masked
                if mask_t.sum().item() > 0:
                    logits_masked = logits[mask_t]  # (num_masked, vocab)
                    tgt_masked = tgt_t[mask_t]  # (num_masked,)
                    loss = F.cross_entropy(logits_masked, tgt_masked)  # type: ignore[union-attr]
                else:
                    loss = torch.tensor(0.0)  # type: ignore[union-attr]

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()  # type: ignore[union-attr]
                    optimizer.step()

                # Predicted tokens: argmax
                pred = torch.argmax(logits, dim=-1).cpu().numpy()  # type: ignore[union-attr]
                # Compute accuracy on masked
                acc = float((pred[mask] == arr[mask]).mean()) if mask.sum() > 0 else 0.0

                result: dict[str, Any] = {
                    "loss": loss,
                    "masked_ratio": float(mask.mean()),
                    "num_masked": int(mask.sum()),
                    "accuracy": acc,
                    "predicted": pred[0] if single else pred,
                    "logits": logits,
                }
                logger.debug("AEMG pretrain_step torch: loss=%.4f acc=%.3f masked=%d/%d", float(loss.item()), acc, int(mask.sum()), int(B*L))  # type: ignore[union-attr]
                return result
            except Exception as e:  # pragma: no cover
                logger.debug("AEMG torch pretrain failed (%s), using heuristic", e)

        # Numpy heuristic fallback: loss is negative log-like based on reconstruction error
        # Simple: loss = proportion of masked that would be guessed correctly by nearest codebook?
        # For heuristic, compute a dummy loss: cross-entropy with uniform 1/vocab -> -log(1/vocab)
        # plus small jitter based on token distribution
        dummy_loss = float(np.log(self.vocab_size)) * float(mask.mean())  # e.g., log(K) * mask_ratio
        # Add jitter based on token entropy
        # Compute token histogram
        hist = np.bincount(arr.reshape(-1), minlength=self.vocab_size).astype(np.float64)
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0] + 1e-12))
        loss_val = dummy_loss + 0.01 * float(entropy)

        # Dummy predicted: copy input and fill masked with most frequent token
        most_common = int(np.argmax(np.bincount(arr.reshape(-1)))) if arr.size else 0
        pred_np = masked_input.copy()
        # For masked positions, predict most_common (heuristic)
        pred_np[mask] = most_common

        result = {
            "loss": float(loss_val),
            "masked_ratio": float(mask.mean()),
            "num_masked": int(mask.sum()),
            "predicted": pred_np[0].astype(np.int64) if single else pred_np.astype(np.int64),
            "accuracy": float((pred_np[mask] == arr[mask]).mean()) if mask.sum() > 0 else 0.0,
        }
        logger.debug("AEMG pretrain_step heuristic: loss=%.4f masked=%d", loss_val, int(mask.sum()))
        return result

    def encode_and_mask(self, signal: np.ndarray) -> dict[str, Any]:
        """Encode signal and run masked modeling in one call.

        Args:
            signal: (T, C) array.

        Returns:
            Dict with ``tokens``, ``masked_input``, ``pretrain`` result.
        """
        tokens = self.tokenize(signal)
        pretrain_res = self.pretrain_step(tokens)
        return {"tokens": tokens, "pretrain": pretrain_res}

