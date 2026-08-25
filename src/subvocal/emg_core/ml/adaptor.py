"""EMG adaptor mapping sEMG features to LLM embedding space.

Per Mohapatra et al., ACL 2025: a lightweight two-layer MLP maps EMG
features (112 handcrafted or 768 speech-encoder style) to the frozen LLM
input dimension (3072 for Llama-3.2-3B, configurable).

References
----------
* Mohapatra et al., ACL 2025.
* Jou et al., 2006; Gaddy & Klein, 2020 (handcrafted feature basis).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from subvocal.core.interfaces import LLMProvider
from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = ["EMGAdaptor", "EMGAdaptorProvider"]

# ---------------------------------------------------------------------------
# lazy torch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - missing torch path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            "torch is required for EMGAdaptor. Install with 'pip install subvocal[ml]'"
        )


# ---------------------------------------------------------------------------
# EMGAdaptor
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class EMGAdaptor(nn.Module):  # type: ignore[no-redef]
        """Two-layer MLP adaptor: EMG features → LLM embedding.

        Architecture (ACL 2025): ``Linear(input_dim → hidden_dim) → ReLU →
        Dropout → Linear(hidden_dim → output_dim)``.

        Handles both raw sEMG segments ``(T, 4)`` / ``(B, T, 4)`` (converted
        via handcrafted extraction) and pre-computed feature vectors
        ``(112,)`` / ``(B, 112)`` / ``(768,)``.

        Args:
            input_dim: Feature dimension (112 handcrafted, 768 speech-encoder).
            hidden_dim: Hidden layer width (default 768).
            output_dim: LLM embedding dim (3072 for Llama-3.2-3B).
            dropout: Dropout probability between layers.
        """

        def __init__(
            self,
            input_dim: int = 112,
            hidden_dim: int = 768,
            output_dim: int = 3072,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
                raise ValueError("input_dim/hidden_dim/output_dim must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError(f"dropout must be in [0,1), got {dropout}")
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout_layer = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            logger.debug(
                "EMGAdaptor init: input_dim=%d hidden_dim=%d output_dim=%d dropout=%.2f",
                input_dim, hidden_dim, output_dim, dropout,
            )

        # -- helpers -----------------------------------------------------------
        def _is_raw_emg(self, x: Any) -> bool:
            """Detect raw EMG vs feature vector from tensor shape."""
            # x is torch.Tensor – treat last dim == 4 as raw EMG
            try:
                shape = tuple(x.shape)  # type: ignore[union-attr]
                ndim = x.dim()  # type: ignore[union-attr]
            except Exception:
                return False
            if ndim == 2 and shape[1] == 4:
                return True
            if ndim == 3 and shape[2] == 4:
                return True
            if ndim == 1 and shape[0] != self.input_dim and shape[0] > 4:
                # Heuristic: 1-D raw channel? single channel length >> input_dim unlikely → not raw
                return False
            return False

        def _convert_raw_to_features(self, x: Any) -> Any:
            """Convert raw EMG tensor(s) to handcrafted feature tensor(s)."""
            # Lazy import to avoid circular deps and keep torch-optional
            from subvocal.emg_core.dsp.handcrafted import extract_handcrafted_features

            # x is torch.Tensor on some device
            device = x.device  # type: ignore[union-attr]
            dtype = x.dtype  # type: ignore[union-attr]

            if x.dim() == 2 and x.shape[1] == 4:  # (T,4) single raw
                arr = x.detach().cpu().numpy()
                feats = extract_handcrafted_features(arr, fs=250)
                t = torch.from_numpy(feats.astype(np.float32)).to(device)
                # ensure dtype matches expected (float32)
                if t.dtype != dtype and dtype in (torch.float32, torch.float64):
                    t = t.to(dtype)
                # Handle input_dim mismatch (e.g., 768 adaptor given raw 112)
                if t.shape[0] != self.input_dim:
                    logger.debug(
                        "raw EMG converted to %d-d but adaptor input_dim=%d – adapting",
                        t.shape[0], self.input_dim,
                    )
                    if t.shape[0] < self.input_dim:
                        pad = torch.zeros(self.input_dim - t.shape[0], device=device, dtype=t.dtype)
                        t = torch.cat([t, pad], dim=0)
                    else:
                        t = t[: self.input_dim]
                return t

            if x.dim() == 3 and x.shape[2] == 4:  # (B,T,4)
                b = x.shape[0]
                feats_list: list[np.ndarray] = []
                for i in range(b):
                    arr = x[i].detach().cpu().numpy()
                    feats = extract_handcrafted_features(arr, fs=250)
                    feats_list.append(feats)
                arr_stack = np.stack(feats_list, axis=0).astype(np.float32)  # (B,112)
                t = torch.from_numpy(arr_stack).to(device)
                if t.dtype != dtype and dtype in (torch.float32, torch.float64):
                    t = t.to(dtype)
                if t.shape[1] != self.input_dim:
                    logger.debug(
                        "batched raw converted to %d-d but adaptor input_dim=%d – adapting",
                        t.shape[1], self.input_dim,
                    )
                    if t.shape[1] < self.input_dim:
                        pad = torch.zeros((b, self.input_dim - t.shape[1]), device=device, dtype=t.dtype)
                        t = torch.cat([t, pad], dim=1)
                    else:
                        t = t[:, : self.input_dim]
                return t

            return x

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            """Map EMG features (or raw segment) to LLM space.

            Args:
                x: Tensor of shape ``(112,)``, ``(B,112)``, ``(768,)``,
                    ``(B,768)``, ``(T,4)`` or ``(B,T,4)``.
                    NumPy arrays are also accepted and converted.

            Returns:
                Tensor of shape ``(output_dim,)`` or ``(B, output_dim)``.
            """
            _require_torch()

            # Accept numpy input gracefully
            if isinstance(x, np.ndarray):
                logger.debug("EMGAdaptor forward: converting numpy input %s", x.shape)
                x = torch.from_numpy(x.astype(np.float32))
                # caller device unknown – keep on CPU

            if not isinstance(x, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"EMGAdaptor expects Tensor or ndarray, got {type(x)}")

            # Preserve original device/dtype for logging
            orig_shape = tuple(x.shape)
            logger.debug("EMGAdaptor forward: input shape %s", orig_shape)

            # Raw EMG handling
            if self._is_raw_emg(x):
                logger.debug("EMGAdaptor detected raw EMG shape %s – extracting handcrafted features", orig_shape)
                x = self._convert_raw_to_features(x)
                logger.debug("EMGAdaptor raw→features shape %s", tuple(x.shape))

            # Handle feature-dim mismatch gracefully (pad/truncate) – e.g., input_dim 112
            # but features are 768 or vice versa, or batch dim handling.
            # Linear layers expect last dim == input_dim.
            if x.dim() == 1:
                if x.shape[0] != self.input_dim:
                    logger.warning(
                        "EMGAdaptor input last dim %d != input_dim %d – adapting",
                        x.shape[0], self.input_dim,
                    )
                    if x.shape[0] < self.input_dim:
                        pad = torch.zeros(self.input_dim - x.shape[0], device=x.device, dtype=x.dtype)
                        x = torch.cat([x, pad], dim=0)
                    else:
                        x = x[: self.input_dim]
            elif x.dim() >= 2:
                if x.shape[-1] != self.input_dim:
                    logger.warning(
                        "EMGAdaptor input last dim %d != input_dim %d – adapting",
                        x.shape[-1], self.input_dim,
                    )
                    last = x.shape[-1]
                    if last < self.input_dim:
                        pad_shape = list(x.shape)
                        pad_shape[-1] = self.input_dim - last
                        pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
                        x = torch.cat([x, pad], dim=-1)
                    else:
                        # truncate last dim
                        x = x[..., : self.input_dim]

            # MLP
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout_layer(x)
            x = self.fc2(x)
            logger.debug("EMGAdaptor forward: output shape %s", tuple(x.shape))
            return x

else:  # torch not available – stub that raises MissingDependencyError
    class EMGAdaptor:  # type: ignore[no-redef]
        """Stub – raises MissingDependencyError when torch is absent."""

        def __init__(
            self,
            input_dim: int = 112,
            hidden_dim: int = 768,
            output_dim: int = 3072,
            dropout: float = 0.1,
        ) -> None:
            _require_torch()

        def forward(self, x: Any) -> Any:
            _require_torch()
            raise MissingDependencyError("torch not available")


# ---------------------------------------------------------------------------
# EMGAdaptorProvider – LLMProvider-compatible wrapper
# ---------------------------------------------------------------------------

class EMGAdaptorProvider(LLMProvider):  # type: ignore[misc]
    """Wrapper coupling an :class:`EMGAdaptor` with a frozen LLM.

    Compatible with :mod:`subvocal.core.llm_providers` ``LLMProvider``
    interface (``get_provider_name`` / ``reconstruct_intent``). Also exposes
    :meth:`encode_emg_features` (plus aliases ``encode`` / ``adapt``) for
    direct EMG→LLM embedding.

    Args:
        adaptor: Pre-initialised adaptor. If ``None``, a default
            ``EMGAdaptor(input_dim=112, hidden_dim=768, output_dim=3072)`` is
            created (requires ``torch``).
        llm_name: Identifier of the frozen LLM (e.g. ``"meta-llama/Llama-3.2-3B"``).
        llm_provider: Optional underlying :class:`subvocal.core.interfaces.LLMProvider`
            to delegate ``reconstruct_intent`` to. If ``None``, a heuristic
            fallback is used.
        device: Torch device for the adaptor.
    """

    def __init__(
        self,
        adaptor: Any | None = None,
        llm_name: str = "meta-llama/Llama-3.2-3B",
        llm_provider: Any | None = None,
        device: str = "cpu",
    ) -> None:
        self.llm_name = llm_name
        self.llm_provider = llm_provider
        self.device = device
        if adaptor is None:
            _require_torch()
            adaptor = EMGAdaptor()  # type: ignore[call-arg]
        self.adaptor = adaptor
        # Move adaptor to device if possible
        try:
            if _TORCH_AVAILABLE and hasattr(self.adaptor, "to"):
                self.adaptor.to(device)  # type: ignore[union-attr]
        except Exception as e:
            logger.debug("EMGAdaptorProvider: adaptor.to(%s) failed: %s", device, e)
        logger.debug("EMGAdaptorProvider init: llm_name=%s adaptor=%s", llm_name, type(self.adaptor).__name__)

    # -- encoding ----------------------------------------------------------
    def encode_emg_features(
        self,
        emg_input: np.ndarray | Any,
        fs: float = 250.0,  # kept for raw segments that carry sample rate
    ) -> Any:
        """Encode EMG features (or raw segment) to LLM input space.

        Handles:
        * handcrafted vectors ``(112,)`` / ``(B,112)`` or ``(768,)`` /
          ``(B,768)`` (torch or numpy)
        * raw segments ``(T,4)`` / ``(B,T,4)`` (torch or numpy) – converted
          via :func:`extract_handcrafted_features`.

        Args:
            emg_input: EMG data as Tensor or ndarray.
            fs: Sample rate for raw-segment conversion (unused for feature
                vectors).

        Returns:
            Tensor of shape ``(output_dim,)`` or ``(B, output_dim)``.
        """
        _require_torch()
        # Normalize numpy → tensor; adaptor.forward already handles both,
        # but we keep explicit branch for logging and fs propagation.
        x = emg_input
        if isinstance(x, np.ndarray):
            logger.debug("EMGAdaptorProvider.encode: numpy input %s fs=%.1f", x.shape, fs)
            # If raw EMG numpy, let adaptor handle conversion (it uses 250 default).
            # For feature vectors, convert directly.
            if x.ndim == 2 and x.shape[1] == 4:
                # raw (T,4) numpy – convert via handcrafted then to tensor
                from subvocal.emg_core.dsp.handcrafted import extract_handcrafted_features

                feats = extract_handcrafted_features(x, fs=fs)
                x = torch.from_numpy(feats.astype(np.float32)).to(self.device)  # type: ignore[union-attr]
                return self.adaptor(x)  # type: ignore[union-attr]
            if x.ndim == 3 and x.shape[2] == 4:
                from subvocal.emg_core.dsp.handcrafted import extract_handcrafted_features

                feats_list = [extract_handcrafted_features(x[i], fs=fs) for i in range(x.shape[0])]
                arr = np.stack(feats_list, axis=0).astype(np.float32)
                x = torch.from_numpy(arr).to(self.device)  # type: ignore[union-attr]
                return self.adaptor(x)  # type: ignore[union-attr]
            # feature vector numpy
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)  # type: ignore[union-attr]

        # Tensor path – adaptor handles raw vs features
        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
            # ensure device
            try:
                if x.device.type != self.device:  # type: ignore[union-attr]
                    x = x.to(self.device)  # type: ignore[union-attr]
            except Exception:
                pass
            return self.adaptor(x)  # type: ignore[union-attr]

        # Fallback – let adaptor deal with it
        return self.adaptor(x)  # type: ignore[union-attr]

    # aliases required by spec (“provides method to encode EMG features”)
    def encode(self, emg_input: np.ndarray | Any, fs: float = 250.0) -> Any:
        """Alias for :meth:`encode_emg_features`."""
        return self.encode_emg_features(emg_input, fs=fs)

    def adapt(self, emg_input: np.ndarray | Any, fs: float = 250.0) -> Any:
        """Alias for :meth:`encode_emg_features`."""
        return self.encode_emg_features(emg_input, fs=fs)

    def encode_emg(self, emg_input: np.ndarray | Any, fs: float = 250.0) -> Any:
        """Alias for :meth:`encode_emg_features`."""
        return self.encode_emg_features(emg_input, fs=fs)

    # -- LLMProvider interface ---------------------------------------------
    def get_provider_name(self) -> str:
        return f"emg_adaptor:{self.llm_name}"

    def reconstruct_intent(self, tokens: Any, context: Any) -> Any:
        """Reconstruct intent, delegating to underlying provider if set.

        If ``llm_provider`` was supplied at construction, delegation is used;
        otherwise a heuristic fallback is attempted before raising.

        Args:
            tokens: List of :class:`subvocal.core.models.CommandToken`.
            context: :class:`subvocal.context.schema.UserContext`.

        Returns:
            :class:`subvocal.core.models.Intent`.
        """
        if self.llm_provider is not None:
            logger.debug("EMGAdaptorProvider delegating reconstruct_intent to %s", type(self.llm_provider).__name__)
            return self.llm_provider.reconstruct_intent(tokens, context)  # type: ignore[union-attr]

        # Fallback: heuristic provider (no network)
        logger.debug("EMGAdaptorProvider: no llm_provider – using HeuristicProvider fallback")
        try:
            from subvocal.core.llm_providers import HeuristicProvider

            provider = HeuristicProvider()
            return provider.reconstruct_intent(tokens, context)
        except Exception as e:
            from subvocal.exceptions import ProviderError

            raise ProviderError(f"EMGAdaptorProvider reconstruct_intent failed: {e}") from e
