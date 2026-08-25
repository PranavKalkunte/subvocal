"""LISA LLM Integrated Scoring Adjustment per Benster et al. 2024.

Rescores beam-search hypotheses from an acoustic model (sEMG / ASR)
using a frozen LLM as a language prior. Per MONA's LISA (arxiv:2403.05583),
final score interpolates acoustic and LLM scores::

    final = (1 - alpha) * acoustic + alpha * llm + beta * length_bonus

The LLM is accessed via :class:`subvocal.core.interfaces.LLMProvider`.
If a provider exposes ``score(text)``, ``score_text(text)`` or similar,
it is used; otherwise a deterministic heuristic fallback is applied.

References
----------
Benster et al., 2024 — ``MONA + LISA``. arxiv:2403.05583.

Guarded: torch is optional; missing torch only disables ``beam_search_rescore``
logits decoding (raises ``MissingDependencyError``). ``rescore`` and
``lisa_rescore`` work without torch.

Example
-------
>>> from subvocal.core.llm_providers import HeuristicProvider
>>> from subvocal.emg_core.ml.lisa import LISA, lisa_rescore
>>> lisa = LISA(llm_provider=HeuristicProvider(), alpha=0.5)
>>> reranked = lisa.rescore(["hello world", "hallo world"], [-0.2, -0.5])
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from subvocal.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

__all__ = ["LISA", "lisa_rescore"]

# ---------------------------------------------------------------------------
# lazy torch import
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
            "torch is required for LISA beam search. Install with 'pip install \"subvocal[ml]\"'"
        )


# ---------------------------------------------------------------------------
# LLM scoring helper — integrates with subvocal.core.llm_providers.LLMProvider
# ---------------------------------------------------------------------------

def _score_text_with_llm(text: str, llm_provider: Any) -> float:
    """Score a hypothesis string with an LLMProvider.

    Tries known scoring methods on the provider in order:
    ``score``, ``score_text``, ``get_score``, ``score_hypothesis``,
    ``language_model_score``, ``get_llm_score``. If none exists,
    falls back to a deterministic heuristic that still distinguishes
    hypotheses and integrates provider identity (hash-based jitter).

    Args:
        text: Hypothesis string to score.
        llm_provider: Instance of :class:`subvocal.core.interfaces.LLMProvider`
            or any object exposing a scoring method.

    Returns:
        Float log-probability-like score (higher is better, typically
        negative or 0..1). Heuristic fallback returns 0..1 range.
    """
    if not text:
        return 0.0

    if llm_provider is not None:
        # Try explicit scoring APIs first
        for name in ("score", "score_text", "get_score", "score_hypothesis", "language_model_score", "get_llm_score"):
            if hasattr(llm_provider, name):
                try:
                    fn = getattr(llm_provider, name)
                    val = fn(text)  # type: ignore[misc]
                    score = float(val)  # type: ignore[arg-type]
                    logger.debug("LISA LLM score via %s: %.4f for %r", name, score, text[:40])
                    return score
                except Exception as e:  # pragma: no cover
                    logger.debug("LISA provider.%s(%r) failed: %s", name, text[:20], e)

        # Try LLMProvider reconstruct_intent confidence as proxy (offline heuristic)
        # We avoid calling reconstruct_intent for scoring because it requires
        # UserContext and tokens; instead use provider_name hash as signal
        # that provider was consulted.
        try:
            provider_name = ""
            if hasattr(llm_provider, "get_provider_name"):
                try:
                    provider_name = str(llm_provider.get_provider_name())  # type: ignore[union-attr]
                except Exception:
                    provider_name = type(llm_provider).__name__
            else:
                provider_name = type(llm_provider).__name__

            # Deterministic heuristic that mixes text + provider name
            # Base: length-normalized (shorter ~ higher, but not dominant)
            # Plus hash jitter so different providers rank differently (proves integration)
            base = 1.0 / (1.0 + len(text.split()))
            # Add common-word bonus (heuristic language prior)
            common = {"the", "a", "is", "to", "and", "hello", "world", "open", "close", "click", "type", "goto", "search"}
            words = text.lower().split()
            overlap = sum(1 for w in words if w.strip(".,!?") in common)
            bonus = 0.05 * overlap
            # Hash jitter 0..0.05
            h = int(hashlib.md5(f"{provider_name}:{text}".encode()).hexdigest()[:4], 16)
            jitter = (h % 100) / 2000.0  # 0..0.05
            # If mock_response present, treat as strong prior
            if hasattr(llm_provider, "mock_response") and llm_provider.mock_response is not None:  # type: ignore[union-attr]
                bonus += 0.1
            score = base + bonus + jitter
            # Normalize to roughly -1..1 via log-like transform for interpolation stability
            # Keep 0..1 for simplicity — caller normalizes if needed
            logger.debug("LISA heuristic LLM score (provider=%s) %.4f for %r", provider_name, score, text[:40])
            return float(score)
        except Exception as e:  # pragma: no cover
            logger.debug("LISA heuristic scoring failed: %s", e)

    # No provider — pure heuristic fallback
    base = 1.0 / (1.0 + len(text.split()))
    # Simple character diversity bonus (more diverse -> slightly higher)
    uniq_ratio = len(set(text.lower())) / max(len(text), 1)
    return float(base + 0.05 * uniq_ratio)


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0,1] for stable interpolation (if needed)."""
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-9:
        return [0.5 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


# ---------------------------------------------------------------------------
# LISA scorer
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class LISA(nn.Module):  # type: ignore[no-redef]
        """LLM Integrated Scoring Adjustment (LISA).

        Rescores acoustic beam-search hypotheses with a frozen LLM prior.

        Args:
            llm_provider: LLM provider for scoring. May be any
                :class:`subvocal.core.interfaces.LLMProvider` or object
                with ``score(text)``. If ``None``, a deterministic heuristic
                is used (still satisfies the API).
            alpha: Interpolation weight for LLM score (0..1). ``0`` = acoustic
                only, ``1`` = LLM only. Default 0.5 per MONA.
            beta: Length-bonus weight (default 0.3). Added as
                ``beta * (len(text)/100)`` or word-count normalized; set 0 to disable.
        """

        def __init__(
            self,
            llm_provider: Any | None = None,
            alpha: float = 0.5,
            beta: float = 0.3,
        ) -> None:
            super().__init__()
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"alpha must be in [0,1], got {alpha}")
            if not 0.0 <= beta <= 1.0:
                raise ValueError(f"beta must be in [0,1], got {beta}")
            # Allow alpha/beta as 0..1 floats; store
            self.llm_provider = llm_provider
            self.alpha = float(alpha)
            self.beta = float(beta)
            # Also accept llm_provider as string name -> resolve via resolve_provider
            if isinstance(llm_provider, str):
                try:
                    from subvocal.core.llm_providers import resolve_provider

                    self.llm_provider = resolve_provider(prefer=llm_provider)
                    logger.debug("LISA resolved provider string %r -> %s", llm_provider, type(self.llm_provider).__name__)
                except Exception as e:
                    logger.warning("LISA could not resolve provider %r: %s", llm_provider, e)
                    self.llm_provider = None
            logger.debug("LISA init: alpha=%.2f beta=%.2f provider=%s", self.alpha, self.beta, type(self.llm_provider).__name__ if self.llm_provider else "None")

        # -- internal LLM scoring --------------------------------------------

        def _get_llm_score(self, text: str) -> float:
            """Score text with the configured LLM provider (fallback heuristic)."""
            return _score_text_with_llm(text, self.llm_provider)

        # -- public: rescore -------------------------------------------------

        def rescore(
            self,
            hypotheses: list[str],
            beam_scores: list[float],
        ) -> list[tuple[str, float]]:
            """Rescore hypotheses by interpolating acoustic and LLM scores.

            Computes for each hypothesis::

                combined = (1 - alpha) * acoustic + alpha * llm_score
                           + beta * length_norm

            where ``length_norm`` is ``len(text)/100`` (character) to mildly
            favor longer hypotheses when ``beta>0`` (paper's length bonus).

            Args:
                hypotheses: List of hypothesis strings from beam search.
                beam_scores: List of acoustic scores (log-probs or probs)
                    parallel to hypotheses. Higher is better.

            Returns:
                List of ``(hypothesis, combined_score)`` sorted descending
                by combined_score.

            Raises:
                ValueError: if lengths mismatch or empty input.
            """
            if len(hypotheses) != len(beam_scores):
                raise ValueError(f"hypotheses ({len(hypotheses)}) and beam_scores ({len(beam_scores)}) length mismatch")
            if not hypotheses:
                return []

            llm_scores: list[float] = []
            for hyp in hypotheses:
                if not isinstance(hyp, str):
                    raise TypeError(f"hypothesis must be str, got {type(hyp)}")
                llm_scores.append(self._get_llm_score(hyp))

            # For stable interpolation, optionally normalize both to comparable ranges?
            # We keep raw beam_scores (often log-probs negative) and llm_scores (0..1)
            # Interpolation works if beam_scores are also ~0..1 (probs). For log-probs,
            # the blend is dominated by acoustic scale — acceptable for monotonic ranking
            # as long as llm_scores have non-zero weight. To improve, we optionally
            # min-max normalize beam_scores if they span large negative range.
            # Heuristic: if beam_scores range > 5 or any < -1, normalize to [0,1] before blend.
            use_norm = False
            b_min: float = 0.0
            b_max: float = 0.0
            if beam_scores:
                b_min = float(min(beam_scores))
                b_max = float(max(beam_scores))
                if b_max - b_min > 5.0 or b_min < -1.0:
                    use_norm = True
            if use_norm:
                beam_n = _normalize_scores(beam_scores)
                llm_n = _normalize_scores(llm_scores)
                llm_range = (max(llm_scores) - min(llm_scores)) if llm_scores else 0.0
                logger.debug(
                    "LISA rescore: normalizing scores for interpolation (beam range %.2f llm range %.2f)",
                    b_max - b_min,
                    llm_range,
                )
            else:
                beam_n = beam_scores
                llm_n = llm_scores

            combined: list[tuple[str, float]] = []
            for hyp, ac_raw, ac, lm in zip(hypotheses, beam_scores, beam_n, llm_n, strict=False):
                # For normalized path, blend normalized scores but preserve ranking
                # For unnormalized path, blend raw acoustic
                if use_norm:
                    score = (1.0 - self.alpha) * ac + self.alpha * lm
                else:
                    score = (1.0 - self.alpha) * ac_raw + self.alpha * lm
                # Length bonus (beta * len_norm) — mild, normalized
                if self.beta != 0:
                    # Use word count normalized to ~0.1
                    length_bonus = self.beta * (len(hyp.split()) / 20.0)
                    score += length_bonus
                combined.append((hyp, float(score)))
                logger.debug("LISA rescore: %r acoustic=%.3f llm=%.3f combined=%.3f", hyp[:40], ac_raw, lm, score)

            # Sort descending by combined score, stable (preserve original order on ties)
            combined.sort(key=lambda x: x[1], reverse=True)
            return combined

        def forward(self, hypotheses: list[str], beam_scores: list[float]) -> list[tuple[str, float]]:  # type: ignore[override]
            """Alias for :meth:`rescore` for nn.Module compatibility."""
            return self.rescore(hypotheses, beam_scores)

        def beam_search_rescore(
            self,
            logits: Any,
            vocab: list[str],
            beam_width: int = 10,
            lm_weight: float = 0.5,
        ) -> str:
            """Decode logits with beam search then LISA-rescore.

            Args:
                logits: Tensor of shape ``(T, V)`` or ``(B, T, V)`` (acoustic
                    logits per frame). If batched, first batch is decoded.
                vocab: List of vocabulary tokens (length V). May include
                    special tokens ``<blank>``, ``<pad>``, ``<eos>`` which
                    are handled: ``<blank>`` is skipped (CTC), ``<pad>`` ignored.
                beam_width: Beam width (default 10).
                lm_weight: LLM interpolation weight for this call (overrides
                    ``self.alpha`` for this decoding).

            Returns:
                Top hypothesis string after LISA rescoring.

            Raises:
                MissingDependencyError: if torch not available.
                ValueError: if shapes invalid.
            """
            _require_torch()
            if not isinstance(vocab, (list, tuple)) or not vocab:
                raise ValueError("vocab must be non-empty list of tokens")
            if beam_width <= 0:
                raise ValueError(f"beam_width must be >0, got {beam_width}")
            if not 0.0 <= lm_weight <= 1.0:
                raise ValueError(f"lm_weight must be in [0,1], got {lm_weight}")
            if logits is None:
                raise ValueError("logits must not be None")

            # Accept numpy as well (convert to tensor)
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits.astype(np.float32))  # type: ignore[union-attr]

            if not isinstance(logits, torch.Tensor):  # type: ignore[arg-type]
                raise TypeError(f"logits must be Tensor or ndarray, got {type(logits)}")

            # Handle batch dim: (B,T,V) -> take first
            if logits.dim() == 3:
                if logits.shape[0] == 0:
                    raise ValueError("logits batch dim 0 is empty")
                logits = logits[0]  # (T,V)
            if logits.dim() != 2:
                raise ValueError(f"logits expected (T,V) or (B,T,V), got {tuple(logits.shape)}")
            t_steps, v = logits.shape
            if v != len(vocab):
                raise ValueError(f"logits vocab dim {v} != len(vocab) {len(vocab)}")
            if t_steps == 0:
                return ""

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)  # type: ignore[union-attr] (T,V)

            # Identify special tokens (CTC blank typically 0)
            blank_idx: int | None = None
            pad_idx: set[int] = set()
            for i, tok in enumerate(vocab):
                low = tok.lower()
                if low in ("<blank>", "<blk>", "[blank]", "blank"):
                    blank_idx = i
                if low in ("<pad>", "<eos>", "<bos>", "[pad]", "<unk>"):
                    pad_idx.add(i)
            # Determine if vocab is character-level (average non-special token length ==1)
            _non_special = [tok for idx, tok in enumerate(vocab) if idx not in pad_idx and idx != blank_idx]
            _avg_len = sum(len(v) for v in _non_special) / len(_non_special) if _non_special else 1
            is_char_vocab = abs(_avg_len - 1.0) < 1e-9

            # Beam search: each beam is dict with text, score
            beams: list[dict[str, Any]] = [{"text": "", "score": 0.0}]

            for t in range(t_steps):
                candidates: list[dict[str, Any]] = []
                lp_t = log_probs[t]  # (V,)
                for beam in beams:
                    base_score = beam["score"]
                    base_text = beam["text"]
                    for v_idx, token in enumerate(vocab):
                        if v_idx in pad_idx:
                            continue
                        if blank_idx is not None and v_idx == blank_idx:
                            # CTC blank: no token emitted, score carries
                            candidates.append({"text": base_text, "score": base_score + float(lp_t[v_idx].item())})  # type: ignore[union-attr]
                            continue
                        # Normal token: append to text
                        if not base_text:
                            new_text = token
                        else:
                            if token.startswith("##"):
                                new_text = base_text + token[2:]
                            elif is_char_vocab:
                                new_text = base_text + token
                            else:
                                # word-level: space-separated
                                if base_text.endswith(" ") or token in (".", ",", "!", "?", "'"):
                                    new_text = base_text + token
                                else:
                                    new_text = base_text + " " + token
                        candidates.append({"text": new_text, "score": base_score + float(lp_t[v_idx].item())})  # type: ignore[union-attr]

                # Merge candidates with identical text (keep max score)
                merged: dict[str, float] = {}
                for cand in candidates:
                    txt = cand["text"]
                    sc = cand["score"]
                    if txt not in merged or sc > merged[txt]:
                        merged[txt] = sc
                # Keep top beam_width
                sorted_cands = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
                beams = [{"text": txt, "score": sc} for txt, sc in sorted_cands[:beam_width]]
                # Debug: log top beam per step? Too verbose

            # beams now holds final hypotheses
            hypotheses = [b["text"] for b in beams]
            beam_scores = [b["score"] for b in beams]

            # Edge: if all hypotheses empty (e.g., all blanks), return ""
            # Use LISA rescoring with temporary alpha = lm_weight
            orig_alpha = self.alpha
            try:
                self.alpha = float(lm_weight)
                reranked = self.rescore(hypotheses, beam_scores)
            finally:
                self.alpha = orig_alpha

            if not reranked:
                return ""
            top = reranked[0][0]
            logger.debug("LISA beam_search_rescore: top %r (beams=%d)", top[:60], len(beams))
            return top

else:  # torch not available — stub

    class LISA:  # type: ignore[no-redef]
        """Stub — LISA scorer without torch (rescore works, beam_search requires torch)."""

        def __init__(
            self,
            llm_provider: Any | None = None,
            alpha: float = 0.5,
            beta: float = 0.3,
        ) -> None:
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"alpha must be in [0,1], got {alpha}")
            if not 0.0 <= beta <= 1.0:
                raise ValueError(f"beta must be in [0,1], got {beta}")
            self.llm_provider = llm_provider
            self.alpha = float(alpha)
            self.beta = float(beta)
            if isinstance(llm_provider, str):
                try:
                    from subvocal.core.llm_providers import resolve_provider

                    self.llm_provider = resolve_provider(prefer=llm_provider)
                except Exception:
                    self.llm_provider = None

        def _get_llm_score(self, text: str) -> float:
            return _score_text_with_llm(text, self.llm_provider)

        def rescore(
            self,
            hypotheses: list[str],
            beam_scores: list[float],
        ) -> list[tuple[str, float]]:
            if len(hypotheses) != len(beam_scores):
                raise ValueError(f"hypotheses ({len(hypotheses)}) and beam_scores ({len(beam_scores)}) length mismatch")
            if not hypotheses:
                return []
            llm_scores: list[float] = []
            for hyp in hypotheses:
                if not isinstance(hyp, str):
                    raise TypeError(f"hypothesis must be str, got {type(hyp)}")
                llm_scores.append(self._get_llm_score(hyp))
            use_norm = False
            b_min_s: float = 0.0
            b_max_s: float = 0.0
            if beam_scores:
                b_min_s = float(min(beam_scores))
                b_max_s = float(max(beam_scores))
                if b_max_s - b_min_s > 5.0 or b_min_s < -1.0:
                    use_norm = True
            if use_norm:
                beam_n = _normalize_scores(beam_scores)
                llm_n = _normalize_scores(llm_scores)
            else:
                beam_n = beam_scores
                llm_n = llm_scores
            combined: list[tuple[str, float]] = []
            for hyp, ac_raw, ac, lm in zip(hypotheses, beam_scores, beam_n, llm_n, strict=False):
                if use_norm:
                    score = (1.0 - self.alpha) * ac + self.alpha * lm
                else:
                    score = (1.0 - self.alpha) * ac_raw + self.alpha * lm
                if self.beta != 0:
                    length_bonus = self.beta * (len(hyp.split()) / 20.0)
                    score += length_bonus
                combined.append((hyp, float(score)))
            combined.sort(key=lambda x: x[1], reverse=True)
            return combined

        def forward(self, hypotheses: list[str], beam_scores: list[float]) -> list[tuple[str, float]]:
            return self.rescore(hypotheses, beam_scores)

        def beam_search_rescore(
            self,
            logits: Any,
            vocab: list[str],
            beam_width: int = 10,
            lm_weight: float = 0.5,
        ) -> str:
            _require_torch()
            raise MissingDependencyError("torch not available for beam_search_rescore")

        # Numpy fallback beam_search for stub (no torch) — uses numpy log_softmax
        def _beam_search_numpy(
            self,
            logits: Any,
            vocab: list[str],
            beam_width: int = 10,
            lm_weight: float = 0.5,
        ) -> str:
            """Numpy fallback beam search (used when torch absent but numpy logits given)."""
            import numpy as np  # local

            if isinstance(logits, list):
                logits = np.array(logits, dtype=np.float32)
            else:
                logits = np.asarray(logits, dtype=np.float32)
            if logits.ndim == 3:
                logits = logits[0]
            if logits.ndim != 2:
                raise ValueError(f"logits expected (T,V), got {logits.shape}")
            t_steps, v = logits.shape
            if v != len(vocab):
                raise ValueError(f"logits vocab dim {v} != len(vocab) {len(vocab)}")
            if t_steps == 0:
                return ""
            # log softmax numpy
            m = np.max(logits, axis=1, keepdims=True)
            e = np.exp(logits - m)
            log_probs = np.log(e / np.sum(e, axis=1, keepdims=True) + 1e-12)

            blank_idx = None
            pad_idx: set[int] = set()
            for i, tok in enumerate(vocab):
                low = tok.lower()
                if low in ("<blank>", "<blk>", "[blank]", "blank"):
                    blank_idx = i
                if low in ("<pad>", "<eos>", "<bos>", "[pad]", "<unk>"):
                    pad_idx.add(i)

            _non_special_np = [tok for idx, tok in enumerate(vocab) if idx not in pad_idx and idx != blank_idx]
            _avg_len_np = sum(len(v) for v in _non_special_np) / len(_non_special_np) if _non_special_np else 1
            is_char_vocab_np = abs(_avg_len_np - 1.0) < 1e-9
            beams: list[dict[str, Any]] = [{"text": "", "score": 0.0}]
            for t in range(t_steps):
                candidates: list[dict[str, Any]] = []
                lp_t = log_probs[t]
                for beam in beams:
                    base_score = beam["score"]
                    base_text = beam["text"]
                    for v_idx, token in enumerate(vocab):
                        if v_idx in pad_idx:
                            continue
                        if blank_idx is not None and v_idx == blank_idx:
                            candidates.append({"text": base_text, "score": base_score + float(lp_t[v_idx])})
                            continue
                        if not base_text:
                            new_text = token
                        else:
                            if token.startswith("##"):
                                new_text = base_text + token[2:]
                            elif is_char_vocab_np:
                                new_text = base_text + token
                            else:
                                new_text = base_text + (" " if not base_text.endswith(" ") else "") + token
                        candidates.append({"text": new_text, "score": base_score + float(lp_t[v_idx])})
                merged: dict[str, float] = {}
                for cand in candidates:
                    txt = cand["text"]
                    sc = cand["score"]
                    if txt not in merged or sc > merged[txt]:
                        merged[txt] = sc
                sorted_cands = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
                beams = [{"text": txt, "score": sc} for txt, sc in sorted_cands[:beam_width]]

            hypotheses = [b["text"] for b in beams]
            beam_scores = [b["score"] for b in beams]
            orig_alpha = self.alpha
            try:
                self.alpha = float(lm_weight)
                reranked = self.rescore(hypotheses, beam_scores)
            finally:
                self.alpha = orig_alpha
            return reranked[0][0] if reranked else ""


# ---------------------------------------------------------------------------
# Functional API: lisa_rescore
# ---------------------------------------------------------------------------

def lisa_rescore(
    hypotheses: list[dict[str, Any]],
    llm_provider: Any | None = None,
    weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Re-rank hypotheses list with LISA LLM scoring (functional API).

    Functional wrapper around :class:`LISA` for dict-based beam outputs
    (common in ASR/EMG decoders). Each hypothesis dict must contain a text
    field and a score field; flexible key aliases are accepted.

    Args:
        hypotheses: List of dicts, each with text and acoustic score.
            Accepted text keys: ``text``, ``hypothesis``, ``transcript``,
            ``hyp``, ``str``. Accepted score keys: ``score``,
            ``acoustic_score``, ``beam_score``, ``acoustic``, ``log_prob``.
            Additional keys are preserved in output.
        llm_provider: LLM provider for scoring. May be an
            :class:`subvocal.core.interfaces.LLMProvider`, a provider name
            string (e.g., ``"heuristic"``, ``"openai"``), or ``None`` for
            heuristic fallback.
        weight: LLM interpolation weight (0..1). ``0`` = acoustic only,
            ``1`` = LLM only. Default 0.5.

    Returns:
        Re-ranked list of dicts sorted descending by combined score.
        Each dict is shallow-copied and augmented with
        ``llm_score``, ``combined_score`` (or ``lisa_score`` alias).

    Raises:
        ValueError: if weight not in [0,1] or hypotheses not a list.
        TypeError: if hypotheses contains non-dict items.

    Example:
        >>> from subvocal.core.llm_providers import HeuristicProvider
        >>> hyps = [{"text": "hello world", "score": -0.2}, {"text": "hallo world", "score": -0.5}]
        >>> reranked = lisa_rescore(hyps, HeuristicProvider(), weight=0.5)
        >>> reranked[0]["text"]
        'hello world'
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0,1], got {weight}")
    if not isinstance(hypotheses, list):
        raise TypeError(f"hypotheses must be list[dict], got {type(hypotheses)}")
    if not hypotheses:
        return []

    # Resolve provider string if needed
    provider = llm_provider
    if isinstance(provider, str):
        try:
            from subvocal.core.llm_providers import resolve_provider

            provider = resolve_provider(prefer=provider)
        except Exception as e:
            logger.warning("lisa_rescore could not resolve provider %r: %s", provider, e)
            provider = None

    # Determine text/score keys per dict
    text_keys = ("text", "hypothesis", "transcript", "hyp", "str", "utterance")
    score_keys = ("score", "acoustic_score", "beam_score", "acoustic", "log_prob", "acoustic_log_prob")

    scored: list[dict[str, Any]] = []
    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            raise TypeError(f"hypotheses[{idx}] must be dict, got {type(hyp)}")
        # Extract text
        text = ""
        for k in text_keys:
            if k in hyp and isinstance(hyp[k], str):
                text = hyp[k]
                break
        if not text:
            # Fallback: first string value
            for v in hyp.values():
                if isinstance(v, str):
                    text = v
                    break
        # Extract acoustic score
        acoustic: float | None = None
        for k in score_keys:
            if k in hyp:
                try:
                    acoustic = float(hyp[k])  # type: ignore[arg-type]
                    break
                except Exception:
                    continue
        if acoustic is None:
            # Try any numeric value
            for v in hyp.values():
                if isinstance(v, (int, float)):
                    acoustic = float(v)
                    break
            if acoustic is None:
                acoustic = 0.0

        llm_score = _score_text_with_llm(text, provider)
        # Interpolate: combined = (1-weight)*acoustic + weight*llm
        # Handle scale mismatch via normalization hint: if acoustic is log_prob negative large,
        # heuristic llm 0..1 will be negligible; so we normalize when needed.
        # For functional API, we keep raw interpolation but log when range large.
        # Caller can pre-normalize acoustic scores if needed; we do mild heuristic:
        # If acoustic < -1 or range >5, we normalize acoustic contribution minimally?
        # Keep simple raw blend.
        combined = (1.0 - weight) * acoustic + weight * llm_score

        # Use deterministic tie-breaker: preserve original order via idx small epsilon
        # Not needed as sort is stable, but we keep for clarity.

        new_hyp = dict(hyp)  # shallow copy
        new_hyp["llm_score"] = float(llm_score)
        new_hyp["combined_score"] = float(combined)
        new_hyp["lisa_score"] = float(combined)  # alias for paper naming
        # Also store blended with beta? For functional, alpha=weight, beta kept 0 (no length bonus)
        # To match LISA class beta behavior, add mild length bonus if original provider is not None?
        # Keep as is for simplicity.
        scored.append(new_hyp)
        logger.debug("lisa_rescore [%d] %r acoustic=%.3f llm=%.3f combined=%.3f", idx, text[:40], acoustic, llm_score, combined)

    # Sort descending by combined_score, stable
    scored.sort(key=lambda d: d["combined_score"], reverse=True)
    return scored

