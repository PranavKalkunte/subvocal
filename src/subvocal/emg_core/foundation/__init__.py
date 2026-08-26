"""Foundation model modules for sEMG.

Exports:

- ``TinyMyoEncoder``, ``TinyMyoFoundation`` — 3.6M Transformer encoder
  per arXiv:2512.15729 with channel-independent patching, SimMIM masking,
  RoPE extrapolation.
- ``EMGTokenizer``, ``AEMGFramework`` — AEMG NCT + VQ per CVPR 2026
  Huang et al., contraction tokens via sliding-window segmentation.
- ``CyRoPE``, ``SPECTREEncoder``, ``stft_kmeans_pseudolabels`` — SPECTRE
  per arXiv:2512.22481.
"""

from __future__ import annotations

from .aemg_tokenizer import AEMGFramework, EMGTokenizer
from .spectre import CyRoPE, SPECTREEncoder, stft_kmeans_pseudolabels
from .tinymyo import TinyMyoEncoder, TinyMyoFoundation

__all__ = [
    "TinyMyoEncoder",
    "TinyMyoFoundation",
    "EMGTokenizer",
    "AEMGFramework",
    "CyRoPE",
    "SPECTREEncoder",
    "stft_kmeans_pseudolabels",
]
