---
title: emg_core.ml.pretrain
sidebar_label: pretrain
---

Pre-training script for deep neural sEMG classifiers on public baseline datasets.

## Functions

### `generate_synthetic_dataset`

```python
def generate_synthetic_dataset(user_id: str = 'pretrained', num_classes: int = 5, segments_per_class: int = 40)
```

Generate a reproducible multi-channel sEMG dataset to simulate a public dataset.

Each class is synthesized with unique harmonic and frequency patterns.

### `run_pretraining`

```python
def run_pretraining()
```

Run pre-training pipeline for all deep architectures.
