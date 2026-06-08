# Calibration & Fine-Tuning Guide

Electromyography signals vary significantly between users due to electrode placement, neck skin impedance, and anatomical variation. This guide details the calibration and fine-tuning engine used to adapt baseline models to individual users.

---

## 1. Reproducible Classifier Training Configuration

The SDK structures ML models under `sdk/emg_core/ml/`. Configurations are managed using `TrainingConfig` (defined in `config_schema.py`) to guarantee reproducible runs:

```python
from emg_core.ml.config_schema import TrainingConfig

config = TrainingConfig(
    seed=42,
    epochs=15,
    batch_size=32,
    learning_rate=0.001,
    model_type="transformer"  # Option: "cnn", "gru", "svm"
)
```

---

## 2. Dynamic Transfer Learning Calibration

To onboard a new user without training a neural network from scratch, the SDK supports transfer learning calibration (`calibrate_model` in `train.py`). The pipeline:

1.  Loads a **pre-trained base model** (trained on synthetic multi-user or public dataset files).
2.  **Freezes** the feature extraction layers (early convolutional or recurrent layers).
3.  Reinitializes the **output classification head** matching the target vocabulary.
4.  Fine-tunes the output parameters on the user's small calibration set (e.g. 5-10 repetitions of each gesture).

```python
from emg_core.ml.train import calibrate_model

# Adapt pre-trained weights to user calibration dataset
calibrated_model = calibrate_model(
    model_type="cnn",
    base_model_path="sdk/models/pretrained_model_cnn.pth",
    calibration_data_path="sdk/data/user_102_calibration.npz",
    config=config
)
```

---

## 3. Capturing User Corrections

To continuously improve intent reconstruction, the pipeline tracks user corrections under `sdk/core/corrections.py`:

```
┌────────────────┐          ┌────────────────────┐          ┌───────────────────────┐
│ LLM Intent     │ ──wrong─>│ User inputs correct│ ────────>│   Correction Log      │
│ Reconstruction │          │ phrase / token     │          │ (corrections_log.json)│
└────────────────┘          └────────────────────┘          └───────────────────────┘
```

When a user overrides a reconstructed intent, the `CorrectionManager` records:
*   The raw shorthand tokens.
*   The original (incorrect) intent.
*   The corrected intent payload.
*   The active user context snapshot.

This log is appended as a structured JSONL entry inside `sdk/data/corrections_log.jsonl`.

---

## 4. Exporting Fine-Tuning Datasets

You can export the accumulated correction logs to standard fine-tuning files for OpenAI or Gemini using `FinetuningHook`:

```python
from core.corrections import FinetuningHook

# Export to OpenAI jsonl format
FinetuningHook.export_to_openai(
    log_path="sdk/data/corrections_log.jsonl",
    output_path="sdk/data/openai_finetuning_dataset.jsonl"
)

# Export to Google Gemini format
FinetuningHook.export_to_gemini(
    log_path="sdk/data/corrections_log.jsonl",
    output_path="sdk/data/gemini_finetuning_dataset.jsonl"
)
```
These datasets can be uploaded directly to cloud providers to fine-tune intent decoders on specific user shorthand dialects.
