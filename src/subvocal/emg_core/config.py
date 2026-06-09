"""Central configuration for EMG Core."""

from subvocal.paths import get_data_dir, get_models_dir

# --- Hardware / Ingestion ---
SAMPLE_RATE: int = 250
NUM_CHANNELS: int = 4

# --- DSP ---
# AlterEgo-style silent speech uses 1.3-50 Hz 4th-order Butterworth bandpass filter.
# Traditional physical/prosthetics EMG uses 20-450 Hz.
BANDPASS_LOW: float = 1.3
BANDPASS_HIGH: float = 50.0
BANDPASS_ORDER: int = 4
NOTCH_FREQ: float = 60.0  # 60 Hz for US, 50 Hz for EU
NOTCH_Q: float = 30.0
SMOOTH_WINDOW: int = 3  # samples for moving average

# --- TD10 Features ---
TD10_FRAME_SIZE_MS: int = 27
TD10_FRAME_SHIFT_MS: int = 10
TD10_CONTEXT: int = 10
LDA_COMPONENTS: int = 32

# --- Classifier ---
CLASSIFIER_TYPE: str = "rf"  # "rf", "cnn", or "gru"
RF_N_ESTIMATORS: int = 200

# --- Prediction & Debounce ---
CONFIDENCE_THRESHOLD: float = 0.75
COOLDOWN_MS: int = 900

# --- Paths ---
# Writable per-user directories (overridable via SUBVOCAL_DATA_DIR /
# SUBVOCAL_MODELS_DIR); the installed package tree is never written to.
MODELS_DIR = get_models_dir()
DATA_DIR = get_data_dir()
