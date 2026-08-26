# Subvocal SDK: Hardware Abstraction Layer (HAL)

**Status:** Draft Spec  
**Version:** v0.1.0-alpha  
**Date:** June 2026  
**Audience:** Wearable Device Engineers, BCI Software Developers  

---

## 1. Overview and Interface Lifecycle

The **Hardware Abstraction Layer (HAL)** provides a uniform, vendor-agnostic interface between electromyographical (EMG) sensor sources and downstream digital signal processing (DSP) and classification pipelines. 

Every sensor driver, dataset replayer, and signal simulator in the Subvocal SDK inherits from the abstract base class `HardwareSource` defined in `sdk/core/interfaces.py`:

```
                         [ HardwareSource Interface ]
                                      │
   ┌──────────┬──────────┼──────────┬──────────┬─────────┬──────────┐
   ▼          ▼          ▼          ▼          ▼         ▼         ▼
[FileReplay][Synthetic][OpenBCI] [Delsys][Ninapro/PutEMG/CSL][Gaddy][Meta sEMG-RD]
                                                     │
                                           [ EmgBench 9-Dataset Harness ]
```

### The Driver Lifecycle:
1. **Instantiation:** Configure ports, files, or parameters (e.g., sample rate `fs`, channels).
2. **`start()`:** Connects to the device (serially, via BLE, or socket), opens resources, and arms the hardware data acquisition buffer.
3. **`is_connected()`:** Returns connection health state.
4. **`read_frame(window_ms)`:** Reads a sliding temporal window of data from the internal buffer, returns a Pydantic `Frame` containing Pydantic `Sample`s.
5. **`stop()`:** Stops streaming, releases connections, and clean-closes file/socket handlers.

---

## 2. Core Drivers

The SDK implements four baseline drivers under `src/subvocal/hardware/drivers.py`
(plus dataset drivers in `src/subvocal/hardware/datasets.py` — Gaddy/Meta — and the
`src/subvocal/emg_core/benchmarks/emgbench.py` EmgBench harness, §§4–5):

### 1. `FileReplayDriver`
* **Purpose:** Reads multi-channel sEMG data from a local CSV file, simulating a real-time hardware stream.
* **Mechanism:** Slices continuous numeric columns. Uses the current system clock to generate system timestamps, simulating live inputs.
* **Options:** Supports configuring the sample rate `fs` and toggling `loop` at End-Of-File (EOF).
* **Resource note:** The driver loads the entire CSV into RAM (`self._data`) on init. If the file exceeds **100 MB**, it logs a warning (`FileReplayDriver: CSV ... is ... MB; entire file will be loaded into RAM`) and advises chunked/streaming ingestion to avoid OOM on long recordings. Header rows and non-numeric lines are skipped defensively.

### 2. `SyntheticSignalGenerator`
* **Purpose:** Simulates clean and noisy muscle signals for offline software development without physical hardware.
* **Signal Components:**
  - Baseline physiological muscle noise (Gaussian white noise).
  - 60 Hz powerline hum (notch noise).
  - High-amplitude transient muscle contraction envelopes injected when commands (e.g., `"clk"`, `"gt"`) are triggered via `trigger_command()`.

### 3. `OpenBCICytonDriver`
* **Purpose:** Acquires sEMG signals from OpenBCI Cyton boards (8-channel EEG/EMG research hardware).
* **Mechanism:** Imports the `brainflow` library dynamically. Supports reading from a real USB dongle serial port or a synthetic board simulation.
* **EXG Mapping:** Extracts the primary ExG analog channels (channels 1 to 8) at a fixed sample rate of 250 Hz.

### 4. `DelsysTrignoDriver`
* **Purpose:** Zero-dependency TCP socket client connecting to the Delsys Trigno wireless base station.
* **Mechanism:** Connects directly to the Delsys Trigno Control Utility over standard TCP sockets, avoiding large vendor DLLs or custom wrappers.
* **Reliability note:** `read_frame` enforces a **100 ms socket timeout** with non-blocking `recv` and raises `HardwareError` if no data arrives (`DelsysTrignoDriver: no data received ... within 100 ms`), preventing infinite blocking on base-station dropout. `stop()` reliably sends `STOP` and closes both command and data sockets.

---

## 3. Delsys Trigno TCP Protocol Reference

The Delsys Trigno base station streams data over native TCP sockets. The SDK driver connects using two ports:

### 1. Command Socket (Port 50040)
Communicates control commands to the base station as ASCII strings terminated by `\r\n\r\n`:
* **`START`:** Commands the utility to begin sampling sensors and streaming data over port 50043.
* **`STOP`:** Ends data sampling and closes active data connections.

### 2. Data Socket (Port 50043)
Streams raw binary single-precision floats (32-bit little-endian, `<f` in struct). 
* **Payload Structure:**
  Each sampling cycle packet contains $N$ floats (where $N$ is the number of active channels, typically 8 or 16).
  
```
  [ Sample 1, Channel 1 ] (4 bytes) ──> [ Ch 1 Float ]
  [ Sample 1, Channel 2 ] (4 bytes) ──> [ Ch 2 Float ]
  ...
  [ Sample 1, Channel N ] (4 bytes) ──> [ Ch N Float ]
```

* **Sample Rate:** Default sEMG data streams at 2000 Hz.
* **Buffering:** The driver uses non-blocking socket reads and byte arrays to partition incoming chunks into precise frame sizes based on requested window durations.

---

## 4. Extra Dataset Drivers — Gaddy & Meta sEMG-RD

The dataset suite lives in `src/subvocal/hardware/datasets.py`. All drivers sanitize `data_dir` with `Path.resolve().is_relative_to` (traversal → `FileNotFoundError`) and use `np.load(..., allow_pickle=False)`.

1. **Ninapro (`NinaproDriver`):** Reads Ninapro subject `.mat` files (MATLAB format) using `scipy.io.loadmat` and extracts the `'emg'` matrix.
2. **PutEMG (`PutEMGDriver`):** Reads PutEMG subject `.h5` files (HDF5 format) by dynamically importing `h5py` and searching for active muscle channel datasets. **Resource fix:** `h5py.File` is always closed on failure — `visititems` exceptions, missing datasets, or shape errors close the file descriptor before re-raising, and `stop()` reliably closes the handle to prevent descriptor leaks in long-lived sessions.
3. **CSL-HDEMG (`CSLHDEMGDriver`):** Reads high-density silent speech sEMG recordings stored as NumPy arrays (`.npy`) or raw binary floats. Uses `np.load(..., allow_pickle=False)` to block pickle-based code execution; only raw numeric arrays are accepted.
4. **Gaddy Silent Speech (`GaddyDriver`, Gaddy & Klein EMNLP 2020 / ACL 2021, Zenodo 4064409):** 8-channel facial sEMG at 1000 Hz (ACL 2021 model resamples to 800 Hz) paired with `info.json` transcripts and FLAC audio, covering both vocalized and silent modes.

   **Download:**
   ```bash
   # Zenodo record 4064409 (silent_speech) — ~2 GB archived tar
   curl -L https://zenodo.org/records/4064409/files/silent_speech.tar.gz -o silent_speech.tar.gz
   tar -xzf silent_speech.tar.gz -C data/gaddy
   # Expected layout: data/gaddy/<session>/<id>_emg.npy (T,8) + <id>_info.json + <id>_audio.flac
   # See https://github.com/dgaddy/silent_speech for session manifests
   ```

   ```python
   from subvocal.hardware.datasets import GaddyDriver

   drv = GaddyDriver(data_dir="data/gaddy", fs=800.0, loop=True, split=None)
   # split="silent" or "vocalized" filters by path/info.json mode
   drv.start()
   frame = drv.read_frame(window_ms=100)          # Frame (T≈80 @800Hz, 8 ch)
   drv.list_utterances()                          # ["session/0", ...]
   drv.get_transcript("session/0")                # info.json text/prompt
   drv.get_audio_path("session/0")                # .flac path if present
   ```

   The driver indexes `*_emg.npy` recursively, mmap-probes the first file for `num_channels`, validates `(T,8)` on load, and concatenates utterances sequentially with padding/loop handling identical to `CSLHDEMGDriver`.

5. **Meta sEMG-RD / emg2pose (`MetaEMGDriver`, Salter et al. NeurIPS 2024 `arXiv:2412.02725`; Sivakumar & Landau et al. Nature 2025; open-sourced Dec 2024):** 16 bipolar channels per wrist at **2 kHz** (12-bit sEMG-RD band) + time-aligned hand-pose labels at 2 kHz (26-camera MoCap → 19 markers/hand → IK joint angles, linearly interpolated).

   **Datasets:** *emg2pose* (surface typing, ~370 h / 193 participants / 25k HDF5) and *sEMG-RD* (general hand pose). Layout per recording HDF5: `/emg (T,16)` (aliases `raw_emg`/`semg`) + `/pose (T,D)` (aliases `joint_angles`/`hand_pose`, typically `D=63` i.e. 21 joints×3, or 84). Also supports unpacked `*_emg.npy` + `*_pose.npy`:

   **Download:**
   ```bash
   # Papers + data via Meta AI blog + emg2pose repo
   # https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024
   # https://github.com/facebookresearch/emg2pose  (see emg2pose/data/README)
   # Apply for the data release, then extract so HDF5 contain /emg and /pose
   # Example after approval:
   # data/emg2pose/train/*.h5  (each with /emg (T,16), /pose (T,63-84))
   # data/emg2pose_npy/*_emg.npy + *_pose.npy
   pip install "subvocal[hardware]"  # pulls h5py>=3.0
   ```

   ```python
   from subvocal.hardware.datasets import MetaEMGDriver

   drv = MetaEMGDriver(data_dir="data/emg2pose", fs=2000.0, split="train", loop=True)
   drv.start()
   frame = drv.read_frame(window_ms=50)               # (T≈100 @2kHz, 16 ch)
   pair  = drv.load_pair(recording_id)                # (emg (T,16), pose (T,D))
   pose  = drv.get_pose(recording_id)                 # pose array alone
   frame, pose_win = drv.read_frame_with_pose(window_ms=50)  # synchronized
   ```

   `split` is sanitized `[^A-Za-z0-9_-]→_` and scopes to `data_dir/split` if present, else path-substring filtered (use `split="all"` to disable). Missing `h5py` raises `MissingDependencyError` with `subvocal[hardware]` hint; `data_dir` absence raises `FileNotFoundError` with the Meta blog and GitHub links. Length mismatches between EMG and pose are truncated/padded to the EMG length.

---

## 5. EmgBench 9-Dataset Harness

`src/subvocal/emg_core/benchmarks/emgbench.py` (Yang et al. *EMGBench*, NeurIPS 2024 Datasets & Benchmarks, `arXiv:2410.23625`, https://emgbench.github.io, https://github.com/jehanyang/emgbench) evaluates intersubject generalization and few-shot adaptation across 9 datasets:

```
ninapro-db2, ninapro-db3, ninapro-db5, capgmyo-db-b,
myoarmband, uciemg, mcs, hyser, flexwear-hd   # DEFAULT_DATASETS
```

Two protocols, both with deterministic synthetic fallback (no 10–100 GiB download needed for CI):

* **LOSO-CV** (`EMGBench.evaluate_loso`): leave-one-subject-out cross-validation. For each held-out subject `model_fn(train_X, train_y, test_X, test_y) → accuracy/dict/preds` is invoked (also supports `((train_X,train_y),(test_X,test_y))` or dict form) and reduced via `_accuracy_from_result`.
* **Few-shot adaptation** (`evaluate_adaptation`, `n_shot=1/2/5`): pretrain on `N-1` subjects, take `n_shot` trials per gesture from the held-out subject as *support* (fine-tune) and score the remaining *query* set. Calls `model_fn(train_X, train_y, support_X, support_y, query_X, query_y)` when available; otherwise augments training with support and delegates to LOSO. Covers FT-X % / intersession FT / TSTS (Table 3).

```python
from subvocal.emg_core.benchmarks import EMGBench, DEFAULT_DATASETS

bench = EMGBench(datasets=DEFAULT_DATASETS, root_dir="DatasetsProcessed_hdf5")
# Real layout expected: root_dir/<dataset>/pN/participant_N.hdf5 (keys=gestures, shape=(trials,electrodes,timesteps))
# Synthetic fallback is used automatically when files are absent.

loso  = bench.evaluate_loso(my_model_fn, dataset="ninapro-db5")
adapt = bench.evaluate_adaptation(my_model_fn, dataset="ninapro-db5", n_shot=5)
all_res = bench.evaluate_all(my_model_fn, mode="loso")          # macro mean/std over 9 datasets
# or bench.evaluate_all(my_model_fn, mode="adaptation", n_shot=5)
# summarize_results(all_res) -> {"mean","std","min","max"}
```

`root_dir` and dataset slugs are sanitized (`[^A-Za-z0-9._-]` → `_`, `Path.resolve().is_relative_to`); `run_all` is an alias for `evaluate_all`. Dataset discovery accepts `pN`/`subject_N` subdirs or flat `*.h5` and normalizes `(trials, electrodes, timesteps)` ↔ `(trials, timesteps, electrodes)` by shape heuristics.

---

## 6. Integration Code Example

The following example shows how a developer can instantiate the `SyntheticSignalGenerator`, trigger simulated commands, and feed raw sEMG frames into the `SubvocalPipeline`:

```python
import time
from hardware.drivers import SyntheticSignalGenerator
from core.pipeline import SubvocalPipeline

# 1. Instantiate the synthetic sEMG source (8 channels, 1000 Hz)
hardware = SyntheticSignalGenerator(fs=1000.0, num_channels=8)

# 2. Define a dummy classification function
def classify(frame):
    # Retrieve channel average amplitudes
    data = frame.to_numpy()
    ch_means = np.max(np.abs(data), axis=0)
    
    # Simple threshold trigger representing command zones
    if ch_means[2] > 2.0:
        return CommandToken(text="clk", confidence=0.95, timestamp=time.time())
    return None

# 3. Initialize pipeline with LLM, Context, and Executor mocks
pipeline = SubvocalPipeline(
    hardware=hardware,
    classify_fn=classify,
    llm_provider=MockLLM(),
    context_provider=MockContext(),
    executor=MockExecutor()
)

# 4. Start streaming
hardware.start()

# 5. Trigger a simulated "click" command contraction
hardware.trigger_command("clk")

# 6. Stream and process frames
for _ in range(10):
    action = pipeline.step(window_ms=50)
    if action:
        print(f"Action Executed: {action.action_type}")
    time.sleep(0.05)

hardware.stop()
```

---

## 7. Unified Ingress Manager for Multi-Source Feeds (v2.0)

For high-reliability consumer and clinical deployments, the SDK provides the `IngressManager` in `subvocal.runtime.ingress` to orchestrate multiple sensor inputs and automate failovers:

* **Source Registration**: Register primary acquisition feeds (e.g. Cyton or Delsys boards) along with fallback simulation replays.
* **Biometric Failover Policy**: The manager detects device dropouts (e.g. flatline anomalies or connection lost states) and seamlessly switches the active pipeline feed to the fallback replayer without interrupting downstream execution loops.

```python
from subvocal.runtime import IngressManager
from subvocal.hardware.drivers import SyntheticSignalGenerator

ingress = IngressManager()

# Register primary and fallback streams
ingress.register_source("cyton_board", OpenBCICytonDriver(port="/dev/tty.usb"), is_fallback=False)
ingress.register_source("simulated_fallback", SyntheticSignalGenerator(), is_fallback=True)

# Start active hardware source
ingress.start()

# Switch active source to fallback simulation on hardware dropouts
ingress.trigger_failover()
```

