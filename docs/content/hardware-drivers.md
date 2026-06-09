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
       ┌───────────────┬────────────┼──────────────┬──────────────┐
       ▼               ▼            ▼              ▼              ▼
 [ FileReplay ]  [ Synthetic ]  [ OpenBCI ]  [ Delsys Trigno ] [ Datasets ]
```

### The Driver Lifecycle:
1. **Instantiation:** Configure ports, files, or parameters (e.g., sample rate `fs`, channels).
2. **`start()`:** Connects to the device (serially, via BLE, or socket), opens resources, and arms the hardware data acquisition buffer.
3. **`is_connected()`:** Returns connection health state.
4. **`read_frame(window_ms)`:** Reads a sliding temporal window of data from the internal buffer, returns a Pydantic `Frame` containing Pydantic `Sample`s.
5. **`stop()`:** Stops streaming, releases connections, and clean-closes file/socket handlers.

---

## 2. Core Drivers

The SDK implements four baseline drivers under `sdk/hardware/drivers.py`:

### 1. `FileReplayDriver`
* **Purpose:** Reads multi-channel sEMG data from a local CSV file, simulating a real-time hardware stream.
* **Mechanism:** Slices continuous numeric columns. Uses the current system clock to generate system timestamps, simulating live inputs.
* **Options:** Supports configuring the sample rate `fs` and toggling `loop` at End-Of-File (EOF).

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

## 4. Public Dataset Streams

The SDK implements custom subject-file readers under `sdk/hardware/datasets.py` to stream standard public electromyographical datasets:

1. **Ninapro (`NinaproDriver`):** Reads Ninapro subject `.mat` files (MATLAB format) using `scipy.io.loadmat` and extracts the `'emg'` matrix.
2. **PutEMG (`PutEMGDriver`):** Reads PutEMG subject `.h5` files (HDF5 format) by dynamically importing `h5py` and searching for active muscle channel datasets.
3. **CSL-HDEMG (`CSLHDEMGDriver`):** Reads high-density silent speech sEMG recordings stored as NumPy arrays (`.npy`) or raw binary floats.

---

## 5. Integration Code Example

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

## 6. Unified Ingress Manager for Multi-Source Feeds (v2.0)

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

