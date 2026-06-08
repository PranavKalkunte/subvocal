# Hardware Abstraction Guide

The Subvocal SDK separates physiological sensor acquisition from signal classification through the `HardwareSource` interface. This allows developers to toggle between recorded CSV archives, synthetic bio-signal generators, public academic datasets, and live physiological hardware.

---

## 1. The `HardwareSource` Interface

Any driver connecting to an sEMG device must subclass the `HardwareSource` abstract base class (defined in `sdk/core/interfaces.py`):

```python
from abc import ABC, abstractmethod
from core.models import Frame

class HardwareSource(ABC):
    @abstractmethod
    def start(self) -> None:
        """Connect to device and initialize streams."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Disconnect clean and release resources."""
        pass

    @abstractmethod
    def read_frame(self, window_ms: int) -> Frame:
        """Read a buffered frame of data.
        
        Args:
            window_ms: Number of milliseconds of data to slice.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection state."""
        pass
```

---

## 2. Standard Drivers

The SDK includes several prebuilt drivers under `sdk/hardware/drivers.py`:

*   **`FileReplayDriver`**: Loops over recorded multi-channel sEMG recordings stored in CSV formats, re-simulating physical clock timestamps.
*   **`SyntheticSignalGenerator`**: Generates continuous physiological baseline noise (Gaussian white noise + 60 Hz hum) and inserts burst contractions on target channels to simulate commands.
*   **`OpenBCICytonDriver`**: Interfaces with OpenBCI Cyton boards (8-channel EEG/EMG) via the standard `brainflow` library.
*   **`DelsysTrignoDriver`**: Connects via raw TCP sockets directly to the Delsys Trigno base station command (50040) and data streaming (50043) ports.

---

## 3. Public Datasets Readers

Acquiring custom raw signals during development is challenging. The SDK provides built-in loaders under `sdk/hardware/datasets.py` to replay academic data:

1.  **Ninapro**: Replays subject files from the `.mat` MATLAB formats using SciPy.
2.  **PutEMG**: Extracts multi-channel HDF5 datasets (`.h5` files) using `h5py`.
3.  **CSL-HDEMG**: Reads high-density silent speech sEMG NumPy records.

---

## 4. Writing a Custom Driver

Here is a template for writing a custom driver (e.g. for an Arduino ADC or custom serial boards):

```python
import time
import serial
from core.interfaces import HardwareSource
from core.models import Sample, Frame

class SerialADCReader(HardwareSource):
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, fs=250.0):
        self.port = port
        self.baudrate = baudrate
        self.fs = fs
        self.ser = None
        self._connected = False

    def start(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        self._connected = True

    def stop(self):
        if self.ser:
            self.ser.close()
        self._connected = False

    def read_frame(self, window_ms: int) -> Frame:
        num_samples = int((window_ms / 1000.0) * self.fs)
        samples = []
        start_time = time.time()
        
        for _ in range(num_samples):
            if not self._connected:
                break
            # Example: Read comma-separated float line
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                channels = [float(v) for v in line.split(",")]
                samples.append(Sample(channels=channels, timestamp=time.time()))
                
        end_time = time.time()
        return Frame(samples=samples, start_time=start_time, end_time=end_time, fs=self.fs)

    def is_connected(self) -> bool:
        return self._connected
```
