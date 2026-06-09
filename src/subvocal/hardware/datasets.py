"""Dataset drivers for public electromyography databases (Ninapro, PutEMG, CSL-HDEMG).
"""

import os
import time

import numpy as np

from subvocal.core.interfaces import HardwareSource
from subvocal.core.models import Frame, Sample
from subvocal.exceptions import HardwareError, MissingDependencyError


class NinaproDriver(HardwareSource):
    """Streams electromyography signals from Ninapro MATLAB (.mat) files."""

    def __init__(self, file_path: str, fs: float = 2000.0, loop: bool = True):
        """Initializes the Ninapro driver.

        Args:
            file_path: Path to the downloaded Ninapro subject MATLAB .mat file.
            fs: Sampling frequency (default: 2000.0 Hz for DB2/DB3, DB1 is 100 Hz).
            loop: Boolean representing whether to loop data when EOF is reached.
        """
        self.file_path = os.path.abspath(file_path)
        self.fs = fs
        self.loop = loop

        self._connected = False
        self._index = 0
        self._sample_counter = 0

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Ninapro dataset file not found: {self.file_path}")

        # scipy is loaded dynamically so a base install can import this module
        try:
            import scipy.io
        except ImportError as e:
            raise MissingDependencyError(
                "scipy is required to use the NinaproDriver. "
                'Install it with: pip install "subvocal[hardware]"'
            ) from e

        # Load mat file
        try:
            mat = scipy.io.loadmat(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse Ninapro MAT file: {e}") from e

        # Ninapro files store raw sEMG signals in the 'emg' key
        if "emg" not in mat:
            raise KeyError(
                f"MAT file does not contain 'emg' key. Keys found: {list(mat.keys())}"
            )

        self._emg_data = mat["emg"]  # Shape: (num_samples, num_channels)
        self.num_channels = self._emg_data.shape[1]

    def start(self) -> None:
        self._connected = True
        self._index = 0

    def stop(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected:
            raise HardwareError("Ninapro stream is not started.")

        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        samples = []

        for _ in range(num_samples):
            if self._index >= self._emg_data.shape[0]:
                if self.loop:
                    self._index = 0
                else:
                    self._index = self._emg_data.shape[0] - 1

            self._sample_counter += 1
            # Read row values
            channels = [float(val) for val in self._emg_data[self._index]]
            samples.append(
                Sample(
                    timestamp=now - ((num_samples - len(samples)) / self.fs),
                    channels=channels,
                    sample_index=self._sample_counter
                )
            )
            self._index += 1

        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self.fs
        )


class PutEMGDriver(HardwareSource):
    """Streams electromyography signals from PutEMG HDF5 (.h5) files."""

    def __init__(self, file_path: str, fs: float = 5120.0, loop: bool = True):
        """Initializes the PutEMG driver.

        Loads h5py dynamically.
        """
        self.file_path = os.path.abspath(file_path)
        self.fs = fs
        self.loop = loop

        self._connected = False
        self._index = 0
        self._sample_counter = 0

        # Load h5py dynamically
        try:
            import h5py
            self._h5py = h5py
        except ImportError as e:
            raise MissingDependencyError(
                "h5py is required to use the PutEMGDriver. "
                'Install it with: pip install "subvocal[hardware]"'
            ) from e

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PutEMG dataset file not found: {self.file_path}")

        # Read HDF5 file
        try:
            self._h5_file = self._h5py.File(self.file_path, "r")
            # PutEMG datasets store signals inside a root-level dataset, e.g. 'emg' or inside groups
            # We look for datasets with a 2D shape containing multi-channel values
            self._emg_dataset = None
            
            # Helper to search for datasets
            def find_emg(name, obj):
                if self._emg_dataset is None and isinstance(obj, self._h5py.Dataset) and len(obj.shape) == 2 and obj.shape[1] > 1:
                    self._emg_dataset = obj

            self._h5_file.visititems(find_emg)

            if self._emg_dataset is None:
                raise KeyError("Could not locate any valid multi-channel sEMG datasets inside the PutEMG HDF5 file.")
                
            # Read first chunk to get channel configuration
            self.num_channels = self._emg_dataset.shape[1]
            self._total_samples = self._emg_dataset.shape[0]

        except Exception as e:
            raise ValueError(f"Failed to open/parse PutEMG file: {e}") from e

    def start(self) -> None:
        self._connected = True
        self._index = 0

    def stop(self) -> None:
        if hasattr(self, "_h5_file") and self._h5_file:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
            self._emg_dataset = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected or self._emg_dataset is None:
            raise HardwareError("PutEMG stream is not started.")

        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        samples = []

        # Read dataset chunk-by-chunk in a single slicing operation for better I/O performance
        end_idx = self._index + num_samples
        if end_idx <= self._total_samples:
            chunk = self._emg_dataset[self._index : end_idx]
            self._index = end_idx
        else:
            if self.loop:
                # Loop around: slice the remainder and the wrap-around start
                rem_len = self._total_samples - self._index
                chunk_rem = self._emg_dataset[self._index : self._total_samples]
                
                needed = num_samples - rem_len
                chunk_start = self._emg_dataset[0 : needed]
                
                chunk = np.concatenate((chunk_rem, chunk_start), axis=0)
                self._index = needed
            else:
                # Pad with the last available values
                chunk = self._emg_dataset[self._index : self._total_samples]
                padding = np.repeat(self._emg_dataset[-1:], num_samples - len(chunk), axis=0)
                chunk = np.concatenate((chunk, padding), axis=0)
                self._index = self._total_samples - 1

        for idx in range(num_samples):
            self._sample_counter += 1
            channels = [float(val) for val in chunk[idx]]
            samples.append(
                Sample(
                    timestamp=now - ((num_samples - idx) / self.fs),
                    channels=channels,
                    sample_index=self._sample_counter
                )
            )

        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self.fs
        )


class CSLHDEMGDriver(HardwareSource):
    """Streams electromyography signals from CSL-HDEMG binary or NumPy (.npy) files."""

    def __init__(self, file_path: str, fs: float = 2000.0, loop: bool = True, num_channels: int = 8):
        """Initializes the CSL-HDEMG driver.

        Args:
            file_path: Path to the NumPy (.npy) or raw binary float data file.
            fs: Sample rate of the recording.
            loop: Boolean representing whether to loop the data when EOF is reached.
            num_channels: Number of channels in the raw binary file (ignored for .npy, which encodes shape).
        """
        self.file_path = os.path.abspath(file_path)
        self.fs = fs
        self.loop = loop
        self._num_channels = num_channels

        self._connected = False
        self._index = 0
        self._sample_counter = 0

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSL-HDEMG dataset file not found: {self.file_path}")

        # Load file content
        try:
            if self.file_path.endswith(".npy"):
                self._data = np.load(self.file_path)
            else:
                # Read raw binary floats (float32, num_channels channels)
                raw_floats = np.fromfile(self.file_path, dtype=np.float32)
                self._data = raw_floats.reshape(-1, num_channels)
        except Exception as e:
            raise ValueError(f"Failed to load CSL-HDEMG dataset array: {e}") from e

        self.num_channels = self._data.shape[1]
        self._total_samples = self._data.shape[0]

    def start(self) -> None:
        self._connected = True
        self._index = 0

    def stop(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected:
            raise HardwareError("CSL-HDEMG stream is offline.")

        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        samples = []

        end_idx = self._index + num_samples
        if end_idx <= self._total_samples:
            chunk = self._data[self._index : end_idx]
            self._index = end_idx
        else:
            if self.loop:
                rem_len = self._total_samples - self._index
                chunk_rem = self._data[self._index : self._total_samples]
                needed = num_samples - rem_len
                chunk_start = self._data[0 : needed]
                chunk = np.concatenate((chunk_rem, chunk_start), axis=0)
                self._index = needed
            else:
                chunk = self._data[self._index : self._total_samples]
                padding = np.repeat(self._data[-1:], num_samples - len(chunk), axis=0)
                chunk = np.concatenate((chunk, padding), axis=0)
                self._index = self._total_samples - 1

        for idx in range(num_samples):
            self._sample_counter += 1
            channels = [float(val) for val in chunk[idx]]
            samples.append(
                Sample(
                    timestamp=now - ((num_samples - idx) / self.fs),
                    channels=channels,
                    sample_index=self._sample_counter
                )
            )

        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self.fs
        )
