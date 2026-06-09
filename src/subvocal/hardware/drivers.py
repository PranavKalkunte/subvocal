"""Concrete implementations of HardwareSource drivers for the Subvocal SDK.
"""

import csv
import os
import socket
import struct
import time
from typing import Any

import numpy as np

from subvocal.core.interfaces import HardwareSource
from subvocal.core.models import Frame, Sample


class FileReplayDriver(HardwareSource):
    """Replays raw sEMG data from a CSV file, simulating real-time acquisition."""

    def __init__(self, file_path: str, fs: float = 1000.0, loop: bool = True):
        """Initializes the file replay driver.

        Args:
            file_path: Path to the recorded CSV file.
            fs: Sample rate of the recording.
            loop: Boolean representing whether to loop the data when EOF is reached.
        """
        self.file_path = os.path.abspath(file_path)
        self.fs = fs
        self.loop = loop

        self._connected = False
        self._data: list[list[float]] = []
        self._index = 0
        self._sample_counter = 0

        # Load file contents into memory
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Recorded sEMG file not found: {self.file_path}")

        with open(self.file_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].strip() or row[0].strip().isalpha():
                    continue  # Skip headers or empty lines
                try:
                    self._data.append([float(val) for val in row])
                except ValueError:
                    continue  # Skip rows with parsing issues

        if not self._data:
            raise ValueError(f"No valid numeric data found in file: {self.file_path}")

        self._num_channels = len(self._data[0])

    def start(self) -> None:
        self._connected = True
        self._index = 0

    def stop(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected:
            raise RuntimeError("Driver is not started. Call start() before reading frames.")

        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        samples = []

        for _ in range(num_samples):
            if self._index >= len(self._data):
                if self.loop:
                    self._index = 0
                else:
                    # Pad with last sample
                    self._index = len(self._data) - 1

            self._sample_counter += 1
            channels = self._data[self._index]
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


class SyntheticSignalGenerator(HardwareSource):
    """Generates continuous physiological sEMG noise and injects command triggers."""

    def __init__(self, fs: float = 1000.0, num_channels: int = 8):
        """Initializes the synthetic signal generator.

        Args:
            fs: Sample frequency in Hz.
            num_channels: Number of channels to generate (maps to 5-zone layouts).
        """
        self.fs = fs
        self.num_channels = num_channels
        self._connected = False
        self._sample_counter = 0

        # Command contraction injection states: maps channel index to multiplier
        self._active_injections: list[dict[str, Any]] = []

    def start(self) -> None:
        self._connected = True
        self._sample_counter = 0

    def stop(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def trigger_command(self, command: str, duration_ms: int = 300) -> None:
        """Injects a high-amplitude transient signal profile on selected channels.

        Args:
            command: Command name (e.g., 'gt', 'clk', 'srch', 'typ').
            duration_ms: Burst duration in milliseconds.
        """
        now = time.time()
        
        # Define muscle zone mapping for commands (channel activations)
        channel_mapping = {
            "gt": [0, 1],       # GOTO uses submental Zone 1
            "clk": [2, 3],      # CLICK uses mid-throat Zone 3
            "srch": [4, 5],     # SEARCH uses larynx flanks Zone 4
            "typ": [6, 7]       # TYPE uses infrahyoid strap Zone 5
        }

        active_ch = channel_mapping.get(command.lower(), [0])
        
        self._active_injections.append({
            "channels": active_ch,
            "start_time": now,
            "duration": duration_ms / 1000.0,
            "amplitude": 3.5  # Signal multiplier factor
        })

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected:
            raise RuntimeError("Signal generator not started.")

        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        samples = []

        # Remove expired injections
        self._active_injections = [
            inj for inj in self._active_injections 
            if now - inj["start_time"] <= inj["duration"]
        ]

        for idx in range(num_samples):
            self._sample_counter += 1
            t = self._sample_counter / self.fs
            sample_time = now - ((num_samples - idx) / self.fs)

            # 1. Base physiological muscle noise (low amplitude white noise)
            raw_channels = [float(np.random.normal(0.0, 0.05)) for _ in range(self.num_channels)]

            # 2. Add 60 Hz powerline notch noise component
            for ch in range(self.num_channels):
                raw_channels[ch] += float(0.02 * np.sin(2 * np.pi * 60.0 * t))

            # 3. Apply active command muscle contractions
            for inj in self._active_injections:
                elapsed = sample_time - inj["start_time"]
                if 0 <= elapsed <= inj["duration"]:
                    # Create bell-shaped contraction envelope
                    x = elapsed / inj["duration"]
                    envelope = np.sin(np.pi * x)  # peaks in the middle
                    
                    # Synthesize sEMG burst (amplitude modulated high-frequency burst)
                    burst = float(inj["amplitude"] * envelope * np.sin(2 * np.pi * 80.0 * t) * np.random.normal(1.0, 0.2))

                    for ch in inj["channels"]:
                        if ch < self.num_channels:
                            raw_channels[ch] += burst

            samples.append(
                Sample(
                    timestamp=sample_time,
                    channels=raw_channels,
                    sample_index=self._sample_counter
                )
            )

        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self.fs
        )


class OpenBCICytonDriver(HardwareSource):
    """Acquires raw biosignals from OpenBCI Cyton boards via BrainFlow."""

    def __init__(self, port: str | None = None, simulated: bool = True):
        """Initializes the Cyton driver.

        Loads BrainFlow dynamically.
        """
        self.port = port
        self.simulated = simulated
        self._connected = False
        self._board = None

        # Load BrainFlow dynamically
        try:
            import brainflow
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
            self._brainflow = brainflow
            self._BoardShim = BoardShim
            self._params = BrainFlowInputParams()
            if self.simulated:
                self._board_id = BoardIds.SYNTHETIC_BOARD
            else:
                self._board_id = BoardIds.CYTON_BOARD
                if self.port:
                    self._params.serial_port = self.port
        except ImportError as e:
            raise ImportError(
                "BrainFlow is required to use the OpenBCICytonDriver. "
                "Please install it using: pip install brainflow"
            ) from e

    def start(self) -> None:
        if self._connected:
            return

        # Initialize the board connection
        self._board = self._BoardShim(self._board_id, self._params)
        self._board.prepare_session()
        self._board.start_stream()
        self._connected = True

    def stop(self) -> None:
        if not self._connected or not self._board:
            return

        try:
            self._board.stop_stream()
            self._board.release_session()
        except Exception:
            pass
        finally:
            self._board = None
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected or not self._board:
            raise RuntimeError("OpenBCI board is not streaming. Call start().")

        now = time.time()
        fs = self._BoardShim.get_sampling_rate(self._board_id)
        num_samples = int((window_ms / 1000.0) * fs)

        # Pull raw board data
        data = self._board.get_current_board_data(num_samples)
        
        # Cyton sEMG channels are usually mapped to rows 1 to 8 in BrainFlow
        exg_channels = self._BoardShim.get_exg_channels(self._board_id)
        
        # Extract rows
        samples = []
        actual_samples = data.shape[1]
        
        for idx in range(actual_samples):
            channels = [float(data[ch][idx]) for ch in exg_channels]
            samples.append(
                Sample(
                    timestamp=now - ((actual_samples - idx) / fs),
                    channels=channels,
                    sample_index=int(data[self._BoardShim.get_package_num_channel(self._board_id)][idx])
                )
            )

        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=fs
        )


class DelsysTrignoDriver(HardwareSource):
    """Direct TCP Socket driver connecting to the Delsys Trigno Control Utility."""

    def __init__(self, host: str = "localhost", data_port: int = 50043, cmd_port: int = 50040, num_channels: int = 8, fs: float = 2000.0):
        """Initializes the Delsys socket driver."""
        self.host = host
        self.data_port = data_port
        self.cmd_port = cmd_port
        self.num_channels = num_channels
        self.fs = fs

        self._connected = False
        self._data_socket: socket.socket | None = None
        self._cmd_socket: socket.socket | None = None
        self._sample_counter = 0

    def start(self) -> None:
        if self._connected:
            return

        try:
            # 1. Connect to command server to initiate start
            self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_socket.connect((self.host, self.cmd_port))
            # Handshake / start command
            self._cmd_socket.sendall(b"START\r\n\r\n")
            time.sleep(0.1)  # brief wait for station to arm

            # 2. Connect to raw data server
            self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._data_socket.connect((self.host, self.data_port))
            # Set non-blocking to prevent locking during reads
            self._data_socket.setblocking(False)

            self._connected = True
            self._sample_counter = 0
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Failed to connect to Delsys Trigno Utility at {self.host}: {e}") from e

    def stop(self) -> None:
        # Send STOP to command station
        if self._cmd_socket:
            try:
                self._cmd_socket.sendall(b"STOP\r\n\r\n")
                self._cmd_socket.close()
            except Exception:
                pass
            self._cmd_socket = None

        if self._data_socket:
            try:
                self._data_socket.close()
            except Exception:
                pass
            self._data_socket = None

        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected or not self._data_socket:
            raise RuntimeError("Trigno data stream is offline.")

        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        
        # Delsys streams single-precision floats (4 bytes) per channel
        # Format: [Ch1, Ch2, ..., ChN] as floats. Packet size = num_channels * 4 bytes
        packet_size = self.num_channels * 4
        total_bytes_needed = num_samples * packet_size
        
        raw_data = bytearray()
        start_read = time.time()

        # Read from socket until buffer is filled, or timeout expires
        while len(raw_data) < total_bytes_needed:
            # Enforce 100ms reading timeout to prevent infinite loops
            if time.time() - start_read > 0.1:
                break
            try:
                chunk = self._data_socket.recv(total_bytes_needed - len(raw_data))
                if not chunk:
                    break
                raw_data.extend(chunk)
            except BlockingIOError:
                # No data ready yet, yield CPU briefly
                time.sleep(0.001)

        # Unpack binary data
        samples = []
        parsed_samples = len(raw_data) // packet_size
        if parsed_samples == 0:
            raise RuntimeError(
                f"DelsysTrignoDriver: no data received from {self.host}:{self.data_port} within 100 ms timeout"
            )

        for idx in range(parsed_samples):
            self._sample_counter += 1
            offset = idx * packet_size
            channels = []
            for ch in range(self.num_channels):
                val_bytes = raw_data[offset + (ch * 4) : offset + (ch * 4) + 4]
                if len(val_bytes) == 4:
                    val = struct.unpack("<f", val_bytes)[0]
                    channels.append(float(val))
                else:
                    channels.append(0.0)

            samples.append(
                Sample(
                    timestamp=now - ((parsed_samples - idx) / self.fs),
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
