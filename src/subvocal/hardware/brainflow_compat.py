import enum
import json
import logging
import math
import random
import threading
import time

import numpy as np

logger = logging.getLogger("subvocal.hardware.brainflow_compat")


class BoardIds(enum.IntEnum):
    NO_BOARD = -100
    PLAYBACK_FILE_BOARD = -3
    STREAMING_BOARD = -2
    SYNTHETIC_BOARD = -1
    CYTON_BOARD = 0
    GANGLION_BOARD = 1
    CYTON_DAISY_BOARD = 2
    GALEA_BOARD = 3
    GANGLION_WIFI_BOARD = 4
    CYTON_WIFI_BOARD = 5
    CYTON_DAISY_WIFI_BOARD = 6
    BRAINBIT_BOARD = 7
    UNICORN_BOARD = 8
    CALLIBRI_EEG_BOARD = 9
    CALLIBRI_EMG_BOARD = 10
    CALLIBRI_ECG_BOARD = 11


class LogLevels(enum.IntEnum):
    LEVEL_TRACE = 0
    LEVEL_DEBUG = 1
    LEVEL_INFO = 2
    LEVEL_WARN = 3
    LEVEL_ERROR = 4
    LEVEL_CRITICAL = 5
    LEVEL_OFF = 6


class BrainFlowPresets(enum.IntEnum):
    DEFAULT_PRESET = 0
    AUXILIARY_PRESET = 1
    ANCILLARY_PRESET = 2


class BrainFlowError(Exception):
    def __init__(self, message: str, exit_code: int):
        super().__init__(f"{message} (Exit Code: {exit_code})")
        self.exit_code = exit_code


class BrainFlowInputParams:
    def __init__(self) -> None:
        self.serial_port = ""
        self.mac_address = ""
        self.ip_address = ""
        self.ip_address_aux = ""
        self.ip_address_anc = ""
        self.ip_port = 0
        self.ip_port_aux = 0
        self.ip_port_anc = 0
        self.ip_protocol = 0
        self.other_info = ""
        self.timeout = 0
        self.serial_number = ""
        self.file = ""
        self.file_aux = ""
        self.file_anc = ""
        self.master_board = BoardIds.NO_BOARD.value

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, indent=4)


# Metadata describing common boards (sampling rate, channels, total rows)
BOARD_DESCRIPTIONS = {
    BoardIds.SYNTHETIC_BOARD.value: {
        "sampling_rate": 250,
        "package_num_channel": 0,
        "battery_channel": 29,
        "timestamp_channel": 30,
        "marker_channel": 31,
        "num_rows": 32,
        "eeg_channels": list(range(1, 17)),
        "emg_channels": list(range(1, 17)),
        "ecg_channels": list(range(1, 17)),
        "eog_channels": list(range(1, 17)),
        "accel_channels": [17, 18, 19],
        "gyro_channels": [20, 21, 22],
        "eda_channels": [23],
        "ppg_channels": [24, 25],
        "temperature_channels": [26],
        "resistance_channels": [27, 28],
    },
    BoardIds.CYTON_BOARD.value: {
        "sampling_rate": 250,
        "package_num_channel": 0,
        "timestamp_channel": 22,
        "marker_channel": 23,
        "num_rows": 24,
        "eeg_channels": list(range(1, 9)),
        "emg_channels": list(range(1, 9)),
        "ecg_channels": list(range(1, 9)),
        "eog_channels": list(range(1, 9)),
        "accel_channels": [9, 10, 11],
        "analog_channels": [19, 20, 21],
        "other_channels": list(range(12, 19)),
    },
    BoardIds.GANGLION_BOARD.value: {
        "sampling_rate": 200,
        "package_num_channel": 0,
        "timestamp_channel": 13,
        "marker_channel": 14,
        "num_rows": 15,
        "eeg_channels": list(range(1, 5)),
        "emg_channels": list(range(1, 5)),
        "ecg_channels": list(range(1, 5)),
        "eog_channels": list(range(1, 5)),
        "accel_channels": [5, 6, 7],
        "resistance_channels": [8, 9, 10, 11, 12],
    },
}


class BoardShim:
    """Fallback compatible BoardShim class implementing a pure-Python biosensor stream."""

    def __init__(self, board_id: int, input_params: BrainFlowInputParams) -> None:
        self.board_id = board_id
        self.input_params = input_params
        self._connected = False
        self._streaming = False
        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._keep_alive = False
        self._serial_conn = None

        # Check if the official brainflow package is available
        self._native_shim = None
        try:
            import brainflow  # noqa: F401
            from brainflow.board_shim import BoardShim as NativeBoardShim
            from brainflow.board_shim import BrainFlowInputParams as NativeInputParams

            native_params = NativeInputParams()
            for key, val in input_params.__dict__.items():
                if hasattr(native_params, key):
                    setattr(native_params, key, val)

            self._native_shim = NativeBoardShim(board_id, native_params)
            logger.info("Successfully bound official C++ BrainFlow native backend proxy.")
        except ImportError:
            logger.info("BrainFlow official library not found; running pure-Python emulator mode.")

    def prepare_session(self) -> None:
        if self._native_shim:
            self._native_shim.prepare_session()
            self._connected = True
            return

        if self._connected:
            raise BrainFlowError("Session already prepared.", 2)

        if self.board_id not in BOARD_DESCRIPTIONS:
            raise BrainFlowError(f"Unsupported board ID: {self.board_id}", 3)

        self._buffer = []
        self._connected = True

    def start_stream(self, buffer_size: int = 200000, streamer_params: str = "") -> None:
        if self._native_shim:
            self._native_shim.start_stream(buffer_size, streamer_params)
            self._streaming = True
            return

        if not self._connected:
            raise BrainFlowError("Session is not prepared.", 4)
        if self._streaming:
            raise BrainFlowError("Stream already running.", 5)

        self._keep_alive = True
        self._streaming = True

        if self.board_id == BoardIds.SYNTHETIC_BOARD.value:
            self._thread = threading.Thread(target=self._run_synthetic_thread)
            self._thread.daemon = True
            self._thread.start()
        elif self.board_id == BoardIds.CYTON_BOARD.value:
            self._thread = threading.Thread(target=self._run_cyton_serial_thread)
            self._thread.daemon = True
            self._thread.start()
        else:
            raise BrainFlowError(f"Streaming not implemented for board: {self.board_id}", 3)

    def stop_stream(self) -> None:
        if self._native_shim:
            self._native_shim.stop_stream()
            self._streaming = False
            return

        if not self._streaming:
            return

        self._keep_alive = False
        if self._thread:
            self._thread.join()
            self._thread = None
        self._streaming = False

    def release_session(self) -> None:
        if self._native_shim:
            self._native_shim.release_session()
            self._connected = False
            return

        if self._connected:
            self.stop_stream()
            self._connected = False

    def get_board_data(self, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> np.ndarray:
        if self._native_shim:
            return self._native_shim.get_board_data(preset)

        with self._buffer_lock:
            if not self._buffer:
                num_rows = self.get_num_rows(self.board_id)
                return np.empty((num_rows, 0))
            data = np.column_stack(self._buffer)
            self._buffer.clear()
            return data

    def get_current_board_data(
        self, num_samples: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value
    ) -> np.ndarray:
        if self._native_shim:
            return self._native_shim.get_current_board_data(num_samples, preset)

        with self._buffer_lock:
            num_rows = self.get_num_rows(self.board_id)
            if not self._buffer:
                return np.empty((num_rows, 0))
            samples = self._buffer[-num_samples:]
            return np.column_stack(samples)

    def is_prepared(self) -> bool:
        if self._native_shim:
            return self._native_shim.is_prepared()
        return self._connected

    # Description methods
    @classmethod
    def get_sampling_rate(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> int:
        if board_id in BOARD_DESCRIPTIONS:
            return BOARD_DESCRIPTIONS[board_id]["sampling_rate"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_num_rows(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> int:
        if board_id in BOARD_DESCRIPTIONS:
            return BOARD_DESCRIPTIONS[board_id]["num_rows"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_eeg_channels(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> list[int]:
        if board_id in BOARD_DESCRIPTIONS and "eeg_channels" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["eeg_channels"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_exg_channels(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> list[int]:
        if board_id in BOARD_DESCRIPTIONS and "eeg_channels" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["eeg_channels"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_emg_channels(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> list[int]:
        if board_id in BOARD_DESCRIPTIONS and "emg_channels" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["emg_channels"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_timestamp_channel(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> int:
        if board_id in BOARD_DESCRIPTIONS and "timestamp_channel" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["timestamp_channel"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_package_num_channel(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> int:
        if board_id in BOARD_DESCRIPTIONS and "package_num_channel" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["package_num_channel"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_battery_channel(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> int:
        if board_id in BOARD_DESCRIPTIONS and "battery_channel" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["battery_channel"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_accel_channels(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> list[int]:
        if board_id in BOARD_DESCRIPTIONS and "accel_channels" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["accel_channels"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    @classmethod
    def get_gyro_channels(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET.value) -> list[int]:
        if board_id in BOARD_DESCRIPTIONS and "gyro_channels" in BOARD_DESCRIPTIONS[board_id]:
            return BOARD_DESCRIPTIONS[board_id]["gyro_channels"]
        raise BrainFlowError(f"Unsupported board ID: {board_id}", 3)

    # Background Generation Threads
    def _run_synthetic_thread(self) -> None:
        desc = BOARD_DESCRIPTIONS[self.board_id]
        sr = desc["sampling_rate"]
        num_rows = desc["num_rows"]
        sleep_sec = 1.0 / sr

        counter = 0
        eeg_chans = desc["eeg_channels"]
        phases = [0.0] * len(eeg_chans)

        while self._keep_alive:
            t_start = time.perf_counter()

            package = np.zeros(num_rows)
            package[desc["package_num_channel"]] = float(counter % 256)

            # Generate ExG signals (sinusoids + white noise)
            for idx, ch in enumerate(eeg_chans):
                amp = 10.0 * (idx + 1)
                freq = 5.0 * (idx + 1)
                shift = 0.05 * idx
                noise_range = (amp * 0.1) / 2.0
                noise_val = random.uniform(-noise_range, noise_range)

                phases[idx] += 2.0 * math.pi * freq / sr
                if phases[idx] > 2.0 * math.pi:
                    phases[idx] -= 2.0 * math.pi

                package[ch] = amp + (amp + noise_val) * math.sqrt(2.0) * math.sin(phases[idx] + shift)

            # Accelerometer and Gyro
            for ch in desc["accel_channels"]:
                package[ch] = random.uniform(0.9, 1.1) - 0.1
            for ch in desc["gyro_channels"]:
                package[ch] = random.uniform(0.9, 1.1) - 0.1

            # EDA, PPG, Temperature, Resistance
            package[desc["eda_channels"][0]] = random.uniform(0.9, 1.1)
            package[desc["ppg_channels"][0]] = 500.0 * random.uniform(0.9, 1.1)
            package[desc["ppg_channels"][1]] = 253500.0 * random.uniform(0.9, 1.1)
            package[desc["temperature_channels"][0]] = random.uniform(0.9, 1.1) / 10.0 + 36.5
            for ch in desc["resistance_channels"]:
                package[ch] = 1000.0 * random.uniform(0.9, 1.1)

            package[desc["battery_channel"]] = (random.uniform(0.9, 1.1) - 0.1) * 100.0
            package[desc["timestamp_channel"]] = time.time()

            with self._buffer_lock:
                self._buffer.append(package)

            counter += 1
            elapsed = time.perf_counter() - t_start
            wait_time = max(0.0, sleep_sec - elapsed)
            time.sleep(wait_time)

    def _run_cyton_serial_thread(self) -> None:
        """Parses OpenBCI Cyton serial byte streams directly."""
        try:
            import serial
        except ImportError:
            logger.error("pyserial is required to connect to a physical CYTON_BOARD serial port.")
            return

        port = self.input_params.serial_port
        if not port:
            logger.error("No serial port specified in BrainFlowInputParams.")
            return

        desc = BOARD_DESCRIPTIONS[self.board_id]
        num_rows = desc["num_rows"]

        try:
            self._serial_conn = serial.Serial(port, 115200, timeout=1)
            # Send stop command, reset command, and start streaming command
            self._serial_conn.write(b"s")
            time.sleep(0.1)
            self._serial_conn.write(b"v")  # Reset board to default parameters
            time.sleep(0.5)
            # Empty read buffer
            self._serial_conn.reset_input_buffer()
            self._serial_conn.write(b"b")  # Start binary streaming
        except Exception as e:
            logger.error(f"Failed to open serial port {port}: {e}")
            return

        # Cyton ExG scale factor (Volts to microvolts at Default Gain 24)
        exg_scale = (4.5 / 24.0 / (2**23 - 1)) * 1000000.0
        # Accel scale factor (typical +/- 4G scale -> 0.002 G/count)
        accel_scale = 0.002

        def parse_24bit_signed(b1, b2, b3):
            val = (b1 << 16) | (b2 << 8) | b3
            if val & 0x800000:
                val -= 0x1000000
            return val

        def parse_16bit_signed(b1, b2):
            val = (b1 << 8) | b2
            if val & 0x8000:
                val -= 0x10000
            return val

        while self._keep_alive:
            # Cyton packet is 33 bytes:
            # Byte 0: 0xA0 (Start)
            # Byte 1: Sample Counter
            # Bytes 2-25: 8 ExG channels (3 bytes each)
            # Bytes 26-31: 3 Aux channels (2 bytes each)
            # Byte 32: Stop Byte (e.g. 0xC0)
            try:
                start_byte = self._serial_conn.read(1)
                if not start_byte or start_byte[0] != 0xA0:
                    continue

                payload = self._serial_conn.read(32)
                if len(payload) < 32:
                    continue

                stop_byte = payload[31]
                if stop_byte < 0xC0 or stop_byte > 0xCF:
                    continue  # Invalid packet alignment, re-sync

                package = np.zeros(num_rows)
                package[desc["package_num_channel"]] = float(payload[0])

                # 8 ExG Channels
                for idx in range(8):
                    offset = 1 + idx * 3
                    b1 = payload[offset]
                    b2 = payload[offset + 1]
                    b3 = payload[offset + 2]
                    raw_val = parse_24bit_signed(b1, b2, b3)
                    package[desc["eeg_channels"][idx]] = raw_val * exg_scale

                # 3 Accelerometer / Aux channels
                for idx in range(3):
                    offset = 25 + idx * 2
                    b1 = payload[offset]
                    b2 = payload[offset + 1]
                    raw_val = parse_16bit_signed(b1, b2)
                    package[desc["accel_channels"][idx]] = raw_val * accel_scale

                package[desc["timestamp_channel"]] = time.time()

                with self._buffer_lock:
                    self._buffer.append(package)

            except Exception as e:
                logger.error(f"Error reading serial stream: {e}")
                break

        # Stop streaming and clean close
        try:
            self._serial_conn.write(b"s")
            time.sleep(0.1)
            self._serial_conn.close()
        except Exception:
            pass
