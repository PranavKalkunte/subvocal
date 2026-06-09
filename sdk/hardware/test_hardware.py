"""Unit and mock integration tests for the Subvocal SDK Hardware Abstraction Layer (HAL).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import time
import tempfile
import csv
import socket
import struct
import threading

import numpy as np
from hardware.drivers import (
    FileReplayDriver,
    SyntheticSignalGenerator,
    OpenBCICytonDriver,
    DelsysTrignoDriver,
)
from hardware.datasets import PutEMGDriver


# --- Mock TCP Server for Delsys Trigno Tests ---

class MockDelsysStation:
    """Mock TCP servers mimicking a Delsys Trigno utility station."""

    def __init__(self, host: str = "127.0.0.1", cmd_port: int = 50040, data_port: int = 50043):
        self.host = host
        self.cmd_port = cmd_port
        self.data_port = data_port

        self._stop_event = threading.Event()
        self._cmd_thread = None
        self._data_thread = None

    def start(self):
        self._stop_event.clear()
        
        # Start command port listener
        self._cmd_thread = threading.Thread(target=self._run_cmd_server, daemon=True)
        self._cmd_thread.start()

        # Start data port listener
        self._data_thread = threading.Thread(target=self._run_data_server, daemon=True)
        self._data_thread.start()
        time.sleep(0.05)  # Let sockets bind

    def stop(self):
        self._stop_event.set()
        
        # Force close sockets by connecting once to trigger socket release if blocking
        for port in (self.cmd_port, self.data_port):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.host, port))
                s.close()
            except Exception:
                pass

        if self._cmd_thread:
            self._cmd_thread.join(timeout=1.0)
        if self._data_thread:
            self._data_thread.join(timeout=1.0)

    def _run_cmd_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.cmd_port))
        s.listen(1)

        while not self._stop_event.is_set():
            try:
                s.settimeout(0.1)
                conn, addr = s.accept()
                s.settimeout(None)
            except socket.timeout:
                continue
            except Exception:
                break

            try:
                data = conn.recv(1024)
                if b"START" in data:
                    conn.sendall(b"STARTING\r\n\r\n")
                elif b"STOP" in data:
                    conn.sendall(b"STOPPED\r\n\r\n")
                conn.close()
            except Exception:
                pass
        s.close()

    def _run_data_server(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.data_port))
        s.listen(1)

        while not self._stop_event.is_set():
            try:
                s.settimeout(0.1)
                conn, addr = s.accept()
                s.settimeout(None)
            except socket.timeout:
                continue
            except Exception:
                break

            # Stream some dummy floats
            try:
                conn.settimeout(0.1)
                sample_count = 0
                while not self._stop_event.is_set() and sample_count < 1000:
                    # Packet: 8 channels * 4 bytes each = 32 bytes
                    channels = [i + 0.1 for i in range(8)]
                    packet = struct.pack("<8f", *channels)
                    conn.sendall(packet)
                    sample_count += 1
                    time.sleep(0.001)
                conn.close()
            except Exception:
                pass
        s.close()


# --- Unit Tests ---

class TestHardwareAbstraction(unittest.TestCase):
    """Test suite for HAL drivers and datasets."""

    def test_file_replay_driver(self):
        """Test FileReplayDriver reading CSV records and looping correctly."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="") as tmp:
            writer = csv.writer(tmp)
            # Write a header
            writer.writerow(["ch1", "ch2", "ch3", "ch4"])
            # Write 5 samples
            for i in range(5):
                writer.writerow([float(i), float(-i), 0.5 * i, 0.1])
            tmp_path = tmp.name

        try:
            # Init driver (fs = 100 Hz for faster tests)
            driver = FileReplayDriver(file_path=tmp_path, fs=100.0, loop=True)
            self.assertEqual(driver._num_channels, 4)
            self.assertEqual(len(driver._data), 5)

            # Assert reads when not started raises RuntimeError
            with self.assertRaises(RuntimeError):
                driver.read_frame(window_ms=20)

            # Start driver
            driver.start()
            self.assertTrue(driver.is_connected())

            # Read a 30 ms frame (3 samples)
            frame = driver.read_frame(window_ms=30)
            self.assertEqual(len(frame.samples), 3)
            self.assertEqual(frame.fs, 100.0)
            self.assertEqual(frame.samples[0].channels, [0.0, 0.0, 0.0, 0.1])
            self.assertEqual(frame.samples[2].channels, [2.0, -2.0, 1.0, 0.1])

            # Read another 30 ms frame (3 samples) to trigger looping
            frame2 = driver.read_frame(window_ms=30)
            self.assertEqual(len(frame2.samples), 3)
            # Sample index 3: [3.0, -3.0, 1.5, 0.1]
            self.assertEqual(frame2.samples[0].channels, [3.0, -3.0, 1.5, 0.1])
            # Sample index 4: [4.0, -4.0, 2.0, 0.1]
            self.assertEqual(frame2.samples[1].channels, [4.0, -4.0, 2.0, 0.1])
            # Looped back to sample index 0: [0.0, 0.0, 0.0, 0.1]
            self.assertEqual(frame2.samples[2].channels, [0.0, 0.0, 0.0, 0.1])

            driver.stop()
            self.assertFalse(driver.is_connected())

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_synthetic_signal_generator(self):
        """Test SyntheticSignalGenerator continuous baseline noise and transient commands spikes."""
        gen = SyntheticSignalGenerator(fs=1000.0, num_channels=8)
        
        # Verify not started raises error
        with self.assertRaises(RuntimeError):
            gen.read_frame(window_ms=10)

        gen.start()
        self.assertTrue(gen.is_connected())

        # 1. Read baseline frame
        frame_base = gen.read_frame(window_ms=50)
        self.assertEqual(len(frame_base.samples), 50)
        arr_base = frame_base.to_numpy()
        self.assertEqual(arr_base.shape, (50, 8))
        # Baseline noise should be small
        self.assertTrue(np.all(np.abs(arr_base) < 1.0))

        # 2. Trigger a command trigger and check burst injection
        gen.trigger_command(command="clk", duration_ms=200)
        time.sleep(0.06)
        
        # Read frame immediately during the burst
        frame_burst = gen.read_frame(window_ms=50)
        arr_burst = frame_burst.to_numpy()
        
        # Zone 3 maps to channels 2 and 3 for clk command. Assert that channels 2 & 3 show much higher amplitude than others
        max_ch2 = np.max(np.abs(arr_burst[:, 2]))
        max_ch0 = np.max(np.abs(arr_burst[:, 0]))  # Unaffected channel
        self.assertTrue(max_ch2 > 3.0 * max_ch0)

        gen.stop()
        self.assertFalse(gen.is_connected())

    def test_openbci_cyton_import_error(self):
        """Verify that OpenBCICytonDriver raises ImportError gracefully if brainflow is missing."""
        # BrainFlow is not installed in the environment, so constructing it should throw ImportError
        with self.assertRaises(ImportError) as ctx:
            OpenBCICytonDriver(simulated=True)
        self.assertIn("brainflow", str(ctx.exception).lower())

    def test_putemg_import_error(self):
        """Verify that PutEMGDriver raises ImportError if h5py is missing."""
        with self.assertRaises(ImportError) as ctx:
            PutEMGDriver(file_path="fake.h5")
        self.assertIn("h5py", str(ctx.exception).lower())

    def test_delsys_trigno_driver_mock(self):
        """Test DelsysTrignoDriver connects, reads, and parses floating-point frames via local mock TCP connections."""
        mock_station = MockDelsysStation(host="127.0.0.1", cmd_port=50040, data_port=50043)
        mock_station.start()

        try:
            driver = DelsysTrignoDriver(host="127.0.0.1", cmd_port=50040, data_port=50043, num_channels=8, fs=1000.0)
            driver.start()
            self.assertTrue(driver.is_connected())

            # Read a 50 ms frame (50 samples * 8 channels = 400 floats = 1600 bytes)
            frame = driver.read_frame(window_ms=50)
            
            # Assert frame properties
            self.assertTrue(len(frame.samples) > 0)
            self.assertEqual(frame.fs, 1000.0)
            self.assertEqual(len(frame.samples[0].channels), 8)
            # The mock sends channels as floats from 0.1 to 7.1
            self.assertAlmostEqual(frame.samples[0].channels[0], 0.1, places=3)
            self.assertAlmostEqual(frame.samples[0].channels[7], 7.1, places=3)

            driver.stop()
            self.assertFalse(driver.is_connected())
        finally:
            mock_station.stop()

    def test_file_replay_driver_no_loop(self):
        """Test FileReplayDriver with loop=False pads with the last sample after EOF."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, newline="") as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["ch1", "ch2"])
            writer.writerow([1.0, 2.0])
            writer.writerow([3.0, 4.0])
            tmp_path = tmp.name

        try:
            driver = FileReplayDriver(file_path=tmp_path, fs=100.0, loop=False)
            driver.start()

            # Read 2 samples (exhausts the file)
            frame1 = driver.read_frame(window_ms=20)
            self.assertEqual(frame1.samples[0].channels, [1.0, 2.0])
            self.assertEqual(frame1.samples[1].channels, [3.0, 4.0])

            # Read 2 more — driver should pad with the last row [3.0, 4.0]
            frame2 = driver.read_frame(window_ms=20)
            self.assertEqual(frame2.samples[0].channels, [3.0, 4.0])
            self.assertEqual(frame2.samples[1].channels, [3.0, 4.0])

            driver.stop()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
