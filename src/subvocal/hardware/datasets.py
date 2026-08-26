"""Dataset drivers for public electromyography databases (Ninapro, PutEMG, CSL-HDEMG, Gaddy).

GaddyDriver supports the Silent Speech EMG dataset (Gaddy & Klein 2020,
Zenodo 4064409): 8-channel facial EMG at 1000 Hz (resampled 800 Hz in ACL 2021
model) paired with audio and info.json transcriptions.
"""

import json
import os
import re
import time
from pathlib import Path

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
        # H5: Ensure file is closed if visititems or parsing raises (resource leak fix)
        self._h5_file = None  # type: ignore
        self._emg_dataset = None  # type: ignore
        try:
            self._h5_file = self._h5py.File(self.file_path, "r")
            # PutEMG datasets store signals inside a root-level dataset, e.g. 'emg' or inside groups
            # We look for datasets with a 2D shape containing multi-channel values
            self._emg_dataset = None
            
            # Helper to search for datasets
            def find_emg(name, obj):
                if self._emg_dataset is None and isinstance(obj, self._h5py.Dataset) and len(obj.shape) == 2 and obj.shape[1] > 1:
                    self._emg_dataset = obj

            try:
                self._h5_file.visititems(find_emg)
            except Exception:
                # visititems raised - close file to avoid descriptor leak
                try:
                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None
                self._emg_dataset = None
                raise

            if self._emg_dataset is None:
                # No dataset found - close file before raising to avoid leak
                try:
                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None
                raise KeyError("Could not locate any valid multi-channel sEMG datasets inside the PutEMG HDF5 file.")
                
            # Read first chunk to get channel configuration
            self.num_channels = self._emg_dataset.shape[1]
            self._total_samples = self._emg_dataset.shape[0]

        except Exception as e:
            # Ensure file closed on any parsing failure
            if getattr(self, "_h5_file", None) is not None:
                try:
                    self._h5_file.close()  # type: ignore
                except Exception:
                    pass
                self._h5_file = None  # type: ignore
                self._emg_dataset = None  # type: ignore
            # Preserve already-wrapped ValueError
            if isinstance(e, ValueError) and "Failed to open/parse PutEMG file" in str(e):
                raise
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
                # Use allow_pickle=False to prevent arbitrary code execution via pickle (C1)
                # Only raw numeric arrays are expected for CSL-HDEMG; object arrays are rejected
                self._data = np.load(self.file_path, allow_pickle=False)
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


class GaddyDriver(HardwareSource):
    """Streams EMG from the Gaddy & Klein silent-speech dataset (Zenodo 4064409).

    Dataset layout (per Zenodo): each utterance provides
    ``{id}_emg.npy`` shape ``(T, 8)`` raw EMG at 1000 Hz (ACL 2021 resampled
    to 800 Hz), ``{id}_audio.flac`` / ``_audio_clean.flac``, and
    ``{id}_info.json`` (contains ``text``/``transcript``/``prompt``). Utterances
    are sampled for both silent (``ES``) and vocalized (``EV``) modes; this
    driver indexes all ``*_emg.npy`` recursively and can filter by split.

    Streaming emulates hardware by sequentially concatenating utterances and
    returning windowed :class:`Frame` objects.

    Args:
        data_dir: Root directory of the extracted Zenodo archive (contains
            ``*_emg.npy`` files recursively).
        fs: Sampling frequency in Hz. Raw data is 1000 Hz; ACL 2021 model
            resamples to 800 Hz. Defaults to 1000.0.
        loop: Whether to loop to the first utterance when EOF is reached.
        split: Optional filter — ``"silent"`` or ``"vocalized"``. When set,
            only utterances whose path or ``info.json`` mode contains the term
            are retained. ``None`` retains all.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist, with download
            instructions for https://zenodo.org/records/4064409.
    """

    def __init__(
        self,
        data_dir: str,
        fs: float = 800.0,
        loop: bool = True,
        split: str | None = None,
    ) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.fs = float(fs)
        self.loop = loop
        self.split = split

        self._connected = False
        self._sample_counter = 0
        self._utt_idx = 0
        self._sample_idx = 0
        self._current_emg: np.ndarray | None = None

        # --- path traversal sanitization (C2) ---
        Path(self.data_dir).resolve()
        # Validate data_dir exists; if not, raise with Zenodo instructions
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Gaddy dataset not found at {self.data_dir}. "
                "Download from https://zenodo.org/records/4064409 and extract "
                "so that the directory contains '*_emg.npy' files (e.g. "
                "data_dir/0_emg.npy, 0_info.json, 0_audio.flac). "
                "See https://github.com/dgaddy/silent_speech for layout."
            )

        self._utterances: list[dict[str, str]] = []
        self._transcripts: dict[str, str] = {}
        self._load_index()

        if not self._utterances:
            raise FileNotFoundError(
                f"No '*_emg.npy' files found under {self.data_dir}. "
                "Ensure the Gaddy dataset was fully extracted. "
                "Download from https://zenodo.org/records/4064409"
            )

        # Infer channel count from first utterance header without loading full array
        first_emg_path = self._utterances[0]["emg_path"]
        try:
            # Memory-map header to avoid loading full array
            arr = np.load(first_emg_path, allow_pickle=False, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"EMG array must be 2D (T,8), got shape {arr.shape}")
            self.num_channels: int = int(arr.shape[1]) if arr.ndim == 2 else 8
            self._total_utterances = len(self._utterances)
        except Exception as e:
            raise ValueError(f"Failed to probe Gaddy EMG file {first_emg_path}: {e}") from e

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _sanitize_id(self, utterance_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9._/-]", "_", utterance_id)

    def _load_index(self) -> None:
        """Scan data_dir recursively for *_emg.npy and build utterance index."""
        base = Path(self.data_dir).resolve()
        # Walk recursively
        for root, _dirs, files in os.walk(self.data_dir):
            for fname in files:
                if not fname.endswith("_emg.npy"):
                    continue
                emg_path = os.path.join(root, fname)
                # Sanitization: ensure emg_path is within base
                try:
                    resolved_emg = Path(emg_path).resolve()
                    # Python 3.9+ has is_relative_to
                    try:
                        if not resolved_emg.is_relative_to(base):
                            continue
                    except AttributeError:
                        try:
                            resolved_emg.relative_to(base)
                        except ValueError:
                            continue
                except Exception:
                    continue

                # Derive utterance id as relative path without suffix
                rel = os.path.relpath(emg_path, self.data_dir)
                # rel e.g. "session1/0_emg.npy" -> id "session1/0"
                utterance_id = rel[: -len("_emg.npy")]
                # Normalize id: remove trailing separators
                utterance_id = utterance_id.strip("/")

                # Derive companion paths
                base_no_suffix = emg_path[: -len("_emg.npy")]
                info_path = base_no_suffix + "_info.json"
                audio_path = base_no_suffix + "_audio.flac"
                audio_clean_path = base_no_suffix + "_audio_clean.flac"
                # Fallback audio .wav
                if not os.path.exists(audio_path):
                    for ext in (".wav", ".flac"):
                        cand = base_no_suffix + f"_audio{ext}"
                        if os.path.exists(cand):
                            audio_path = cand
                            break

                # Split filtering: check path and info.json
                if self.split is not None:
                    s = self.split.lower()
                    # Path-based filter
                    if s not in rel.lower():
                        # Also check info.json mode if exists
                        mode_match = False
                        if os.path.exists(info_path):
                            try:
                                with open(info_path, encoding="utf-8") as f:
                                    info = json.load(f)
                                # Gaddy info.json may have 'mode', 'speaking_mode', 'tag'
                                for k in ("mode", "speaking_mode", "tag", "split", "type"):
                                    if k in info and s in str(info[k]).lower():
                                        mode_match = True
                                        break
                            except Exception:
                                pass
                        if not mode_match:
                            continue

                entry: dict[str, str] = {
                    "id": utterance_id,
                    "emg_path": os.path.abspath(emg_path),
                    "info_path": os.path.abspath(info_path) if os.path.exists(info_path) else "",
                    "audio_path": os.path.abspath(audio_path) if os.path.exists(audio_path) else "",
                    "audio_clean_path": os.path.abspath(audio_clean_path) if os.path.exists(audio_clean_path) else "",
                }
                self._utterances.append(entry)

                # Cache transcript if available
                if entry["info_path"] and os.path.exists(entry["info_path"]):
                    try:
                        with open(entry["info_path"], encoding="utf-8") as f:
                            info = json.load(f)
                        # Try multiple keys for transcription
                        text = (
                            info.get("text")
                            or info.get("transcript")
                            or info.get("prompt")
                            or info.get("sentence")
                            or info.get("utterance")
                            or ""
                        )
                        # Some files use 'text' with nested structure
                        if isinstance(text, dict):
                            text = text.get("text") or text.get("prompt") or str(text)
                        self._transcripts[utterance_id] = str(text)
                    except Exception:
                        self._transcripts[utterance_id] = ""
                else:
                    self._transcripts[utterance_id] = ""

        # Sort for deterministic order
        self._utterances.sort(key=lambda x: x["id"])

    # ------------------------------------------------------------------
    # HardwareSource interface
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._connected = True
        self._utt_idx = 0
        self._sample_idx = 0
        self._sample_counter = 0
        self._current_emg = None
        self._load_current_emg()

    def stop(self) -> None:
        self._connected = False
        self._current_emg = None

    def is_connected(self) -> bool:
        return self._connected

    def _load_current_emg(self) -> None:
        if not self._utterances:
            self._current_emg = None
            return
        # Clamp index
        if self._utt_idx >= len(self._utterances):
            if self.loop:
                self._utt_idx = 0
                self._sample_idx = 0
            else:
                self._utt_idx = len(self._utterances) - 1
                self._sample_idx = 0
        emg_path = self._utterances[self._utt_idx]["emg_path"]
        try:
            # allow_pickle=False prevents code execution (C1)
            self._current_emg = np.load(emg_path, allow_pickle=False)
            if self._current_emg.ndim != 2:
                raise ValueError(f"EMG array must be 2D (T,8), got shape {self._current_emg.shape}")
            # Ensure float representation
            if self._current_emg.dtype == object:
                raise ValueError("Object arrays not allowed (pickle)")
        except Exception as e:
            raise ValueError(f"Failed to load Gaddy EMG file {emg_path}: {e}") from e

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected:
            raise HardwareError("Gaddy stream is not started. Call start().")
        if not self._utterances:
            raise HardwareError("No utterances indexed.")

        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        if num_samples <= 0:
            num_samples = 1
        samples: list[Sample] = []

        # Ensure current EMG is loaded
        if self._current_emg is None:
            self._load_current_emg()

        collected = 0
        while collected < num_samples:
            if self._current_emg is None:
                self._load_current_emg()
                if self._current_emg is None:
                    break
            assert self._current_emg is not None
            remaining_in_utt = self._current_emg.shape[0] - self._sample_idx
            need = num_samples - collected
            take = min(remaining_in_utt, need)

            chunk = self._current_emg[self._sample_idx : self._sample_idx + take]
            for i in range(take):
                self._sample_counter += 1
                row = chunk[i]
                channels = [float(v) for v in row]
                # Timestamp interpolation: newest sample = now, oldest = now - window
                idx = collected + i
                ts = now - ((num_samples - idx) / self.fs)
                samples.append(Sample(timestamp=ts, channels=channels, sample_index=self._sample_counter))
            collected += take
            self._sample_idx += take

            if self._sample_idx >= self._current_emg.shape[0]:
                # Move to next utterance
                self._utt_idx += 1
                self._sample_idx = 0
                if self._utt_idx >= len(self._utterances):
                    if self.loop:
                        self._utt_idx = 0
                    else:
                        # Pad with last sample if not looping and we still need more
                        if collected < num_samples and self._current_emg is not None and self._current_emg.shape[0] > 0:
                            last_row = [float(v) for v in self._current_emg[-1]]
                            while collected < num_samples:
                                self._sample_counter += 1
                                ts = now - ((num_samples - collected) / self.fs)
                                samples.append(Sample(timestamp=ts, channels=last_row, sample_index=self._sample_counter))
                                collected += 1
                        break
                self._current_emg = None
                if collected < num_samples:
                    self._load_current_emg()

        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self.fs,
        )

    # ------------------------------------------------------------------
    # Gaddy-specific helpers
    # ------------------------------------------------------------------
    def list_utterances(self) -> list[str]:
        """Return list of utterance IDs indexed from data_dir."""
        return [u["id"] for u in self._utterances]

    def get_transcript(self, utterance_id: str | None = None) -> str:
        """Return transcript for an utterance.

        Args:
            utterance_id: ID as returned by list_utterances(). If None, returns
                transcript of the current streaming utterance.

        Returns:
            Transcription string (may be empty if info.json missing).
        """
        if utterance_id is None:
            if not self._utterances:
                return ""
            # Current utterance
            idx = min(self._utt_idx, len(self._utterances) - 1)
            utterance_id = self._utterances[idx]["id"]
        # Sanitize
        safe = self._sanitize_id(utterance_id)
        # Allow original id if sanitized differs only by sanitization? Use exact match first.
        if utterance_id in self._transcripts:
            return self._transcripts[utterance_id]
        if safe in self._transcripts:
            return self._transcripts[safe]
        raise KeyError(f"Utterance '{utterance_id}' not found. Available: {list(self._transcripts.keys())[:5]}...")

    def get_audio_path(self, utterance_id: str) -> str | None:
        """Return audio file path for utterance if present."""
        for u in self._utterances:
            if u["id"] == utterance_id:
                return u["audio_path"] or None
        return None

    def get_info(self, utterance_id: str) -> dict[str, object]:
        """Return parsed info.json dict for utterance (empty if missing)."""
        for u in self._utterances:
            if u["id"] == utterance_id and u["info_path"] and os.path.exists(u["info_path"]):
                try:
                    with open(u["info_path"], encoding="utf-8") as f:
                        return json.load(f)  # type: ignore[return-value]
                except Exception:
                    return {}
        return {}


class MetaEMGDriver(HardwareSource):
    """Driver for Meta's sEMG-RD emg2pose / surface-typing datasets.

    Open-sourced Dec 2024 (blog_), Nature 2025 (Sivakumar & Landau et al.
    "Generic hand pose tracking..." + Salter et al. NeurIPS 2024 emg2pose
    ``arXiv:2412.02725``). Wristband is the sEMG-RD: 16 bipolar channels per
    wrist at 2 kHz (12-bit), paired with hand-pose labels at 2 kHz from a
    26-camera motion-capture rig (19 markers/hand → IK joint angles,
    linearly interpolated to 2 kHz).

    Layout (HDF5 per recording, 25k files / 193 participants / 370 h):

    * ``/emg``  → ``(T, 16)`` float32 (also accepted: ``raw_emg``, ``semg``)
    * ``/pose`` → ``(T, D)`` float32 (also accepted: ``joint_angles``,
      ``hand_pose``, ``joints``; typically 63 = 21 joints × 3 or 84-dim)

    Also supports unpacked NPY layout: ``*_emg.npy`` paired with
    ``*_pose.npy`` / ``*_joint_angles.npy``.

    Streaming concatenates recordings sequentially and exposes
    :class:`Frame` (EMG) via the :class:`HardwareSource` interface; pose
    is available via :meth:`load_pair`, :meth:`get_pose`, and
    :meth:`read_frame_with_pose`.

    Args:
        data_dir: Root of the extracted archive (may contain ``train`` /
            ``val`` / ``test`` subfolders; see blog).
        fs: Sampling frequency in Hz (sEMG-RD is 2000.0).
        split: Dataset split — ``"train"`` / ``"val"`` / ``"test"``.
            Sanitized to ``[A-Za-z0-9_-]`` and used to scope to
            ``data_dir/split`` if that subdirectory exists, otherwise
            filtered by path substring. ``"all"`` disables filtering.
        loop: Whether to loop when recordings are exhausted.

    Raises:
        FileNotFoundError: If ``data_dir`` is missing or contains no
            recordings, with instructions to download from
            https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024
            and https://github.com/facebookresearch/emg2pose.

    .. _blog: https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024
    """

    _DOWNLOAD_URL = "https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024"
    _GITHUB_URL = "https://github.com/facebookresearch/emg2pose"
    _EMG_KEYS = ("emg", "raw_emg", "semg", "sEMG", "data")
    _POSE_KEYS = ("pose", "joint_angles", "hand_pose", "joints", "angles", "pose_3d", "jointAngles")

    def __init__(
        self,
        data_dir: str,
        fs: float = 2000.0,
        split: str = "train",
        loop: bool = True,
    ) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.fs = float(fs)
        self.loop = bool(loop)
        # traversal sanitization on split
        raw_split = str(split)
        self.split = re.sub(r"[^A-Za-z0-9_-]", "_", raw_split) if raw_split else "train"
        if not self.split:
            self.split = "train"

        self._connected = False
        self._sample_counter = 0
        self._rec_idx = 0
        self._sample_idx = 0
        self._current_emg: np.ndarray | None = None
        self._current_pose: np.ndarray | None = None

        # validate base dir with traversal sanitization
        try:
            base_resolved = Path(self.data_dir).resolve()
        except Exception:
            base_resolved = Path(self.data_dir).absolute()
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Meta EMG dataset not found at {self.data_dir} (split='{self.split}'). "
                f"Download the emg2pose / surface-typing archive from {self._DOWNLOAD_URL} "
                f"(code & docs: {self._GITHUB_URL}) and extract so that the directory "
                f"contains HDF5 recordings (e.g. data_dir/{self.split}/*.h5 with "
                f"/emg (T,16) and /pose (T,D) at 2000 Hz). See emg2pose.data.Emg2PoseSessionData."
            )

        # optional h5py (only needed for HDF5 layout)
        self._h5py = None
        try:
            import h5py as _h5py  # type: ignore[import-not-found]

            self._h5py = _h5py
        except ImportError:
            self._h5py = None  # defer error until HDF5 file encountered

        self._recordings: list[dict[str, str]] = []
        self._load_index(base_resolved)

        if not self._recordings:
            raise FileNotFoundError(
                f"No Meta EMG recordings found under {self.data_dir} (split='{self.split}'). "
                f"Expected HDF5 files with '/emg' and '/pose' at 2000 Hz, or paired "
                f"'*_emg.npy' + '*_pose.npy'. Download from {self._DOWNLOAD_URL} "
                f"({self._GITHUB_URL}) and ensure extraction preserved the split layout."
            )

        # infer dims from first recording header
        first = self._recordings[0]
        try:
            if first.get("type") == "h5":
                if self._h5py is None:
                    raise MissingDependencyError(
                        "h5py is required to read Meta emg2pose HDF5 files. "
                        'Install with: pip install "subvocal[hardware]" or pip install h5py'
                    )
                with self._h5py.File(first["emg_path"], "r") as hf:  # type: ignore[union-attr]
                    emg_ds = self._find_dataset(hf, self._EMG_KEYS)
                    pose_ds = self._find_dataset(hf, self._POSE_KEYS)
                    if emg_ds is None:
                        raise KeyError(f"No EMG dataset {self._EMG_KEYS} in {first['emg_path']}")
                    self.num_channels: int = int(emg_ds.shape[1]) if len(emg_ds.shape) == 2 else 16
                    self.pose_dim: int = int(pose_ds.shape[1]) if pose_ds is not None and len(pose_ds.shape) == 2 else 0
            else:
                arr = np.load(first["emg_path"], allow_pickle=False, mmap_mode="r")
                if arr.ndim != 2:
                    raise ValueError(f"EMG array must be 2D (T,16), got {arr.shape}")
                self.num_channels = int(arr.shape[1])
                # pose may be in companion file
                pose_path = first.get("pose_path", "")
                if pose_path and os.path.exists(pose_path):
                    parr = np.load(pose_path, allow_pickle=False, mmap_mode="r")
                    self.pose_dim = int(parr.shape[1]) if parr.ndim == 2 else 0
                else:
                    self.pose_dim = 0
        except MissingDependencyError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to probe Meta EMG recording {first}: {e}") from e

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _sanitize_id(self, rec_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9._/-]", "_", rec_id)

    def _find_dataset(self, h5obj: object, candidates: tuple[str, ...]) -> object | None:
        """Find first matching dataset key in HDF5 file (top-level or one group deep)."""
        # direct keys
        for key in candidates:
            if key in h5obj:  # type: ignore[operator]
                try:
                    ds = h5obj[key]  # type: ignore[index]
                    # h5py Dataset has shape
                    if hasattr(ds, "shape") and len(getattr(ds, "shape", ())) >= 1:
                        return ds
                except Exception:
                    continue
        # search one level deep
        try:
            for name in list(h5obj.keys()):  # type: ignore[union-attr]
                try:
                    grp = h5obj[name]  # type: ignore[index]
                    if hasattr(grp, "keys"):
                        for key in candidates:
                            if key in grp:
                                ds = grp[key]  # type: ignore[index]
                                if hasattr(ds, "shape"):
                                    return ds
                except Exception:
                    continue
        except Exception:
            pass
        # fallback: first 2D dataset with second dim plausible
        best = None
        try:
            def _visitor(n: str, o: object) -> None:
                nonlocal best
                if best is not None:
                    return
                if hasattr(o, "shape") and hasattr(o, "dtype"):
                    shp = getattr(o, "shape", ())
                    if len(shp) == 2 and shp[1] in (16, 63, 84, 21, 48, 42):
                        # heauristic: pick emg-like (16) first, else any
                        best = o

            h5obj.visititems(_visitor)  # type: ignore[union-attr]
        except Exception:
            pass
        return best

    def _rel_id(self, resolved: Path, base: Path, suffix_len: int = 0) -> str:
        """Compute sanitized recording id from resolved path relative to base."""
        try:
            rel = str(resolved.relative_to(base))
        except ValueError:
            try:
                rel = os.path.relpath(str(resolved), str(base))
            except Exception:
                rel = resolved.name
        if suffix_len and len(rel) > suffix_len:
            rel = rel[:-suffix_len]
        # strip leading slashes
        rel = rel.lstrip("/")
        return self._sanitize_id(rel)

    def _load_index(self, base_resolved: Path) -> None:
        """Scan for HDF5 and paired NPY recordings under data_dir[/split]."""
        base = base_resolved
        # decide search root
        split_path = Path(self.data_dir) / self.split
        if self.split.lower() != "all" and split_path.is_dir():
            search_root = split_path.resolve()
            filter_by_split = False
        else:
            search_root = base
            filter_by_split = self.split.lower() != "all"

        # walk recursively
        for root, _dirs, files in os.walk(search_root):
            for fname in files:
                low = fname.lower()
                fpath = os.path.join(root, fname)
                # sanitization: ensure within base
                try:
                    resolved = Path(fpath).resolve()
                    try:
                        if not resolved.is_relative_to(base):
                            continue
                    except AttributeError:
                        try:
                            resolved.relative_to(base)
                        except ValueError:
                            continue
                except Exception:
                    continue

                rec: dict[str, str] | None = None
                if low.endswith((".h5", ".hdf5", ".h5py")):
                    # for split filtering, if enabled and no split term in path, skip
                    if filter_by_split and self.split.lower() not in fpath.lower():
                        continue
                    suffix_len = len(os.path.splitext(resolved.name)[1])
                    rec_id = self._rel_id(resolved, base, suffix_len)
                    rec = {
                        "id": rec_id,
                        "type": "h5",
                        "emg_path": str(resolved),
                        "pose_path": str(resolved),  # same file, diff keys
                    }
                elif fname.endswith("_emg.npy"):
                    if filter_by_split and self.split.lower() not in fpath.lower():
                        continue
                    # companion pose candidates (resolve each)
                    # derive companion base from resolved path
                    res_str = str(resolved)
                    base_no_suffix = res_str[: -len("_emg.npy")]
                    pose_cands = [
                        base_no_suffix + "_pose.npy",
                        base_no_suffix + "_joint_angles.npy",
                        base_no_suffix + "_joints.npy",
                        base_no_suffix + "_angles.npy",
                        base_no_suffix + "_hand_pose.npy",
                    ]
                    pose_path = ""
                    for cand in pose_cands:
                        if os.path.exists(cand):
                            try:
                                rp = Path(cand).resolve()
                                try:
                                    if not rp.is_relative_to(base):
                                        continue
                                except AttributeError:
                                    try:
                                        rp.relative_to(base)
                                    except ValueError:
                                        continue
                            except Exception:
                                continue
                            pose_path = str(rp)
                            break
                    rec_id = self._rel_id(resolved, base, len("_emg.npy"))
                    rec = {
                        "id": rec_id,
                        "type": "npy",
                        "emg_path": str(resolved),
                        "pose_path": pose_path,
                    }
                else:
                    continue

                if rec is not None:
                    self._recordings.append(rec)

        # If we filtered too aggressively (zero results but files exist), retry without split filter
        if not self._recordings and filter_by_split:
            # fallback: ignore split substring, keep all
            self._recordings.clear()
            for root, _dirs, files in os.walk(search_root):
                for fname in files:
                    low = fname.lower()
                    fpath = os.path.join(root, fname)
                    try:
                        resolved = Path(fpath).resolve()
                        try:
                            if not resolved.is_relative_to(base):
                                continue
                        except AttributeError:
                            try:
                                resolved.relative_to(base)
                            except ValueError:
                                continue
                    except Exception:
                        continue
                    if low.endswith((".h5", ".hdf5", ".h5py")):
                        suffix_len = len(os.path.splitext(resolved.name)[1])
                        rec_id = self._rel_id(resolved, base, suffix_len)
                        self._recordings.append(
                            {"id": rec_id, "type": "h5", "emg_path": str(resolved), "pose_path": str(resolved)}
                        )
                    elif fname.endswith("_emg.npy"):
                        res_str = str(resolved)
                        base_no_suffix = res_str[: -len("_emg.npy")]
                        pose_path = ""
                        for cand in [
                            base_no_suffix + "_pose.npy",
                            base_no_suffix + "_joint_angles.npy",
                            base_no_suffix + "_joints.npy",
                        ]:
                            if os.path.exists(cand):
                                try:
                                    rp = Path(cand).resolve()
                                    try:
                                        if not rp.is_relative_to(base):
                                            continue
                                    except AttributeError:
                                        try:
                                            rp.relative_to(base)
                                        except ValueError:
                                            continue
                                except Exception:
                                    continue
                                pose_path = str(rp)
                                break
                        rec_id = self._rel_id(resolved, base, len("_emg.npy"))
                        self._recordings.append(
                            {"id": rec_id, "type": "npy", "emg_path": str(resolved), "pose_path": pose_path}
                        )

        self._recordings.sort(key=lambda x: x["id"])

    # ------------------------------------------------------------------
    # HardwareSource interface
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._connected = True
        self._rec_idx = 0
        self._sample_idx = 0
        self._sample_counter = 0
        self._current_emg = None
        self._current_pose = None
        self._load_current_pair()

    def stop(self) -> None:
        self._connected = False
        self._current_emg = None
        self._current_pose = None

    def is_connected(self) -> bool:
        return self._connected

    def _load_current_pair(self) -> None:
        if not self._recordings:
            self._current_emg = None
            self._current_pose = None
            return
        if self._rec_idx >= len(self._recordings):
            if self.loop:
                self._rec_idx = 0
                self._sample_idx = 0
            else:
                self._rec_idx = len(self._recordings) - 1
                self._sample_idx = 0
        rec = self._recordings[self._rec_idx]
        try:
            if rec["type"] == "h5":
                if self._h5py is None:
                    raise MissingDependencyError(
                        "h5py is required to read Meta emg2pose HDF5 files. "
                        'Install with: pip install "subvocal[hardware]" or pip install h5py'
                    )
                with self._h5py.File(rec["emg_path"], "r") as hf:  # type: ignore[union-attr]
                    emg_ds = self._find_dataset(hf, self._EMG_KEYS)
                    pose_ds = self._find_dataset(hf, self._POSE_KEYS)
                    if emg_ds is None:
                        raise KeyError(f"No EMG dataset in {rec['emg_path']} (tried {self._EMG_KEYS})")
                    emg = np.array(emg_ds, dtype=np.float32)
                    if emg.ndim != 2:
                        raise ValueError(f"EMG must be 2D (T,16), got {emg.shape}")
                    self._current_emg = emg
                    if pose_ds is not None:
                        pose = np.array(pose_ds, dtype=np.float32)
                        if pose.ndim == 1:
                            pose = pose[:, None]
                        # align lengths (IK may be shorter due to failures)
                        if pose.shape[0] != emg.shape[0]:
                            # truncate or pad with last valid pose
                            if pose.shape[0] < emg.shape[0]:
                                pad = np.repeat(pose[-1:], emg.shape[0] - pose.shape[0], axis=0)
                                pose = np.concatenate([pose, pad], axis=0)
                            else:
                                pose = pose[: emg.shape[0]]
                        self._current_pose = pose
                    else:
                        self._current_pose = np.zeros((emg.shape[0], self.pose_dim if self.pose_dim else 63), dtype=np.float32)
            else:
                emg = np.load(rec["emg_path"], allow_pickle=False)
                if emg.ndim != 2:
                    raise ValueError(f"EMG array must be 2D (T,16), got {emg.shape}")
                if emg.dtype == object:
                    raise ValueError("Object arrays not allowed")
                self._current_emg = emg.astype(np.float32, copy=False)
                pose_path = rec.get("pose_path", "")
                if pose_path and os.path.exists(pose_path):
                    pose = np.load(pose_path, allow_pickle=False)
                    if pose.ndim == 1:
                        pose = pose[:, None]
                    if pose.shape[0] != emg.shape[0]:
                        if pose.shape[0] < emg.shape[0]:
                            pad = np.repeat(pose[-1:], emg.shape[0] - pose.shape[0], axis=0)
                            pose = np.concatenate([pose, pad], axis=0)
                        else:
                            pose = pose[: emg.shape[0]]
                    self._current_pose = pose.astype(np.float32, copy=False)
                else:
                    self._current_pose = np.zeros((emg.shape[0], self.pose_dim if self.pose_dim else 0), dtype=np.float32)
        except MissingDependencyError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to load Meta recording {rec['id']} ({rec['emg_path']}): {e}") from e

    def read_frame(self, window_ms: int) -> Frame:
        if not self._connected:
            raise HardwareError("Meta EMG stream is not started. Call start().")
        if not self._recordings:
            raise HardwareError("No recordings indexed.")
        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        if num_samples <= 0:
            num_samples = 1
        samples: list[Sample] = []
        if self._current_emg is None:
            self._load_current_pair()
        collected = 0
        while collected < num_samples:
            if self._current_emg is None:
                self._load_current_pair()
                if self._current_emg is None:
                    break
            assert self._current_emg is not None
            remaining = self._current_emg.shape[0] - self._sample_idx
            need = num_samples - collected
            take = min(remaining, need)
            chunk = self._current_emg[self._sample_idx : self._sample_idx + take]
            for i in range(take):
                self._sample_counter += 1
                row = chunk[i]
                channels = [float(v) for v in row]
                idx = collected + i
                ts = now - ((num_samples - idx) / self.fs)
                samples.append(Sample(timestamp=ts, channels=channels, sample_index=self._sample_counter))
            collected += take
            self._sample_idx += take
            if self._sample_idx >= self._current_emg.shape[0]:
                self._rec_idx += 1
                self._sample_idx = 0
                if self._rec_idx >= len(self._recordings):
                    if self.loop:
                        self._rec_idx = 0
                    else:
                        if collected < num_samples and self._current_emg is not None and self._current_emg.shape[0] > 0:
                            last_row = [float(v) for v in self._current_emg[-1]]
                            while collected < num_samples:
                                self._sample_counter += 1
                                ts = now - ((num_samples - collected) / self.fs)
                                samples.append(Sample(timestamp=ts, channels=last_row, sample_index=self._sample_counter))
                                collected += 1
                        break
                self._current_emg = None
                self._current_pose = None
                if collected < num_samples:
                    self._load_current_pair()
        return Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self.fs,
        )

    # ------------------------------------------------------------------
    # Paired EMG + pose helpers
    # ------------------------------------------------------------------
    def list_recordings(self) -> list[str]:
        """Return sorted recording IDs indexed from data_dir/split."""
        return [r["id"] for r in self._recordings]

    def load_pair(self, idx: int | str) -> tuple[np.ndarray, np.ndarray]:
        """Load paired EMG and pose arrays for a recording.

        Args:
            idx: Integer index or recording ID string.

        Returns:
            Tuple ``(emg, pose)`` where ``emg`` is ``(T, 16)`` float32 and
            ``pose`` is ``(T, D)`` float32 (D is joint-angle dim, e.g. 63).
        """
        rec: dict[str, str] | None = None
        if isinstance(idx, int):
            if not 0 <= idx < len(self._recordings):
                raise IndexError(f"Recording index {idx} out of range [0, {len(self._recordings)})")
            rec = self._recordings[idx]
        else:
            safe = self._sanitize_id(str(idx))
            for r in self._recordings:
                if r["id"] == idx or r["id"] == safe:
                    rec = r
                    break
            if rec is None:
                raise KeyError(f"Recording '{idx}' not found. Available: {[rr['id'] for rr in self._recordings[:5]]}...")

        # load without affecting streaming cursor
        if rec["type"] == "h5":
            if self._h5py is None:
                raise MissingDependencyError(
                    "h5py is required to read Meta emg2pose HDF5 files. "
                    'Install with: pip install "subvocal[hardware]" or pip install h5py'
                )
            with self._h5py.File(rec["emg_path"], "r") as hf:  # type: ignore[union-attr]
                emg_ds = self._find_dataset(hf, self._EMG_KEYS)
                pose_ds = self._find_dataset(hf, self._POSE_KEYS)
                if emg_ds is None:
                    raise KeyError(f"No EMG dataset in {rec['emg_path']}")
                emg = np.array(emg_ds, dtype=np.float32)
                pose = np.array(pose_ds, dtype=np.float32) if pose_ds is not None else np.zeros((emg.shape[0], 0), dtype=np.float32)
                if pose.ndim == 1:
                    pose = pose[:, None]
                if pose.shape[0] != emg.shape[0]:
                    if pose.shape[0] < emg.shape[0]:
                        pad = np.repeat(pose[-1:], emg.shape[0] - pose.shape[0], axis=0) if pose.shape[0] > 0 else np.zeros((emg.shape[0] - pose.shape[0], pose.shape[1] if pose.ndim == 2 else 0), dtype=np.float32)
                        pose = np.concatenate([pose, pad], axis=0) if pose.size else pose
                    else:
                        pose = pose[: emg.shape[0]]
                return emg, pose
        else:
            emg = np.load(rec["emg_path"], allow_pickle=False).astype(np.float32, copy=False)
            pose_path = rec.get("pose_path", "")
            if pose_path and os.path.exists(pose_path):
                pose = np.load(pose_path, allow_pickle=False).astype(np.float32, copy=False)
                if pose.ndim == 1:
                    pose = pose[:, None]
            else:
                pose = np.zeros((emg.shape[0], 0), dtype=np.float32)
            if pose.shape[0] != emg.shape[0]:
                if pose.shape[0] < emg.shape[0]:
                    pad = np.repeat(pose[-1:], emg.shape[0] - pose.shape[0], axis=0) if pose.shape[0] > 0 else np.zeros((emg.shape[0] - pose.shape[0], pose.shape[1] if pose.ndim == 2 else 0), dtype=np.float32)
                    pose = np.concatenate([pose, pad], axis=0) if pose.size else pose
                else:
                    pose = pose[: emg.shape[0]]
            return emg, pose

    def get_emg(self, idx: int | str) -> np.ndarray:
        """Return EMG array ``(T, 16)`` for recording ``idx``."""
        emg, _ = self.load_pair(idx)
        return emg

    def get_pose(self, idx: int | str) -> np.ndarray:
        """Return pose array ``(T, D)`` for recording ``idx``."""
        _, pose = self.load_pair(idx)
        return pose

    def read_frame_with_pose(self, window_ms: int) -> tuple[Frame, np.ndarray]:
        """Stream EMG as :class:`Frame` and return time-aligned pose window.

        Returns:
            Tuple of ``(frame, pose_window)`` where ``pose_window`` is an
            ``(num_samples, D)`` array aligned to the EMG samples.

        Note:
            Pose windows are contiguous across recordings (like EMG).
        """
        if not self._connected:
            raise HardwareError("Meta EMG stream is not started. Call start().")
        # snapshot current pose cursor before reading EMG, then collect parallel
        # We implement by duplicating the read_frame loop but also gathering pose.
        now = time.time()
        num_samples = int((window_ms / 1000.0) * self.fs)
        if num_samples <= 0:
            num_samples = 1
        samples: list[Sample] = []
        pose_chunks: list[np.ndarray] = []
        if self._current_emg is None:
            self._load_current_pair()
        collected = 0
        while collected < num_samples:
            if self._current_emg is None:
                self._load_current_pair()
                if self._current_emg is None:
                    break
            assert self._current_emg is not None
            assert self._current_pose is not None
            remaining = self._current_emg.shape[0] - self._sample_idx
            need = num_samples - collected
            take = min(remaining, need)
            emg_chunk = self._current_emg[self._sample_idx : self._sample_idx + take]
            pose_chunk = self._current_pose[self._sample_idx : self._sample_idx + take]
            for i in range(take):
                self._sample_counter += 1
                row = emg_chunk[i]
                channels = [float(v) for v in row]
                idx = collected + i
                ts = now - ((num_samples - idx) / self.fs)
                samples.append(Sample(timestamp=ts, channels=channels, sample_index=self._sample_counter))
            pose_chunks.append(pose_chunk)
            collected += take
            self._sample_idx += take
            if self._sample_idx >= self._current_emg.shape[0]:
                self._rec_idx += 1
                self._sample_idx = 0
                if self._rec_idx >= len(self._recordings):
                    if self.loop:
                        self._rec_idx = 0
                    else:
                        if collected < num_samples and self._current_emg is not None:
                            last_row = [float(v) for v in self._current_emg[-1]]
                            last_pose = self._current_pose[-1:] if self._current_pose.size else np.zeros((1, self.pose_dim), dtype=np.float32)
                            while collected < num_samples:
                                self._sample_counter += 1
                                ts = now - ((num_samples - collected) / self.fs)
                                samples.append(Sample(timestamp=ts, channels=last_row, sample_index=self._sample_counter))
                                pose_chunks.append(last_pose)
                                collected += 1
                        break
                self._current_emg = None
                self._current_pose = None
                if collected < num_samples:
                    self._load_current_pair()
        frame = Frame(
            samples=samples,
            start_time=now - (window_ms / 1000.0),
            end_time=now,
            fs=self.fs,
        )
        if pose_chunks:
            try:
                pose_window = np.concatenate(pose_chunks, axis=0)
            except ValueError:
                # mismatched dims (zero-dim fallback)
                pose_window = np.zeros((len(samples), self.pose_dim if self.pose_dim else 0), dtype=np.float32)
        else:
            pose_window = np.zeros((0, self.pose_dim if hasattr(self, "pose_dim") else 0), dtype=np.float32)
        # ensure length matches frame
        if pose_window.shape[0] != len(samples):
            if pose_window.shape[0] < len(samples):
                pad = np.repeat(pose_window[-1:], len(samples) - pose_window.shape[0], axis=0) if pose_window.shape[0] > 0 else np.zeros((len(samples), pose_window.shape[1] if pose_window.ndim == 2 else 0), dtype=np.float32)
                pose_window = np.concatenate([pose_window, pad], axis=0) if pose_window.size else np.zeros((len(samples), self.pose_dim), dtype=np.float32)
            else:
                pose_window = pose_window[: len(samples)]
        return frame, pose_window
