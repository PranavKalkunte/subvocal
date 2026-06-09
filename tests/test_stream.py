import unittest

import numpy as np

from subvocal.core.models import Frame, Sample
from subvocal.stream import FrameRing, SignalLevel, SignalQualityScorer, StreamStats, StreamTracker


class TestStreamLayer(unittest.TestCase):
    def _create_frame(self, data: list[list[float]], start_time: float, end_time: float, fs: float = 1000.0) -> Frame:
        samples = [
            Sample(timestamp=start_time + idx * (1.0 / fs), channels=ch, sample_index=idx)
            for idx, ch in enumerate(data)
        ]
        return Frame(samples=samples, start_time=start_time, end_time=end_time, fs=fs)

    def test_frame_ring_circular_behavior(self):
        """Verify FrameRing maintains max_size and evicts oldest items."""
        ring = FrameRing(max_size=3)
        f1 = self._create_frame([[0.1]], 0.0, 0.1)
        f2 = self._create_frame([[0.2]], 0.1, 0.2)
        f3 = self._create_frame([[0.3]], 0.2, 0.3)
        f4 = self._create_frame([[0.4]], 0.3, 0.4)

        ring.push(f1)
        ring.push(f2)
        ring.push(f3)
        self.assertEqual(len(ring), 3)

        ring.push(f4)
        self.assertEqual(len(ring), 3)

        all_frames = ring.get_all()
        self.assertEqual(all_frames[0].samples[0].channels, [0.2])
        self.assertEqual(all_frames[1].samples[0].channels, [0.3])
        self.assertEqual(all_frames[2].samples[0].channels, [0.4])

        self.assertEqual(ring.pop(), f2)
        self.assertEqual(len(ring), 2)

    def test_stream_stats_counters(self):
        """Verify StreamStats counts frames, samples, gaps, and jitter correctly."""
        stats = StreamStats()
        f1 = self._create_frame([[0.1], [0.1]], 0.0, 0.002, fs=1000.0)
        f2 = self._create_frame([[0.1], [0.1]], 0.002, 0.004, fs=1000.0)
        # Gap here: f3 starts at 0.010 instead of 0.004
        f3 = self._create_frame([[0.1], [0.1]], 0.010, 0.012, fs=1000.0)

        stats.observe(f1)
        stats.observe(f2)
        stats.observe(f3)

        s = stats.get_stats()
        self.assertEqual(s["total_frames"], 3)
        self.assertEqual(s["total_samples"], 6)
        self.assertEqual(s["total_gaps"], 1)
        self.assertTrue(s["jitter_seconds"] >= 0.0)

    def test_signal_level_activation(self):
        """Verify SignalLevel reports active state when thresholds are met."""
        level = SignalLevel(active_level=0.5, min_percentile=50.0, update_interval_ms=10)
        
        # Inactive frame (MAV is 0.1)
        f_inactive = self._create_frame([[0.1], [0.1]], 0.0, 0.005)
        # Active frame (MAV is 2.0)
        f_active = self._create_frame([[2.0], [2.0]], 0.005, 0.010)

        level.observe(f_inactive)
        level.observe(f_active)

        # Update interval is 10ms, total observed duration is 10ms, active duration is 5ms (50% >= 50%)
        # Pass 0.010 (end time of the active frame) as now to avoid the stale check firing
        smoothed_val, is_active = level.get_level(0.010)
        self.assertTrue(is_active)
        self.assertTrue(smoothed_val > 0.0)

    def test_stream_tracker_hysteresis(self):
        """Verify StreamTracker requires continuous inputs to transition states."""
        tracker = StreamTracker(samples_required=3, cycles_required=3)
        states = []
        tracker.on_status_changed(states.append)

        # Start stopped. 2 active frames -> still stopped.
        tracker.observe(True)
        tracker.observe(True)
        self.assertEqual(tracker.status, StreamTracker.STATUS_STOPPED)
        self.assertEqual(states, [])

        # 3rd active frame -> ACTIVE.
        tracker.observe(True)
        self.assertEqual(tracker.status, StreamTracker.STATUS_ACTIVE)
        self.assertEqual(states, [StreamTracker.STATUS_ACTIVE])

        # 2 inactive frames -> still ACTIVE.
        tracker.observe(False)
        tracker.observe(False)
        self.assertEqual(tracker.status, StreamTracker.STATUS_ACTIVE)

        # 3rd inactive frame -> STOPPED.
        tracker.observe(False)
        self.assertEqual(tracker.status, StreamTracker.STATUS_STOPPED)
        self.assertEqual(states, [StreamTracker.STATUS_ACTIVE, StreamTracker.STATUS_STOPPED])

    def test_signal_quality_scorer_penalties(self):
        """Verify SignalQualityScorer reduces quality on bad signals (clipping, dropout)."""
        scorer = SignalQualityScorer(clip_max=5.0)

        # Clean normal frame with active signal variation (non-zero std-dev)
        samples_data = [[float(np.sin(i * 0.5)), float(np.cos(i * 0.5)), float(np.sin(i))] for i in range(20)]
        f_clean = self._create_frame(samples_data, 0.0, 0.02)
        scorer.update(f_clean)
        self.assertEqual(scorer.quality, SignalQualityScorer.QUALITY_EXCELLENT)

        # Saturated frame (clipping)
        f_clip = self._create_frame([[5.0, 5.0, 5.0]] * 20, 0.0, 0.02)
        # Apply multiple updates to drive score down quickly due to EMA
        for _ in range(10):
            scorer.update(f_clip)
        self.assertIn(scorer.quality, (SignalQualityScorer.QUALITY_POOR, SignalQualityScorer.QUALITY_LOST))

        # Flatline / Dropout frame
        scorer = SignalQualityScorer(clip_max=5.0)
        f_flat = self._create_frame([[0.0, 0.0, 0.0]] * 20, 0.0, 0.02)
        for _ in range(10):
            scorer.update(f_flat)
        self.assertEqual(scorer.quality, SignalQualityScorer.QUALITY_LOST)
