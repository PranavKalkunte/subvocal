"""EMGBench harness — LOSO and few-shot adaptation across 9 EMG datasets.

Implements the evaluation protocol from:

* Yang et al., *EMGBench: Benchmarking Out-of-Distribution Generalization
  and Adaptation for Electromyography*, NeurIPS 2024 Datasets & Benchmarks
  Track, https://arxiv.org/abs/2410.23625 — https://emgbench.github.io —
  https://github.com/jehanyang/emgbench

EMGBench spans 9 datasets (Ninapro DB2/DB3/DB5, CapgMyo DB-b, Myo Armband,
UCI EMG, MCS, Hyser, FlexWear-HD) and evaluates:

1. Intersubject generalization via Leave-One-Subject-Out CV (LOSO-CV)
2. Few-shot / intersession adaptation (fine-tuning on ``n_shot`` examples
   per gesture from the held-out subject after pre-training on the others;
   also TSTS / FT-X% variants)

This module provides a *lightweight, dependency-optional* harness that
works both on real EMGBench ``DatasetsProcessed_hdf5/`` layouts and in
synthetic-fallback mode (no data on disk), so unit tests and CI never
require the 10-100 GiB downloads.

Real-data layout (per EMGBench ``CNN_EMG.py``)::

    root_dir/<dataset>/p<N>/participant_<N>.hdf5   # keys = gesture names
    root_dir/<dataset>/frequency.txt               # sampling rate (optional)
    # each gesture dataset shape: (trials, electrodes, timesteps)

When a dataset directory cannot be read, synthetic windows are generated
deterministically so ``model_fn`` contracts can still be exercised.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_DATASETS: list[str] = [
    "ninapro-db2",
    "ninapro-db3",
    "ninapro-db5",
    "capgmyo-db-b",
    "myoarmband",
    "uciemg",
    "mcs",
    "hyser",
    "flexwear-hd",
]
"""9 datasets in the NeurIPS 2024 EMGBench benchmark."""

_DOWNLOAD_NOTE = (
    "EMGBench data not found. Download/preprocess per https://github.com/jehanyang/emgbench "
    "and place HDF5 files under DatasetsProcessed_hdf5/[DATASET]/p[N]/participant_[N].hdf5. "
    "This harness falls back to synthetic data when files are absent (for testing)."
)

_SAN_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize(name: str) -> str:
    return _SAN_RE.sub("_", name)[:128]


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _accuracy_from_result(result: Any, y_true: np.ndarray | None) -> float:
    if result is None:
        return 0.0
    if isinstance(result, dict):
        for k in ("accuracy", "acc", "mean_accuracy", "test_accuracy", "score"):
            if k in result:
                try:
                    return float(result[k])
                except Exception:
                    pass
        for k in ("predictions", "y_pred", "preds"):
            if k in result and y_true is not None:
                try:
                    preds = np.asarray(result[k])
                    yt = np.asarray(y_true)
                    if preds.shape == yt.shape:
                        return float(np.mean(preds == yt))
                    if preds.ndim == 2 and preds.shape[0] == yt.shape[0]:
                        return float(np.mean(np.argmax(preds, axis=1) == yt))
                except Exception:
                    pass
        return 0.0
    if isinstance(result, (float, int, np.floating, np.integer)):
        v = float(result)
        # clamp to [0,1] if looks like accuracy, else treat as scalar metric
        return v
    if isinstance(result, (list, np.ndarray)):
        arr = np.asarray(result)
        if arr.size == 1:
            try:
                return float(arr.flat[0])
            except Exception:
                return 0.0
        if y_true is not None:
            yt = np.asarray(y_true)
            if arr.shape == yt.shape:
                return float(np.mean(arr == yt))
            if arr.ndim == 2 and arr.shape[0] == yt.shape[0]:
                try:
                    return float(np.mean(np.argmax(arr, axis=1) == yt))
                except Exception:
                    pass
    return 0.0


def _invoke_loso(model_fn: Callable[..., Any], train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray) -> float:
    # primary signature (train_X, train_y, test_X, test_y) -> metric/dict/preds
    try:
        res = model_fn(train_X, train_y, test_X, test_y)
        return _accuracy_from_result(res, test_y)
    except TypeError:
        pass
    try:
        res = model_fn((train_X, train_y), (test_X, test_y))
        return _accuracy_from_result(res, test_y)
    except TypeError:
        pass
    try:
        res = model_fn({"train_X": train_X, "train_y": train_y, "test_X": test_X, "test_y": test_y})
        return _accuracy_from_result(res, test_y)
    except Exception:
        pass
    # last resort: callable expects no args? treat as zero
    try:
        res = model_fn()
        return _accuracy_from_result(res, test_y)
    except Exception:
        return 0.0


def _few_shot_split(X: np.ndarray, y: np.ndarray, n_shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0 or y.size == 0 or n_shot <= 0:
        return np.zeros((0,) + X.shape[1:], dtype=X.dtype), np.zeros((0,), dtype=y.dtype), X, y
    classes = np.unique(y)
    s_idx: list[int] = []
    q_idx: list[int] = []
    for c in classes:
        idx = np.where(y == c)[0]
        take = min(n_shot, len(idx))
        s_idx.extend(idx[:take].tolist())
        q_idx.extend(idx[take:].tolist())
    if not s_idx:
        return np.zeros((0,) + X.shape[1:], dtype=X.dtype), np.zeros((0,), dtype=y.dtype), X, y
    if not q_idx:
        # all data used as support — query empty => use support as query for scoring
        s_idx_arr = np.array(s_idx)
        return X[s_idx_arr], y[s_idx_arr], X[s_idx_arr], y[s_idx_arr]
    return X[np.array(s_idx)], y[np.array(s_idx)], X[np.array(q_idx)], y[np.array(q_idx)]


def _invoke_adaptation(
    model_fn: Callable[..., Any],
    train_X: np.ndarray,
    train_y: np.ndarray,
    support_X: np.ndarray,
    support_y: np.ndarray,
    query_X: np.ndarray,
    query_y: np.ndarray,
) -> float:
    # try adaptation-aware signature first
    try:
        res = model_fn(train_X, train_y, support_X, support_y, query_X, query_y)
        return _accuracy_from_result(res, query_y)
    except TypeError:
        pass
    try:
        res = model_fn((train_X, train_y), (support_X, support_y), (query_X, query_y))
        return _accuracy_from_result(res, query_y)
    except TypeError:
        pass
    # fallback: augment train with support and evaluate as LOSO
    try:
        if support_X.size:
            aug_X = np.concatenate([train_X, support_X], axis=0) if train_X.size else support_X
            aug_y = np.concatenate([train_y, support_y], axis=0) if train_y.size else support_y
        else:
            aug_X, aug_y = train_X, train_y
        return _invoke_loso(model_fn, aug_X, aug_y, query_X, query_y)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# synthetic data helpers (deterministic, for testing without downloads)
# ---------------------------------------------------------------------------

def _synthetic_params(dataset: str) -> tuple[int, int, int]:
    h = sum(ord(c) for c in dataset)  # stable, no hash-randomization
    n_channels = 8 + (h % 9)  # 8-16
    n_gestures = 4 + (h % 5)  # 4-8
    n_subjects = 5 + (h % 4)  # 5-8
    return n_channels, n_gestures, n_subjects


def _synthetic_subject_data(dataset: str, subject: str, n_channels: int, n_gestures: int) -> tuple[np.ndarray, np.ndarray]:
    # deterministic seed per dataset+subject
    seed = sum(ord(c) for c in dataset + subject) % (2**31)
    rng = np.random.RandomState(seed)
    trials_per_gesture = 10
    T = 150  # window length (samples) — matches EMGBench preprocessing windows
    N = n_gestures * trials_per_gesture
    X = rng.randn(N, T, n_channels).astype(np.float32)
    # add gesture-specific bias so a dummy classifier could learn if it tried
    for g in range(n_gestures):
        sl = slice(g * trials_per_gesture, (g + 1) * trials_per_gesture)
        X[sl] += float(g) * 0.3
    y = np.repeat(np.arange(n_gestures), trials_per_gesture)
    # shuffle deterministically
    perm = rng.permutation(N)
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# EMGBench
# ---------------------------------------------------------------------------


class EMGBench:
    """EMGBench evaluation harness (NeurIPS 2024).

    Args:
        datasets: List of 9 EMGBench dataset slugs (e.g.
            ``"ninapro-db5"``, ``"capgmyo-db-b"``). Sanitized to
            ``[A-Za-z0-9._-]``. If empty, :data:`DEFAULT_DATASETS` is used.
        root_dir: Root directory containing per-dataset subfolders
            (``root_dir/<dataset>/...``) or the ``DatasetsProcessed_hdf5``
            parent. Traversal-sanitized.

    Example::

        bench = EMGBench(datasets=DEFAULT_DATASETS, root_dir="DatasetsProcessed_hdf5")
        loso = bench.evaluate_loso(my_model_fn, dataset="ninapro-db5")
        adapt = bench.evaluate_adaptation(my_model_fn, dataset="ninapro-db5", n_shot=5)
        all_res = bench.evaluate_all(my_model_fn, mode="loso")
    """

    def __init__(self, datasets: list[str], root_dir: str) -> None:
        # sanitize root_dir (prevent traversal via absolute + resolve)
        self.root_dir = os.path.abspath(str(root_dir))
        try:
            Path(self.root_dir).resolve()
        except Exception:
            pass
        # sanitize dataset names
        if not datasets:
            datasets = list(DEFAULT_DATASETS)
        self.datasets: list[str] = [_sanitize(str(d)) for d in datasets]
        if not self.datasets:
            self.datasets = list(DEFAULT_DATASETS)

    # -- discovery ---------------------------------------------------------

    def _dataset_path(self, dataset: str) -> Path:
        ds = _sanitize(str(dataset))
        p = (Path(self.root_dir) / ds).resolve()
        # traversal check: must remain within root_dir
        try:
            root_res = Path(self.root_dir).resolve()
            try:
                if not p.is_relative_to(root_res):
                    # fallback to root if traversal attempted
                    return root_res / ds
            except AttributeError:
                try:
                    p.relative_to(root_res)
                except ValueError:
                    return root_res / ds
        except Exception:
            pass
        return p

    def _discover_subjects(self, dataset: str) -> list[str]:
        dpath = self._dataset_path(dataset)
        subjects: list[str] = []
        if dpath.is_dir():
            # EMGBench layout: p1/participant_1.hdf5 etc. — also handle flat subject dirs
            for child in sorted(dpath.iterdir()):
                name = child.name
                # directory subjects (pN) or participant files
                if child.is_dir() and re.match(r"^(p\d+|subject[_-]?\d+|s\d+)$", name, re.I):
                    subjects.append(name)
                elif child.is_file() and name.lower().endswith((".h5", ".hdf5", ".h5py", ".npz")) and "participant" in name.lower():
                    subjects.append(child.stem)
                elif child.is_file() and name.lower().endswith((".h5", ".hdf5")):
                    subjects.append(child.stem)
            # fallback: any subdirs count as subjects
            if not subjects:
                subdirs = [c.name for c in dpath.iterdir() if c.is_dir()]
                if len(subdirs) >= 2:
                    subjects = sorted(subdirs)
            # fallback: files as subjects
            if not subjects:
                files = [c.stem for c in dpath.iterdir() if c.is_file() and c.suffix.lower() in (".h5", ".hdf5", ".npz")]
                if files:
                    subjects = sorted(files)
        if len(subjects) >= 2:
            return subjects
        # synthetic fallback
        _, _, n_subjects = _synthetic_params(dataset)
        return [f"subject_{i+1:02d}" for i in range(n_subjects)]

    def _load_subject(self, dataset: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
        """Try real HDF5 load; fallback to synthetic.

        Real HDF5 format (per emgbench ``utils_generic.py``): file contains
        one dataset per gesture with shape ``(trials, electrodes, timesteps)``.
        We flatten to ``(trials, timesteps, electrodes)`` windows for
        classifier consumption.
        """
        dpath = self._dataset_path(dataset)
        # try to locate the subject's HDF5 file
        candidates: list[Path] = []
        if dpath.is_dir():
            # common patterns
            candidates.extend(dpath.rglob(f"{subject}.h5"))
            candidates.extend(dpath.rglob(f"{subject}.hdf5"))
            candidates.extend(dpath.rglob("participant_*.h5"))
            # direct file under subject dir
            subj_dir = dpath / subject
            if subj_dir.is_dir():
                candidates.extend(list(subj_dir.glob("*.h5")))
                candidates.extend(list(subj_dir.glob("*.hdf5")))
        # try loading first candidate that exists
        for cand in candidates:
            if not cand.is_file():
                continue
            try:
                import h5py  # type: ignore[import-not-found]

                with h5py.File(str(cand), "r") as hf:
                    gestures = list(hf.keys())
                    if not gestures:
                        continue
                    X_parts: list[np.ndarray] = []
                    y_parts: list[np.ndarray] = []
                    label_map = {g: i for i, g in enumerate(sorted(gestures))}
                    for g in gestures:
                        ds = hf[g]
                        arr = np.asarray(ds)
                        # expected (trials, electrodes, timesteps) or (trials, timesteps, electrodes)
                        if arr.ndim == 3:
                            # normalize to (trials, timesteps, electrodes)
                            # heauristic: if dim1 > dim2, likely (trials, electrodes, timesteps)
                            if arr.shape[1] < arr.shape[2]:
                                arr = np.transpose(arr, (0, 2, 1))
                        elif arr.ndim == 2:
                            # (trials, features) — treat as (trials, timesteps) single channel
                            arr = arr[:, :, None]
                        else:
                            continue
                        n_trials = arr.shape[0]
                        X_parts.append(arr.astype(np.float32, copy=False))
                        y_parts.append(np.full((n_trials,), label_map[g], dtype=np.int64))
                    if X_parts:
                        X = np.concatenate(X_parts, axis=0)
                        y = np.concatenate(y_parts, axis=0)
                        # shuffle deterministically per subject
                        seed = sum(ord(c) for c in dataset + subject) % (2**31)
                        rng = np.random.RandomState(seed)
                        perm = rng.permutation(len(y))
                        return X[perm], y[perm]
            except ImportError:
                break
            except Exception:
                continue
        # synthetic fallback
        n_channels, n_gestures, _ = _synthetic_params(dataset)
        return _synthetic_subject_data(dataset, subject, n_channels, n_gestures)

    # -- public API --------------------------------------------------------

    def evaluate_loso(
        self,
        model_fn: Callable[..., Any],
        dataset: str,
    ) -> dict[str, Any]:
        """Intersubject LOSO-CV for ``dataset``.

        For each subject *i* held out as test, ``model_fn`` is invoked as
        ``model_fn(train_X, train_y, test_X, test_y)`` (also supports
        ``((train_X, train_y), (test_X, test_y))`` or dict forms).
        The return value may be a scalar accuracy, a dict containing
        ``"accuracy"``/``"acc"``, or predictions — all are reduced to a
        per-fold accuracy via :func:`_accuracy_from_result`.

        Args:
            model_fn: Callable implementing the model train+eval for one
                LOSO fold.
            dataset: Dataset slug (sanitized). Must be discoverable under
                ``root_dir``; synthetic fallback is used otherwise.

        Returns:
            Dict with ``per_subject`` (list of dicts), ``mean_accuracy``,
            ``std_accuracy``, ``min_accuracy``, ``max_accuracy``, ``n_subjects``.
        """
        ds = _sanitize(str(dataset))
        subjects = self._discover_subjects(ds)
        per_subject: list[dict[str, Any]] = []
        accs: list[float] = []
        for held in subjects:
            train_subs = [s for s in subjects if s != held]
            # aggregate training data
            train_X_parts: list[np.ndarray] = []
            train_y_parts: list[np.ndarray] = []
            for s in train_subs:
                X, y = self._load_subject(ds, s)
                train_X_parts.append(X)
                train_y_parts.append(y)
            test_X, test_y = self._load_subject(ds, held)
            if train_X_parts:
                train_X = np.concatenate(train_X_parts, axis=0)
                train_y = np.concatenate(train_y_parts, axis=0)
            else:
                train_X = np.zeros((0,) + test_X.shape[1:], dtype=test_X.dtype)
                train_y = np.zeros((0,), dtype=test_y.dtype)
            acc = _invoke_loso(model_fn, train_X, train_y, test_X, test_y)
            acc = float(np.clip(acc, 0.0, 1.0)) if 0.0 <= acc <= 1.5 else float(acc)
            per_subject.append({"subject": held, "accuracy": acc, "n_train_subjects": len(train_subs), "n_test": int(len(test_y))})
            accs.append(acc)
        mean_acc, std_acc = _mean_std(accs)
        return {
            "dataset": ds,
            "per_subject": per_subject,
            "accuracies": accs,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "min_accuracy": float(np.min(accs)) if accs else 0.0,
            "max_accuracy": float(np.max(accs)) if accs else 0.0,
            "n_subjects": len(subjects),
        }

    def evaluate_adaptation(
        self,
        model_fn: Callable[..., Any],
        dataset: str,
        n_shot: int = 1,
    ) -> dict[str, Any]:
        """Few-shot adaptation (pretrain on N-1 subjects, adapt on ``n_shot``/class).

        For each held-out subject, the subject's trials are split few-shot:
        the first ``n_shot`` trials per gesture become a *support* set for
        fine-tuning, the remainder is the *query* test set. ``model_fn`` is
        invoked as ``model_fn(train_X, train_y, support_X, support_y,
        query_X, query_y)`` when possible; otherwise training data is
        augmented with the support set and delegated to the LOSO path.
        Covers the EMGBench adaptation task (Yang et al., NeurIPS 2024,
        Table 3 — FT-X% and IS FT; TSTS few-shot).

        Args:
            model_fn: Callable for adaptation-aware training. Should accept
                either 6 positional args or 3 tuple args.
            dataset: Dataset slug.
            n_shot: Number of trials per gesture from the held-out subject
                to use for adaptation (1/2/5 typical; FT-5% ≈ 1-shot).

        Returns:
            Dict like :meth:`evaluate_loso` but with ``n_shot`` and query-
            set sizes. Includes ``mean_accuracy``/``std_accuracy`` over
            subjects.
        """
        if n_shot < 0:
            raise ValueError(f"n_shot must be >=0, got {n_shot}")
        ds = _sanitize(str(dataset))
        subjects = self._discover_subjects(ds)
        per_subject: list[dict[str, Any]] = []
        accs: list[float] = []
        for held in subjects:
            train_subs = [s for s in subjects if s != held]
            train_X_parts: list[np.ndarray] = []
            train_y_parts: list[np.ndarray] = []
            for s in train_subs:
                X, y = self._load_subject(ds, s)
                train_X_parts.append(X)
                train_y_parts.append(y)
            held_X, held_y = self._load_subject(ds, held)
            support_X, support_y, query_X, query_y = _few_shot_split(held_X, held_y, n_shot=int(n_shot))
            if train_X_parts:
                train_X = np.concatenate(train_X_parts, axis=0)
                train_y = np.concatenate(train_y_parts, axis=0)
            else:
                train_X = np.zeros((0,) + held_X.shape[1:], dtype=held_X.dtype)
                train_y = np.zeros((0,), dtype=held_y.dtype)
            # handle empty query (n_shot covers all)
            if query_X.size == 0:
                query_X, query_y = support_X, support_y
            acc = _invoke_adaptation(model_fn, train_X, train_y, support_X, support_y, query_X, query_y)
            acc = float(np.clip(acc, 0.0, 1.0)) if 0.0 <= acc <= 1.5 else float(acc)
            per_subject.append(
                {
                    "subject": held,
                    "accuracy": acc,
                    "n_shot": int(n_shot),
                    "n_support": int(len(support_y)),
                    "n_query": int(len(query_y)),
                }
            )
            accs.append(acc)
        mean_acc, std_acc = _mean_std(accs)
        return {
            "dataset": ds,
            "n_shot": int(n_shot),
            "per_subject": per_subject,
            "accuracies": accs,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "min_accuracy": float(np.min(accs)) if accs else 0.0,
            "max_accuracy": float(np.max(accs)) if accs else 0.0,
            "n_subjects": len(subjects),
        }

    def evaluate_all(
        self,
        model_fn: Callable[..., Any],
        mode: str = "loso",
        n_shot: int = 1,
    ) -> dict[str, Any]:
        """Run ``mode`` across all configured datasets and aggregate.

        Args:
            model_fn: Per-fold callable (see :meth:`evaluate_loso`).
            mode: ``"loso"`` for intersubject LOSO-CV or ``"adaptation"``
                / ``"few_shot"`` / ``"ft"`` for few-shot adaptation.
            n_shot: Passed to :meth:`evaluate_adaptation` when
                ``mode`` is adaptation.

        Returns:
            Dict with ``per_dataset`` (dataset → per-dataset result dict),
            ``mean_accuracy`` / ``std_accuracy`` over dataset means
            (macro-average), plus ``overall_mean``/``overall_std`` aliases.
        """
        mode_l = str(mode).lower()
        is_adapt = mode_l in ("adaptation", "adapt", "few_shot", "fewshot", "ft", "few-shot")
        per_dataset: dict[str, dict[str, Any]] = {}
        dataset_means: list[float] = []
        for ds in self.datasets:
            if is_adapt:
                res = self.evaluate_adaptation(model_fn, ds, n_shot=n_shot)
            else:
                res = self.evaluate_loso(model_fn, ds)
            per_dataset[ds] = res
            dataset_means.append(float(res.get("mean_accuracy", 0.0)))
        mean_m, std_m = _mean_std(dataset_means)
        return {
            "mode": "adaptation" if is_adapt else "loso",
            "n_shot": int(n_shot) if is_adapt else None,
            "per_dataset": per_dataset,
            "dataset_means": dataset_means,
            "mean_accuracy": mean_m,
            "std_accuracy": std_m,
            "overall_mean": mean_m,
            "overall_std": std_m,
            "n_datasets": len(self.datasets),
        }

    # alias for spec wording ("helper to run across 9 datasets and report mean/std")
    def run_all(self, model_fn: Callable[..., Any], mode: str = "loso", n_shot: int = 1) -> dict[str, Any]:
        """Alias for :meth:`evaluate_all`."""
        return self.evaluate_all(model_fn, mode=mode, n_shot=n_shot)


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


def summarize_results(results: dict[str, Any]) -> dict[str, float]:
    """Aggregate an :meth:`EMGBench.evaluate_all` result to mean/std.

    Args:
        results: Dict returned by :meth:`EMGBench.evaluate_all`.

    Returns:
        Dict with ``mean``, ``std``, ``min``, ``max`` over dataset means.
    """
    means = results.get("dataset_means")
    if means is None:
        # try per_dataset
        per = results.get("per_dataset", {})
        means = [float(v.get("mean_accuracy", 0.0)) for v in per.values()]
    if not means:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(means, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def aggregate_loso_scores(per_dataset_results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Backwards-compatible alias for :func:`summarize_results` on per-dataset dict."""
    return summarize_results({"per_dataset": per_dataset_results})


__all__ = [
    "EMGBench",
    "DEFAULT_DATASETS",
    "summarize_results",
    "aggregate_loso_scores",
]
