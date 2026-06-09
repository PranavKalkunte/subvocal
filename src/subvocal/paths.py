"""Resolution of writable data and model directories.

The SDK writes pipeline traces, correction logs, and trained model weights at
runtime. After installation the package directory lives in ``site-packages``
and must never be written to, so all writable paths resolve to a per-user data
directory instead. Resolution order:

1. Environment variables ``SUBVOCAL_DATA_DIR`` / ``SUBVOCAL_MODELS_DIR``.
2. The OS-conventional user data directory (via ``platformdirs``),
   e.g. ``~/Library/Application Support/subvocal`` on macOS or
   ``~/.local/share/subvocal`` on Linux.

Directories are created on first use.
"""

import os

from platformdirs import user_data_dir

_APP_NAME = "subvocal"

DATA_DIR_ENV = "SUBVOCAL_DATA_DIR"
MODELS_DIR_ENV = "SUBVOCAL_MODELS_DIR"


def _resolve(env_var: str, subdir: str) -> str:
    override = os.environ.get(env_var)
    path = override if override else os.path.join(user_data_dir(_APP_NAME), subdir)
    os.makedirs(path, exist_ok=True)
    return path


def get_data_dir() -> str:
    """Writable directory for traces, correction logs, and datasets."""
    return _resolve(DATA_DIR_ENV, "data")


def get_models_dir() -> str:
    """Writable directory for trained and downloaded model weights."""
    return _resolve(MODELS_DIR_ENV, "models")
