# Getting Started

This guide walks through setting up your environment, running the SDK, and standing up the telemetry dashboard.

---

## 1. Prerequisites

Subvocal SDK requires **Python 3.10** or higher. We recommend creating a virtual environment:

```bash
# Clone the repository
git clone https://github.com/PranavKalkunte/subvocal.git
cd subvocal

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Installation

Install dependencies from the root directory:

```bash
pip install -r requirements.txt
```

*(Note: If you are running on macOS Apple Silicon, PyTorch CPU builds are recommended to bypass local MPS pool issues).*

To install optional developer dependencies (for ONNX model export and dataset reading):
```bash
pip install onnx scipy h5py brainflow
```

## 3. Running Verification Tests

To verify that the entire SDK is installed correctly, execute the full test suite:

```bash
PYTHONPATH=sdk python3 -m unittest discover -s sdk
```

You should see all unit and integration tests passing successfully:
```text
Ran 33 tests in 16.621s
OK (skipped=1)
```

## 4. Starting the Observability Dashboard

The SDK includes a local zero-dependency telemetry dashboard. To launch it on port `8000`:

```bash
PYTHONPATH=sdk python3 -m sdk.core.dashboard --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser. The page features real-time visualization of classification histories, latency metrics, and model confidence curves.
