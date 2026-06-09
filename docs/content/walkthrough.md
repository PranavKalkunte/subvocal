# Subvocal Middleware Platform Walkthrough

Welcome to the end-to-end interactive notebook walkthrough for the Subvocal Middleware SDK. This notebook will guide you through:
1. **Synthetic sEMG Generation**: Simulating biometric sensor data.
2. **DSP Filtering & Extraction**: Applying bandpass filters and calculating TD10 features.
3. **Gesture Classification**: Training and evaluating an SVM classifier.
4. **Context collation**: Building active user application states.
5. **Intent Reconstruction**: Translating noisy phonetic tokens to semantic system actions.

Let's get started!

## 1. Environment Setup

Install the SDK from PyPI (or from the repository checkout with `pip install -e ".[ml]"`).

```python
%pip install "subvocal[ml]" matplotlib --quiet

import numpy as np
import time
import matplotlib.pyplot as plt

import subvocal
print(f"Subvocal SDK {subvocal.__version__} loaded!")
```

## 2. Physiological Bio-Signal Simulation

We use the `SyntheticSignalGenerator` to mimic multi-channel electromyography muscle potentials (Gaussian white noise + powerline notch shape) and inject contractions.

```python
from subvocal.hardware.drivers import SyntheticSignalGenerator

# 1. Initialize synthetic generator (8 channels, 250Hz sample rate)
hardware = SyntheticSignalGenerator(fs=250.0, num_channels=8)
hardware.start()

# 2. Read a 500ms frame of data
frame = hardware.read_frame(window_ms=500)
data = frame.to_numpy()

print(f"Frame shape (samples x channels): {data.shape}")

# 3. Plot signal trace
plt.figure(figsize=(10, 4))
plt.plot(data[:, 0], label='Channel 0')
plt.plot(data[:, 1], label='Channel 1')
plt.title("Raw Simulated sEMG Waveforms")
plt.xlabel("Sample Index")
plt.ylabel("Micro-Volts (uV)")
plt.legend()
plt.grid(True)
plt.show()

hardware.stop()
```

## 3. DSP and Bandpass Filtering

We apply the reconciled bandpass filter (1.3Hz to 50Hz) designed for silent speech to strip powerline noise and biological drift.

```python
from subvocal.emg_core.dsp.filters import BandpassFilter

# 1. Initialize physiological filter
bp_filter = BandpassFilter(low_cutoff=1.3, high_cutoff=50.0, fs=250.0, order=4)

# 2. Filter raw data
filtered_data = bp_filter.apply(data)

plt.figure(figsize=(10, 4))
plt.plot(data[:, 0], alpha=0.5, label='Raw Channel 0')
plt.plot(filtered_data[:, 0], label='Filtered Channel 0')
plt.title("Physiological Bandpass Filtering Verification")
plt.xlabel("Sample Index")
plt.ylabel("Micro-Volts (uV)")
plt.legend()
plt.grid(True)
plt.show()
```

## 4. Feature Extraction (TD10 Segment Pool)

We extract TD10 temporal-domain features (Mean Absolute Value, Waveform Length, Zero Crossings, Slope Sign Changes) across channels to create model inputs.

```python
from subvocal.emg_core.dsp.features import extract_td10_features

# Extract features over the filtered frame
features = extract_td10_features(filtered_data)
print(f"Extracted feature vector length: {len(features)}")
print(f"Features preview (first 10 items): {features[:10]}")
```

## 5. Machine Learning Classifier Training

We train an SVM gesture classifier on synthetic data and calibrate probabilities.

```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Generate mock training dataset (3 classes: rest, clench, LipMove)
np.random.seed(42)
num_samples = 60
X_train = np.random.randn(num_samples, len(features))
# Inject higher amplitude in clench class features
y_train = np.random.choice([0, 1, 2], size=num_samples)
X_train[y_train == 1] *= 5.5

# Fit calibrated baseline SVM model
base_svc = SVC(kernel='rbf', probability=True, C=1.0)
classifier = CalibratedClassifierCV(base_svc)
classifier.fit(X_train, y_train)

# Perform inference test
test_feat = X_train[0].reshape(1, -1)
pred_label = classifier.predict(test_feat)[0]
probs = classifier.predict_proba(test_feat)[0]

print(f"Predicted Gesture Label Index: {pred_label}")
print(f"Calibrated Probabilities: {probs}")
```

## 6. End-to-End Context-Aware Intent Reconstruction

We set up the full `SubvocalPipeline` leveraging user contexts and simulated LLM decoders.

```python
from subvocal.context.schema import UserContext, AppState
from subvocal.core.models import CommandToken, Action, Intent
from subvocal.core.interfaces import ActionExecutor, ContextProvider, LLMProvider
from subvocal.core.pipeline import SubvocalPipeline

# Define mock client executor
class MyExecutor(ActionExecutor):
    def execute(self, action: Action):
        print(f"\n[⚡ EXECUTOR DISPATCH] Action Executed: {action.action_type.upper()}")
        print(f"Arguments: {action.params.get('arguments')}")
        
    def can_execute(self, action: Action) -> bool:
        return True

# Define mock LLM intent decoder
class MyLLM(LLMProvider):
    def reconstruct_intent(self, tokens: list, context: UserContext) -> Intent:
        shorthand = " ".join([t.text for t in tokens])
        # Perform translation based on shorthand token content
        if "clk" in shorthand:
            return Intent(
                command="CLICK",
                arguments=["link: Getting Started"],
                resolved_text="Click the link",
                raw_shorthand=shorthand,
                confidence=0.96,
                timestamp=time.time()
            )
        return Intent(
            command="SCROLL",
            arguments=["down"],
            resolved_text="Scroll page",
            raw_shorthand=shorthand,
            confidence=0.85,
            timestamp=time.time()
        )
        
    def get_provider_name(self) -> str:
        return "my_mock_llm"

# Define static context provider
class MyContext(ContextProvider):
    def get_context(self) -> UserContext:
        return UserContext(app_state=AppState(current_app="Terminal"))
    def get_provider_name(self) -> str:
        return "my_mock_context"

# Setup Pipeline with synthetic signal source
hw_source = SyntheticSignalGenerator()
hw_source.start()

pipeline = SubvocalPipeline(
    hardware=hw_source,
    classify_fn=lambda frame: CommandToken(text="clk", confidence=0.95, timestamp=time.time()),
    llm_provider=MyLLM(),
    context_provider=MyContext(),
    executor=MyExecutor(),
    phrase_timeout_seconds=0.5
)

# Inject mock token and process phrase
pipeline._token_buffer.append(CommandToken(text="clk", confidence=0.95, timestamp=time.time()))
action = pipeline.process_phrase()
hw_source.stop()

print("\nPipeline execution successfully demonstrated!")
```

## 7. Advanced LiveKit Subsystems Usage (v2.0)

For multi-device or distributed edge deployments, the SDK provides persistent storage, ingress/egress lifecycles, load selectors, and TCP-based biometric metrics streaming.

### 7.1 Persistent Worker and State Storage

Here we initialize a SQLite state database and configure a session worker with persistent tracking:

```python
from subvocal.runtime import SessionWorker, SQLiteSessionStore
from subvocal.config import load_config

# Initialize configuration
config = load_config()

# 1. Setup persistent SQLite store
store = SQLiteSessionStore("my_sessions.db")

# 2. Instantiate worker coordinating sessions
worker = SessionWorker(config, max_sessions=5, store=store)

# Create session (instantly persists config & metadata to database)
session = worker.create_session("sess-001", hw_source, lambda f: None, MyLLM(), MyContext(), MyExecutor())
print(f"Persisted sessions: {store.list_sessions()}")
```

### 7.2 Ingress Failover Management

Unifies biometric sources and handles dynamic connection drops automatically:

```python
from subvocal.runtime import IngressManager
from subvocal.hardware.drivers import SyntheticSignalGenerator

ingress = IngressManager()
primary_collar = SyntheticSignalGenerator(fs=1000.0)
fallback_replay = SyntheticSignalGenerator(fs=250.0)

# Register sources
ingress.register_source("primary_cyton", primary_collar, is_fallback=False)
ingress.register_source("fallback_sim", fallback_replay, is_fallback=True)

ingress.start()
print(f"Active sensor: {ingress.active_name}")

# Simulating a signal dropout failover
ingress.trigger_failover()
print(f"Failed over to: {ingress.active_name}")
ingress.stop()
```

### 7.3 Real-Time TCP Biometric Data Channel

Broadcasts biometric metrics (signal levels, quality scores, classifier outputs) to visualization tools:

```python
from subvocal.stream import BiometricDataChannelServer, BiometricDataChannelClient

# 1. Start TCP streaming server
server = BiometricDataChannelServer(host="127.0.0.1", port=8105)
server.start()

# 2. Connect visualization client
client = BiometricDataChannelClient(host="127.0.0.1", port=8105)
client.connect()

# 3. Broadcast metric update
server.broadcast({"event": "frame_processed", "quality_score": 4.5, "token": "clk"})

# 4. Receive stream on client
for msg in client.read_messages():
    print("Received Live Biometric Update:", msg)
    break

client.close()
server.close()
```
