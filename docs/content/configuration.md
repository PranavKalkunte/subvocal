# Configuration & Runtime (v2)

The v2 architecture introduces a single source of configuration truth and a
production runtime layer (sessions, signal monitoring, telemetry, and auth
grants) layered on top of the v1 pipeline. This page is the developer guide to
both.

## The configuration tree

Every subsystem is constructed from one validated Pydantic model,
`subvocal.config.SubvocalConfig`. It is composed of typed sections — `hardware`,
`dsp`, `classifier`, `provider`, `policy`, `telemetry`, `runtime`, and `auth` —
each with sensible defaults.

```python
from subvocal.config import load_config

# All keys are optional; omit the path for an all-defaults config.
config = load_config("subvocal.yaml")
print(config.hardware.sample_rate)     # 250
print(config.dsp.bandpass_high)        # 50.0
```

Configuration is **strict**: `extra="forbid"` is set on every section, so an
unknown or misspelled key raises `subvocal.exceptions.ConfigurationError`
instead of being silently dropped. A full annotated template lives in
`subvocal-sample.yaml`.

### Environment overrides

Any nested field can be overridden from the environment using the
`SUBVOCAL_<SECTION>__<KEY>` convention, where a double underscore separates the
section from the key:

```bash
export SUBVOCAL_HARDWARE__SAMPLE_RATE=500
export SUBVOCAL_DSP__BANDPASS_LOW=2.5
export SUBVOCAL_TELEMETRY__ENABLED=true
```

Values are coerced to booleans, numbers, or `None` automatically. Overrides are
merged on top of the YAML file (env wins).

> **Reserved variables.** The flat `SUBVOCAL_DATA_DIR` and `SUBVOCAL_MODELS_DIR`
> variables are consumed by `subvocal.paths` for writable-directory resolution
> and are **not** configuration keys. Only variables containing the `__`
> section separator are treated as config overrides.

### Coercion order and `trace_enabled`

Env values are coerced in strict order: `int` → `float` → `bool` (only
`true`/`false`/`yes`/`no`) → `None` (`none`/`null`) → `str`. Parsing `int`
before `bool` preserves numeric types — `SUBVOCAL_HARDWARE__SAMPLE_RATE=1`
remains `int 1`, not `True` — and `float` parsing handles scientific notation
(`1e3` → `1000.0`).

Tracing is gated by two flags, either of which can disable it (biometric-PII
opt-out):

```bash
export SUBVOCAL_TELEMETRY__TRACE_ENABLED=false
export SUBVOCAL_RUNTIME__TRACE_ENABLED=false   # alternative / legacy
```

When enabled, `Session._write_trace` appends to `pipeline_traces.jsonl` (under
`get_data_dir()`) but enforces a **10 MB rotation cap**: if the file already
exceeds `10 * 1024 * 1024` bytes the write is skipped with a warning until the
file is rotated or tracing is disabled.

### Validated models

Pydantic validators enforce invariants at ingestion:

- `Frame`: `end_time > start_time` and `fs > 0`, else `ValueError`.
- `Sample`: `channels` non-empty and `len(channels) <= 128`.
- `CommandToken` / `Intent`: `0.0 <= confidence <= 1.0`.

Violations surface as `ValidationError` → `ConfigurationError` on
`load_config` / model construction, failing fast rather than propagating
corrupt frames.

## Sessions and the runtime layer

`subvocal.runtime.Session` wraps the pipeline with the machinery a long-running
deployment needs:

- **Lifecycle state machine** — `STARTING → ACTIVE → DEGRADED → CLOSED`, with an
  observable `on_state_changed` callback.
- **Liveness watchdog** — transitions the session to `DEGRADED` and emits a
  `HardwareError` if frames stop arriving within `runtime.session_liveness_timeout`.
- **Asynchronous processing** — frames are processed on a serialized
  **bounded** `OpsQueue` worker thread (`subvocal.utils`, `maxsize=min_size`
  default `128`). The queue applies backpressure — `enqueue()` returns `False`
  and logs when full (new task dropped) — and exposes `qsize()` / `is_full()`
  for monitoring, keeping ingestion non-blocking. On `stop()`, sentinel
  insertion evicts the oldest entry if full to guarantee drain.
- **Signal monitoring** (`subvocal.stream`):
  - `SignalLevel` — EMA-smoothed, percentile-of-window subvocalization activity
    detection.
  - `StreamTracker` — hysteresis-based active/stopped segmentation that drives
    phrase boundaries (replacing naive wall-clock timeouts).
  - `SignalQualityScorer` — a MOS-like quality score (saturation, drift,
    dropouts, SNR) mapped to `EXCELLENT / GOOD / POOR / LOST`; a lost signal
    pauses classification.
- **Trace control** — JSONL tracing in `Session._write_trace` honors
  `telemetry.trace_enabled` and `runtime.trace_enabled` (env
  `SUBVOCAL_TELEMETRY__TRACE_ENABLED` / `SUBVOCAL_RUNTIME__TRACE_ENABLED`);
  either `False` disables tracing. When enabled, the file is capped at
  **10 MB** — further writes are skipped until rotation.

`subvocal.runtime.SessionWorker` manages a pool of sessions with capacity
limits and load reporting. Because it exposes `id`, `load`, `cpu_usage`, and
`status`, a worker satisfies the routing `WorkerNode` protocol and can be ranked
by `subvocal.routing.CPULoadSelector` or `SessionCountSelector`.

## Telemetry

`subvocal.telemetry` defines a `TelemetryService` interface with a no-op
`NullTelemetry` default. Install the `metrics` extra to export to Prometheus:

```bash
pip install "subvocal[metrics]"
```

```python
from subvocal.telemetry import PrometheusTelemetry
from subvocal.config import load_config

telemetry = PrometheusTelemetry(load_config())   # honors telemetry.enabled / .prometheus_port
```

It reports active/total sessions, phrases, intents, executed and blocked
actions, provider retries, signal-quality scores, and error counts. A
ready-to-import dashboard ships at
`src/subvocal/telemetry/grafana_dashboard.json`.

## Capability grants

`subvocal.auth` provides HMAC-signed, capability-scoped tokens (`ActionGrants`)
that constrain which commands a caller may trigger, a minimum confidence floor,
and whether execution is forced into dry-run. Grants propagate through a
`contextvars` context and are enforced by `GrantsPolicy`, which plugs into the
standard `PolicyEngine`:

```python
from subvocal.auth import ActionGrants, generate_token, verify_token, set_context_grants
from subvocal.auth.grants import GrantsPolicy

grants = ActionGrants(allowed_commands=["goto", "search"], min_confidence=0.7)
token = generate_token(grants, secret_key="...")     # hand to a client
# ...on the server, per request:
set_context_grants(verify_token(token, secret_key="..."))
```

## BrainFlow-compatible acquisition

`subvocal.hardware.brainflow_compat` and `subvocal.emg_core.dsp.brainflow_filter`
mirror BrainFlow's `BoardShim` and `DataFilter` APIs in pure Python (NumPy +
SciPy), so existing BrainFlow code runs with no native build. If the official
`brainflow` package is installed it is used transparently as the backend;
otherwise the built-in synthetic generator and direct Cyton serial parser take
over.
