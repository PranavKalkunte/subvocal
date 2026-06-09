# Subvocal SDK SemVer and Release Process Specification

**Status:** Proposed  
**Version:** v0.1.0-alpha  
**Date:** June 2026  
**Audience:** Platform Maintainers, Community Contributors  

---

## 1. Semantic Versioning (SemVer) Policy

The Subvocal SDK follows the standard Semantic Versioning 2.0.0 specification (`MAJOR.MINOR.PATCH`), customized for a pluggable middleware framework:

### Major Version Bump (`X.0.0`)
A major version bump is triggered by **backwards-incompatible changes** to the public API surface. Examples include:
* Modifying abstract base class signatures in `sdk/core/interfaces.py` (e.g., changing parameter definitions or return types of `HardwareSource.read_frame` or `LLMProvider.reconstruct_intent`).
* Removing or renaming fields in core data models in `sdk/core/models.py`.
* Altering the execution flow or constructor requirements of the `SubvocalPipeline` orchestrator.
* Breaking changes to the Model Context Protocol (MCP) tool schemas exposed to LLM clients.

### Minor Version Bump (`0.Y.0` or `X.Y.0`)
A minor version bump represents **backwards-compatible feature additions**. Examples include:
* Adding new subclass implementations of abstract interfaces (e.g., a new `OpenBCI` hardware driver, a new local `Llama` LLM adapter, or an app-specific context provider).
* Introducing non-breaking, optional utility methods to existing classes or models.
* Exposing new optional parameters in Pydantic models or configuration schema.

### Patch Version Bump (`0.Y.Z` or `X.Y.Z`)
A patch version bump covers **backwards-compatible bug fixes and optimizations**. Examples include:
* Adjusting signal-processing filter thresholds (e.g. tuning the 60 Hz notch filter parameters).
* Tuning prompt templates in the `shorthand/decoder.py` module to improve intent reconstruction accuracy.
* Resolving platform or operating system specific bugs (e.g., resolving MPS/CPU pooling issues in PyTorch inference).
* Correcting typos in documentation or improving error messages.

---

## 2. Release Cadence and Lifecycle

We structure our release cadence around three lifecycle phases to balance developer safety with rapid feature iteration:

```
[ Experimental Features ] ──> ( v0.1.0-alpha ) ──> ( v0.1.0-beta ) ──> ( v0.1.0 )
                                   │                   │                   │
                                 [ Dev ]            [ Pilot ]          [ Stable ]
```

### 1. Alpha Phase (`v0.X.X-alpha.N`)
* **Purpose:** High-velocity iteration, experimental features, and API design prototyping.
* **Stability:** Unstable. APIs may change day-to-day without warning.
* **Testing:** Basic unit test validation.

### 2. Beta Phase (`v0.X.X-beta.N`)
* **Purpose:** Feature-complete candidate builds released for simulated pilot testing, community feedback, and developer integration.
* **Stability:** Reasonably stable. Breaking changes are kept to a minimum and require documentation.
* **Testing:** Full test suite execution, integration benchmarks, and verification of reference application flows.

### 3. Stable Release (`vX.Y.Z`)
* **Purpose:** Production-ready release for mainstream developers, clinical rigs, and commercial integrations.
* **Stability:** Production grade. Complete backwards-compatibility guarantees within the major version.
* **Testing:** End-to-end automated smoke tests, performance/latency benchmarks, and verification across all supported LLM providers.

---

## 3. Pre-1.0.0 (v0.y.z) Compatibility Exception

Per SemVer specification, during the initial development cycle (pre-1.0.0):
* The public API surface is not considered fully stable.
* **Minor versions (`0.Y.0`) may introduce breaking changes** to allow the API to mature.
* Patch versions (`0.Y.Z`) must remain strictly backwards-compatible.
* Stable production integrations should pin dependencies to the minor version (e.g., `subvocal-sdk~=0.1.0`).

---

## 4. Automated CI/CD Requirements

To maintain platform stability across distributed hardware and software integrations, the main repository enforces an automated CI/CD baseline:

### Pull Request (PR) Validation
Every PR targeting the `main` or `release/*` branches must pass:
1. **Linting & Code Formatting:** Code must comply with standard style rules verified via `ruff` or `flake8`.
2. **Type Checking:** Strict static analysis passing `mypy` check on the `sdk/` codebase.
3. **Unit & Integration Tests:** Automated execution of the full test suite via GitHub Actions:
   ```bash
   python3 -m unittest discover -s sdk
   ```

### Release Publication
When a release tag (e.g. `v0.1.0`) is pushed by a platform maintainer:
1. The CI pipeline executes the complete regression suite.
2. The pipeline automatically builds source and binary wheels.
3. If tests pass, the artifacts are signed and published automatically to the official PyPI registry.
