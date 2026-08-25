# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 2.0.x   | ✅ |
| 1.0.x   | ✅ |
| < 1.0   | ❌ |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately via GitHub's [private vulnerability reporting](https://github.com/PranavKalkunte/subvocal/security/advisories/new)
or email the maintainer at `pranav.kalkunte@utexas.edu` with:

- A description of the issue and its impact
- Reproduction steps or a proof of concept
- Affected version(s)

You can expect an acknowledgment within 72 hours and a remediation plan or
status update within 14 days. Credit is given in the release notes unless you
prefer otherwise.

## Scope notes for integrators

The SDK processes physiological (sEMG) signals — biometric data with legal
protection in several jurisdictions (BIPA, GDPR, HIPAA). The platform
threat model, data-residency guidance, and policy-engine hardening
recommendations are published in the
[Security & Threat Model](https://pranavkalkunte.github.io/subvocal/platform/security.html)
document. Key defaults:

- All processing is local; no telemetry or cloud calls occur unless an LLM
  provider is explicitly configured.
- Action execution is gated by the pluggable policy engine; dry-run mode and
  `raise_on_policy_violation` are available for high-assurance deployments.
- Pipeline traces are written only to the per-user data directory
  (`SUBVOCAL_DATA_DIR`); they may contain reconstructed intent text and should
  be treated as sensitive. Disable entirely with `telemetry.trace_enabled: false` / `runtime.trace_enabled: false` or env `SUBVOCAL_TELEMETRY__TRACE_ENABLED=false` / `SUBVOCAL_RUNTIME__TRACE_ENABLED=false` (opt-out, traces rotate at 10 MB when enabled).
- **2.0.1 hardening**: secure deserialization (`torch.load(weights_only=True)` — C1), path-traversal sanitization for `model_path` and file I/O (C2), HMAC `=` padding fix (C3), deadlock-free `pipeline.step()` timeout (C4), thread-safe `deque` + `inject_token()` (C5), bounded `BoardShim` buffers (C6), TTS flag-injection sanitization (C7), Prometheus low-cardinality (removed `session_id` label, bounded port/registry — H9), and correctly-plumbed DSP notch (H2). See [CHANGELOG.md](CHANGELOG.md) for full C1–C8 / H1–H10 list.
