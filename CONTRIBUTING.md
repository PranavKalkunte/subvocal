# Contributing to the Subvocal SDK

Thanks for your interest in improving the Subvocal SDK. This document covers the
development workflow, quality gates, and conventions the project enforces.

## Development setup

```bash
git clone https://github.com/PranavKalkunte/subvocal.git
cd subvocal
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
```

The base package is intentionally lightweight (pydantic + numpy). Heavier
subsystems live behind extras — keep it that way: any new dependency must be
either stdlib, a base dependency with strong justification, or an optional
extra with a lazy, guarded import that raises
`subvocal.exceptions.MissingDependencyError` naming the extra.

## Quality gates

Every pull request must pass the same gates CI runs:

```bash
ruff check src tests benchmarks tools   # lint (E, F, I, UP, B; E501/E741 ignored)
pyright                                  # type check (standard mode, 0 errors required)
pytest --cov=subvocal --cov-report=term-missing --cov-fail-under=65  # test suite (65% floor enforced in CI)
pip-audit                                # dependency CVE audit
python tools/check_licenses.py           # dependency license audit
```

New file I/O code must include model validation (Pydantic field validators, e.g. `Frame` ordering / `confidence` 0–1) and path sanitization (reject `..` / absolute escapes, constrain to `get_data_dir()` / `get_models_dir()`); deserialization of checkpoints must use `torch.load(..., weights_only=True)`.

Research modules (`emg_core/foundation/tinymyo`, `aemg_tokenizer`, `spectre`, `emg_core/adaptation/sal_lbn`, `cpep`, `variance_transfer`, `emg_core/ml/spd_gru`, `adaptor`, `mona`, `lisa`, `speechnet`, `emg_core/dsp/handcrafted`, `spd`, `hardware/datasets` Gaddy/MetaEMG, `emg_core/benchmarks/emgbench`) are optional under `subvocal[ml]` and require `torch` — gate them with lazy guarded imports raising `MissingDependencyError` naming the extra (`pip install "subvocal[ml]"`), keep numpy/scipy fallbacks where feasible (handcrafted, SPD `eigh`, AEMG VQ, SPECTRE STFT/K-means, SAL/LBN numpy fallback, CPEP kNN), and include the paper citation in the module docstring (e.g. TinyMyo arXiv:2512.15729, AEMG CVPR 2026, SPECTRE 2512.22481, SAL 2409.08058, CPEP 2509.04699, Variance Transfer EMBC 2024/2505.15381, MONA 2403.05583, SilentWear 2026, Gaddy Zenodo 4064409, Meta emg2pose 2412.02725, EMGBench 2410.23625).

Generated artifacts must be in sync with their sources — CI fails if these
produce a diff:

```bash
python tools/build_api_page.py   # docs/api.html from docstrings
python tools/build_site.py       # docs/platform/*.html + walkthrough from docs/content/
```

## Conventions

- **Errors**: raise types from `subvocal.exceptions`. New error categories
  subclass `SubvocalError` and, where they replace a builtin (e.g.
  `RuntimeError`), inherit it too so existing handlers keep working.
- **Logging**: library code uses `logging.getLogger(__name__)`; `print()` is
  allowed only in `__main__` CLI blocks, benchmarks, and tests.
- **Paths**: never write inside the package tree. Use
  `subvocal.paths.get_data_dir()` / `get_models_dir()`.
- **Public API**: anything re-exported from `subvocal/__init__.py:40` `__all__` is
  frozen on `v2.0.1` and covered by semantic versioning. Breaking changes require
  a major version bump and a `CHANGELOG.md` entry — add new drivers/examples
  under `src/subvocal/hardware/` or `examples/` instead of renaming top-level exports.
- **Tests**: new behavior ships with tests in `tests/`. Tests must run offline
  with no API keys; network and heavy-model paths are mocked or skipped.
- **Docs**: public-facing writing lives in `docs/content/` (markdown) and is
  rendered into the site; code documentation lives in docstrings and flows
  into the API reference automatically.

## Commit and release flow

- Conventional-style commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`,
  `ci:`, `chore:`); `!` marks breaking changes.
- Releases follow [SemVer](https://semver.org). The single source of version
  truth is `subvocal.__version__`; tagging `v<version>` triggers the release
  workflow, which builds, verifies, and publishes to PyPI.

## Reporting issues

Use GitHub Issues for bugs and feature requests. For anything
security-sensitive, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.
