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
ruff check src tests benchmarks tools   # lint (E, F, I, UP, B; E501/E741 ignored by policy)
pyright                                  # type check (0 errors required)
pytest --cov=subvocal                    # test suite (coverage floor enforced in CI)
python tools/check_licenses.py           # dependency license audit
```

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
- **Public API**: anything re-exported from `subvocal/__init__.py` is covered
  by semantic versioning. Breaking changes require a major version bump and a
  CHANGELOG entry.
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
