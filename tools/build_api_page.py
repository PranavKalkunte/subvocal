#!/usr/bin/env python3
"""Builds the static API reference page (docs/api.html) for the GitHub Pages site.

Reuses the AST-based ModuleParser from generate_api_docs.py and renders a single
self-contained HTML page styled like the rest of the site. Run from anywhere:

    python tools/build_api_page.py
"""

import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_api_docs import SDK_DIR, ModuleParser  # noqa: E402

OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "api.html"))
TARGET_PACKAGES = [
    "core", "context", "hardware", "emg_core", "mcp", "shorthand", "tts",
    "routing", "runtime", "stream", "auth", "telemetry", "utils",
]
SKIP_FILES = {"eval_set.py"}

PAGE_TOP = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>API Reference | Subvocal SDK</title>
  <link rel="stylesheet" href="./index.css?v=2">
  <style>
    .api-module { margin-bottom: 3rem; }
    .api-module h2 { font-family: Inter, sans-serif; font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }
    .api-member { margin: 1.25rem 0; }
    .api-member h4 { font-family: var(--mono); margin-bottom: 0.25rem; }
    .sidebar ul { margin-bottom: 1rem; }
  </style>
</head>
<body>
  <header class="site-header">
    <div class="container">
      <a class="brand" href="./index.html">
        <span class="brand-mark">subvocal</span>
        <span class="brand-tag">Silent Productivity</span>
      </a>
      <nav class="nav-center" aria-label="Primary">
        <a href="./apex.html">SPX Benchmarks</a>
        <a href="./platform/index.html">Platform</a>
        <a href="./docs.html">Docs</a>
        <a href="./api.html" aria-current="page">API</a>
      </nav>
      <div class="nav-actions">
        <a class="btn-ghost hide-mobile" href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a>
        <a class="btn-dark" href="./docs.html">Start building →</a>
        <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="mobileMenu" onclick="toggleMobile()"><span class="sr-only">Open menu</span><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button>
      </div>
    </div>
  </header>
  <div id="mobileMenu" class="mobile-menu">
    <div class="container" style="padding:12px 16px 16px">
      <a href="./apex.html">SPX Benchmarks</a>
      <a href="./platform/index.html">Platform</a>
      <a href="./docs.html">Docs</a>
      <a href="./api.html" aria-current="page">API →</a>
      <a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub →</a>
    </div>
  </div>
  <div class="docs-layout">
"""

PAGE_BOTTOM = """  </div>
  <footer class="footer">
    <div class="container">
      <div class="footer-grid">
        <div class="footer-col">
          <div style="font-weight:800;letter-spacing:-0.03em;color:#0a0a0a">subvocal</div>
          <p style="margin-top:8px">Hardware-agnostic middleware for silent speech. MIT licensed.</p>
          <div style="margin-top:12px;display:flex;gap:8px"><a class="btn-ghost" href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a><a class="btn-dark" href="./apex.html">SPX Leaderboard</a></div>
        </div>
        <div class="footer-col"><h4>Research</h4><a href="./apex.html">SPX Benchmarks</a><a href="./platform/intent-benchmark.html">Intent Benchmark</a><a href="./platform/benchmark-report.html">Benchmark Report</a></div>
        <div class="footer-col"><h4>Platform</h4><a href="./platform/architecture.html">Architecture</a><a href="./platform/hardware-drivers.html">Hardware</a><a href="./api.html">API</a></div>
        <div class="footer-col"><h4>Company</h4><a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank">MIT License</a><a href="https://github.com/PranavKalkunte/subvocal/blob/main/CHANGELOG.md" target="_blank">Changelog</a></div>
      </div>
    </div>
    <div class="legal">
      <span>©2026 <a href="https://github.com/PranavKalkunte">Pranav Kalkunte</a> · San Francisco, CA</span>
      <span><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank">MIT License</a> · <a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a> · English</span>
    </div>
  </footer>
  <script>function toggleMobile(){const m=document.getElementById('mobileMenu');const b=document.querySelector('.nav-toggle');const o=m.classList.toggle('open');b.setAttribute('aria-expanded',o?'true':'false')}</script>
</body>
</html>
"""


def esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def docstring_html(doc: str) -> str:
    if not doc:
        return ""
    paragraphs = [p.strip() for p in doc.split("\n\n") if p.strip()]
    return "".join(f"<p>{esc(p)}</p>" for p in paragraphs)


TOP_LEVEL_MODULES = ["exceptions.py", "paths.py", "config.py"]


def collect_modules() -> list[ModuleParser]:
    parsers = []
    for filename in TOP_LEVEL_MODULES:
        full_path = os.path.join(SDK_DIR, filename)
        if not os.path.exists(full_path):
            continue
        parser = ModuleParser(full_path, filename)
        parser.parse()
        if parser.classes or parser.functions:
            parsers.append(parser)
    for pkg in TARGET_PACKAGES:
        pkg_dir = os.path.join(SDK_DIR, pkg)
        if not os.path.isdir(pkg_dir):
            continue
        for root, _, files in sorted(os.walk(pkg_dir)):
            for file in sorted(files):
                if not file.endswith(".py") or file.startswith("test_") or file == "__init__.py" or file in SKIP_FILES:
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, SDK_DIR)
                parser = ModuleParser(full_path, rel_path)
                try:
                    parser.parse()
                except Exception as e:
                    print(f"Skipping {rel_path}: {e}")
                    continue
                if parser.classes or parser.functions:
                    parsers.append(parser)
    return parsers


def module_name(parser: ModuleParser) -> str:
    return "subvocal." + parser.relative_path.replace(".py", "").replace(os.sep, ".")


def module_anchor(parser: ModuleParser) -> str:
    return module_name(parser).replace(".", "-")


def render_module(parser: ModuleParser) -> str:
    name = module_name(parser)
    out = [f'<section class="api-module" id="{module_anchor(parser)}">']
    out.append(f"<h2>{esc(name)}</h2>")
    out.append(docstring_html(parser.module_doc))

    for cls in parser.classes:
        bases = f"({', '.join(cls['bases'])})" if cls["bases"] else ""
        out.append('<div class="api-member">')
        out.append(f"<h3><code>class {esc(cls['name'] + bases)}</code></h3>")
        out.append(docstring_html(cls["docstring"]))
        for method in cls["methods"]:
            out.append('<div class="api-member">')
            out.append(f"<h4>{esc(method['name'])}</h4>")
            out.append(f"<pre><code>{esc(method['signature'])}</code></pre>")
            out.append(docstring_html(method["docstring"]))
            out.append("</div>")
        out.append("</div>")

    for func in parser.functions:
        out.append('<div class="api-member">')
        out.append(f"<h3><code>{esc(func['name'])}</code></h3>")
        out.append(f"<pre><code>{esc(func['signature'])}</code></pre>")
        out.append(docstring_html(func["docstring"]))
        out.append("</div>")

    out.append("</section>")
    return "\n".join(out)


def main():
    parsers = collect_modules()

    sidebar = ['<aside class="sidebar">', "<h4>API Reference</h4>"]
    current_pkg = None
    for p in parsers:
        pkg = p.relative_path.split(os.sep)[0]
        if pkg.endswith(".py"):
            pkg = pkg[:-3]
        if pkg != current_pkg:
            if current_pkg is not None:
                sidebar.append("</ul>")
            sidebar.append(f"<h4>subvocal.{esc(pkg)}</h4>")
            sidebar.append("<ul>")
            current_pkg = pkg
        leaf = module_name(p).split(".")[-1]
        sidebar.append(f'<li><a href="#{module_anchor(p)}">{esc(leaf)}</a></li>')
    sidebar.append("</ul>")
    sidebar.append("</aside>")

    body = ['<article class="docs-body">', "<h1>API Reference</h1>",
            f"<p>Auto-generated from docstrings across {len(parsers)} modules. "
            'Regenerate with <code>python tools/build_api_page.py</code>.</p>']
    body.extend(render_module(p) for p in parsers)
    body.append("</article>")

    page = PAGE_TOP + "\n".join(sidebar) + "\n" + "\n".join(body) + "\n" + PAGE_BOTTOM
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {OUTPUT_FILE} ({len(parsers)} modules)")


if __name__ == "__main__":
    main()
