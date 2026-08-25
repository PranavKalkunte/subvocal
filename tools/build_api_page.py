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
  <link rel="stylesheet" href="./index.css">
  <style>
    .api-module { margin-bottom: 3rem; }
    .api-module h2 { font-family: var(--font-mono); }
    .api-member { margin: 1.25rem 0; }
    .api-member h4 { font-family: var(--font-mono); margin-bottom: 0.25rem; }
    .docs-sidebar ul { margin-bottom: 1rem; }
  </style>
</head>
<body>
  <main data-page="subvocal">
    <div data-component="container">
      <section data-component="top">
        <div>
          <a href="./index.html" style="text-decoration:none; font-size:1.15rem; font-weight:700; color:#0a0a0a; letter-spacing:-0.02em; font-family: Inter, sans-serif;">subvocal</a>
          <span style="margin-left:8px; font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; vertical-align:middle;">Silent Productivity</span>
        </div>
        <nav data-component="nav-desktop" style="flex:1; justify-content:center; margin:0 1rem;">
          <ul style="gap:20px;">
            <li><a href="./apex.html">SPX Benchmarks</a></li>
            <li><a href="./platform/index.html">Platform</a></li>
            <li><a href="./docs.html">Docs</a></li>
            <li><a href="./api.html" aria-current="page">API</a></li>
          </ul>
        </nav>
        <nav data-component="nav-desktop" style="flex-shrink:0;">
          <ul style="gap:8px;">
            <li><a href="https://github.com/PranavKalkunte/subvocal" target="_blank" style="border:1px solid #e5e7eb; padding:6px 12px; border-radius:9999px; font-size:13px; white-space:nowrap;">GitHub</a></li>
            <li><a href="./api.html" style="background:#0a0a0a; color:#fff; padding:7px 14px; border-radius:9999px; font-size:13px; font-weight:600; white-space:nowrap;">Start building →</a></li>
          </ul>
        </nav>
      </section>
      <div class="docs-layout">
"""

PAGE_BOTTOM = """      </div>
      <footer data-component="footer" style="background:#fff; border-top:1px solid #e5e7eb; display:grid; grid-template-columns:1.2fr 0.8fr 0.8fr 0.8fr; gap:0; padding:0;">
        <div style="padding:24px; border-right:1px solid #f3f4f6;">
          <div style="font-weight:700; color:#0a0a0a; margin-bottom:8px; font-size:14px;">subvocal</div>
          <div style="font-size:13px; color:#6b7280; line-height:1.5;">Hardware-agnostic middleware for silent speech. MIT licensed.</div>
        </div>
        <div style="padding:24px; border-right:1px solid #f3f4f6;">
          <div style="font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;">Research</div>
          <div style="display:flex; flex-direction:column; gap:8px; font-size:13px;"><a href="./apex.html" style="text-decoration:none; color:#6b7280;">SPX Benchmarks</a><a href="./platform/intent-benchmark.html" style="text-decoration:none; color:#6b7280;">Intent Benchmark</a><a href="./platform/benchmark-report.html" style="text-decoration:none; color:#6b7280;">Benchmark Report</a></div>
        </div>
        <div style="padding:24px; border-right:1px solid #f3f4f6;">
          <div style="font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;">Platform</div>
          <div style="display:flex; flex-direction:column; gap:8px; font-size:13px;"><a href="./platform/architecture.html" style="text-decoration:none; color:#6b7280;">Architecture</a><a href="./platform/hardware-drivers.html" style="text-decoration:none; color:#6b7280;">Hardware</a><a href="./api.html" style="text-decoration:none; color:#6b7280;">API</a></div>
        </div>
        <div style="padding:24px;">
          <div style="font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;">Company</div>
          <div style="display:flex; flex-direction:column; gap:8px; font-size:13px;"><a href="https://github.com/PranavKalkunte/subvocal" target="_blank" style="text-decoration:none; color:#6b7280;">GitHub</a><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank" style="text-decoration:none; color:#6b7280;">MIT License</a></div>
        </div>
      </footer>
    </div>
    <div data-component="legal">
      <span>©2026 <a href="https://github.com/PranavKalkunte">Pranav Kalkunte</a> · San Francisco, CA</span>
      <span><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank">MIT License</a> · <a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a> · English</span>
    </div>
  </main>
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

    sidebar = ['<aside class="docs-sidebar">', "<h4>API Reference</h4>"]
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
        sidebar.append(f'<li><a href="#{module_anchor(p)}"><span>[*]</span> {esc(leaf)}</a></li>')
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
