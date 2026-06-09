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
TARGET_PACKAGES = ["core", "context", "hardware", "emg_core", "mcp", "shorthand", "tts"]
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
          <a href="./index.html" style="text-decoration: none; font-size: 1.25rem; font-weight: 700; color: var(--color-text-strong); font-family: var(--font-mono); letter-spacing: -0.03em;">subvocal</a>
        </div>
        <nav data-component="nav-desktop">
          <ul>
            <li><a href="https://github.com/PranavKalkunte/subvocal" target="_blank" style="white-space: nowrap;">GitHub</a></li>
            <li><a href="./docs.html">Docs</a></li>
            <li><a href="./api.html" aria-current="page">API</a></li>
          </ul>
        </nav>
      </section>
      <div class="docs-layout">
"""

PAGE_BOTTOM = """      </div>
      <footer data-component="footer">
        <div data-slot="cell"><a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a></div>
        <div data-slot="cell"><a href="./docs.html">Docs</a></div>
        <div data-slot="cell"><a href="https://github.com/PranavKalkunte/subvocal/commits/main" target="_blank">Changelog</a></div>
      </footer>
    </div>
    <div data-component="legal">
      <span>©2026 <a href="https://github.com/PranavKalkunte">Pranav Kalkunte</a></span>
      <span><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank">MIT License</a></span>
      <span>English</span>
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


def collect_modules() -> list[ModuleParser]:
    parsers = []
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
