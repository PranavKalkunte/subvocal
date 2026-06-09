#!/usr/bin/env python3
"""Renders the markdown corpus in docs/content/ into styled GitHub Pages HTML.

Outputs docs/platform/<slug>.html (one page per platform document, plus an
index) and docs/walkthrough.html. Pages share the site stylesheet and chrome
used by index.html / docs.html. Mermaid code fences render via the Mermaid CDN.

Run from anywhere:

    python tools/build_site.py

Requires the 'markdown' package (installed with the [dev] extra).
"""

import os
import re

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONTENT_DIR = os.path.join(ROOT, "docs", "content")
PLATFORM_DIR = os.path.join(ROOT, "docs", "platform")

# (source markdown, output slug, page title, one-line description)
PLATFORM_PAGES = [
    ("positioning.md", "positioning", "Platform Positioning",
     "The middleware thesis: why software rails, not proprietary hardware."),
    ("declaration.md", "declaration", "Declaration Post",
     "The framing shift and what is shipping."),
    ("architecture.md", "architecture", "Architecture",
     "Anatomical, engineering, and systems-level design of the platform."),
    ("intent-layer.md", "intent-layer", "Intent Layer",
     "Shorthand-to-intent reconstruction: providers, prompts, corrections."),
    ("intent-benchmark.md", "intent-benchmark", "Intent Reconstruction Benchmark",
     "The open 50-case shorthand-to-intent evaluation specification."),
    ("benchmark-report.md", "benchmark-report", "Benchmark Report",
     "Latest generated benchmark results across providers and models."),
    ("hardware-drivers.md", "hardware-drivers", "Hardware Drivers",
     "The hardware abstraction layer: replay, synthetic, OpenBCI, Delsys, datasets."),
    ("classifiers.md", "classifiers", "Classifier Infrastructure",
     "Reference models, training pipeline, calibration, export, quantization."),
    ("security.md", "security", "Security & Threat Model",
     "Authorization policies, threat model, data residency, biometric compliance."),
    ("mcp-profile.md", "mcp-profile", "MCP Intent Profile (RFC)",
     "Proposed Model Context Protocol profile for low-bandwidth intent inputs."),
    ("release-process.md", "release-process", "Release Process",
     "Semantic versioning policy, cadence, and changelog format."),
    ("reference-bom.md", "reference-bom", "Reference Hardware BOM",
     "Minimum and full bill-of-materials for a DIY sEMG capture rig."),
    ("phase0-results.md", "phase0-results", "Phase 0: Method & Results",
     "The original feasibility study that seeded the SDK."),
]

WALKTHROUGH_PAGE = ("walkthrough.md", "walkthrough", "End-to-End Walkthrough",
                    "Signal generation, DSP, classifier training, and the full pipeline in code.")

MERMAID_SCRIPT = (
    '<script type="module">'
    'import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
    'mermaid.initialize({ startOnLoad: true, theme: "neutral" });'
    "</script>"
)

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} | Subvocal SDK</title>
  <link rel="stylesheet" href="{root}index.css">
  <style>
    .docs-body table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
    .docs-body th, .docs-body td {{ border: 1px solid var(--color-border, #444); padding: 0.4rem 0.7rem; text-align: left; }}
    .docs-body img {{ max-width: 100%; }}
    .docs-body h1, .docs-body h2 {{ margin-top: 2rem; }}
    .mermaid {{ background: transparent; margin: 1rem 0; }}
  </style>
</head>
<body>
  <main data-page="subvocal">
    <div data-component="container">
      <section data-component="top">
        <div>
          <a href="{root}index.html" style="text-decoration: none; font-size: 1.25rem; font-weight: 700; color: var(--color-text-strong); font-family: var(--font-mono); letter-spacing: -0.03em;">subvocal</a>
        </div>
        <nav data-component="nav-desktop">
          <ul>
            <li><a href="https://github.com/PranavKalkunte/subvocal" target="_blank" style="white-space: nowrap;">GitHub</a></li>
            <li><a href="{root}docs.html">Docs</a></li>
            <li><a href="{root}platform/index.html"{platform_current}>Platform</a></li>
            <li><a href="{root}api.html">API</a></li>
          </ul>
        </nav>
      </section>
      <div class="docs-layout">
        <aside class="docs-sidebar">
{sidebar}
        </aside>
        <article class="docs-body">
{content}
        </article>
      </div>
      <footer data-component="footer">
        <div data-slot="cell"><a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a></div>
        <div data-slot="cell"><a href="{root}docs.html">Docs</a></div>
        <div data-slot="cell"><a href="https://github.com/PranavKalkunte/subvocal/commits/main" target="_blank">Changelog</a></div>
      </footer>
    </div>
    <div data-component="legal">
      <span>©2026 <a href="https://github.com/PranavKalkunte">Pranav Kalkunte</a></span>
      <span><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank">MIT License</a></span>
      <span>English</span>
    </div>
  </main>
  {mermaid}
</body>
</html>
"""


def render_markdown(md_text: str) -> tuple[str, bool]:
    """Returns (html, uses_mermaid)."""
    html = markdown.markdown(md_text, extensions=["fenced_code", "tables", "sane_lists"])
    uses_mermaid = '<code class="language-mermaid">' in html
    if uses_mermaid:
        html = re.sub(
            r'<pre><code class="language-mermaid">(.*?)</code></pre>',
            r'<pre class="mermaid">\1</pre>',
            html,
            flags=re.DOTALL,
        )
    return html, uses_mermaid


def sidebar_html(active_slug: str, root: str) -> str:
    lines = ["          <h4>Platform Corpus</h4>", "          <ul>"]
    for _, slug, title, _ in PLATFORM_PAGES:
        current = ' aria-current="page"' if slug == active_slug else ""
        lines.append(f'            <li><a href="{root}platform/{slug}.html"{current}><span>[*]</span> {title}</a></li>')
    lines.append("          </ul>")
    lines.append("          <h4>Tutorials</h4>")
    lines.append("          <ul>")
    current = ' aria-current="page"' if active_slug == "walkthrough" else ""
    lines.append(f'            <li><a href="{root}walkthrough.html"{current}><span>[*]</span> End-to-End Walkthrough</a></li>')
    lines.append("          </ul>")
    return "\n".join(lines)


def write_page(out_path: str, title: str, content_html: str, active_slug: str,
               root: str, uses_mermaid: bool, platform_current: bool) -> None:
    page = PAGE_TEMPLATE.format(
        title=title,
        root=root,
        sidebar=sidebar_html(active_slug, root),
        content=content_html,
        mermaid=MERMAID_SCRIPT if uses_mermaid else "",
        platform_current=' aria-current="page"' if platform_current else "",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)


def build_platform_pages() -> list[str]:
    written = []
    for filename, slug, title, _ in PLATFORM_PAGES:
        src = os.path.join(CONTENT_DIR, filename)
        if not os.path.exists(src):
            print(f"[skip] missing source: {filename}")
            continue
        with open(src, encoding="utf-8") as f:
            html, uses_mermaid = render_markdown(f.read())
        out = os.path.join(PLATFORM_DIR, f"{slug}.html")
        write_page(out, title, html, slug, "../", uses_mermaid, platform_current=True)
        written.append(out)
    return written


def build_platform_index() -> str:
    cards = ["<h1>Platform Corpus</h1>",
             "<p>The complete public specification set behind the Subvocal SDK: positioning, "
             "architecture, benchmarks, security, and the MCP intent-profile proposal.</p>",
             '<div class="docs-card-grid">']
    for _, slug, title, desc in PLATFORM_PAGES:
        cards.append(
            f'<a href="./{slug}.html" class="docs-card">'
            f'<div class="docs-card-header"><span class="docs-card-icon">[*]</span>'
            f'<span class="docs-card-title">{title}</span></div>'
            f"<p>{desc}</p></a>"
        )
    cards.append("</div>")
    out = os.path.join(PLATFORM_DIR, "index.html")
    write_page(out, "Platform Corpus", "\n".join(cards), "", "../", False, platform_current=True)
    return out


def build_walkthrough() -> str:
    filename, slug, title, _ = WALKTHROUGH_PAGE
    with open(os.path.join(CONTENT_DIR, filename), encoding="utf-8") as f:
        html, uses_mermaid = render_markdown(f.read())
    out = os.path.join(ROOT, "docs", f"{slug}.html")
    write_page(out, title, html, slug, "./", uses_mermaid, platform_current=False)
    return out


def main() -> None:
    written = build_platform_pages()
    written.append(build_platform_index())
    written.append(build_walkthrough())
    for path in written:
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
