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
    ("configuration.md", "configuration", "Configuration & Runtime",
     "The v2 config tree, sessions, signal monitoring, telemetry, and auth grants."),
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
  <link rel="stylesheet" href="{root}index.css?v=2">
</head>
<body>
  <header class="site-header">
    <div class="container">
      <a class="brand" href="{root}index.html">
        <span class="brand-mark">subvocal</span>
        <span class="brand-tag">Silent Productivity</span>
      </a>
      <nav class="nav-center" aria-label="Primary">
        <a href="{root}apex.html">SPX Benchmarks</a>
        <a href="{root}platform/index.html"{platform_current}>Platform</a>
        <a href="{root}docs.html">Docs</a>
        <a href="{root}api.html">API</a>
      </nav>
      <div class="nav-actions">
        <a class="btn-ghost hide-mobile" href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a>
        <a class="btn-dark" href="{root}docs.html">Start building →</a>
        <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="mobileMenu" onclick="toggleMobile()"><span class="sr-only">Open menu</span><svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg></button>
      </div>
    </div>
  </header>
  <div id="mobileMenu" class="mobile-menu">
    <div class="container" style="padding:12px 16px 16px">
      <a href="{root}apex.html">SPX Benchmarks</a>
      <a href="{root}platform/index.html">Platform</a>
      <a href="{root}docs.html">Docs</a>
      <a href="{root}api.html">API</a>
      <a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub →</a>
    </div>
  </div>
  <div class="docs-layout">
    <aside class="sidebar">
{sidebar}
    </aside>
    <article class="docs-body">
{content}
    </article>
  </div>
  <footer class="footer">
    <div class="container">
      <div class="footer-grid">
        <div class="footer-col">
          <div style="font-weight:800;letter-spacing:-0.03em;color:#0a0a0a">subvocal</div>
          <p style="margin-top:8px">Hardware-agnostic middleware for silent speech. MIT licensed.</p>
          <div style="margin-top:12px;display:flex;gap:8px"><a class="btn-ghost" href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a><a class="btn-dark" href="{root}apex.html">SPX Leaderboard</a></div>
        </div>
        <div class="footer-col"><h4>Research</h4><a href="{root}apex.html">SPX Benchmarks</a><a href="{root}platform/intent-benchmark.html">Intent Benchmark</a><a href="{root}platform/benchmark-report.html">Benchmark Report</a></div>
        <div class="footer-col"><h4>Platform</h4><a href="{root}platform/architecture.html">Architecture</a><a href="{root}platform/hardware-drivers.html">Hardware</a><a href="{root}api.html">API</a></div>
        <div class="footer-col"><h4>Company</h4><a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank">MIT License</a><a href="https://github.com/PranavKalkunte/subvocal/blob/main/CHANGELOG.md" target="_blank">Changelog</a></div>
      </div>
    </div>
    <div class="legal">
      <span>©2026 <a href="https://github.com/PranavKalkunte">Pranav Kalkunte</a> · San Francisco, CA</span>
      <span><a href="https://github.com/PranavKalkunte/subvocal/blob/main/LICENSE" target="_blank">MIT License</a> · <a href="https://github.com/PranavKalkunte/subvocal" target="_blank">GitHub</a> · English</span>
    </div>
  </footer>
  <script>function toggleMobile(){{const m=document.getElementById('mobileMenu');const b=document.querySelector('.nav-toggle');const o=m.classList.toggle('open');b.setAttribute('aria-expanded',o?'true':'false')}}</script>
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
    lines = ["      <h4>Platform Corpus</h4>", "      <ul>"]
    for _, slug, title, _ in PLATFORM_PAGES:
        current = ' aria-current="page"' if slug == active_slug else ""
        lines.append(f'        <li><a href="{root}platform/{slug}.html"{current}>{title}</a></li>')
    lines.append("      </ul>")
    lines.append("      <h4>Tutorials</h4>")
    lines.append("      <ul>")
    current = ' aria-current="page"' if active_slug == "walkthrough" else ""
    lines.append(f'        <li><a href="{root}walkthrough.html"{current}>End-to-End Walkthrough</a></li>')
    lines.append("      </ul>")
    # SPX special links
    lines.append("      <h4>Benchmarks</h4>")
    lines.append("      <ul>")
    lines.append(f'        <li><a href="{root}apex.html">SPX Leaderboard</a></li>')
    lines.append(f'        <li><a href="{root}platform/benchmark-report.html">Report</a></li>')
    lines.append(f'        <li><a href="{root}api.html">API Reference</a></li>')
    lines.append("      </ul>")
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
    intro = [
        "<h1>Platform Corpus</h1>",
        "<p>The complete public specification set behind the Subvocal SDK: positioning, "
        "architecture, benchmarks, security, and the MCP intent-profile proposal.</p>",
        '<div style="margin:12px 0 18px;display:flex;gap:8px;flex-wrap:wrap"><span class="badge">14 docs</span><span class="badge">MIT licensed</span><a href="../apex.html" class="btn-dark" style="padding:6px 14px;font-size:13px">SPX Leaderboard →</a></div>',
        '<div class="cards">'
    ]
    cards = []
    for _, slug, title, desc in PLATFORM_PAGES:
        cards.append(
            f'<a href="./{slug}.html" class="card">'
            f'<div class="card-title">{title}</div>'
            f"<p style=\"font-size:13.5px;color:#6b7280;line-height:1.5;margin:6px 0 0\">{desc}</p></a>"
        )
    out_html = "\n".join(intro + cards + ["</div>"])
    out = os.path.join(PLATFORM_DIR, "index.html")
    write_page(out, "Platform Corpus", out_html, "", "../", False, platform_current=True)
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
