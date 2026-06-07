# Architecture section: model recommendations and execution notes

---

## Task-by-task model recommendations

### Task 1: Neckband-vs-earbud reasoning (01-neckband-vs-earbud.md)

**Best model:** Claude Opus 4.6  
**Why:** This requires synthesizing anatomy (muscle belly locations, surface accessibility) with engineering reasoning and literature citations. The argument is subtle — you are not summarizing papers, you are drawing a conclusion the papers imply but don't state. Opus handles this kind of lateral synthesis better than Sonnet.  
**Prompt approach:** Provide the stub from this file plus the synthesis PDF as context. Ask Opus to "fill in each marked section, citing only claims you can trace to the provided synthesis or to well-established anatomy (flag anything you cannot source)." Run with extended thinking enabled if available — this task benefits from step-by-step anatomical reasoning before the prose.  
**Watch out for:** Hallucinated muscle anatomy details. Cross-check any specific mm measurements or fiber direction claims against a real anatomy atlas before publishing.

---

### Task 2: 5-zone electrode layout (02-electrode-layout.md)

**Best model:** Claude Opus 4.6  
**Why:** Same as Task 1 — requires anatomical precision. The key added challenge is reconciling the proposed zone layout with actual electrode counts from the literature (Wu 2024 used 10, SilentWear used 14). Opus can hold the cross-referencing logic across a longer context.  
**Prompt approach:** Provide the stub plus synthesis PDF. Also provide the HD mapping paper (Zhu/Wang 2020) if you obtain it — the SFS-selected subsets are the key quantitative anchor for the zone layout.  
**Watch out for:** The open questions in the stub must be answered before the section is complete. Do not let the model paper over unanswered questions with confident-sounding prose.

---

### Task 3: Chin-extension rationale (03-chin-extension.md)

**Best model:** Claude Sonnet 4.6  
**Why:** This is more mechanical than Tasks 1–2. The argument structure is clear (include if X, omit if Y), and the anatomy is simpler. Sonnet is fast enough to iterate the decision tree quickly.  
**Prompt approach:** Provide the stub. Ask for a prose version of the decision tree first, then integrate with the anatomical rationale. Cross-check the Meltzner ablation finding before writing — if his 4-channel minimum set did not include submental channels, the quantitative anchor is missing.  
**Watch out for:** The omission condition is the most important part. Do not let the model make the chin extension sound mandatory. It is explicitly optional.

---

### Task 4: Intent-reconstruction reframing (04-intent-reconstruction.md)

**Best model:** Claude Opus 4.6  
**Why:** This is the most original and argumentative section. The reframing (not a silent mic → low-bandwidth intent channel) is a conceptual move that needs to be executed without hedging. Opus is better at holding a coherent argumentative line across a long document.  
**Prompt approach:** Provide the stub. Also provide the Phase 0 task spec (task 5 in the master list) so the model can connect the reframing to what the Phase 0 demo actually does. Ask for the sections in order: "wrong framing" → "correct framing" → "accuracy requirement shift" → "design implications table."  
**Watch out for:** The model may drift toward over-claiming what the LLM can fix. The limits are real — keep §4 of 06-architecture-piece.md honest about what this architecture cannot do.

---

### Task 5: Signal-path diagram (05-signal-path-diagram.md)

**Best model:** Claude Sonnet 4.6 (for the written stage spec); then a separate call for the diagram itself  
**Why:** Filling in the stage parameters is a lookup and synthesis task — Sonnet handles this well. The diagram rendering (SVG or Mermaid) is a structured output task that does not benefit from Opus-level reasoning.  
**For the diagram:** Generate a Mermaid flowchart first (it's version-controlled and editable). When producing the polished SVG for publication, use the Figma MCP or the SVG skill. The diagram should show: stage boxes, directional arrows with latency annotations, and the dashed feedback path for the training data pipeline. The brand/design task (Part 2, task 8) produces the anatomical illustration separately.  
**Watch out for:** The latency numbers in the diagram will be placeholders until Phase 0 measurement. Mark them clearly as "target" not "measured."

---

### Task 6: Draft the architecture piece (06-architecture-piece.md)

**Best model:** Claude Opus 4.6  
**Why:** Drafting a 2,500–3,500-word argumentative corpus piece that synthesizes five sub-documents requires sustained coherence and rhetorical control. Opus.  
**Prompt approach:**  
1. Provide all five completed sub-documents (01–05) as context.  
2. Provide the 06 stub as the outline.  
3. Ask Opus to draft §§1–7 in order, writing each section fully before moving to the next.  
4. Do not ask for the whole piece in one pass if the sub-documents total >20,000 words of context — section by section is more reliable.  
**Watch out for:** Section 4 (intent reconstruction) is the most important and most likely to drift into vagueness. If the draft of §4 uses phrases like "seamlessly," "intelligently infers," or "understands context," rewrite it.

---

### Task 7: Edit the architecture piece

**Best model:** Claude Sonnet 4.6  
**Why:** Editing is a tighter, faster task. Use Sonnet to run through the edit checklist in 06-architecture-piece.md. For the final read-aloud check, use your own judgment.  
**Prompt approach:** Provide the draft and the edit checklist. Ask Sonnet to go line by line through the checklist and return a diff of proposed changes. Do not accept changes that alter the argument — only accept changes that fix citation gaps, redundant phrasing, or hedging.  
**Watch out for:** Sonnet may over-smooth the prose. The architecture piece should sound argued, not polished. Preserve sentences that make strong claims.

---

## Dependency order (which to do first)

```
Literature synthesis (task 2) must be substantially complete first.

Then, in parallel:
  01-neckband-vs-earbud.md
  02-electrode-layout.md
  03-chin-extension.md    ← blocked by 02

Then:
  04-intent-reconstruction.md    ← can run in parallel with 01–03

Then:
  05-signal-path-diagram.md    ← blocked by 02 (electrode count) and 04 (LLM stage)

Then:
  06-architecture-piece.md draft    ← blocked by all five
  06-architecture-piece.md edit    ← blocked by draft
```

---

## Other notes

**Context to always provide:** When running any of these tasks, always attach the synthesis PDF. It is the primary source. Any claim in the sub-documents that cannot be traced to the synthesis PDF should be flagged as "verify against primary source."

**Calibration numbers:** Several stubs say "fill in from bench experiment." Do not fill those in from memory or estimation — leave them as explicit placeholders until Phase 1 data exists. The architecture piece can publish with placeholders marked "Phase 1 TBD."

**Corpus publication standard:** The architecture piece is public. Every number in it must be either (a) cited to a published source, (b) marked "measured in Phase 1" with a link to the data, or (c) explicitly labeled "target" or "estimated." Do not publish unsourced quantitative claims.

**Format of the final piece:** Markdown for the working version. When converting to the corpus publication format (Substack, Framer, or GitHub Pages — decision pending per Part 2, task 1), use the docx or pdf skill for any offline distribution, or export directly to HTML. The Mermaid diagram exports to SVG cleanly.

**Versioning:** All architecture documents should be committed to a Git repository with timestamps (per IP defense plan, Part 2, task 19). The architecture piece in particular should be committed before any public release, with the commit serving as prior-art documentation.
