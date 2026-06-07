# The architecture of a practical subvocal interface

**Status:** stub — fill in during architecture tasks 6 (draft) and 7 (edit)  
**Depends on:** 01 through 05, literature synthesis (task 2)  
**This is the publishable corpus piece. Target: 2,500–3,500 words.**

---

## Outline

### §1: Opening — what this document is and is not

*Fill in (draft):*

- Not a restatement of the literature (the synthesis piece covers that)
- Not a product announcement
- What it is: a design rationale — the anatomical, engineering, and systems reasoning behind specific architecture choices

Open with the central tension: subvocal speech recognition has worked in labs since 2004. It has not become a product. This document argues that the gap is not primarily a machine learning problem. It is a form-factor, system-architecture, and framing problem.

Approximate length: 200 words

---

### §2: Form factor — why the neckband

*Fill in (draft):* condense from 01-neckband-vs-earbud.md  
Key claims to preserve:
- The speech muscles are anterior and inferior; ear-adjacent placements are anatomically wrong
- The neckband is the intersection of correct anatomy and viable wearability
- Literature convergence: Jorgensen, Meltzner, Wu, SilentWear all land on anterior neck
- AlterEgo demonstrates the cost of choosing facial placement for marginal gain

Approximate length: 500–600 words

---

### §3: Electrode layout — the 5 zones

*Fill in (draft):* condense from 02-electrode-layout.md and 03-chin-extension.md  
Key claims to preserve:
- The 5 zones map to distinct muscle groups with distinct roles in speech production
- Not all zones are always needed — the omission condition for zones 1–2 is real and decision-dependent
- The zone-anchored layout is how you address inter-session variability at the mechanical level (not just in software)
- Quantify: 10–14 electrodes, proposed zone assignment, trade-off table

Include a figure placeholder for the anatomical illustration (brand/design task 8, produced as SVG).

Approximate length: 600–700 words

---

### §4: Intent reconstruction — not a silent microphone

*Fill in (draft):* condense from 04-intent-reconstruction.md  
Key claims to preserve:
- The system is a low-bandwidth intent channel, not a speech transcription system
- This reframing changes the accuracy requirement from WER-based to intent-accuracy-based
- The LLM is not a post-hoc spell-checker; it is the primary decoder; the classifier is its noisy input
- Derive the warehouse-wedge accuracy target from the Vocollect benchmark

This is probably the most original section in the corpus. Write it without hedging.

Approximate length: 600–700 words

---

### §5: The signal path

*Fill in (draft):* condense from 05-signal-path-diagram.md  
Key claims to preserve:
- The 7-stage path and what happens at each stage
- The latency budget and the end-to-end target of <1 s
- The branch point decision: on-device vs. phone-side classifier; current choice and why
- Why bone conduction specifically

Include the Mermaid diagram from 05-signal-path-diagram.md, or the polished SVG version if produced.

Approximate length: 400–500 words

---

### §6: What this architecture cannot do (yet)

*Fill in (draft):*

Be direct about limits. The architecture piece should not oversell.

- Open vocabulary: the intent-reconstruction framing does not solve unlimited vocabulary; it raises the practical ceiling from ~25 to ~200–500 commands, not to 50,000 words
- Inter-session electrode drift: the zone-based layout mitigates but does not eliminate; SilentWear's <10 min re-calibration is the current state of the art
- Covert speech: the system requires subvocally mouthed speech (visible articulatory motion); purely internal speech (no muscle activation) is not currently decodable from sEMG — Jorgensen documented this failure explicitly
- Cross-user generalization: per-user calibration is still required; the system is not zero-shot

These are not fatal to the warehouse wedge. Document why not, briefly.

Approximate length: 300–400 words

---

### §7: The one ask

*Fill in (draft):*

Every corpus piece ends with one named technical or pilot ask (per the corpus assembly plan, task 15).

Options (pick one when writing):
- "If you have collected subvocal sEMG data on a neckband form factor and are willing to share it, get in touch."
- "If you run a warehouse operation and want to run a Phase 0 pilot of the intent-reconstruction layer (no hardware required), get in touch."
- "If you are a researcher with access to Delsys or BIOPAC hardware and want to collaborate on signal validation, get in touch."

Approximate length: 100 words

---

## Editorial notes for the draft

- Write in the same register as the synthesis piece but more argumentative — this is a design document, not a literature review
- Every quantitative claim must be traceable to a source in the synthesis or an explicit "to be measured in Phase 1"
- Do not use "paradigm shift," "revolutionary," "seamless," or similar
- The piece should be self-contained — a reader who has not read the synthesis should still understand the argument
- If a claim is contested in the literature, acknowledge it; do not paper over disagreements

---

## Edit checklist (task 7)

- [ ] Every numeric claim has a citation or a "Phase 1 TBD" tag
- [ ] The intent-reconstruction section does not conflate WER and intent accuracy
- [ ] The 5-zone section matches the CAD electrode layout (cross-check with task 9 output)
- [ ] The latency budget table is filled in (not placeholder)
- [ ] The one ask is specific and actionable
- [ ] No paragraph begins with "Furthermore," "Moreover," or "It is important to note"
- [ ] The figure placeholders are replaced with actual figures
- [ ] Run a read-aloud check on §4 (intent reconstruction) — this is the section most likely to drift into abstraction
