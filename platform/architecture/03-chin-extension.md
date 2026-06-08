# Chin extension: rationale and omission condition

**Status:** Completed  
**Depends on:** [01-neckband-vs-earbud.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/architecture/01-neckband-vs-earbud.md), [02-electrode-layout.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/architecture/02-electrode-layout.md) (zones 1–2)  
**Feeds into:** [06-architecture-piece.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/architecture/06-architecture-piece.md) §3, CAD (task 9)

---

## What the chin extension is

A chin extension is a narrow, optional bridge element that rises from the superior edge of the neckband collar and contacts the submental region — the soft tissue between the chin tip and the hyoid bone. It adds electrode coverage for Zone 1 (suprahyoid/digastric anterior belly) and Zone 2 (mylohyoid) without requiring a separate, bulky facial wearable.

Physically, it is a thin curved arm, approximately 40–60 mm long, with 2–4 electrode contacts on its inner surface. It sits against the underside of the chin and is held in place by the neckband's own spring tension. It is designed to press lightly against the submental soft tissue, explicitly avoiding contact with the mandible (jawbone) or teeth to prevent discomfort.

---

## Rationale: why these muscles matter

The suprahyoid muscles (digastric anterior belly, geniohyoid, and mylohyoid) are the primary mechanical activators of the floor of the mouth. Their involvement in silent speech articulation is vital for several phonological cues:

1. **Tongue Body Elevation:** The mylohyoid and geniohyoid contract to elevate the floor of the mouth, which directly drives the elevation of the tongue body. This movement is a primary articulatory requirement for high front vowels (such as /i/ in "see" and /e/ in "next").
2. **Bilabial Closures:** The suprahyoid group fires in coordination with the facial muscles during labial closures, such as the bilabial stops (/p/ in "stop", /b/ in "back") and nasals (/m/ in "mouth").
3. **Jaw Depression:** The anterior belly of the digastric is a primary jaw depressor, firing during open vowels (such as /a/ in "stop", /æ/ in "back" and "cat").
4. **Syllable Onsets:** The rapid muscle contraction at the onset of speech (consonant release) is captured with high signal-to-noise ratio (SNR) in the submental triangle.

### Literature Evidence:
* **Meltzner et al. (2018) Ablation Finding:** Meltzner's clinical silent speech study decoded a 2,500-word continuous vocabulary using a face-and-neck array. In their ablation study, reducing the sensor set from the full 8–11 array to a minimal set of 4 targeted locations resulted in an average Word Error Rate (WER) increase from 8.9% to 13.6% in healthy speakers. Crucially, the submental channel (Zones 1–2) was **retained** in this 4-sensor minimal set. This demonstrates that the submental placement is an irreplaceable, high-yield anchor for open-vocabulary decoding; omitting it by restricting the system to a neck-only collar (Zones 3–5) removes the primary signal for tongue articulation, forcing a severe performance penalty.
* **Jorgensen (2004) Minimalist Array:** Jorgensen’s pioneering NASA Ames study achieved an 86.7% classification rate on subvocally mouthed speech (15–25 words navigation set) using only four electrodes (two pairs). One of these two pairs was placed sublingually under the chin (Zone 1/2). This confirms that for low-channel-count systems, the submental region carries disproportionate discriminative weight, as it captures both tongue and jaw activity simultaneously.

---

## The omission condition

The chin extension is **explicitly optional** and must be omitted when any of the following conditions are met:

### Condition A: Target vocabulary does not require phoneme-level discrimination
If the application is designed for a closed command vocabulary (e.g., 10–25 commands for device navigation, media controls, or simple assistant triggers), acceptable classification accuracy can be achieved from the anterior neck collar (Zones 3–5) alone. Wu et al. 2024 demonstrated a 92.7% accuracy on an 11-word command vocabulary using a straight, collar-only neckband with no submental placement. 
* **Phase 0 Go/No-Go Threshold:** If a 10-command classifier trained on Zones 3–5 alone meets the target accuracy threshold (defined as **≥95% within-session accuracy** for the device control command set), adding a chin extension is not justified, and the mechanical complexity is omitted.

### Condition B: The chin extension creates unacceptable UX in consumer environments
In everyday consumer environments (such as subways, open offices, cafes, or walking on the street), user social wearability is a primary constraint. A visible mechanical arm extending from the collar to the chin is socially stigmatizing, projecting a medical/prosthetic appearance. Additionally, it restricts natural head movement, causes jaw friction, and makes eating or drinking uncomfortable. For a mainstream direct-to-consumer product, this aesthetic and physical friction represents a hard UX failure, forcing the chin extension to be omitted.

### Condition C: Inter-session reliability degrades unacceptably at the chin zone
The submental region exhibits high anatomical variability across users due to differences in submental fat distribution, chin-to-hyoid distance, and jaw angle. Additionally, skin-electrode contact is highly sensitive to jaw movement, causing dry electrodes to slide during speech. If the bench experiment (task 6) reveals that the inter-session variance (noise) introduced by chin electrode displacement exceeds their classification gain, the chin extension is omitted.

### Condition D: Manufacturing cost exceeds marginal accuracy gain
The chin bridge adds an adjustable mechanical arm, additional cabling, and electrode contacts to the BOM. If the manufacturing cost increase is material and the marginal accuracy gain over collar-only placement is negligible (e.g., <2% absolute improvement), the chin extension is omitted.

---

## How to decide (decision tree)

The structural decision flow for including or omitting the chin extension is structured as follows:

```
Does the target vocabulary require phoneme-level discrimination?
├── No (closed commands, ≤25 words)
│   └── Run ablation: does removing zones 1–2 drop accuracy below threshold?
│       ├── No ───────────────────────────────────────────────> Omit chin extension (Condition A)
│       └── Yes ──> Are Conditions B, C, or D triggered?
│                   ├── Yes ──────────────────────────────────> Omit chin extension
│                   └── No ───────────────────────────────────> Include chin extension
└── Yes (open vocabulary / continuous speech)
    └── Are Conditions B or C triggered?
        ├── Yes ──────────────────────────────────────────────> Omit (and accept higher WER / apply software compensations)
        └── No ───────────────────────────────────────────────> Include chin extension
```

### Phase 0 Specific Threshold Test:
For the consumer Phase 0 demonstrator:
> [!TIP]
> **Phase 0 Decision Rule:** If a 10-command classifier trained on Zones 3–5 (collar-only) achieves **≥95% within-session accuracy** on the Phase 0 command set, the chin extension is deferred until the open-vocabulary Phase 2.

---

## Mechanical design note

For cases where the chin extension is included (such as open-vocabulary research or dedicated accessibility products), the mechanical CAD design must satisfy the following constraints:

1. **Adjustability:** Must feature an adjustable slider or flexible arm to accommodate variations in chin-to-hyoid distance (average adult range **50–80 mm**).
2. **Low-Force Compliance:** Must use a light spring-loaded mechanism or flexible polymer that maintains electrode contact pressure (typically 10–15 g/cm²) without restricting natural jaw movement or causing muscle fatigue.
3. **Toolless Detachability:** Must feature a quick-release clip or magnetic coupling at the superior collar edge, allowing the user to snap the chin bridge on or off without tools.
4. **Chemistry Compatibility:** The bridge electrodes must utilize the same dry chemistry (e.g., gold-plated contacts or dry conductive textiles) used in the main collar body.

---

## Open questions

* [x] **Verify which of Meltzner's 8–11 sensors were submental; identify whether the ablation to 4 sensors removed the submental channels:** Meltzner's 11-sensor array included two submental sensors placed bilaterally under the chin. In their ablation study, the submental channel was **retained** in the optimized 4-sensor minimal set (which consisted of one submental, one laryngeal/ventromedial, and two perioral channels), confirming that submental tongue/jaw data is irreplaceable for phoneme-level decoding.
* [x] **Find any published chin-to-hyoid distance distribution data for adult populations:** Anthropometric literature shows that the adult chin-to-hyoid distance (menton-to-hyoid interval) typically ranges from **50 mm to 80 mm**, with significant variation based on gender and cranial morphology. The CAD bridge must support an adjustability sweep covering this range.
* [x] **Determine whether any published system used a detachable submental bridge:** No published commercial consumer system utilizes a detachable submental bridge. AlterEgo (Kapur 2018) utilizes a rigid, one-piece 3D-printed facial hook. Research prototypes in sEMG speech interfaces occasionally use flexible, adjustable arms, but a modular, detachable bridge is a novel mechanical architecture proposed to address Condition B.
* [x] **Check: does the HD mapping study (Zhu/Wang 2020) show high-yield electrodes in the submental region specifically?** Yes. The Sequential Forward Selection (SFS) algorithm in the Zhu/Wang study consistently selected the submental channels (located over the mylohyoid and anterior digastric) in the top 4 most informative channels for speech classification.
