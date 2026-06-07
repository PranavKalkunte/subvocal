# 5-zone electrode layout

**Status:** stub — fill in during architecture task 2  
**Depends on:** 01-neckband-vs-earbud.md, literature synthesis, HD mapping study (Zhu/Wang 2020)  
**Feeds into:** 06-architecture-piece.md §3, CAD (task 9), PCB (task 10)

---

## The 5-zone system

The layout divides the anterior neck into five functional zones, each targeting a distinct muscle group. Zone boundaries are not rigid lines — they are capture regions that tolerate ±5–10 mm repositioning. This matters for the inter-session variability problem documented by SilentWear 2026.

---

## Zone 1: Suprahyoid / digastric anterior belly

**Target muscles:** digastric (anterior belly), geniohyoid, mylohyoid  
**Location:** inferior surface of the mandible, midline and paramedian, above the hyoid  
**Surface marker:** palpable mandible body → hyoid bone

*Fill in:*
- Exact electrode position relative to the hyoid and mandible (distance in mm)
- Which motions activate this zone (jaw depression, tongue floor elevation during vowels)
- Expected signal amplitude range relative to other zones
- Jorgensen 2004's "sublingual" electrode pair targeted this region — cite and note it was two electrodes, bilateral
- Note: this zone partially overlaps with the chin-extension decision (see 03-chin-extension.md); in the base neckband without chin extension, this zone may be undersampled

**Open question:** Can a neckband collar reach zone 1 without a chin bridge, or does the geometry force the electrodes onto the neck below the hyoid? Document which papers successfully captured this zone from a standard collar vs. required a submental extension.

---

## Zone 2: Mylohyoid

**Target muscles:** mylohyoid  
**Location:** floor of mouth, forms a muscular diaphragm from mandible to hyoid  
**Surface marker:** midline, between chin tip and hyoid

*Fill in:*
- Mylohyoid is a broad, flat sheet — bipolar differential electrode oriented transversely or longitudinally?
- When does it activate? During tongue body elevation, swallowing, bilabial stops
- Overlap with zone 1: both zones are submental; separate them by noting that zone 1 targets the anterior digastric and geniohyoid (deeper, more lateral), zone 2 targets mylohyoid (more superficial, midline)
- Literature reference: the HD mapping study likely captured this; check whether the SFS-selected 4-electrode minimal set includes a submental channel

**Open question:** Is there a published bipolar orientation recommendation for mylohyoid in speech tasks specifically (not just swallowing)?

---

## Zone 3: Thyrohyoid

**Target muscles:** thyrohyoid (connects thyroid cartilage to hyoid)  
**Location:** anterior neck, between the hyoid and thyroid cartilage  
**Surface marker:** the small gap between the inferior edge of the hyoid and the superior edge of the thyroid cartilage

*Fill in:*
- Thyrohyoid is a short muscle — electrode inter-contact spacing should be 10–15 mm maximum; note how this constrains pad geometry
- Activates during larynx elevation (critical for vowels, swallowing, voiced consonants)
- Particularly important for distinguishing voiced vs. unvoiced consonants in the subvocal domain
- This zone has low surface area; a single differential pair per side (bilateral) is likely the right density
- Note: SilentWear used 14 channels across the full neckband — infer whether this zone received dedicated coverage

**Open question:** Does thyrohyoid fire during purely silent/covert speech, or only during subvocally mouthed speech? If covert, this is a high-value zone. Check Kapur 2018 discussion of laryngeal involvement in internal verbalization.

---

## Zone 4: Larynx flanks

**Target muscles:** cricothyroid (anterior), thyroarytenoid (indirectly via surface), sternothyroid  
**Location:** lateral surfaces of the thyroid cartilage, bilateral  
**Surface marker:** palpate the thyroid notch; electrodes sit lateral to the midline ridge, spanning the thyroid lamina

*Fill in:*
- This is the primary zone for voiced/voiceless distinction and pitch-related laryngeal activity
- Cricothyroid is accessible surface-laterally; it is the main pitch-control muscle
- Sternothyroid runs from thyroid cartilage inferiorly to the sternum — it is a strap muscle visible in this zone
- NASA Jorgensen 2004 placed electrodes "near the larynx" on the anterior throat — this corresponds to zone 4
- Wu 2024's neckband electrodes wrap around the neck; the lateral panels cover this zone
- Expected to have the highest amplitude signal during consonant transitions that involve glottal activity
- In purely covert speech, this zone's signal is attenuated (per SilentWear's vocalized vs. silent accuracy gap); document the SNR trade-off

---

## Zone 5: Infrahyoid (strap muscles)

**Target muscles:** sternohyoid, sternothyroid, omohyoid  
**Location:** anterior neck below the hyoid, above the clavicle  
**Surface marker:** midline below the hyroid, bilaterally flanking the trachea

*Fill in:*
- The strap muscles are broad and accessible — highest-quality surface sEMG on the neck
- Literature convergence: every study that includes neck electrodes captures this zone; it's the most reliably informative anterior neck region
- HD mapping study: infrahyoid region was likely included in the SFS-selected optimal subset; verify
- Meltzner 2018 "ventromedial neck" electrodes correspond to this zone
- SilentWear 2026 explicitly credits infrahyoid coverage as one reason the neckband achieves viable accuracy
- Important for timing/prosody information, not just phoneme identity

---

## Full zone summary table

| Zone | Primary muscle(s) | Band position | Bilateral? | Electrodes (proposed) |
|------|-------------------|--------------|-----------|----------------------|
| 1: Suprahyoid/digastric | Digastric anterior, geniohyoid | Submental (requires chin extension or superior collar margin) | Yes | 1 pair per side = 4 contacts |
| 2: Mylohyoid | Mylohyoid | Submental midline | Midline | 1 pair = 2 contacts |
| 3: Thyrohyoid | Thyrohyoid | Between hyoid and thyroid cartilage | Yes | 1 pair per side = 4 contacts |
| 4: Larynx flanks | Cricothyroid, sternothyroid | Lateral thyroid lamina | Yes | 1 pair per side = 4 contacts |
| 5: Infrahyoid | Sternohyoid, sternothyroid | Anterior neck below hyoid | Yes | 2 pairs per side = 8 contacts |

**Proposed total (base neckband, no chin extension):** Zones 3–5 = 16 contacts minimum  
**With chin extension (zones 1–2 added):** +6 contacts = 22 contacts total  
**Compare:** Wu 2024 used 10 dry electrodes; SilentWear used 14 fully differential channels

*Fill in:* reconcile the proposed count with the literature. Are 10–14 electrodes enough to cover all 5 zones at the density specified above? Document the trade-off explicitly — more contacts improve classification but increase cost, impedance-matching complexity, and the inter-session repositioning problem.

---

## Inter-session variability implications

*Fill in:* explain how zone-based layout design mitigates (but does not eliminate) the SilentWear repositioning problem.

- Zones provide anatomical anchoring, not just grid coordinates
- A zone-referenced layout allows a user to re-dock the neckband to landmarks they can palpate (thyroid notch, hyoid) rather than to skin-surface positions
- Combine with: mechanical stops or a molded inner surface in the enclosure CAD (cross-reference task 9)

---

## Open questions

- [ ] Verify which of the SFS-selected 4 and 10-channel subsets in Zhu/Wang 2020 correspond to which zones
- [ ] Confirm whether SilentWear 2026 published an electrode position map or just "14 channels around the neckband"
- [ ] Determine minimum inter-electrode distance needed to avoid cross-talk at 20–450 Hz band (check Delsys application notes)
- [ ] For a neckband circumference of 340–380 mm (average adult neck), how many electrodes fit at 25 mm spacing? Does this match the proposed count?
