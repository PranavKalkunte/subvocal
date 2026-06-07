# 5-zone electrode layout

**Status:** Completed  
**Depends on:** [01-neckband-vs-earbud.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/architecture/01-neckband-vs-earbud.md), literature synthesis, HD mapping study (Zhu/Wang 2020)  
**Feeds into:** [06-architecture-piece.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/architecture/06-architecture-piece.md) §3, CAD (task 9), PCB (task 10)

---

## The 5-zone system

The layout divides the anterior neck and submental region into five functional zones, each targeting a distinct muscle group. Zone boundaries are not rigid lines — they are capture regions that tolerate ±5–10 mm repositioning. This spatial tolerance is a critical hardware design requirement to mitigate the inter-session variability problem documented by SilentWear 2026, where micro-shifts in dry electrode placement cause leave-one-session-out classification accuracy to drop significantly.

---

## Zone 1: Suprahyoid / digastric anterior belly

* **Target muscles:** digastric (anterior belly), geniohyoid, mylohyoid  
* **Location:** inferior surface of the mandible, midline and paramedian, above the hyoid  
* **Surface marker:** palpable mandible body → hyoid bone

### Details:
* **Exact Electrode Position:** Electrodes sit submentally, positioned 10–15 mm lateral to the midline on each side, superior to the hyoid bone, and just below the inferior margin of the mandible.
* **Activating Motions:** This zone is primarily activated by jaw depression (opening the mouth) and elevation of the floor of the mouth during tongue body movement (specifically during high vowels like /i/ and /e/) and the initiation of swallowing.
* **Expected Signal Amplitude:** High amplitude range (typically 50–150 µV during active movement). The submental skin is thin, and the anterior belly of the digastric lies superficially, resulting in excellent SNR relative to lower neck zones.
* **Literature Reference:** Jorgensen 2004 placed a "sublingual" electrode pair under the chin to capture these floor-of-mouth potentials, which was essential for decoding digits and navigation commands (86.7% accuracy).
* **Collar vs. Bridge Geometry:** A standard straight collar sits below the hyoid bone and cannot reach Zone 1 because of the submental angle. Capturing this zone requires a dedicated, curved submental chin bridge rising 40–60 mm from the collar's superior margin. Neckbands without a chin bridge (such as Wu 2024 and SilentWear 2026) omit direct Zone 1 coverage, relying instead on indirect volume conduction at the superior edge of the collar or omitting these signals entirely in favor of anterior neck signals.

---

## Zone 2: Mylohyoid

* **Target muscles:** mylohyoid  
* **Location:** floor of mouth, forms a muscular diaphragm from mandible to hyoid  
* **Surface marker:** midline, between chin tip and hyoid

### Details:
* **Bipolar Electrode Orientation:** A transverse bipolar orientation (perpendicular to the midline, parallel to the hyoid bone) is recommended to align with the transverse path of the mylohyoid muscle fibers, which run from the mylohyoid line of the mandible to the midline raphe.
* **Activating Motions:** Active during tongue body elevation, swallowing, and bilabial consonant closures (e.g., stops like /p/ and /b/).
* **Overlap with Zone 1:** While both Zones 1 and 2 sit in the submental region, Zone 1 targets the lateral, rope-like anterior digastric bellies, whereas Zone 2 targets the broad, midline mylohyoid muscular sheet. Separating them is accomplished by placing Zone 2 electrodes directly along the midline raphe, while Zone 1 electrodes are placed 10–15 mm laterally.
* **Literature Reference:** The Sequential Forward Selection (SFS) algorithm in the Zhu/Wang 2020 HD mapping study identified a submental channel (placed over the mylohyoid/digastric region) as one of the top 4 highest-yield channels for decoding speech, confirming its high discriminative weight.

---

## Zone 3: Thyrohyoid

* **Target muscles:** thyrohyoid (connects thyroid cartilage to hyoid)  
* **Location:** anterior neck, between the hyoid and thyroid cartilage  
* **Surface marker:** the small gap between the inferior edge of the hyoid and the superior edge of the thyroid cartilage

### Details:
* **Electrode Spacing Constraints:** The thyrohyoid is a short muscle (typically 15–20 mm long in adults). To avoid capturing cross-talk from the overlapping sternohyoid and omohyoid strap muscles, the inter-electrode distance (IED) must be constrained to 10–15 mm maximum.
* **Activating Motions:** Fires during larynx elevation, which is a major physiological marker during vowel production, swallowing, and voiced consonant articulation.
* **Voiced/Unvoiced Distinction:** This zone is critical for separating voiced vs. unvoiced consonants (such as /d/ vs. /t/ or /g/ vs. /k/), as voicing involves distinct laryngeal elevation patterns.
* **Channel Density:** Because of the low surface area in this narrow anatomical gap, a single bilateral differential pair (4 contacts total) provides optimal coverage.
* **Literature Reference:** Kapur 2018 (AlterEgo) documented that laryngeal muscles exhibit subtle innervation and micro-contractions during purely covert/silent verbalization (internal cognitive speech), making Zone 3 a high-value source of phonological intent. SilentWear 2026's 14-channel circumferential array covered this superior laryngeal zone to ensure reliable silent speech decoding.

---

## Zone 4: Larynx flanks

* **Target muscles:** cricothyroid (anterior), thyroarytenoid (indirectly via surface), sternothyroid  
* **Location:** lateral surfaces of the thyroid cartilage, bilateral  
* **Surface marker:** palpate the thyroid notch; electrodes sit lateral to the midline ridge, spanning the thyroid lamina

### Details:
* **Voiced/Voiceless and Pitch Control:** The cricothyroid is the primary muscle responsible for tilting the thyroid cartilage forward, which tenses the vocal folds. This makes Zone 4 the primary site for tracking voicing onset time (VOT) and pitch-related laryngeal dynamics.
* **Sternothyroid Tracking:** The sternothyroid runs from the thyroid cartilage to the sternum, stabilizing the larynx, and its activity is captured on the lateral margins of this zone.
* **Literature Reference:** Jorgensen's 2004 NASA system placed a differential pair "near the larynx" on the anterior throat, which corresponds directly to Zone 4. Wu et al. 2024's neckband lateral panels targeted this region to capture laryngeal activity, contributing to their 92.7% overt speech classification accuracy.
* **Expected Signal Amplitude & Covert Attenuation:** During voiced speech, this zone yields high-amplitude signals. However, in purely covert/silent speech (where glottal vibration is absent), signals in Zone 4 undergo severe attenuation. This is a primary driver of the accuracy gap documented by SilentWear 2026 (where within-session accuracy dropped from 84.8% vocalized to 77.5% silent).
* **BOM Optimization:** A single bilateral pair (4 contacts) provides sufficient lateral coverage.

---

## Zone 5: Infrahyoid (strap muscles)

* **Target muscles:** sternohyoid, sternothyroid, omohyoid  
* **Location:** anterior neck below the hyoid, above the clavicle  
* **Surface marker:** midline below the hyoid, bilaterally flanking the trachea

### Details:
* **Signal Quality & Stability:** The infrahyoid strap muscles are long, broad, and superficial, making Zone 5 the most reliable and highest-SNR sEMG recording site on the neck. It is highly resistant to minor electrode displacement.
* **Literature Convergence:** Every major silent speech study incorporating neck channels (Jorgensen 2004, Meltzner 2018, Wu 2024, SilentWear 2026) utilizes this zone. Meltzner's "ventromedial neck" electrodes correspond to Zone 5, and SilentWear 2026 explicitly credits infrahyoid coverage for enabling stable dry-electrode classification.
* **Prosody and Timing:** In addition to phoneme identity, Zone 5 captures long-envelope muscle activations associated with speech pacing, phrasing, and prosodic emphasis.
* **Proposed Density:** Two bilateral differential pairs (8 contacts total) to capture spatial gradients across the broad sternohyoid and omohyoid bellies.

---

## Full zone summary table

| Zone | Primary muscle(s) | Band position | Bilateral? | Electrodes (proposed) |
|------|-------------------|--------------|-----------|----------------------|
| 1: Suprahyoid/digastric | Digastric anterior, geniohyoid | Submental (requires chin extension or superior collar margin) | Yes | 1 pair per side = 4 contacts |
| 2: Mylohyoid | Mylohyoid | Submental midline | Midline | 1 pair = 2 contacts |
| 3: Thyrohyoid | Thyrohyoid | Between hyoid and thyroid cartilage | Yes | 1 pair per side = 4 contacts |
| 4: Larynx flanks | Cricothyroid, sternothyroid | Lateral thyroid lamina | Yes | 1 pair per side = 4 contacts |
| 5: Infrahyoid | Sternohyoid, sternothyroid | Anterior neck below hyoid | Yes | 2 pairs per side = 8 contacts |

* **Proposed total (base neckband, no chin extension):** Zones 3–5 = 16 contacts minimum (8 differential channels)  
* **With chin extension (zones 1–2 added):** +6 contacts (3 differential channels) = 22 contacts total (11 differential channels)

### Reconciling Proposed Count with Literature Constraints:
1. **Wu et al. 2024 (10 dry electrodes):** Wu achieved 92.7% accuracy on 11 words using only 10 dry contacts. This was accomplished by using a **single-ended (monopolar) common-reference configuration**, where 9 active contacts positioned around the collar recorded potentials relative to a single reference electrode placed over the electrically inactive cervical vertebrae (C7) on the back of the neck.
2. **SilentWear 2026 (14 differential channels):** SilentWear utilized 14 fully differential channels (requiring 28 active contacts) circumferentially distributed around the neckband.
3. **BOM & Complexity Resolution:**
   * Our proposed 22-contact layout is less dense than SilentWear’s 28 contacts, confirming it is mechanically viable for a standard neckband surface area.
   * To optimize the BOM and reduce ADC channel count for the Phase 0 firmware and ESP32 hardware (which has limited ADC inputs), we can implement a **common-reference (monopolar) routing** or a **shared-reference differential routing** (where adjacent zones share a common reference contact). For example, by using a common-reference configuration on the base neckband, we can reduce the 16 contacts of Zones 3–5 down to **9 active contacts + 1 reference/ground contact (10 contacts total)**. This directly aligns with the Wu 2024 configuration, lowering BOM cost and computational overhead while maintaining high spatial resolution for consumer assistant vocabularies.

---

## Inter-session variability implications

The 5-zone electrode layout is specifically designed to address the inter-session variability (repositioning) problem:

1. **Anatomical Anchoring:** Rather than placing electrodes on a rigid grid with arbitrary coordinates, the layout uses palpable physiological landmarks (mandible, hyoid bone, thyroid notch, and trachea).
2. **Repeatable Self-Docking:** By incorporating mechanical contours in the neckband enclosure (such as a flexible, molded thyroid-notch index or spring-loaded wings, cross-referenced to CAD task 9), the user can repeatably align the neckband to their own anatomy.
3. **Software Compensation:** In addition to physical anchoring, the spatial overlapping of the zones provides the classifier with stable, redundant representations of muscle activity, ensuring that micro-shifts of ±5–10 mm do not cause complete features mismatch.

---

## Open questions

* [x] **Verify which of the SFS-selected 4 and 10-channel subsets in Zhu/Wang 2020 correspond to which zones:**
  * The **4-channel optimal subset** identified by SFS consists of:
    1. A submental channel (Zone 2 mylohyoid/Zone 1 digastric) to capture jaw depression and tongue floor activation.
    2. A laryngeal channel (Zone 3 thyrohyoid/Zone 4 cricothyroid) to capture voicing and larynx elevation.
    3. An anterior midline strap channel (Zone 5 sternohyoid) to capture pitch and lower articulation.
    4. A facial channel (masseter/cheek) to capture jaw closing.
  * The **10-channel subset** expands this by duplicating these positions bilaterally (providing left/right symmetric channels for the submental, laryngeal, and infrahyoid zones) and adding channels for the buccinator/orbicularis oris.
* [x] **Confirm whether SilentWear 2026 published an electrode position map or just "14 channels around the neckband":** SilentWear did not publish a specific asymmetric anatomical map; instead, they deployed a circumferential, evenly spaced array of 14 dry textile differential electrode pairs around the elastic neckband collar, relying on the neural network to adapt to the resulting multi-channel feature space.
* [x] **Determine minimum inter-electrode distance needed to avoid cross-talk at 20–450 Hz band:** Electromyography standards (SENIAM) and Delsys guidelines specify that surface EMG contacts require an inter-electrode distance (IED) of **10 to 20 mm** (center-to-center). Spacing below 10 mm causes electrical shunting (signal cancellation across the skin), while spacing above 20 mm increases the pick-up volume, resulting in severe cross-talk from adjacent non-target muscles (such as the SCM).
* [x] **For a neckband circumference of 340–380 mm (average adult neck), how many electrodes fit at 25 mm spacing? Does this match the proposed count?**
  * Spacing calculation: $340 \text{ mm} / 25 \text{ mm} = 13.6$ electrodes; $380 \text{ mm} / 25 \text{ mm} = 15.2$ electrodes.
  * At a 25 mm pitch, the collar can accommodate **13 to 15 electrodes** in a single row. This matches the SilentWear 2026 count of 14 channels and Wu 2024 count of 10 channels. Our base neckband layout (16 contacts) can easily be accommodated if we arrange them as 8 differential pairs (where the two contacts in each pair have a 12 mm IED, and the pairs are spaced at a 25–30 mm pitch around the anterior/lateral neck).
