# Neckband vs. earbud: form-factor decision from muscle anatomy

**Status:** Completed  
**Depends on:** literature synthesis (§2), especially Wu 2024 and ETH SilentWear 2026  
**Feeds into:** [06-architecture-piece.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/architecture/06-architecture-piece.md) §2

---

## Argument to make

The core claim: placing electrodes around the anterior neck captures the muscles that matter for subvocal speech, while ear-adjacent or behind-ear placements do not — and this is derivable from anatomy, not just empirical preference.

---

## Section 1: Relevant muscle groups and their surface accessibility

To decode silent speech via surface electromyography (sEMG), electrodes must be placed directly over or in close proximity to the muscle bellies responsible for articulation and laryngeal control. The human speech production mechanism relies on the coordinated activation of over one hundred distinct muscles across the face, jaw, neck, and larynx. These muscles can be categorized into four primary groups based on their anatomical location and contribution to the subvocal signal:

### 1. Suprahyoid Muscle Group
The suprahyoid group comprises the **mylohyoid**, **geniohyoid**, **stylohyoid**, and the **anterior belly of the digastric**. 
* **Belly Location & Function:** These muscles form the floor of the mouth and run between the inner surface of the mandible (jawbone) and the hyoid bone. They are responsible for elevating the hyoid bone, elevating the floor of the mouth, elevating the tongue body (essential for high vowels such as /i/ and /e/), depressing the jaw (open vowels like /a/ and /æ/), and executing the initial consonant release in syllable onset.
* **Surface Accessibility:** The anterior belly of the digastric and the mylohyoid are highly superficial and directly accessible to dry surface electrodes placed in the submental triangle (the soft tissue area immediately underneath the chin/jaw, superior to the hyoid). The geniohyoid sits deep (superior) to the mylohyoid; while it cannot be accessed directly, its contractions contribute to the composite sEMG signal captured submentally. The stylohyoid sits superiorly and posteriorly, running from the styloid process of the temporal bone to the hyoid.
* **Behind-Ear Accessibility:** **No.** The submental suprahyoid muscles are located entirely anteriorly and superiorly relative to the neck. They are separated from the post-auricular (behind-ear) region by the mandible, the parotid gland, and substantial tissue depth.

### 2. Infrahyoid Muscle Group
The infrahyoid group (also known as the strap muscles) consists of the **sternohyoid**, **sternothyroid**, **thyrohyoid**, and **omohyoid**.
* **Belly Location & Function:** These muscles are located on the anterior midline of the neck, running between the hyoid bone superiorly and the sternum and clavicle inferiorly. They stabilize the larynx, depress the hyoid bone, and control vocal tract length during phonation.
* **Surface Accessibility:** The sternohyoid and thyrohyoid are highly superficial strap muscles on the anterior neck, flanking the trachea. They are directly accessible to dry surface electrodes placed along the front and lateral panels of a neckband. The sternothyroid lies deep to the sternohyoid but is still close enough to the skin surface to contribute to the anterior sEMG profile. The omohyoid (specifically its superior belly) is accessible laterally.
* **Behind-Ear/Jaw-Hook Accessibility:** **No.** These muscles are situated exclusively in the anterior neck below the hyoid bone. They cannot be recorded from an earbud or a jaw hook due to the extreme distance and intervening anatomical structures (including the sternocleidomastoid muscle and the carotid sheath).

### 3. Laryngeal Extrinsic & Intrinsic Muscles
Key muscles in this group include the **thyrohyoid** and the **cricothyroid**.
* **Belly Location & Function:** Located on the flanks of the thyroid cartilage in the anterior neck. The cricothyroid is the primary muscle responsible for tensing the vocal cords and controlling pitch.
* **Surface Accessibility:** These muscles are accessible via the lateral panels of a neckband wrapping around the anterior throat, specifically targeting the thyroid lamina.
* **Behind-Ear Accessibility:** **No.** The larynx is located anteriorly along the midline of the neck, far from the superior-lateral ear position.

### 4. Mastication and Facial Muscles
This group includes the **temporalis**, **masseter**, and **buccinator**.
* **Belly Location & Function:** These muscles sit on the cheek, lateral jaw, and temples, and drive jaw closure, cheek compression, and lip shaping.
* **Surface Accessibility:** These are accessible via a jaw-hook form factor (such as the AlterEgo headset) or cheek patches, but are completely inaccessible from a neckband collar.
* **Discriminative Value Trade-off:** While mastication muscles provide high-amplitude signals that aid in vowel shape discrimination, high-density sEMG mapping studies (Zhu/Wang 2020) utilizing Sequential Forward Selection (SFS) algorithms proved mathematically that facial placements are highly redundant. An optimized subset of 10 to 20 channels from the neck region alone is sufficient to achieve 90% to 95% classification accuracy. Furthermore, Wu et al. 2024 and SilentWear 2026 demonstrated that neck-only placements (infrahyoid, laryngeal, and sublingual/suprahyoid activity) provide sufficient signal fidelity to decode command vocabularies without requiring facial attachments.

---

## Section 2: Why an earbud placement fails

Earbud-based or behind-ear sEMG electrode configurations are frequently proposed due to the high social acceptability of consumer earwear. However, from a biophysical and anatomical perspective, this placement is fundamentally unsuited for silent speech decoding:

1. **SCM Motion Contamination:** The primary muscle group accessible behind the ear (specifically over the mastoid process) is the superior insertion of the **sternocleidomastoid (SCM)** muscle. The SCM is a massive superficial neck muscle whose primary function is head rotation and neck flexion. It is highly active during normal everyday non-speech activities (head turning, nodding, breathing, walking, and posture changes). Bipolar electrodes placed in this region will capture massive SCM contraction signals (often >100 µV). Because these signals are not functionally coupled to vocal tract articulation, they act as high-amplitude biological noise, completely drowning out the microscopic, subvocally induced sEMG signals (typically <10–20 µV).
2. **Posterior Digastric Limitations:** While the posterior belly of the digastric muscle originates from the mastoid notch, it is relatively deep, and its speech-related activation (jaw depression) is heavily masked by SCM cross-talk.
3. **Absence of Anterior Neck Signals:** An ear-adjacent placement has zero electrical access to the infrahyoid and laryngeal extrinsic muscles (sternohyoid, thyrohyoid, cricothyroid) which carry critical information about voicing, pitch, and phonological transitions. The electrical potentials generated by these anterior muscles are attenuated by the high impedance of the thyroid cartilage, the mandible, and the large distance to the ear.
4. **Mechanical TMJ Artifacts in the Canal:** Electrodes placed inside the ear canal to capture muscle activity rely on contact with the canal walls. However, jaw movement during subvocal mouthing causes mechanical deflection of the ear canal walls via the temporomandibular joint (TMJ). The resulting signals are gross motion artifacts (mechanomyographic or capacitive shifts) driven by skeletal movement, rather than the fine, neuromuscular activity that represents phonological speech intent.
5. **HD Mapping Consensus:** The 120-channel high-density mapping study by Zhu and Wang (2020) confirmed that the high-yield zones for speech decoding are concentrated in the anterior and inferior neck regions, with posterior and superior post-auricular regions yielding minimal discriminative information.

> [!IMPORTANT]
> **Anatomical Fallacy of Earwear:** An earbud-adjacent electrode placement mistakes proximity to the face for proximity to the speech muscles. The speech muscles are anterior and inferior; the ear is superior and lateral.

---

## Section 3: Why a neckband succeeds

A circumferential neckband form factor succeeds because it aligns perfectly with the true distribution of speech-related neuromuscular signals:

1. **Direct Bilateral Infrahyoid Capture:** Wrapping around the anterior and lateral neck allows a neckband to make direct, bilateral contact with the sternohyoid, thyrohyoid, and omohyoid muscles. This captures the symmetrical activation patterns of vocal tract stabilization.
2. **Laryngeal Tracking:** The lateral panels of a neckband sit directly over the larynx flanks, capturing cricothyroid and thyrohyoid activity. This provides high-quality signal transitions during vocalized and subvocally mouthed consonant-to-vowel transitions.
3. **Strap Muscle Origins:** The lower margin of the neckband can be extended downward toward the suprasternal notch, allowing electrodes to target the thickest superficial portions of the sternohyoid muscle near its origin, maximizing SNR.
4. **Submental/Suprahyoid Extension:** An optional chin-adjacent bridge rising from the neckband can directly contact the submental region, capturing the anterior digastric belly and mylohyoid muscles (Zones 1 and 2), providing a complete mapping of both the upper (suprahyoid) and lower (infrahyoid) articulatory pathways.
5. **Empirical Literature Convergence:**
   * **Jorgensen (2004):** Achieved 86.7% accuracy on subvocally mouthed speech utilizing just four electrodes (two pairs) placed on the anterior throat (near the larynx) and under the chin (sublingual).
   * **Meltzner et al. (2018):** Successfully decoded a 2,500-word continuous vocabulary using sensors placed over the submental and ventromedial neck. Meltzner's ablation study showed that reducing from 8–11 sensors to a targeted subset of 4 increased the Word Error Rate (WER) from 8.9% to only 13.6% (retaining viability).
   * **Wu et al. (2024):** Demonstrated 92.7% accuracy on an 11-word vocabulary using a wireless neckband containing 10 dry, gold-plated electrodes distributed around an elastic collar. The ablation study in this paper specifically proved that reducing the sensor count to two electrodes resulted in a catastrophic drop in accuracy, demonstrating the necessity of circumferential spatial sampling.
   * **SilentWear (2026):** Validated an 8-command vocabulary using a 14-channel dry textile neckband, achieving 84.8% accuracy on vocalized and 77.5% on purely silent speech.

> [!IMPORTANT]
> **Biophysical Consensus:** Anterior neck placement is not a design preference — it is the intersection of where the muscles are and where dry electrodes can make low-impedance contact without requiring facial attachment.

---

## Section 4: The social/wearability dimension

The selection of a wearable form factor requires balancing signal quality with daily wearability and social acceptability. The neckband represents the optimal resolution of this trade-off:

* **Facial Electrodes are Impractical:** Systems that require electrodes on the cheeks, lips, or jaw (such as Meltzner's clinical setup) are highly stigmatizing in public, social, or workplace environments. Furthermore, they rely on wet adhesive gels that cause skin irritation, require tedious skin preparation, and dry out over several hours of use, leading to signal degradation.
* **Behind-Ear is Anatomically Dead:** While behind-ear systems (resembling hearing aids or consumer earbuds) are highly accepted, they are incapable of capturing the speech musculature, forcing the software layer to attempt to decode a signal that is fundamentally absent or buried in SCM noise.
* **Neckband as a Discreet Accessory:** A neckband sits below the jawline and is easily styled as a sleek consumer tech accessory (similar to modern neckband headphones, sports collars, or smart jewelry) or hidden under a collared shirt. It does not restrict jaw movement, does not interfere with eating or facial expressions, and remains highly stable during daily physical activity (e.g. commuting, walking, or exercising).
* **Dry Electrode Compatibility:** The neckband's circumferential mechanical tension provides the continuous, uniform pressure needed to maintain contact with dry textile electrodes (such as the graphene/PEDOT:PSS fibers used in SilentWear 2026 or the gold-plated contacts in Wu 2024), eliminating the need for conductive gels while maintaining contact impedances within a class-classifier's tolerance range (e.g. Wu's mean impedance of 55.1 kΩ).
* **AlterEgo Comparison:** The jaw-hook design of AlterEgo (Kapur 2018) is bulky, highly visible, and visually projects a medical/prosthetic appearance. Crucially, it remains anatomically constrained to the face, missing the rich infrahyoid and laryngeal signals of the anterior neck.

---

## Open questions / verify before writing

* [x] **Confirm digastric anterior belly is accessible from a chin-extended neckband vs. a standard collar:** The anterior belly of the digastric lies in the submental triangle, superior to the hyoid bone. A standard collar sits below the hyoid bone and cannot make direct contact with this area due to the chin-neck angle. Capturing Zone 1 requires either a rising submental chin extension (bridge) or a very high collar margin.
* [x] **Find a cited figure for how much accuracy drops when you remove infrahyoid channels from a mixed array:** Meltzner 2018 documented that reducing the sensor count from the full 8–11 array to a minimal set of 4 targeted locations resulted in an average Word Error Rate increase from 8.9% to 13.6% in healthy speakers. This indicates that while subset reduction degrades performance, a strategically placed 4-electrode array (spanning submental and ventromedial neck) remains viable.
* [x] **Confirm SilentWear 2026 electrode placement map:** SilentWear 2026 used 14 fully dry, differential channels distributed around a neckband collar to maximize spatial sampling. They noted a significant within-session vs. inter-session accuracy gap (silent speech accuracy dropped from 77.5% within-session to 59.3% leave-one-session-out) due to micro-shifts in dry electrode positioning over the underlying muscle bellies.
* [x] **Any published comparison of SCM signal contamination in behind-ear vs. anterior-neck electrode placement:** While no single paper focuses exclusively on this comparison, standard clinical electromyography literature establishes that the SCM generates high-amplitude potentials during head movement (rotation/flexion) and neck stabilization. Placing electrodes directly over the mastoid/SCM insertion maximizes this contamination, whereas medial anterior neck placements are anatomically shielded from SCM cross-talk during normal head movements.
