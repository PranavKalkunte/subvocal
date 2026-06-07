# Neckband vs. earbud: form-factor decision from muscle anatomy

**Status:** stub — fill in during architecture task 1  
**Depends on:** literature synthesis (§2), especially Wu 2024 and ETH SilentWear 2026  
**Feeds into:** 06-architecture-piece.md §2

---

## Argument to make

The core claim: placing electrodes around the anterior neck captures the muscles that matter for subvocal speech, while ear-adjacent or behind-ear placements do not — and this is derivable from anatomy, not just empirical preference.

---

## Section 1: Relevant muscle groups and their surface accessibility

*Fill in:* name each muscle group active during subvocal speech, where on the body its belly sits, and whether the belly is accessible to dry surface electrodes from a neckband position.

Key muscles to cover (verify against anatomy sources, not just the synthesis PDF):

- **Suprahyoid group** — mylohyoid, geniohyoid, digastric (anterior belly), stylohyoid
  - Belly location: floor of mouth, undersurface of jaw/chin
  - Accessible from: chin-adjacent extension on a neckband, or inferior jaw
  - Accessible from behind ear? No — too far posterior and superior

- **Infrahyoid group** — sternohyoid, sternothyroid, thyrohyoid, omohyoid
  - Belly location: anterior neck, below hyoid bone
  - Accessible from: neckband front and lateral panels
  - Accessible from earbud or jaw hook? No

- **Laryngeal extrinsic muscles** — thyrohyoid, cricothyroid
  - Belly location: flanks of the thyroid cartilage / anterior neck
  - Accessible from: neckband lateral panels
  - Accessible from behind ear? No

- **Temporalis, masseter, buccinator** — chewing/jaw muscles
  - Belly location: temple, cheek, lateral jaw
  - Accessible from: jaw-hook form factor (AlterEgo style) or cheek electrodes
  - Accessible from neckband? No — but literature (HD mapping study) shows these add limited discriminative value for neck-only subvocal; cite SFS results from Zhu/Wang 2020

*Cross-reference:* Kapur 2018 used jaw/cheek electrodes; Wu 2024 and SilentWear 2026 showed neck-only is viable for command vocabularies. Note the tradeoff: jaw muscles add signal for vowel discrimination but impose a facial form factor.

---

## Section 2: Why an earbud placement fails

*Fill in:* quantify what an earbud or behind-ear electrode would actually pick up.

Points to cover:

- Behind the ear: mastoid bone, sternocleidomastoid (SCM) insertion, posterior digastric belly
- SCM fires during head rotation and general neck tension — high motion artifact, low speech specificity
- Posterior digastric: some jaw-depression signal, but mixed with SCM noise and far from the primary subvocal muscles
- No access to infrahyoid group from this position
- Cite: the HD mapping study (Zhu/Wang 2020) found the high-yield zones are anterior and inferior, not posterior
- Practical: any in-ear electrode picks up jaw movement via the ear canal walls (temporomandibular joint), which is gross motion, not fine neuromuscular signal

*Claim to write:* An earbud-adjacent electrode placement mistakes proximity to the face for proximity to the speech muscles. The speech muscles are anterior and inferior; the ear is superior and lateral.

---

## Section 3: Why a neckband succeeds

*Fill in:* describe what a correctly designed neckband can see.

Points to cover:

- Full circumferential coverage of the anterior neck captures infrahyoid bilaterally
- Inferior margin can extend toward the suprasternal notch (sternohyoid origin)
- Chin extension (covered in 03-chin-extension.md) optionally adds suprahyoid
- Literature convergence: Jorgensen 2004 (throat + chin), Meltzner 2018 (ventral neck included), Wu 2024 (full neckband), SilentWear 2026 (14-ch neckband) all land on anterior-neck placement
- Wu 2024: 92.7% accuracy from neck-only electrodes, no facial placement
- SilentWear 2026: viable command classification with dry textile electrodes, neck only

*Claim to write:* Anterior neck placement is not a design preference — it is the intersection of where the muscles are and where dry electrodes can make low-impedance contact without requiring facial attachment.

---

## Section 4: The social/wearability dimension

*Fill in:* complete the argument by noting that muscle anatomy and wearability converge rather than trade off here.

- Facial electrodes (AlterEgo, Meltzner) are stigmatizing and gel-dependent
- Behind-ear electrodes look like hearing aids — socially acceptable, but anatomically wrong for this signal
- Neckband sits below the chin line; in a warehouse or daily-use context, indistinguishable from a medical alert pendant or audio collar
- AlterEgo's jaw-hook aesthetic is the worst of both worlds: visible and anatomically constrained to non-neck muscles

---

## Open questions / verify before writing

- [ ] Confirm digastric anterior belly is accessible from a chin-extended neckband vs. a standard collar (check the chin extension doc)
- [ ] Find a cited figure for how much accuracy drops when you remove infrahyoid channels from a mixed array (Meltzner ablation: WER goes from 8.9% to 13.6% with 4-channel reduction — is that subset infrahyoid?)
- [ ] Confirm SilentWear 2026 electrode placement map (14 channels, positions described?)
- [ ] Any published comparison of SCM signal contamination in behind-ear vs. anterior-neck electrode placement?

---

## Notes for the writer

This section is an argument, not a survey. The synthesis PDF covers the papers — this doc needs to draw the anatomical conclusion that the literature implies but doesn't state directly: *the neckband is the only form factor that is both anatomically correct and socially viable*.

Approximate target length when written: 600–900 words for use in the architecture piece; this stub expands to roughly 1,500 words as a standalone document.
