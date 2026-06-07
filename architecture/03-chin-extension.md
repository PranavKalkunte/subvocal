# Chin extension: rationale and omission condition

**Status:** stub — fill in during architecture task 3  
**Depends on:** 01-neckband-vs-earbud.md, 02-electrode-layout.md (zones 1–2)  
**Feeds into:** 06-architecture-piece.md §3, CAD (task 9)

---

## What the chin extension is

A chin extension is a narrow bridge element that rises from the superior edge of the neckband collar and contacts the submental region — the soft tissue between the chin tip and the hyoid. It adds electrode coverage for zones 1 (suprahyoid/digastric anterior) and 2 (mylohyoid) without requiring a separate wearable.

Physically, it is a thin curved arm, probably 40–60 mm long, with 2–4 electrode contacts on its inner surface. It sits against the underside of the chin and is held in place by the neckband's own tension. It should not contact the jawbone or teeth; it presses lightly against the submental soft tissue.

---

## Rationale: why these muscles matter

*Fill in:* the argument for including zones 1 and 2.

Suprahyoid muscles (digastric anterior, geniohyoid, mylohyoid) are the primary floor-of-mouth activators. They fire during:
- Tongue body elevation (vowel /i/, /e/)
- Bilabial closures (stops /p/, /b/)
- Jaw depression (open vowels /a/, /æ/)
- Initial consonant release in syllable onset

*Fill in:* cite the specific Meltzner 2018 ablation finding — reducing from 8–11 to 4 sensors increased WER from 8.9% to 13.6%. If the removed sensors included submental placements (check: Meltzner's "submental" corresponds to zones 1–2), this quantifies the cost of omitting the chin zone.

*Fill in:* also cite Jorgensen 2004's "sublingual" electrode pair — he got 86.7% on subvocally mouthed speech using just 4 electrodes, one pair of which was submental/sublingual. This suggests the chin region carries disproportionate discriminative weight at small electrode counts.

---

## The omission condition

The chin extension is **optional** and is omitted when any of the following applies:

**Condition A: Target vocabulary does not require phoneme-level discrimination**  
If the system is designed for a closed command vocabulary (10–25 commands), it may achieve acceptable accuracy from zones 3–5 alone. Wu 2024 demonstrated 92.7% on an 11-word vocabulary using a straight neckband with no chin extension. If the closed-vocabulary accuracy threshold is met without the chin extension, adding mechanical complexity is not justified.

*Fill in:* define what "acceptable accuracy threshold" means for the intended deployment (warehouse operations). If the go/no-go threshold is set at ≥95% on the command vocabulary, determine empirically whether zones 3–5 alone meet it.

**Condition B: The chin extension creates unacceptable UX in the target environment**  
A warehouse worker who frequently looks down at a picking shelf, bends, or wears a hard hat with a chin strap cannot use a chin extension. Check with warehouse-wedge demand interviews (task 3.3) whether submental contact is compatible with the physical demands of the role.

**Condition C: Inter-session reliability degrades unacceptably at the chin zone**  
The submental region is more anatomically variable than the neck — fat distribution, chin-to-hyoid distance, and jaw angle all vary more between people and sessions. If the chin electrodes add signal variance (noise) that exceeds their discriminative gain, they reduce net accuracy. This is an empirical question for the bench experiment (task 6).

**Condition D: Manufacturing cost exceeds marginal accuracy gain**  
The chin bridge adds a moving part and additional electrode contacts to the BOM. If the cost delta is material and the accuracy gain is small (e.g., <2% on the target vocabulary), omit.

---

## How to decide (decision tree)

```
Does the target vocabulary require phoneme-level discrimination?
├── No (closed commands, ≤25 words)
│   └── Run ablation: does removing zones 1–2 drop accuracy below threshold?
│       ├── No → Omit chin extension (Condition A)
│       └── Yes → Include if Conditions B, C, D are not triggered
└── Yes (open vocabulary / phrase-level)
    └── Chin extension is likely required; include unless B or C applies
```

*Fill in:* for the warehouse-wedge Phase 0, write the specific threshold test. Propose: "If a 10-command classifier trained on zones 3–5 achieves ≥95% within-session accuracy on the Phase 0 command set, the chin extension is deferred until open-vocabulary Phase 2."

---

## Mechanical design note

*Fill in for CAD task:* the chin bridge must be:
- Adjustable for chin-to-hyoid distance (variable across people; typical range 55–75 mm)
- Low-force contact — it must not restrict speech movement or create jaw fatigue
- Detachable from the main neckband without tools (user can remove it and reattach; Condition B)
- Compatible with the same dry electrode chemistry used in the main neckband body

---

## Open questions

- [ ] Verify which of Meltzner's 8–11 sensors were submental; identify whether the ablation to 4 sensors removed the submental channels
- [ ] Find any published chin-to-hyoid distance distribution data for adult populations
- [ ] Determine whether any published system used a detachable submental bridge (separate from AlterEgo's fixed jaw hook)
- [ ] Check: does the HD mapping study (Zhu/Wang 2020) show high-yield electrodes in the submental region specifically?
