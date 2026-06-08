# Surface-EMG silent speech interfaces: architectures, form factors, and measured performance

This piece synthesizes six studies that mark the development of surface-electromyography (sEMG) silent speech recognition: the NASA Ames subvocal work (Jorgensen et al., 2003, with related reports through 2005), the AlterEgo interface (Kapur et al., 2018), the Delsys clinical decoding paper (Meltzner et al., 2018), the high-density electrode mapping work (Zhu et al., 2021, with a 2020 conference precursor), the Berkeley textile necklace (Wu et al., 2024), and the ETH Zurich / Bologna SilentWear system (Spacone et al., 2026, building on Meier et al., 2025). It covers the physiological basis, extracts each system against the same fields, compares them, reads them for agreement and disagreement, and draws out the second-order patterns that govern where the field can go next.

A point worth holding throughout: these papers differ in kind, not just in result. Four are device-and-result papers (NASA Ames, AlterEgo, Wu, SilentWear). One advances sensors and algorithms using research-grade instrumentation (Meltzner). One is a mapping study whose output is placement guidance rather than a wearable (Zhu). Their accuracy numbers were produced under different conditions, vocabularies, and speaking modes, so the headline figures are not directly comparable, and the comparison table below should be read with that in mind.

## Physiological basis

Speech is a motor task. When a person intends to speak, the central nervous system sends action potentials down efferent nerves to the neuromuscular junctions of roughly a hundred muscles spanning the face, jaw, neck, and larynx. At each junction, acetylcholine release depolarizes the muscle fiber membrane, and the resulting motor unit action potentials propagate along the fibers. Their summed electrical field reaches the skin surface as microvolt-level potential differences, which surface electrodes detect.

The property that makes silent speech interfaces possible is that this electrical activity is present whether or not sound is produced. The same muscles are innervated during fully vocalized speech, during silently mouthed speech with articulatory movement but no airflow, and, more weakly, during internal articulation with movements too small to see. Because the electrical signal arises from the motor command, it begins before audible sound and is unaffected by ambient acoustic noise. That makes sEMG attractive for noisy settings (industrial, aerospace, covert communication) and for people who cannot produce sound after laryngectomy or severe dysarthria.

The recording problem is hard for reasons that recur across all six papers. The sEMG signal is non-stationary and easily contaminated by baseline drift, power-line interference, cardiac artifact, and changing electrode-skin impedance. The amplitude also falls sharply as speech moves from vocalized to silent, because the muscle excursions shrink. Much of the engineering history below is an attempt to pull a small, speech-specific signal out of this noise while keeping the hardware comfortable enough to wear.

## Per-paper extraction

### Jorgensen, Lee & Agabon, 2003 (NASA Ames)

The task list dates this 2004. The foundational peer-reviewed paper is the 2003 IJCNN paper; Jorgensen himself dates the program's shift toward speech to 2004, and related NASA Ames reports (Jorgensen & Binsted, 2005; Betts & Jorgensen, 2005) extend it. The program began as a way to support operators in noisy or low-pressure environments.

- Vocabulary: sixteen items, namely six control words (stop, go, left, right, alpha, omega) for a simulated rover task and the ten digits. Lee (2008) later reached 60 words at 87.07%.
- Accuracy: about 92% on the six control words. Some popular accounts cite figures up to 99% for very limited sets or for isolated vowels and consonants; the documented peer-reviewed figure is 92%.
- Electrode count: two pairs of self-adhesive Ag/AgCl electrodes (four electrodes) plus a separate ground on the wrist or jaw. Three subjects participated.
- Placement: the larynx flanks and the sublingual area under the chin, roughly 0.25 cm behind the chin cleft and about 1.5 cm to each side of the larynx. This is anterior neck and throat placement.
- Form factor: a laboratory rig with a Neuroscan-class amplifier. The authors anticipated embedding sensors in clothing or a collar.
- Dry vs gel: wet, self-adhesive gel electrodes. Sampling ran up to 20 kHz with a 60 Hz notch filter.
- Silent vs vocalized: sub-audible, that is, silent. The original work recorded sub-acoustically pronounced words. A separate line of EMG research has since compared vocalized against subvocal production directly and found vocalized higher, but that comparison is not part of Jorgensen's original result.
- Calibration: heavy. One hundred or more exemplars per word per subject were recorded over a six-day interval, in morning and afternoon sessions. Features came from a complex dual-tree wavelet transform; the classifier was a scaled conjugate gradient neural network.
- Limitations: a very small closed vocabulary, very heavy per-user training, laboratory-bound hardware, and strict single-speaker dependency.

### Kapur, Kapur & Maes, 2018 (AlterEgo, MIT Media Lab)

AlterEgo aimed past command-and-control toward an internal, always-available interface that a user could operate while holding a normal conversation. The hardware was a 3D-printed wearable that hooked over the ear and ran along the jawline, leaving the mouth and vocal tract clear.

- Vocabulary: ten digits as the primary benchmark; two roughly 20-word application sets (an arithmetic task and chess-move notation). Kapur has separately described performance on about 100 words in press interviews.
- Accuracy: 92% median word accuracy, reported across a 10-subject dataset.
- Electrode count: seven face-and-jaw locations were found to reliably separate subvocalized words; later tests reported comparable results with four electrodes along one jaw.
- Placement: face and jaw, around the mouth and along the jawline. This is a face-mounted configuration, not a neck one.
- Form factor: a head-and-jaw wearable with face-contact electrodes, plus bone-conduction headphones so an assistant could reply without sound.
- Dry vs gel: face-contact Ag/AgCl electrodes. The IUI paper does not foreground a dry-versus-gel distinction, and later neckband papers characterize the setup as wet and face-mounted. This field should be treated as not cleanly specified in the available sources.
- Silent vs vocalized: silent. The user articulates internally with the mouth closed, which the authors call internal articulation, avoiding visible lip movement.
- Calibration: the system is personalized per user, with a 1D convolutional network trained on that user's data. A specific calibration duration is not confirmed in the sources reviewed; one secondary account gives about 15 minutes, which has not been verified against the primary work.
- Limitations: face-mounted electrodes carry comfort and social costs, the vocabulary is small and closed, training is per-user, and the prototype is bulky.

### Meltzner et al., 2018 (Delsys, NIH-funded)

Meltzner and colleagues approached sEMG decoding clinically, to restore communication for people who had undergone total laryngectomy. This is the vocabulary-scale benchmark of the group, moving past closed-set whole-word classification toward continuous recognition.

- Vocabulary: a 2,200-word vocabulary, with more than 1,200 test phrases. Phoneme-based models let the system recognize words it had not been trained on.
- Accuracy: 91.1% recognition, that is, an 8.9% word error rate, on the phrase set. A companion paper (Meltzner et al., 2017) reported about 89.7% for laryngectomized speakers.
- Electrode count: a face-and-neck sensor configuration. The closely related 2017 and 2021 studies from the same group specify eight single-differential Delsys Trigno sensors; an earlier study used eleven recording locations. The 2018 paper's exact channel count should be confirmed against the primary PDF.
- Placement: face and neck. Neck targets sit on suprahyoid and infrahyoid muscles (anterior belly of the digastric, mylohyoid, geniohyoid, platysma, stylohyoid, thyrohyoid, omohyoid, sternohyoid); face targets include zygomaticus, levator labii, orbicularis oris, and mentalis.
- Form factor: multi-point miniaturized sensors arranged in flexible geometries and adhered to the skin. This is research-grade instrumentation, not a consumer wearable.
- Dry vs gel: wet, adhesive sensors requiring skin preparation (alcohol cleaning, tape exfoliation). The related study sampled at 2,222 Hz, bandpass filtered 20 to 450 Hz, with a gain of 300.
- Silent vs vocalized: silent (silently mouthed), which suits laryngectomized users who have lost glottal excitation. The group's broader work also covers vocalized production.
- Calibration: recognition models trained on a large collected corpus, which is per-system training rather than a short per-user routine. A sensor-subset analysis reported a word-error increase when the array was reduced to a minimal set.
- Limitations: the face-and-neck adhesive sensors are obtrusive, and the result is tied to research-grade hardware rather than a wearable form factor.

### Zhu et al., 2021 (high-density electrode mapping, SIAT)

To replace the educated-guess electrode placement of earlier work, this study mapped the facial and neck articulatory network with a dense electrode grid, then searched for the smallest configuration that preserved accuracy. The task list's "120-electrode study" maps cleanly to this work, with a 2020 conference paper from the same group as a precursor.

- Vocabulary: the ten digits, recorded in both English and Chinese.
- Accuracy: with ten optimally selected electrodes, 86% for English and 94% for Chinese. Reaching comparable accuracy without optimization required about 40 electrodes. The 2020 precursor reports higher full-array figures (on the order of the high 90s with all channels), which is why a flashier "near 99%" number circulates; the more useful result is what a small optimized subset achieves.
- Electrode count: 120 closely spaced electrodes in high-density grids over face and neck, used as the search space. A sequential forward selection algorithm identified small optimal subsets.
- Placement: grids over both face and neck. The optimally selected electrodes were mostly on the neck rather than the face.
- Form factor: a laboratory high-density grid. The contribution is placement guidance, not a device.
- Dry vs gel: high-density gel or adhesive electrode arrays.
- Silent vs vocalized: silent.
- Calibration: per-session recording, with the analysis aimed at electrode selection rather than a deployment routine. Classification used linear discriminant analysis on time-domain features.
- Limitations: a 120-electrode grid is impractical to wear, so the study informs placement rather than delivering a product. The English-versus-Chinese difference means electrode-count conclusions may not transfer across languages.

### Wu et al., 2024 (textile necklace, UC Berkeley)

Wu and colleagues moved the sensor payload entirely off the face and onto a discreet neckband with reusable dry electrodes, addressing the comfort and stigma of facial arrays.

- Vocabulary: eleven words (heed, had, hood, tail, kale, doe, goat, aba, ada, aga, aka), ten repetitions each, chosen as minimal and near-minimal pairs to isolate vowels and plosives.
- Accuracy: 92.7% with the ten dry neck electrodes alone (95% CI 90.9 to 94.8); 93.9% with all thirteen channels; 88.6% with face electrodes only. The classifier was a random forest with max depth 32, on time-domain statistical features. The paper discusses self-supervised speech models (WavLM) for an EMG-to-acoustic correlation analysis, but the headline classification accuracy comes from the random forest, not a deep network.
- Electrode count: thirteen recording channels, namely ten dry electrodes spaced around the neck plus three wet face electrodes used as a benchmark, with a shared wet reference behind the right ear. The WANDmini frontend can record up to 64 channels.
- Placement: ten equally spaced electrodes around the full neck (3 cm pitch, 2 cm² each), with the two center electrodes over the larynx, plus three perioral electrodes for comparison.
- Form factor: an elastic velcro neckband with a behind-the-neck WANDmini module (2.5 by 2.5 cm, 3.8 g). The dry neck electrodes are 3D-printed and electroless gold-plated, with a measured mean skin-electrode impedance of 55.1 kΩ at 50 Hz.
- Dry vs gel: dry on the neck; the three face benchmark electrodes are wet Ag/AgCl.
- Silent vs vocalized: vocalized. EMG and audio were recorded at the same time while subjects spoke aloud, which is the key qualifier on the 92.7% figure. The authors note that vocalized speech benefits from glottal vibration, tissue displacement, and airflow that silent speech lacks. Silent and sentence-length data are named as future work.
- Calibration: multi-speaker classifiers trained on 80/20 splits, with no per-user calibration routine reported. Two subjects, both male native English speakers.
- Limitations: two male subjects, vocalized speech only, an eleven-word vocabulary, and word-level rather than continuous decoding.

Two findings reach beyond the paper's own numbers. Neck-only dry electrodes nearly matched the combined neck-and-face set (92.7% against 93.9%), and an ablation showed accuracy improving clearly as electrodes increased past two and continuing through roughly the eighth. The phonological confusion pattern from neck-only data closely matched the pattern from neck-and-face data, which suggests the neck carries much of the same articulatory information as the face.

### Spacone et al., 2026 (SilentWear, ETH Zurich / University of Bologna)

SilentWear is the closest of the six to a deployable wearable, combining a dry textile neckband with on-device, low-power inference. It builds directly on a September 2025 predecessor from the same group (Meier et al.).

- Vocabulary: eight human-machine interaction commands, collected over multiple days.
- Accuracy: within-session cross-validated, 84.8 ± 4.6% for vocalized speech and 77.5 ± 6.6% for silent speech. In an inter-session setting, with the band removed and repositioned between sessions, accuracy fell to 71.1 ± 8.3% (vocalized) and 59.3 ± 2.2% (silent). An incremental fine-tuning step recovered more than 10 percentage points with under 10 minutes of new data.
- Electrode count: fourteen differential EMG channels plus four ground-reference channels. The band carries dry Datwyler SoftPulse electrodes on sewn-in snap fasteners, with the central electrodes arranged in overlapping differential rows so adjacent channels share a middle electrode.
- Placement: along the neck, in a textile band.
- Form factor: a soft-fabric, fully dry, velcro-adjustable neckband on the BioGAP-Ultra acquisition platform, with a 15,000-parameter CNN (SpeechNet) running on a commercial multi-core microcontroller.
- Dry vs gel: fully dry, no gel or skin preparation.
- Silent vs vocalized: both, reported separately. Vocalized accuracy is consistently higher than silent.
- Power and latency: 20.5 mW for acquisition, inference, and wireless transmission together, allowing more than 27 hours of continuous operation; 63.9 µJ per inference at 2.47 ms latency, with 24-bit acquisition.
- Calibration: incremental fine-tuning to handle day-to-day repositioning, requiring under 10 minutes of additional user data.
- Limitations: a small command set, silent accuracy that remains modest, and a clear inter-session drop that fine-tuning only partly closes.

The Meier et al. (2025) predecessor used the same dry neckband and eight commands, reporting 87 ± 3% (vocalized) and 68 ± 3% (silent) under cross-validation with a random forest, and leave-one-session-out accuracies of 64 ± 18% and 54 ± 7%, at 22 mW.

## Comparison table

| Source | Vocabulary | Accuracy (metric, condition) | Electrodes | Placement | Form factor | Silent or vocalized | Calibration | Key limitation |
|---|---|---|---|---|---|---|---|---|
| Jorgensen et al. 2003 (NASA Ames) | 16 items (6 words + 10 digits); 60 words by 2008 | ~92%, 6 words, silent | 4 (two pairs) + ground | Larynx flanks + sublingual (anterior neck) | Lab rig; envisioned collar | Silent (sub-audible) | Heavy: 100+ exemplars/word over 6 days | Tiny vocab; very heavy training; lab hardware; 3 subjects |
| Kapur et al. 2018 (AlterEgo, MIT) | 10 digits; ~20-word task sets; ~100 words claimed | 92% median word acc., digits (10 subjects) | 7 face/jaw (later 4) | Face and jaw, around mouth | Head/jaw unit with face-contact electrodes; bone-conduction audio | Silent (internal articulation) | Per-user; CNN; duration unconfirmed | Face placement; small closed vocab; per-user training; bulky |
| Meltzner et al. 2018 (Delsys, NIH) | 2,200-word vocab; >1,200 phrases | 91.1% (8.9% WER), phrases, silent; ~89.7% laryngectomized (2017 paper) | ~8 face+neck (confirm; 11 earlier) | Face and neck (supra/infrahyoid + perioral) | Research adhesive sensors, flexible geometries | Silent (silently mouthed) | Per-system training on large corpus; phoneme HMM | Research-grade adhesive sensors; not a wearable |
| Zhu et al. 2021 (HD mapping, SIAT) | 10 digits, English + Chinese | 86% English / 94% Chinese, 10 optimal electrodes; near high-90s full array (2020 precursor) | 120 grid → ~10 optimal | Face+neck grid; optimal subset on neck | HD lab grid (placement guidance) | Silent | Per-session; SFS electrode selection; LDA | 120-grid impractical to wear; language-dependent |
| Wu et al. 2024 (necklace, Berkeley) | 11 words (minimal pairs) | 92.7% neck-only; 93.9% all; 88.6% face-only (voiced) | 13 (10 dry neck + 3 wet face) + mastoid ref | Full neck ring (center over larynx) + perioral | Velcro neckband + behind-neck WANDmini (3.8 g) | Vocalized | Train/test split; no per-user routine; 2 male subjects | 2 subjects; voiced only; 11 words; word-level; random forest |
| Spacone et al. 2026 (SilentWear, ETH/Bologna) | 8 commands | 84.8% voc. / 77.5% silent (CV); 71.1% / 59.3% inter-session | 14 differential + 4 ground; dry Datwyler SoftPulse | Neck (textile band) | Fully dry neckband; on-device CNN; 20.5 mW, >27 h | Both | Incremental fine-tuning, <10 min | Small command set; modest silent acc.; inter-session drop |

## Points of convergence

The anterior neck is a productive recording site, and the field has moved toward it. Jorgensen recorded from the larynx and sublingual area in 2003. Meltzner's neck targets sit on the suprahyoid and infrahyoid muscles. Zhu found that the optimal small electrode subsets cluster on the neck rather than the face. Wu showed neck-only dry electrodes matching a combined neck-and-face set. SilentWear and its predecessor place all channels on the neck. AlterEgo is the outlier with face-and-jaw placement, and its own successor work has since moved off the face.

The neckband form factor reaches usable word accuracy. Wu reached 92.7% on eleven words with a dry neckband, voiced, and SilentWear reached the high 70s to mid 80s on eight commands across silent and vocalized conditions. Zhu's result explains why this works: a small, well-chosen set of neck electrodes captures most of the available signal, so the dense facial coverage assumed by early work is unnecessary.

More than two electrodes helps, and accuracy saturates quickly after that. Wu's ablation improves up to about the eighth electrode and collapses at two, the count used in simple electroglottographs. Zhu found accuracy rising fast then plateauing as electrodes increase toward 120, with ten optimal electrodes matching about forty unoptimized ones. The working range that Wu and SilentWear settled on, roughly eight to fourteen channels, sits in this zone.

About 90% word accuracy is reachable on small vocabularies under controlled, single-session conditions. Jorgensen reported 92% on six words, AlterEgo 92% on digits, Wu 92.7% on eleven words (voiced), and Meltzner 91.1% on phrases from a large vocabulary using research-grade sensors. These figures cluster closely enough to suggest a soft ceiling for the classic approach when conditions are favorable.

The classic acquisition recipe is a bandpass of roughly 20 to 450 Hz with adhesive, prepared skin, seen in Meltzner and Jorgensen, and the recent trend drops the gel and skin prep for dry electrodes, as in Wu and SilentWear. Per-user personalization is assumed throughout, whether through AlterEgo's personalized model, Jorgensen's six-day training, or SilentWear's short fine-tuning.

The algorithms have grown more complex and have moved onto the device. Early systems (Jorgensen, the mapping work) used handcrafted features fed into shallow classifiers such as linear discriminant analysis. Meltzner used phoneme-based hidden Markov models with grammar constraints. AlterEgo and SilentWear use convolutional networks that learn features from filtered time-series data. SilentWear runs a 15,000-parameter network on a microcontroller at 63.9 µJ per inference, which shows that these models no longer need a tethered computer.

## Disagreements and gaps

Silent and vocalized speech are not interchangeable, and several headline numbers are vocalized. Wu's 92.7% is voiced, not silent. SilentWear shows silent speech trailing vocalized by roughly 7 points within a session and about 12 points across sessions. AlterEgo and Jorgensen are silent but on very small vocabularies. So the strongest neckband accuracy in this set (Wu) is voiced, while the strongest silent accuracy on a dry wearable neckband (SilentWear, 77.5% within session and 59.3% across sessions) is considerably lower. This is the widest distance between the field's promise and its wearable reality.

Cross-session robustness is mostly unaddressed before 2025. The classic results report within-session or heavily trained accuracy. Only SilentWear and Meier quantify what happens when the device is taken off and put back on, and the drop is large (for silent speech, 77.5% to 59.3%). Repositioning is the everyday failure mode for a wearable, and it has barely been measured across most of this literature.

Subject counts are small and often skewed. Wu used two male subjects, Jorgensen three, AlterEgo ten. Zhu and Meltzner are larger but still modest. Cross-subject generalization is weak across the set, and none of these papers demonstrates a model that transfers cleanly to a new user without per-user data.

Vocabulary size trades against wearability. Meltzner reached 2,200 words, but with research-grade adhesive face-and-neck sensors. The dry neckbands operate at eight to eleven words. No paper here shows large-vocabulary silent decoding on a dry neckband, let alone one that survives repositioning. The likely reason is structural: treating recognition as closed-set whole-word classification does not scale to conversational vocabularies, which need phoneme-level decoding and coarticulation modeling that the current dry-neckband signal quality does not reliably support.

The electrode-chemistry question is unresolved. Classical practice uses Ag/AgCl with conductive gel, which lowers contact impedance and gives clean signals but dries out, irritates skin, and is unsuited to a put-on-and-forget wearable. Jorgensen, Meltzner, and Zhu used wet or gel-assisted electrodes; the AlterEgo case is ambiguous in the sources. Wu and SilentWear use dry electrodes (gold-plated polymer and a commercial dry textile electrode, respectively) and accept a higher noise floor and impedance (Wu measured 55.1 kΩ) as the cost of comfort. The split runs along application lines: clinical, high-accuracy continuous-speech work leans wet, and consumer-facing wearable work leans dry.

The accuracy metrics are not directly comparable. Word accuracy (AlterEgo, Wu, Jorgensen, SilentWear) sits beside word error rate on phrases (Meltzner) and digit classification across two languages (Zhu). Vocabulary sizes, chance levels, classifiers, and the silent-versus-voiced condition all differ. Side-by-side accuracy comparisons across these papers are loose.

Results may be language-dependent. Zhu found English needs more electrodes than Chinese for the same digit accuracy. Most of the rest of this work is English-only, so electrode-count and placement conclusions drawn from one language may not transfer.

## Deeper reading

Three patterns sit underneath the individual results and matter more than any single number.

The first is a wearability-performance tension. As the hardware becomes more comfortable and less conspicuous, the signal becomes harder to decode. Removing gel and moving sensors off the expressive facial muscles onto a neck band reduces stigma and improves comfort, and it also introduces motion artifact, capacitive noise, crosstalk from swallowing and head movement, and lower signal amplitude. The visible response in the literature is that as systems become more wearable, vocabularies shrink back to a handful of distinct commands, which keeps published accuracy high. The ambition behind AlterEgo, a fluent internal interface, depends on decoders becoming good enough to pull phoneme-level continuous speech from the degraded signal a comfortable dry neckband provides.

The second is inter-session variability as the binding constraint. Initial accuracy is comparatively easy to achieve in a single sitting, where the electrode-to-muscle geometry is fixed. The real barrier is stability across days. SilentWear quantified this directly: a model at 84.8% in-session fell to 71.1% simply because the user removed the band and put it back on, with the silent-speech figure falling further. A few millimeters of electrode shift changes the amplitude, phase, and shape of the differential signals enough to break the learned feature maps. SilentWear's fix is short re-calibration on each donning, which works but is friction. The implication is that static models do not match the reality of wearable electrophysiology, and that continuous or self-supervised adaptation is likely required rather than optional.

The third is neuromuscular redundancy. The facial and neck muscles form an overlapping web rather than a set of independent actuators, so the intent to produce a phoneme sends correlated activity across the region. Wu's finding that neck-only confusions match neck-and-face confusions implies the neck carries a compressed version of the whole vocal tract's activity, and Zhu's result that a small optimized neck subset matches a dense grid points the same way. Read together, this is encouraging for hardware (you do not need electrodes on the lips to know what the lips are doing) and demanding for software (the decoder has to untangle a dense, overlapping signal), and it reframes the problem as decoding capacity rather than sensor placement.

## Where the gaps point

Two directions follow from the gaps above, and both are visible in adjacent recent work rather than in the six core papers.

The first is multimodal sensing. Pure electrical recording through high-impedance dry contact has limited headroom on a comfortable neckband. Combining sEMG with modalities that are not sensitive to skin impedance, such as inertial sensing of tissue vibration or magnetic recording of muscle activity, could disambiguate similar articulations without returning electrodes to the face. Recent magnetometer-based silent speech work is an early example.

The second is a shift away from fully supervised, per-user models toward self-supervised models pretrained on large unlabeled neuromuscular datasets. If a model learns the structure of human motor control before deployment, it could adapt to a new user with little data and tolerate electrode shift better, reducing both the calibration burden and the inter-session penalty. Recent self-supervised EMG-to-speech work points in this direction. Neither path is proven for a dry silent-speech neckband, but they address the two failure modes the data above make clearest.

## What this means for a neckband venture

Read together, these papers describe a roughly two-decade move from face to neck and from laboratory instrumentation to a dry, low-power, on-device wearable. Jorgensen showed a few throat electrodes could separate a small command set. AlterEgo made the idea concrete and personalized, on the face. Meltzner showed the vocabulary could reach the thousands with enough sensors and the right phoneme models, at the cost of research-grade hardware. Zhu supplied the quantitative argument that the useful signal concentrates on the neck and a small placement captures most of it. Wu and SilentWear are the payoff, dry electrodes on a neckband at usable accuracy, with SilentWear adding on-device inference within an all-day power budget.

The honest state of the art for a wearable, silent, dry-electrode neckband is not the 92% that headlines the field. It is closer to SilentWear's numbers: high 70s within a session and high 50s once the band has been removed and replaced, on an eight-command set. The 92% figures are real but carry qualifiers, whether vocalized speech (Wu), a tiny vocabulary (Jorgensen, AlterEgo), or adhesive research sensors (Meltzner). For a product, the two limits that bind are the silent-versus-voiced gap and the inter-session drop, both measured directly only in the 2025 to 2026 work.

That gap is where the engineering risk and the product opportunity both sit. SilentWear's short fine-tuning is one practical response to repositioning. A second, and the one this venture has adopted, is to stop treating the device as a silent microphone aiming for transcription accuracy and instead treat it as a low-bandwidth intent channel feeding a language model that reconstructs the command. That fits what the literature shows: small command vocabularies are where dry-neckband accuracy is currently usable, and a model that expands a short, noisy signal into an intended action suits a channel that is reliable for a handful of commands but not yet for open dictation. The literature does not validate the full product, but it supports the specific choices: neck placement, a small electrode count in the eight-to-fourteen range, dry electrodes, per-user calibration with ongoing adaptation, and a command-level rather than free-text target.

## Verification notes

Confirmed against the cited sources: AlterEgo's 92% median digit accuracy and seven-electrode face/jaw configuration; Jorgensen's six control words plus ten digits, two electrode pairs, larynx and sublingual placement, sub-audible (silent) mode, and roughly 92% accuracy; Meltzner's 2,200-word vocabulary and 8.9% word error rate; Zhu's 120 electrodes, ten-optimal-electrode result of 86% English and 94% Chinese, and neck-clustering finding; Wu's eleven words, 92.7% neck-only voiced accuracy, ten dry neck electrodes, random-forest classifier, and voiced (not silent) condition; SilentWear's fourteen differential channels, dry Datwyler SoftPulse electrodes, 84.8% vocalized and 77.5% silent cross-validated accuracy, inter-session figures, and 20.5 mW power budget.

Points to resolve before publishing:

- AlterEgo's dry-versus-gel electrode type is not cleanly stated in the available sources; later papers describe it as wet and face-mounted. Confirm against the IUI 2018 paper.
- AlterEgo's per-user calibration duration is not confirmed. A secondary account gives about 15 minutes; verify against the thesis or leave it unstated.
- Meltzner 2018's exact electrode count should be read off the primary PDF. The related 2017 and 2021 studies use eight Trigno sensors; an earlier study used eleven. The 89.7% laryngectomized figure is from the 2017 companion paper, not the 2018 paper.
- Meltzner's vocabulary is 2,200 words in the 2018 paper. A 2,500-word figure appears in a later prosodic-speech paper from adjacent authors and should not be attributed to the 2018 study.
- The high-density numbers cited here (86% English, 94% Chinese for ten optimal electrodes) are from the 2021 optimization paper. The 2020 precursor reports higher full-array figures; a "near 99%" full-array number circulates and should be attributed to the precursor and to the full grid, not presented as the headline result of the optimized system.
- Jorgensen's original work is sub-audible (silent) speech at about 92% on six words. A vocalized-versus-mouthed accuracy comparison that circulates (roughly 92% versus 87%) comes from a separate neck-and-face EMG paper on speaking modes and should not be attributed to Jorgensen's 2003 result.
- The "Jorgensen 2004" label maps to a body of work anchored by the 2003 IJCNN paper, with the program's speech emphasis dated by Jorgensen to 2004. Choose which date to cite.

## References

Kapur, A., Kapur, S., & Maes, P. (2018). AlterEgo: A personalized wearable silent speech interface. In Proceedings of the 23rd International Conference on Intelligent User Interfaces (IUI '18), Tokyo, Japan (pp. 43–53). ACM. https://doi.org/10.1145/3172944.3172977

Jorgensen, C., Lee, D. D., & Agabon, S. (2003). Sub auditory speech recognition based on EMG signals. In Proceedings of the International Joint Conference on Neural Networks (IJCNN) (Vol. 4, pp. 3128–3133). IEEE.

Jorgensen, C., & Binsted, K. (2005). Web browser control using EMG based sub vocal speech recognition. In Proceedings of the 38th Annual Hawaii International Conference on System Sciences (HICSS) (pp. 294c.1–294c.8). IEEE.

Betts, B. J., & Jorgensen, C. (2005). Small vocabulary recognition using surface electromyography in an acoustically harsh environment. Neuro-Engineering Laboratory, NASA Ames Research Center, Moffett Field, CA.

Meltzner, G. S., Heaton, J. T., Deng, Y., De Luca, G., Roy, S. H., & Kline, J. C. (2018). Development of sEMG sensors and algorithms for silent speech recognition. Journal of Neural Engineering, 15(4), 046031. https://doi.org/10.1088/1741-2552/aac965

Meltzner, G. S., Heaton, J. T., Deng, Y., De Luca, G., Roy, S. H., & Kline, J. C. (2017). Silent speech recognition as an alternative communication device for persons with laryngectomy. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 25(12), 2386–2398.

Zhu, M., Zhang, H., Wang, X., Wang, X., Yang, Z., Wang, C., Samuel, O. W., Chen, S., & Li, G. (2021). Towards optimizing electrode configurations for silent speech recognition based on high-density surface electromyography. Journal of Neural Engineering, 18(1), 016005. https://doi.org/10.1088/1741-2552/abca14

Zhu, M., et al. (2020). The effects of channel number on classification performance for sEMG-based speech recognition. In 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC).

Wu, P., Kaveh, R., Nautiyal, R., Zhang, C., Guo, A., Kachinthaya, A., Mishra, T., Yu, B., Black, A. W., Muller, R., & Anumanchipalli, G. K. (2024). Towards EMG-to-speech with a necklace form factor. Interspeech 2024 / arXiv:2407.21345.

Spacone, G., Frey, S., Pollo, G., Burrello, A., Jahier Pagliari, D., Kartsch, V., Cossettini, A., & Benini, L. (2026). SilentWear: An ultra-low power wearable system for EMG-based silent speech recognition. arXiv:2603.02847.

Meier, F., et al. (2025). A parallel ultra-low power silent speech interface based on a wearable, fully-dry EMG neckband. arXiv:2509.21964.

### Related sources

Lee, K.-S. (2008). EMG-based speech recognition using hidden Markov models, extending the vocabulary to 60 words (referenced via secondary sources).

Gaddy, D., & Klein, D. (2020/2021). Digital voicing of silent speech; An improved model for voicing silent speech. EMNLP 2020 / ACL-IJCNLP 2021. Closed-vocabulary silent EMG synthesis at 3.6% WER.

Recent adjacent directions cited in the synthesis: magnetometer-based silent speech recognition (bioRxiv, 2025) and self-supervised EMG-to-speech models (arXiv, 2025), as examples of multimodal sensing and pretraining rather than as validated neckband results.
