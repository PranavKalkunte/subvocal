# Intent reconstruction: the low-bandwidth channel reframing

**Status:** stub — fill in during architecture task 4  
**Depends on:** thesis (task 1), synthesis (task 2)  
**Feeds into:** 06-architecture-piece.md §4, Phase 0 software demonstrator (task 5)

---

## The reframing in one sentence

The system is not a silent microphone that transcribes speech. It is a low-bandwidth intent channel that transmits compressed, degraded tokens to an LLM, which then reconstructs the full user intent from context.

This distinction is not cosmetic — it changes the accuracy requirement, the vocabulary design, the prompt architecture, and the failure mode.

---

## Section 1: What "silent microphone" framing implies (and why it's wrong)

*Fill in:*

The "silent mic" framing treats the classifier's output as a speech transcript with errors. Under this model, the system fails whenever the classifier produces a wrong word. The user has to repeat themselves. The system is evaluated against word error rate (WER) as the primary metric, and the accuracy bar is roughly where automatic speech recognition (ASR) is — i.e., <5% WER for production use.

The problem: subvocal sEMG classifiers are nowhere near ASR accuracy on open vocabulary. The best system in the literature (Meltzner 2018) achieved 8.9% WER on a 2,500-word vocabulary under ideal lab conditions with gel electrodes on a face-mounted array, calibrated per session. A practical dry-electrode neckband for open vocabulary is likely 15–25% WER at best, probably worse on cross-session data without calibration.

If you frame the system as a silent mic, 15–25% WER is a failure. If you frame it correctly, it is not.

*Fill in:* also note the ASR comparison is unfair because:
- ASR benefits from acoustic redundancy (an entire waveform per phoneme)
- sEMG captures motor intent before acoustic output; it has fundamentally less signal per phoneme
- The appropriate comparison is to other intent channels under noise, not to clean speech recognition

---

## Section 2: The low-bandwidth channel framing

*Fill in:*

Consider how humans communicate over degraded, low-bandwidth channels. A text message uses 20–200 characters to convey what would take 30 seconds of speech. A warehouse pick-confirmation signal is a single button press. A two-letter airport code replaces a city name. In each case, the receiver reconstructs the full intent from a compressed token plus context.

The sEMG classifier's output is exactly this kind of compressed, noisy token stream. The classifier does not need to produce a perfect transcript. It needs to produce a token sequence that is sufficiently informative, combined with context, for the LLM to reconstruct intent with high confidence.

*Fill in:* spell out what "context" means here:
- Conversation history (what was said in the last N exchanges)
- Task state (what operation is in progress, e.g., picking order #4471, aisle 7)
- Environment model (user is in a warehouse, not a hospital; they are likely to say things like bin numbers, item names, counts)
- User-specific vocabulary and command patterns (personalized via calibration history)
- Phonetic proximity (if the classifier outputs "fife" when the user said "five", the LLM knows "five" from context and corrects)

---

## Section 3: The accuracy requirement shift

*Fill in:*

Under the silent-mic framing, target accuracy = WER ≤ 5% (ASR bar).  
Under the intent-reconstruction framing, target accuracy = intent correctly executed ≥ 95% of the time.

These are different metrics. A system with 20% WER on the raw classifier can still achieve ≥95% intent accuracy if:
- The vocabulary is constrained to high-priority commands (the LLM ignores low-confidence outputs)
- The LLM has enough context to disambiguate common error patterns
- The system requests confirmation on low-confidence outputs rather than acting

*Fill in:* derive the specific accuracy target for the warehouse wedge:
- What is the current error rate for Vocollect voice-picking? (Vocollect claims ~99.9% pick accuracy — find the source)
- What is the cost of a mis-pick in a warehouse? (labor cost, return cost, safety implications)
- What accuracy does the intent-reconstruction pipeline need to match or beat that?
- Map this back to a classifier WER budget

---

## Section 4: Design implications

*Fill in: for each design decision, state what the silent-mic framing would prescribe vs. what the intent-reconstruction framing prescribes.*

| Decision | Silent-mic framing | Intent-reconstruction framing |
|----------|-------------------|-------------------------------|
| Vocabulary size | Maximize (open vocab) | Constrain to high-value commands |
| Classifier output | Full word transcript | Phoneme-level or token lattice with confidence scores |
| LLM role | Post-hoc correction (spell-checker) | Primary intent decoder (the classifier is its noisy input) |
| Failure mode | Transcription error → user repeats | Low confidence → system asks for confirmation |
| Calibration | Minimize per-user data needed | Accept per-user calibration as a feature, not a burden |
| Latency budget | Minimize classifier latency | Allow LLM latency in the budget; target is intent-to-action, not phoneme-to-word |

---

## Section 5: LLM prompt architecture (high-level)

*Fill in: outline the prompt structure for the intent-reconstruction LLM. This is the basis for Phase 0 task 5.*

The LLM receives:
1. A noisy token string from the classifier (could be raw phonemes, word candidates, or a confusion matrix top-N)
2. The current task context
3. The recent conversation history
4. A description of the expected command vocabulary in the current context

The LLM produces:
1. A best-guess intent in a structured schema (command name + parameters)
2. A confidence indicator
3. If confidence < threshold: a clarification question to surface via bone conduction

*Fill in:* the specific format of (1), (3), and the threshold logic will be designed in Phase 0 (task 5, step "write and iterate the intent-reconstruction prompt"). This document sets the architecture principle; Phase 0 operationalizes it.

---

## Section 6: Contrast with BCIs and other silent-speech approaches

*Fill in:*

This reframing is analogous to what good speech BCI work does. The UCSF Chang Lab speech BCI (Metzger et al. 2023) achieves useful communication not because it transcribes every phoneme perfectly, but because it is deployed with a constrained vocabulary and a language model that fills in the gaps. Note that their system has far higher channel count (cortical arrays) and still uses a language model — the principle generalizes.

Contrast: the NASA Ames system (Jorgensen 2004) used a closed command vocabulary of 15 words. Under the intent-reconstruction framing, this is correct architecture, even if it wasn't theorized that way. The failure of early silent-speech systems to scale wasn't a failure of the classifier; it was a failure to recognize that the LLM layer (which didn't exist at the time) is what makes open-vocabulary reconstruction tractable.

---

## Open questions

- [ ] Find the Vocollect pick accuracy claim and its source
- [ ] Find the cost-per-mis-pick figure for warehouse operations (internal company data or Honeywell white paper)
- [ ] Review how the Chang Lab (UCSF) and Henderson Lab (Stanford) BCIs handle LM integration — any published prompt architecture?
- [ ] Determine whether the Phase 0 demonstrator (task 5) should output a confidence score from the simulated classifier, or just a noisy token, and how that flows into the LLM prompt
- [ ] Define what "confirmation request" looks like via bone conduction: a yes/no prompt? Full re-statement of the interpreted command?
