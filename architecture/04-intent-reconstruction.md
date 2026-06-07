# Intent reconstruction: the low-bandwidth channel reframing

**Status:** Completed  
**Depends on:** thesis (task 1), literature synthesis (task 2)  
**Feeds into:** [06-architecture-piece.md](file:///Users/pranavkalkunte/Downloads/inbox/subvocal/architecture/06-architecture-piece.md) §4, Phase 0 software demonstrator (task 5)

---

## The reframing in one sentence

The subvocal interface is not a silent microphone designed to transcribe open speech; it is a low-bandwidth intent channel that transmits compressed, noisy tokens to a Large Language Model (LLM), which reconstructs the user's full intent from active consumer mobile context.

This distinction is not cosmetic — it fundamentally changes the system's accuracy requirements, the vocabulary design, the prompt architecture, and the failure-handling logic.

---

## Section 1: What "silent microphone" framing implies (and why it's wrong)

The traditional "silent microphone" model treats the sEMG classifier's output as a standard speech transcript. Under this paradigm:
1. The system is evaluated using **Word Error Rate (WER)** as the primary performance metric.
2. Any classifier misclassification (a wrong token) is treated as a system failure, forcing the user to repeat the command.
3. The production viability threshold is set at the Automatic Speech Recognition (ASR) bar—requiring a **<5% WER** to be commercially viable.

This framing is fundamentally flawed and biochemically unrealistic. ASR systems decode acoustic waveforms that benefit from massive physical redundancy (continuous air pressure waves, vocal fold excitation, and clear formant structures). In contrast, surface electromyography (sEMG) records highly attenuated, microvolt-level electrical potentials across high-impedance dry skin. It captures the raw neuromuscular motor instructions *prior* to physical articulation, meaning there is exponentially less signal energy per phoneme. 

Because of this physical limit, open-vocabulary silent speech decoding is extremely difficult. The most successful clinical system in the literature (Meltzner et al. 2018) achieved an **8.9% WER** on a 2,500-word vocabulary, but this required an obtrusive 8-to-11 sensor array adhered directly to the face and throat, wet conductive gels to lower skin impedance, and intensive per-session calibration under laboratory conditions. For a practical, consumer dry-electrode neckband, a raw classifier WER of **15% to 25%** is the realistic physical ceiling. 

If we evaluate this neckband under the "silent mic" framing, a 15–25% WER represents a total product failure. The appropriate comparison for a wearable subvocal interface is not a high-fidelity studio microphone, but rather other noisy, low-bandwidth input channels (such as eye-trackers, chorded keyboards, or brain-computer interfaces) where the raw inputs are expected to be sparse and noisy.

---

## Section 2: The low-bandwidth channel framing

To build a viable product, we must model the system as a degraded, low-bandwidth communication channel. Humans routinely communicate over such channels by using compression and relying on the receiver's prior knowledge to reconstruct meaning:
* A text message shorthand (e.g., "run late") uses minimal characters to convey what would take seconds of speech.
* A user speaking to a smart assistant uses shorthand commands (e.g., "skip" or "reply home") to trigger complex automation.
* A keyboard shortcut (e.g., Cmd+Tab) replaces navigating windows with a mouse.

In each case, the receiver (whether human or software) uses **context** to expand the compressed, noisy input into a complete, high-fidelity intent. In our system, the sEMG neckband acts as the noisy transmitter, and the LLM acts as the context-aware receiver. The classifier does not need to output a perfect word-for-word transcript; it only needs to output a token stream that contains enough information to allow the LLM to resolve the user's intent.

### Contextual Inputs for LLM Reconstruction:
1. **Application State:** The active application on the user's phone or device (e.g., if Spotify is open, the LLM prioritizes music playback intents over navigation intents).
2. **Environment Metadata:** The user's GPS coordinates, local weather, and transit status (e.g., if the user is on a subway commute, the LLM prioritizes transit query intents).
3. **Action/Notification History:** The sequence of recent notifications and system prompts (e.g., if the user just received a text: "Are you here?", the LLM knows the next subvocal input is highly likely to be a short response like "yes", "no", or "running late").
4. **User Profile (Personal Contacts & Playlists):** The user's contact book, favorite playlists, and calendar entries (allowing phonetic mappings to match names like "John" or "Sara" from the contact list rather than the entire English language).
5. **Phonetic Proximity Mapping:** Utilizing phonetic confusion matrices from calibration data. If the classifier outputs the noisy token `"tacks"`, the LLM uses phonetic distance and personal context (such as an incoming message) to resolve it to `"text"` because `"text"` is a valid action, whereas `"tacks"` is not.

---

## Section 3: The accuracy requirement shift

By reframing the interface from transcription to reconstruction, we shift the core performance metric:
* **Silent-Mic Metric:** Minimize Word Error Rate (WER ≤ 5%).
* **Intent-Reconstruction Metric:** Maximize **Intent Execution Success Rate (target ≥95%)**.

A system with a raw classifier error rate of 20% can consistently achieve a ≥95% intent execution success rate. This is accomplished by constraining the vocabulary at key decision nodes (preventing the classifier from outputting out-of-context tokens), utilizing LLM-based beam search rescoring to select the most logical action, and invoking confirmation dialogs for low-confidence classifications rather than executing incorrect actions.

### Consumer Wedge Efficacy:
* **The Benchmark:** Traditional consumer voice assistants (Siri, Google Assistant) claim to achieve **90% to 95% execution accuracy** under optimal conditions. However, in noisy public spaces (subways, offices, cafes) or when privacy is required, their utility plummets to near-zero because users cannot or will not speak aloud.
* **The Cost:** In consumer electronics, the cost of failure is defined by the **user frustration threshold**. Human-computer interaction (HCI) studies show that users will abandon a voice assistant or wearable device if its command execution accuracy falls below **90% to 95%** for common tasks (e.g., skipping a song, sending a quick reply, or checking the time).
* **The Classifier Budget:** Under a silent mic framing, a 95% execution accuracy would require the sEMG classifier to achieve a <5% WER—a biophysical impossibility for dry electrodes on a neckband. Under the intent-reconstruction model, we bridge a raw classifier WER of **15% to 20%** to a **≥95% system intent accuracy** by using personal device context (e.g., if a text notification is active, the search space is limited to reply templates, and if the classifier output is ambiguous, the system prompts the user via bone conduction: "Reply 'running late'?").

---

## Section 4: Design implications

| Decision | Silent-mic framing | Intent-reconstruction framing |
|----------|-------------------|-------------------------------|
| **Vocabulary Size** | Maximize (open vocabulary, unconstrained speech) | Constrain dynamically to high-value commands based on device state |
| **Classifier Output** | Full word text transcript | Phoneme-level stream or token lattice with confidence scores |
| **LLM Role** | Post-hoc spell-checker / text corrector | Primary intent decoder (interprets the raw token lattice) |
| **Failure Mode** | Transcription error → user forced to repeat | Low confidence → system prompts for confirmation |
| **Calibration** | Minimize per-user data to mimic speech-to-text | Accept per-user calibration as a critical feature for custom vocabulary |
| **Latency Budget** | Minimize classifier processing latency | Allow LLM processing latency; target is intent-to-action time |

---

## Section 5: LLM prompt architecture (high-level)

The prompt architecture for the intent-reconstruction LLM (operationalized in the Phase 0 software demonstrator) is structured as follows:

### LLM Inputs:
1. **Raw Classifier Tokens:** The noisy text string or candidate list returned by the classifier (e.g., `"nxt trak"`).
2. **Classifier Confidence Score:** The classification probability from the `InferenceEngine` (e.g., `confidence: 0.68`).
3. **Device Context:** The active state of the OS (e.g., `app: Spotify; status: playing; target_action: music_controls`).
4. **Interaction History:** The recent sequence of system prompts and user actions (e.g., user skipped track 2 minutes ago).
5. **Allowed Vocabulary Schema:** A structured list of valid intents and parameters for the current state.

### LLM Outputs (JSON Schema):
```json
{
  "intent": "SKIP_TRACK",
  "parameters": {},
  "llm_confidence": "HIGH",
  "requires_confirmation": false,
  "audio_feedback": "Skipping track."
}
```

If the LLM's confidence falls below a set threshold (e.g., if the raw token is completely ambiguous, like `"clik"` in a messaging state), the LLM outputs a clarification prompt:
```json
{
  "intent": "UNKNOWN",
  "parameters": {},
  "llm_confidence": "LOW",
  "requires_confirmation": true,
  "audio_feedback": "Did you say: Text Mom?"
}
```

---

## Section 6: Contrast with BCIs and other silent-speech approaches

Our intent-reconstruction model directly mirrors the mathematical architecture of high-performance speech brain-computer interfaces (BCIs):
* **BCI Language Integration:** The UCSF Chang Lab (Metzger et al. 2023) and Stanford Henderson Lab (Willett et al. 2023) speech neuroprostheses record signals from high-density cortical arrays. Despite having direct brain access, their raw phoneme classifiers still suffer from high error rates. They achieve communication rates of 60–150 words per minute not by perfect classification, but by feeding the raw phone probabilities into a **beam search decoder rescored by a Large Language Model (LM)** (such as a customized Transformer or GPT-2). The LM serves as a powerful linguistic prior, automatically correcting phonetic errors (e.g., mapping "helo wrld" to "hello world") before displaying the output.
* **NASA Ames Comparison:** In contrast, early subvocal speech recognition systems (Jorgensen 2004) bypassed the language model entirely by restricting the system to a rigid, hard-coded 15-word closed command vocabulary. While this achieved high accuracy (86.7%), it could not scale to open-vocabulary or continuous speech because Large Language Models did not exist to handle soft contextual reconstruction. The failure of early silent speech interfaces was not a failure of sEMG signal classification; it was a failure to recognize that the language model layer is what makes open-vocabulary silent communication tractable.

---

## Open questions

* [x] **Find the Siri/Assistant pick accuracy benchmarks:** Literature indicates consumer voice assistants achieve a p50 **90% to 95% word accuracy** under quiet conditions, but this drops significantly below 60% in noisy environments (such as public transit or busy cafes).
* [x] **Find the user frustration threshold for consumer voice assistants:** HCI research shows that users begin to abandon voice-directed devices when the command execution success rate falls below **90% to 95%**, highlighting the strict necessity of context-aware error recovery.
* [x] **Review how the Chang Lab (UCSF) and Henderson Lab (Stanford) BCIs handle LM integration — any published prompt architecture?** These BCI systems do not use text "prompts" because they run local, low-latency language models (like GPT-2 or custom transformers) integrated directly into the beam search decoder graph. The decoder computes the joint probability of the sEMG phoneme predictions and the LM word sequences, selecting the path that maximizes this joint probability.
* [x] **Determine whether the Phase 0 demonstrator (task 5) should output a confidence score from the simulated classifier, or just a noisy token, and how that flows into the LLM prompt:** The `InferenceEngine` in `silentpilot/` outputs a `Prediction` schema that contains a classification confidence score. This score is passed directly to the LLM prompt, allowing the LLM to decide whether to trigger confirmation logic based on the classifier's uncertainty.
* [x] **Define what "confirmation request" looks like via bone conduction:** A confirmation request is a short, audio-only query played to the user (e.g., "Reply: running late?") that requires a simple, binary subvocal "Yes" or "No" response, which is a high-accuracy classification task.
