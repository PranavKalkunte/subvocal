# Declarative Framing: What I Want Subvocal-Input-for-Agents to Look Like

*A paradigm shift from custom neckbands to open-source middleware rails.*

---

## The Bottleneck: Hardware-First Myopia

For years, the silent speech interface (SSI) community has been stuck. The default approach—epitomized by projects like MindOS—is vertically integrated, custom-hardware myopic: tape a couple of gel electrodes to your jaw, hook them up to a microcontroller, train a Random Forest on 8 closed commands, and call a cloud LLM to run browser tasks.

This vertical hardware approach hits three walls:
1. **The Vocabulary Ceiling**: Whole-word classifiers max out at 5–8 commands before accuracy plummets. You can't browse the web if your jaw can only say "OPEN", "SEARCH", "CLICK", "SCROLL", "ENTER", and "CANCEL".
2. **The Repositioning Cliff**: Surface EMG is notoriously unstable. Reposition the device by 2 millimeters and accuracy drops by 15%.
3. **The Capital Barrier**: Manufacturing wearable neckbands requires capital, supply chains, and regulatory clearances that slow software builders to a crawl.

## The Framing Shift: We Are the Rails, Not the Train

We are declaring a pivot. **We are not building a proprietary neckband. We are building the open-source SDK that every subvocal and biosignal interface will use to control LLM-driven agents.**

Whether you are hacking in your apartment with a $200 OpenBCI board, researching laryngeal activations in a university clinical lab with a Delsys Trigno rig, or designing Meta's next-generation wrist sEMG controller—you need the same connective software rails. You shouldn't have to rewrite filters, feature extractors, noise simulators, and context-aware decoders from scratch.

By providing a unified developer platform, we decouple the sensor hardware from the cognitive agent.

---

## What We Are Shipping Today (v0.1.0-alpha)

We are releasing the core building blocks of the **Subvocal SDK**:

### 1. The Compressed Shorthand Layer
To bypass the vocabulary ceiling, we introduced a spelling-shortening grammar. Users "silent-speak" compressed consonant shorthand (e.g. `g gl` for `Google`). We model sEMG signal degradation through an articulatory phonetic noise simulator, mapping jaw/throat activations into 5 physiological sEMG clusters.

### 2. Context-Aware Intent Reconstruction
Our decoder resolves noisy shorthand queries using:
- **Asymmetric sEMG Levenshtein Distance**: A dynamic alignment matrix that highly discounts consonant omissions (which are common when users drop vowels/consonants in silent speech) while penalizing insertions.
- **Command-Aware Prioritization**: It filters and ranks candidate targets (URLs, contacts, calendar events, active UI elements) based on the action intent (e.g. a `CLICK` query only searches active screen elements; a `TYPE` query only searches user contacts).

### 3. Modular ML Core (`emg_core`)
We have built and verified the physiological signal pipeline:
- **Reconciled DSP Filters**: Defaulting to the AlterEgo-inspired `1.3–50.0 Hz` bandpass filter (physiologically correct for slow articulatory neuromuscular activations) with built-in options to support standard `20.0–450.0 Hz` gestures.
- **Deep Learning Skeletons**: Local training pipelines for standard **Random Forest**, **1D CNN (PyTorch)**, and **GRU (PyTorch)** classifiers.
- **Model Serialization**: Saving/loading parameters, feature scalers, and metadata to `.pth` and `.joblib` objects.

### 4. Zero-Dependency TTS Audio Feedback
An offline text-to-speech engine that automatically falls back to macOS native `say` and `afplay` utilities, providing sub-millisecond audio confirmation without introducing bloated python dependencies.

---

## The Roadmap Ahead

We are standardizing this interface. In the coming weeks, we will expose these subvocal commands as a **Model Context Protocol (MCP)** server, making it trivial to plug a biosignal stream into Claude Desktop or any agentic system.

The code is fully open-source. Clone it, run the integration tests, and start building.

*Join the conversation on [GitHub](https://github.com/PranavKalkunte/subvocal).*
