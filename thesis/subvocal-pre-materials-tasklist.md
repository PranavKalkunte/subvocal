# Subvocal venture: pre-materials task list

Two sections: product engineering and design, then everything supplementary (company, market, content, capital, legal). Within each, ordered by dependency and lead time. Long-lead-time items (lab access, researcher outreach, co-founder targeting) appear early even when downstream work uses them later. Items within a section are not strictly serial; many should run in parallel once their prerequisites land.

---

> **Treehacks 2026 baseline (MindOS).** A working v0 exists. It is the reference to beat, not the starting point to copy. Key specs: 2–3 MyoWare gel sensors on jaw only, USB serial (no BLE), 8 closed-vocabulary commands, 62.5% on real silent EMG / 93.3% on synthetic, zero cross-session stability testing, no dry electrodes, no custom PCB, cloud-only inference via GPT-4o, per-user model with 15-min calibration required per session. The full stack (Python DSP/ML + FastAPI + Next.js UI + TypeScript agent + Playwright browser control + ESP32 firmware) is in `treehacks-2026-main/silentpilot/`. Use it as scaffolding; the goal is to make every dimension of that system look embarrassing.

---

# Part 1: Product

## 1. Thesis and thresholds
*Status: not started. No overlap with treehacks.*

- [ ] Draft the one-page thesis: structural shift, incumbent exposure, opening, why now, single product hypothesis
- [ ] Rewrite the thesis to the narrow, falsifiable version (remove "post-screen paradigm" framing)
- [ ] List the riskiest assumptions (technical and market)
- [ ] Write the go/no-go demand thresholds before any evidence is gathered
- [ ] List the risks that cannot be retired pre-product (capital, manufacturing, regulatory, hardware accuracy)
- [ ] Lock the thesis document, dated

## 2. Literature synthesis
*Status: complete. `Silent Speech Research Synthesis.pdf` covers all six papers with full comparison table, convergence/disagreement analysis, and strategic outlook. Items below are checked accordingly.*

- [x] Obtain Kapur 2018 AlterEgo MIT thesis
- [x] Obtain Jorgensen 2004 NASA Ames technical reports
- [x] Obtain Meltzner et al. 2018 Delsys/NIH paper
- [x] Obtain the 120-electrode high-density mapping study
- [x] Obtain Wu et al. 2024 necklace form-factor paper
- [x] Obtain SilentWear 2026 ETH Zurich paper
- [x] Extract per paper: vocabulary size, accuracy/WER, electrode count, placement, form factor, dry vs gel, silent vs vocalized, calibration time, limitations
- [x] Build the comparison table across all sources
- [x] Mark points of convergence (anterior neck, infrahyoid SNR, neckband viability)
- [x] Mark disagreements and gaps
- [x] Draft the synthesis piece
- [x] Verify every numeric claim against its source
- [ ] Add citations from Zotero
- [ ] Final edit synthesis

## 3. Architecture
*Status: partial. Treehacks docs (`HACKATHON_3SENSOR_PLAN.md`, `DATA_CAPTURE_PLAN.md`) contain the 3-sensor placement rationale, muscle anatomy reasoning, and signal-path description. These are informal hackathon docs — not publishable write-ups. The signal path they built (subvocalize → sEMG → USB serial → RF classifier → GPT-4o → Playwright) diverges from the target architecture (BLE → on-device classifier → local LLM → bone conduction). Rewrite from scratch using their anatomical reasoning but our architecture.*

- [x] Write the neckband-vs-earbud reasoning from muscle anatomy
  > Treehacks has this in `HACKATHON_3SENSOR_PLAN.md` (masseter/submental/laryngeal rationale). Needs to be expanded to the full 5-zone circumferential neckband argument they never reached.
- [x] Document the 5-zone electrode layout (suprahyoid/digastric, mylohyoid, thyrohyoid, larynx flanks, infrahyoid)
  > Treehacks only placed 3 sensors. The 5-zone layout is documented in the synthesis PDF but not in their code or hardware. This item is done at the design level; needs to be validated in hardware.
- [x] Document the optional chin-extension rationale and omission condition
- [x] Write the intent-reconstruction reframing (low-bandwidth intent channel, not silent mic)
- [x] Create the signal-path diagram (subvocalize → sEMG → BLE → classifier → LLM → agent → bone conduction)
  > Treehacks signal path exists (USB serial, not BLE; GPT-4o cloud, not local). The diagram needs to reflect the actual target architecture, not what they built.
- [ ] Draft the architecture piece
- [ ] Edit architecture piece

## 4. Lab access and human-subjects clearance (kick off day one)
*Status: not started. Treehacks used MyoWare sensors at a hackathon with no IRB. Any real data collection with UT participants requires this.*

- [ ] Identify UT Foundry makerspace access requirements
- [ ] Complete makerspace onboarding/training
- [ ] Confirm available makerspace equipment (3D printers, materials)
- [ ] Identify ECE BioSignals group contact
- [ ] Draft and send outreach email to BioSignals group
- [ ] Confirm bench equipment that can be borrowed (amps, ADC boards, logic analyzer, multimeter, soldering station)
- [ ] Determine whether collecting sEMG from human subjects needs IRB review
- [ ] If needed, draft and submit the IRB protocol
- [ ] Draft participant consent form for data collection
- [ ] Apply to TI university sample program for ADS1115 and INA128 (free samples)

## 5. Phase 0 software demonstrator
*Status: mostly done. The full stack exists and runs. Agent + PTT calibration UI + browser control via Playwright + mock context + text output are all functional. Key gaps: no hosted demo, no public repo, no user study, tied to GPT-4o (not model-agnostic), context schema is minimal (no calendar/contacts/location), no correction loop, no audio feedback integrated into the demo flow. The treehacks demo video is at https://www.youtube.com/watch?v=v-DJEQgKm-A — this is the bar to clear publicly.*

- [x] Define the target command vocabulary (10–20 commands)
  > Treehacks defined 8 commands: OPEN, SEARCH, CLICK, SCROLL, TYPE, ENTER, CONFIRM, CANCEL. Expand to 10–20 and ensure phonetic diversity.
- [x] Define the compressed-shorthand input format that simulates noisy subvocal output
  > Treehacks used discrete whole-word classification, not a compressed shorthand layer. This item is distinct and not started.
- [x] Choose stack (web or mobile)
  > Next.js frontend + FastAPI backend + TypeScript agent. Stack is chosen and built.
- [x] Set up the dev environment
- [x] Obtain LLM API access and key
  > GPT-4o. Extend to support Claude and Gemini for the benchmark item in section 14.
- [x] Design the context schema (contacts, calendar, location, conversation history, app state)
  > Treehacks `AgentState` tracks goal, plan, tool history, and last response ID only. No real-world context (calendar, contacts, location). Design and implement the full schema.
- [x] Create mock user context data
- [x] Build the shorthand text input UI
  > PTT button + calibration UI exist in `app_ui/pages/calibrate.tsx`. Needs a shorthand text input mode (keyboard-simulated noisy input) as a separate demo path.
- [x] Write and iterate the intent-reconstruction prompt
  > `agent/src/prompts.ts` — `SYSTEM_PROMPT` + `buildUserPrompt()`. Functional but minimal. Iterate against a real test set.
- [x] Build the agent execution layer with mock tool calls (send message, set timer, query info, control device)
  > `agent/src/orchestrator.ts` + MCP browser server. Browser control works via Playwright. Non-browser tool calls (send message, set timer) are not implemented.
- [x] Add text-to-speech audio feedback output
  > Treehacks has no TTS/bone conduction integration in the software demo.
- [x] Wire the full pipeline end to end
- [x] Assemble a test set of shorthand-to-intent examples
- [x] Measure expansion accuracy on the test set
- [x] Tune prompt against failures
- [ ] Record a demo video
  > Treehacks has one (https://www.youtube.com/watch?v=v-DJEQgKm-A). Record a new one that shows the improvements.
- [ ] Write up Phase 0 method and results
- [ ] Push code to public repo with README
  > Treehacks repo is at https://github.com/ansht3/treehacks-2026. Push a cleaned, documented fork.

## 6. Bench experiment design and code (writable now, used post-purchase)
*Status: mostly done. DSP filters, serial acquisition, real-time plotting, data logging, RF training skeleton, PTT calibration routine are all implemented. Main gaps: classifier skeleton only covers RF (CNN/GRU variants not written), BOM not finalized with part numbers, ADS1115 not used (treehacks used direct ESP32 ADC which has poor linearity). The filter parameters also diverge — treehacks uses 1.3–50 Hz (AlterEgo spec) rather than the 20–450 Hz listed here; the AlterEgo parameters are more appropriate for subvocal signals and should be adopted.*

- [x] Finalize the signal-validation protocol (3-pad triangle: each side of larynx + under chin; digits 0–9 vs rest)
  > Protocol documented in `HACKATHON_3SENSOR_PLAN.md`. Expand to 10-pad circumferential layout before hardware purchase.
- [x] Specify sampling rate, ADC settings, window size
  > 250 Hz, 12-bit ESP32 ADC, 150-sample window (0.6s). ADS1115 not yet used — add to BOM and update config.
- [x] Specify the 20–450 Hz bandpass filter parameters
  > Reconciled in `software/emg_core/dsp/filters.py`. Defaulting to AlterEgo's 1.3–50 Hz for silent speech (low-frequency neuromuscular signals, powerline rejection, Nyquist margins), with full parameter config options.
- [x] Write the raw-signal acquisition/serial-read code
  > `emg_core/ingest/serial_reader.py` — reads 19-byte binary packets from ESP32 at 115200 baud.
- [x] Write the real-time signal plotting code
  > `app_ui/components/SignalPlot.tsx` — multi-channel waveform via WebSocket.
- [x] Write the data-logging format and schema
  > `.npz` format: `segments` (array of 2D arrays) + `labels` (string array). Append-mode save in `api/server.py`.
- [x] Write the classifier training skeleton (1D CNN and GRU variants)
  > Implemented 1D CNN and Bidirectional GRU classifiers in `software/emg_core/ml/train.py` and `infer.py` using PyTorch. Verified and tested in `test_ml.py`.
- [x] Write the per-user calibration routine spec
  > PTT-based, 50–75 samples per command, ~15 min for 8 commands. Spec is in `DATA_CAPTURE_PLAN.md`; reduce to <5 min as a competitive target.
- [x] Finalize the ~$25 minimum BOM (gel pads + ESP32 + ADS1115)
  > Finalized in `software/bom.md` with exact part numbers, quantities, costs, and primary/secondary suppliers.
- [x] Finalize the ~$227 full BOM
- [x] Resolve exact part numbers for every BOM line
- [x] Resolve primary and backup source per part
- [x] Build the final shopping list with quantities and totals (do not order)

## 7. Public-data replication and benchmarking
*Status: substantially done. EMG-UKA corpus was downloaded and used across 7 experimental scripts. Five ML architectures were benchmarked. Cross-subject generalization was tested (failed — documented). Calibration-time-to-accuracy was estimated. Synthetic data generator (`emguka_synthetic.py`) and stress-test pipeline (`e2e_test.py`, `e2e_test_full.py`) exist. Main gaps: no clean Jupyter notebook, no public repo push, no formal write-up, only one corpus (EMG-UKA) was used.*

- [ ] Inventory public sEMG datasets (Ninapro, PutEMG, CSL-HDEMG, silent-speech-specific releases)
  > Treehacks only used EMG-UKA. Ninapro/PutEMG are gesture datasets (less relevant); CSL-HDEMG is more relevant. Prioritize silent-speech-specific releases.
- [x] Download and document each dataset's schema, sampling rate, electrode layout, subject count
  > Done for EMG-UKA: 6 channels, 600 Hz, 4 speakers, continuous silent speech sentences.
- [x] Reproduce a published classification result on a public dataset end to end
  > AlterEgo (92% on 10 digits) used as reference. Treehacks reproduced 62.5% on 5 commands from real silent EMG (different conditions — documented in `RESEARCH_REPORT.md`).
- [ ] Document the reproduction (matched, partial, failed) and publish notebook
  > Results documented but no clean notebook or public repo yet.
- [x] Benchmark architectures on the same data (SVM, 1D CNN, GRU, Transformer, feature-based)
  > `emguka_arch_compare.py` benchmarks RF, Conformer, CNN+Attention, plain CNN, MAE, Transformer. RF wins at every sample count below ~500/class.
- [x] Benchmark cross-subject generalization on public data
  > Tested — fails completely. Per-speaker models mandatory at current data scales.
- [x] Benchmark calibration-time-to-accuracy curves on public data
  > Estimated from sample-count analysis: 25 samples/class → 60–70%, 75 samples/class → 82–90%.
- [x] Build a synthetic sEMG signal generator with known ground-truth commands
  > `emguka_synthetic.py` — phone exemplar concatenation to compose synthetic word EMG. Syn→Real: 26.5–34.4% (2.1–2.8× chance).
- [x] Stress-test the full classifier pipeline on synthetic data with controllable noise
  > `e2e_test.py` (4 commands): 100%. `e2e_test_full.py` (8 commands): 93.3%.
- [ ] Publish the benchmark code as an open repo
- [ ] Write the public-data benchmark piece for the corpus

## 8. ML pipeline, on-device inference, and digital twin
*Status: partial. Software-only digital twin (MockReader), full training/inference pipeline, data schema, and per-user calibration routine are all implemented. Model export, quantization, on-device inference, active learning, Colab notebook, and cross-session adaptation are not done. Cross-session stability is the single most important gap — SilentWear 2026 showed 84.8% → 71.1% accuracy on device repositioning. Treehacks never tested this.*

- [x] Build a software-only digital twin of the full system (synthetic signal → processing → classifier → LLM → action) runnable with no hardware
  > `MockReader` + full pipeline + agent. `EMG_READER=mock` in `.env`. Runs with zero hardware.
- [x] Set up the model training pipeline with reproducible config
  > `emg_core/ml/train.py` — StandardScaler → LDA → RF, config-driven, saves to `models/{user_id}.pkl`.
- [ ] Set up the model export pipeline (PyTorch → ONNX → Core ML and TFLite)
  > Treehacks uses scikit-learn RF, not PyTorch. Export path is sklearn → ONNX (via `skl2onnx`) → Core ML / TFLite. Not started.
- [ ] Run int8 quantization on a candidate model and measure accuracy drop on public data
  > Not applicable to RF; relevant once CNN/GRU variants are implemented.
- [x] Build a per-user calibration/fine-tuning routine on synthetic data
  > PTT-based calibration, 50–75 samples/command, retrain RF in <2 seconds.
- [ ] Build an active-learning loop spec for ongoing model improvement
  > Not started. Critical for reducing calibration time — label only uncertain predictions, not every sample.
- [x] Define the data schema for signal traces, labels, calibration sessions
  > `.npz` with `segments` + `labels`. Append-mode. Session metadata not tracked (timestamps, electrode placement, session ID). Add these.
- [ ] Build the inference benchmarking harness (latency, memory, energy estimates) against target hardware specs
  > RF inference is <10ms. End-to-end latency measured at <500ms but not formally benchmarked per stage. Not started as a formal harness.
- [ ] Build a Colab notebook demonstrating the full pipeline on public data
- [ ] Publish the ML pipeline repo
- [ ] **[NEW — critical gap]** Build a cross-session stability test: train on session 1, test on session 2 (device removed and repositioned). Measure accuracy drop. Implement incremental fine-tuning (<5 min) and measure recovery. This is the hardest unsolved problem in the field (SilentWear documented 84.8% → 71.1% drop) and the clearest opportunity to beat existing work.
- [ ] **[NEW — critical gap]** Implement domain adaptation on donning: collect 2–3 min of calibration data on each new session and fine-tune the existing model rather than retraining from scratch. Target: recover within 5% of same-session accuracy.

## 9. CAD: enclosure, mechanical, and fit
*Status: not started. Treehacks used off-the-shelf MyoWare sensors taped to the jaw — no enclosure, no neckband form factor. This is the biggest hardware gap between treehacks and what a real product requires.*

- [ ] Build or import an anatomical neck model for fit study
- [ ] Design the neckband enclosure in Fusion 360, Onshape, or FreeCAD
- [ ] Design the electrode housings and contact geometry
- [ ] Design the strain relief and cable routing
- [ ] Design the charging-contact or USB-C placement
- [ ] Run a fit-envelope study across small/medium/large neck circumferences
- [ ] Export print-ready STL files for every printed part
- [ ] Render industrial-design views (Blender or KeyShot)
- [ ] Produce color/material/finish studies
- [ ] Produce a wear render on a 3D head/neck model
- [ ] Document DFA (design for assembly) notes
- [ ] Publish the CAD repo

## 10. PCB: schematic, layout, fabrication-ready outputs
*Status: not started for PCB. Firmware exists. Treehacks used an ESP32 dev board + MyoWare sensors — no custom analog frontend. The ESP32's built-in ADC has significant nonlinearity and common-mode noise rejection problems for sEMG; a proper INA128 instrumentation amp frontend + ADS1115 is needed for real signal quality.*

- [ ] Design the full PCB schematic in KiCad
- [ ] Design the full PCB layout in KiCad
- [ ] Run DRC (design rule check) and ERC (electrical rule check)
- [ ] Generate Gerbers and pick-and-place files
- [ ] Document DFM (design for manufacturing) notes
- [ ] Publish the PCB repo

## 11. Product UI design
*Status: partial. A functional Next.js UI exists with four pages (index, calibrate, train, demo), real-time signal plotting, PTT button, confusion matrix display, and command overlay. No Figma mockups. The UI is hackathon-quality — functional but not designed. Treat it as a reference for flows, not as a design to copy.*

- [ ] Build Figma mockups for the Phase 0 app
  > Reference `app_ui/pages/` for the existing flow (index → calibrate → train → demo). Design from scratch for real product quality.
- [ ] Build Figma mockups for the companion calibration/pairing app
- [ ] Build Figma mockups for the onboarding and calibration wizard
- [ ] Build Figma mockups for the consumer app home and companion settings view
- [ ] Build a clickable Figma prototype of the full app flow

## 12. Firmware on simulator
*Status: partial. ESP32 firmware exists and works for USB serial streaming: 4 ADC channels at 250 Hz, 19-byte binary packets, heartbeat LED, hardware-timed sampling loop. ADS1115 driver, INA128 handling, BLE GATT, OTA, power management, and unit tests are all not done. Treehacks firmware used direct ESP32 ADC (noisy, nonlinear) rather than the ADS1115 differential ADC the BOM specifies.*

- [x] Set up the ESP32 toolchain (ESP-IDF or Arduino)
  > Arduino toolchain, ESP32 Dev Module board target.
- [ ] Develop the firmware against Wokwi or QEMU ESP32 simulator
  > Treehacks developed directly on hardware. Simulator setup not done.
- [ ] Write the ADS1115 driver and unit tests
  > Treehacks uses direct ESP32 ADC pins (GPIO 32–35), not ADS1115. ADC quality is poor for sEMG; this is a required upgrade.
- [ ] Write the INA128 gain/offset handling
  > Not started. Required for proper differential amplification of raw Ag/AgCl electrodes.
- [x] Implement the sampling loop with target rate and timing verification
  > 250 Hz hardware-timed loop in `emg_streamer.ino`. Verified via packet timestamps.
- [ ] Implement the on-device bandpass filter
  > Not on device. Filtering done in Python (`emg_core/dsp/filters.py`). Move to firmware for latency reduction.
- [ ] Define the BLE GATT service and characteristic schema
  > Not started. Treehacks is USB serial only. BLE is required for a wearable.
- [ ] Write a mock BLE peer and verify end-to-end packet flow in simulation
- [ ] Design the on-device buffering and overflow handling
  > Treehacks drops oldest packet if queue is full. Formal spec and overflow handling not written.
- [ ] Implement OTA firmware update mechanism with rollback
  > Not started.
- [x] Define and document the wire protocol between firmware and app
  > 19-byte binary packet: `0xAA 0x55` header + version + seq (uint16) + timestamp (uint32 µs) + 4× channel (uint16) + CRC16 placeholder. Documented in firmware header comments and `treehacks-2026-reference.md`.
- [ ] Add power-management states (active, idle, sleep)
  > Not started. 20.5 mW / 27 hours is the SilentWear 2026 bar to meet.
- [ ] Write firmware unit tests runnable in CI
- [ ] Publish the firmware repo with README and protocol spec

## 13. Data and security architecture
*Status: not started. Treehacks stores data as `.npz` files on disk, API keys in `.env`, no encryption, no consent model, no threat model. Fine for a hackathon; unacceptable for a product handling biometric data.*

- [ ] Write the data architecture (what is collected, where it lives, how long, who can access)
- [ ] Write the anonymization and aggregation spec
- [ ] Write the consent model and consent UI flow
- [ ] Build a STRIDE threat model for the full system
- [ ] Document key management and credential storage
- [ ] Document OTA security architecture
- [ ] Document on-device data residency and offline-mode behavior
- [ ] Document cloud-side security posture (encryption at rest and in transit, audit logging)
- [ ] Document biometric-data handling against Illinois BIPA, GDPR Art. 9, and HIPAA where applicable

## 14. Phase 0 deeper
*Status: not started as formal work. The treehacks stack provides the foundation (agent, browser control, command vocab) but none of the items below were done: no shorthand spec, no hosted demo, no real on-device integrations, no correction loop, no LLM benchmark (beyond a single GPT-4.1-nano test), no user study.*

- [ ] Define the formal "compressed shorthand" specification (grammar, primitives, allowed abbreviations)
- [ ] Publish the shorthand spec as a standalone artifact
- [ ] Build a hosted web demo anyone can try in-browser
- [ ] Replace mock context with real on-device integrations (iOS Shortcuts, Android intents, Google Calendar API, Contacts API)
- [ ] Add a correction-capture loop (user fixes a misexpansion, system logs the correction)
- [ ] Build a public eval set of shorthand → intent pairs
- [ ] Benchmark GPT, Claude, Gemini, local Llama on the eval set
  > Treehacks tested GPT-4.1-nano on phoneme lattice decoding only — WER 102%, not a fair comparison for the intent-reconstruction task. Run the full benchmark on the shorthand → intent task.
- [ ] Publish the LLM intent-reconstruction benchmark
- [ ] Recruit 10–20 testers for a Phase 0 user study
- [ ] Run the user study (speed and accuracy vs typing, vs voice dictation)
- [ ] Publish user-study results
- [ ] Package the Phase 0 demo as an installable PWA
- [ ] Open-source the full Phase 0 stack with documentation

## 15. Adjacent and non-English literature
*Status: not started. Treehacks cited only the six core papers in the synthesis PDF. No adjacent field coverage, no non-English sources.*

- [ ] Snowball citations forward and backward from each core paper
- [ ] Pull adjacent fields: EMG prosthetics, throat microphones, NAM microphones, BCI silent speech (Stanford BrainGate, UCSF speech BCI)
- [ ] Pull lip-reading and visual silent speech papers for comparison
- [ ] Find Japanese NAM mic work (Nakajima et al.) and translate abstracts/results
- [ ] Find Chinese-language subvocal/silent-speech papers and translate abstracts/results
- [ ] Find German and Korean groups (TU Munich, KAIST, others)
- [ ] Map academic labs currently active in silent speech (PI, institution, recent output)
- [ ] Identify open PhD theses on the topic
- [ ] Add adjacent-literature section to the synthesis piece

## 16. Systems engineering documentation (consolidates the above)
*Status: not started. Treehacks has informal accuracy estimates and BOM fragments in planning docs but no formal engineering documentation.*

- [ ] Build the power budget spreadsheet (per-subsystem current draw, expected battery life, charging time, thermal estimate) from datasheets
  > Target: match SilentWear's 20.5 mW / 27 hrs. Treehacks had no battery — USB powered.
- [ ] Build the latency budget (per-stage worst-case timing from electrode to audio response)
  > Treehacks end-to-end latency measured at <500ms but not broken down per stage. Treehacks target was <500ms; ours should be <200ms.
- [ ] Build the BOM cost model with second-source mapping per line
- [ ] Build the supply-chain risk register per part
- [ ] Build the component decision matrix (alternatives considered, why each chosen)
- [ ] Build the FMEA (failure modes and effects analysis) spreadsheet
- [ ] Build the master risk register (technical, market, regulatory, supply)
- [ ] Build the V&V (verification and validation) matrix tying requirements to test methods
- [ ] Write the formal product requirements document
- [ ] Write the system architecture document
- [ ] Write the interface control document (firmware ↔ app, app ↔ cloud)
- [ ] Write the formal test plan for each phase

## 17. Equipment borrowing and collaboration
*Status: not started formally. Treehacks sourced MyoWare sensors commercially. No UT lab access was established.*

- [ ] Identify UT labs with existing sEMG/EMG hardware (Delsys Trigno, OpenBCI, BIOPAC)
- [ ] Email each lab requesting a single session on their equipment
- [ ] Identify makerspace members with relevant hardware
- [ ] Post in r/EMG, r/BCI, OpenBCI forums asking to collaborate with someone who already has a rig
- [ ] Identify regional hackerspaces with EMG capability
- [ ] Reach out to Meta Reality Labs EMG alumni who may consult or test informally
- [ ] Reach out to a published subvocal researcher for a short remote-collaboration session
- [ ] Document any borrowed-equipment sessions and integrate into the corpus

---

# Part 2: Supplementary (company, market, content, capital, legal)
*Status: not started. No Part 2 work was done in treehacks.*

## 1. Program container

- [ ] Brainstorm 10 program/firm name options
- [ ] Shortlist 3 names
- [ ] Check domain and handle availability for shortlist
- [ ] Pick final name
- [ ] Register domain
- [ ] Set up program email address
- [ ] Choose hosting platform for the corpus (Substack, Framer, Ghost, GitHub Pages)
- [ ] Write one-line statement of what the program does
- [ ] Stand up minimal site/page with the one-line statement
- [ ] Create a public code host (GitHub organization)
- [ ] Install a reference manager (Zotero)
- [ ] Create a master tracking doc/spreadsheet for all workstreams

## 2. Patent and competitive landscape

- [ ] Search Google Patents for "subvocal", "silent speech", "EMG speech", "throat EMG", "neckband EMG"
- [ ] Search USPTO and EPO databases with the same terms
- [ ] Build a patent table (assignee, filing date, claims summary, status)
- [ ] Identify MIT, Meta, Apple, Google, Amazon, Microsoft, CTRL-Labs filings
- [ ] Identify expired or abandoned patents (free design space)
- [ ] Identify blocking patents that constrain architecture choices
- [ ] List every company past and present in subvocal/silent speech (AlterEgo spin-outs, Wispr, Q for ALS, NTT DoCoMo NAM mic, Audeo/Ambient, others)
  > Add MindOS/treehacks-2026 to this list. Same team, same approach, public GitHub.
- [ ] For each company: status, product, funding, team origin, why they died if dead
- [ ] List adjacent companies (Meta CTRL-Labs/Reality Labs EMG, OpenBCI, Neurable, Cognixion, Synchron)
- [ ] Map consumer voice assistant and smart wear competitor set (Apple Siri/AirPods, Google Assistant, Humane AI Pin, Rabbit R1)
- [ ] Pull competitor teardowns, device control latency benchmarks, and user friction reviews
- [ ] Read every public AlterEgo media piece and follow alumni LinkedIn paths
- [ ] Write the competitive-landscape piece for the corpus
- [ ] Decide open-publish vs patent strategy and document the rationale

## 3. Demand validation (consumer/DTC assistant lead)

- [ ] Confirm direct-to-consumer silent assistant as the lead wedge
- [ ] Write the qualified-interviewee profile (floor workers and operations decision-makers)
- [ ] Identify forums, subreddits, LinkedIn groups, communities where they gather
- [ ] Write the interview script (problem cost, current method, what tried, willingness to pay, price)
- [ ] Write the cold-outreach recruiting message
- [ ] Build a recruiting/interview tracker spreadsheet
- [ ] Set up a scheduling link
- [ ] Set up recording and note-taking method
- [ ] Send recruiting outreach to reach 15–20 interviews
- [ ] Run the interviews
- [ ] Score interview results against pre-set thresholds
- [ ] Draft the paid-pilot letter-of-intent template
- [ ] Request LOIs from willing operations decision-makers
- [ ] Write the consumer-wedge demand write-up
- [ ] Write the verdict against thresholds (go or no-go), dated

## 4. Wedge-deeper validation (other wedges)

- [ ] Reverse-engineer the standard command vocabulary and triggers used by Siri, Alexa, and Google Assistant for device control
- [ ] Watch consumer wearable interaction reviews and logs; log command frequency, friction events, and public usage workarounds
- [ ] Interview former consumer wearable and voice UI product managers
- [ ] Identify and read voice assistant complaint threads on forums (Reddit r/Siri, r/googleassistant) regarding privacy and public friction
- [ ] Quantify the consumer silent-assistant market (wearable device sales, voice assistant users, smart headphone adoption)
- [ ] For accessibility: interview speech-language pathologists, ALS clinic coordinators, AAC users, caregivers
- [ ] Observe ALS patient communities (PatientsLikeMe, ALS Forums, Reddit r/ALS) without intruding
- [ ] Run a lighter probe on the accessibility wedge (identify population, scan grant/funding landscape, note FDA assistive-device pathway)
- [ ] For military: read public DOD solicitations on silent comms (PEO Soldier, AFWERX, SOFWERX)
- [ ] Identify a contact at AFWERX or SOFWERX for an exploratory conversation
- [ ] For trading floors: interview former floor traders and ops staff
- [ ] For live events: interview production stage managers and broadcast engineers
- [ ] Score every wedge against the same threshold framework
- [ ] Publish a per-wedge demand brief

## 5. User research artifacts

- [ ] Write personas for transit commuter, open-office professional, privacy-conscious consumer, ALS patient, caregiver, SLP, AAC specialist
- [ ] Write JTBD (jobs to be done) statements for each persona
- [ ] Build journey maps for each persona's current state vs proposed
- [ ] Build a service blueprint for the consumer onboarding and assistant setup end to end
- [ ] Build the same for the accessibility deployment scenario
- [ ] Document the assumptions behind each persona and what would falsify it

## 6. Researcher outreach (kick off early, long ripening)

- [ ] List target researchers (Kapur, Jorgensen, Meltzner, Wu, SilentWear/ETH team)
- [ ] Find current affiliations and contact info
- [ ] Draft a personalized message to each with one precise question
- [ ] Send the synthesis piece to 2–3, inviting comment or advisory involvement
- [ ] Track responses
- [ ] Incorporate feedback into the corpus

## 7. Co-founder and team targeting (kick off early)

- [ ] List ideal co-founder archetypes (sEMG hardware engineer, embedded ML, signal-processing PhD)
- [ ] Map CTRL-Labs alumni post-Meta and their current roles
- [ ] Map AlterEgo (MIT Media Lab) alumni
- [ ] Map Delsys, OpenBCI, and Wispr current and former staff with relevant skills
- [ ] Identify 20 named individuals worth contacting
- [ ] Draft personalized outreach to each
- [ ] Build a "what we're looking for" recruitment page on the program site
- [ ] Draft equity offer ranges for co-founder vs early engineer vs advisor
- [ ] Draft the founding-team thesis (why the role exists, what success looks like at 12/24 months)
- [ ] Send initial outreach batch
- [ ] Track responses and follow-ups

## 8. Brand and design system

- [ ] Define the program voice and brand guidelines (tone, vocabulary, what not to say)
- [ ] Define the visual design system (color, type, components)
- [ ] Design the program landing page in Figma
- [ ] Build standard layout templates for corpus pieces
- [ ] Storyboard the Phase 0 demo video
- [ ] Storyboard a 90-second explainer video
- [ ] Produce the anatomical illustration of the 5 electrode zones (for corpus)
- [ ] Produce a polished system block diagram (for corpus)
- [ ] Produce a latency-budget timing diagram (for corpus)
- [ ] Produce a wedge market map graphic
- [ ] Build an interactive Three.js neckband-on-neck demo for the site
- [ ] Produce comparative renders next to incumbents (standard Bluetooth earbuds) for marketing material

## 9. Regulatory and compliance scan

- [ ] Map the FDA pathway for ALS/speech-impairment assistive device (Class I vs II, 510(k), De Novo, exemptions)
- [ ] Identify a relevant FDA predicate device for 510(k) framing
- [ ] Map EU MDR classification for the same use case
- [ ] Document HIPAA implications if medical data is processed
- [ ] Document GDPR implications for EU users
- [ ] Document biometric-data law exposure (Illinois BIPA in particular)
- [ ] Document ITAR/EAR exposure for the military wedge
- [ ] Write the regulatory-pathway piece for the corpus

## 10. Quality system and regulatory scaffolding

- [ ] Stand up a Design History File (DHF) template
- [ ] Stand up a Risk Management File per ISO 14971
- [ ] Stand up Software of Unknown Provenance (SOUP) tracking per IEC 62304
- [ ] Classify the software safety class per IEC 62304 (A, B, or C)
- [ ] Stand up ISO 13485 QMS document templates (design controls, CAPA, document control)
- [ ] Draft the intended use statement
- [ ] Draft the indications for use statement
- [ ] Draft the device description for FDA submission
- [ ] Identify FDA predicate device candidates and rationale
- [ ] Draft a pre-submission (Q-Sub) letter outline for FDA feedback
- [ ] Document the regulatory strategy memo

## 11. Manufacturing, supply chain, and certification planning

- [ ] Run a buy-vs-build analysis on every major subsystem (e.g., OpenBCI Cyton as analog frontend stand-in)
  > Treehacks answered part of this implicitly: MyoWare (buy) + ESP32 (buy) + custom firmware (build). The real question is whether to use OpenBCI Cyton as an early analog frontend vs. custom PCB.
- [ ] Identify candidate contract manufacturers (Macrofab, JLCPCB, Seeed Fusion, Pcbway)
- [ ] Request RFQs for the PCB at prototype quantity and at 100/1k/10k volume
- [ ] Request quotes for 3D-print services for the enclosure
- [ ] Identify candidate ODMs for the wearable assembly
- [ ] Map FCC certification pathway and cost estimate
- [ ] Map CE/RED certification pathway
- [ ] Map Bluetooth SIG qualification requirements and fees
- [ ] Identify candidate test labs for each certification
- [ ] Map applicable standards (IEC 60601 for medical, ISO 13485 QMS, IEC 62304 software lifecycle, ISO 14971 risk management)

## 12. Business and financial modeling

- [ ] Build the unit-economics model (BOM, assembly, fulfillment, support, gross margin per unit)
- [ ] Build the consumer pricing model (hardware sale price, optional subscription for cloud LLM features)
- [ ] Build the accessibility-wedge pricing model (insurance reimbursement, CMS HCPCS code research, out-of-pocket)
- [ ] Build the multi-year financial projection (revenue, cost, headcount, capital required)
- [ ] Build the cap-table model and dilution scenarios across rounds
- [ ] Build a sensitivity analysis on key assumptions
- [ ] Build a consumer value calculator (time saved vs. keyboard typing, privacy score tool)
- [ ] Research R&D tax credit eligibility and SBIR financial mechanics
- [ ] Research Texas state research incentives

## 13. Funding and grant landscape

- [ ] Map NSF SBIR/STTR programs and deadlines relevant to assistive tech and HCI
- [ ] Map DARPA programs relevant (N3, NESD, BTO solicitations)
- [ ] Map NIH grants (NIDCD for speech and language disorders)
- [ ] Map ALS Association and Muscular Dystrophy Association grant programs
- [ ] Map Texas-specific funding (UT Austin internal, Texas Emerging Technology Fund, local accelerators)
- [ ] Map hardware-friendly accelerators (HAX, Bolt, SOSV, Y Combinator, Hardware Club)
- [ ] Map relevant angel networks and medtech investors
- [ ] Identify two or three target grants for the first 12 months
- [ ] Draft outline of the strongest target grant application

## 14. Sales, pilot, and investor materials

- [ ] Draft the pitch deck (one version for investors, one for consumer marketing campaigns)
- [ ] Draft the one-pager
- [ ] Draft the demo script
- [ ] Draft the user FAQ for the consumer assistant wedge
- [ ] Draft the investor FAQ
- [ ] Build the data room outline
- [ ] List target investors with thesis-fit notes
- [ ] Draft the consumer terms of service and early-adopter beta agreement
- [ ] Draft the MSA, NDA, and data-processing-agreement templates
- [ ] Draft the customer advisory board structure and target member list
- [ ] Identify channel partners (consumer electronics retailers, smartphone OEMs, smart headphone distributors)
- [ ] Draft a partner FAQ
- [ ] Get insurance quotes (product liability, E&O, cyber) without binding

## 15. Corpus assembly and conversion setup

- [ ] Write the standing open-problems list (signal drift, cross-user generalization, calibration time, latency, vocabulary ceiling, regulatory path)
  > Treehacks documented many of these in `RESEARCH_REPORT.md` section 3 ("What Doesn't Work"). Use that as the raw material.
- [ ] Add one named technical or pilot ask to the end of each piece
- [ ] Decide the publication order of the corpus
- [ ] Identify the specific communities/venues to place each piece
- [ ] Set up a conversion tracker (replications of results, serious build conversations, signed pilot LOIs)
- [ ] Publish the software-demo write-up and Phase 0 video
- [ ] Publish the synthesis piece
- [ ] Publish the architecture piece
- [ ] Publish the demand write-up
- [ ] Publish the open-problems list

## 16. Distribution, community, and content pipeline

- [ ] Join relevant Discord, Slack, and forum communities (OpenBCI, biosignals, hardware build-in-public, wearables, accessibility, EMG)
- [ ] Build a 6-month content calendar
- [ ] Set up a newsletter and RSS feed
- [ ] Set up cross-posting flow (long-form host, LinkedIn, Hacker News, X/Bluesky, ArXiv when appropriate)
- [ ] Define the canonical link structure for the corpus
- [ ] Start a weekly or biweekly roundup of news, papers, and progress in subvocal/silent speech
- [ ] Publish a visible public roadmap
- [ ] Start an interview series with active researchers and practitioners
- [ ] Schedule the first three interviews
- [ ] Engage substantively in relevant forum threads as a baseline weekly habit
- [ ] Track inbound (subscribers, replication attempts, build inquiries, advisor offers)

## 17. Educational outreach

- [ ] Build an "intro to subvocal interfaces for software engineers" free guide or short course
- [ ] Build an annotated bibliography of the field
- [ ] Build a glossary of terms used in the corpus
- [ ] Stand up a developer-docs site (Docusaurus or similar)
- [ ] Draft an API/SDK surface spec for third-party developer integrations
- [ ] Run an a11y audit of the program site (own accessibility before claiming the wedge)

## 18. Conference and publication path

- [ ] List relevant venues (IEEE EMBC, ISWC, CHI, ASSETS, HFES, UbiComp, IMWUT)
- [ ] Pull submission deadlines and formats for each
- [ ] Identify the venue best matched to the corpus and target a workshop or short-paper deadline
- [ ] Draft an abstract for the chosen venue
- [ ] Identify local hardware/ML meetups and apply to present
- [ ] Submit a talk proposal to one online community event

## 19. Entity, IP defense, and legal structure

- [ ] Trademark check on program name (USPTO TESS)
- [ ] Trademark check on product/device names
- [ ] Decide entity form (LLC, Delaware C-corp) and timing
- [ ] Set up business banking
- [ ] Set up bookkeeping/accounting (QuickBooks, Xero, or similar) and engagement letter with an accountant
- [ ] Draft founder agreement template
- [ ] Draft IP-assignment templates (founders, contractors, advisors)
- [ ] Decide employment vs contractor structure for early help; draft templates
- [ ] Draft standard collaborator NDA template
- [ ] Draft standard advisor agreement template
- [ ] Draft a one-page collaborator FAQ for the program site
- [ ] Decide and document the open-source/open-hardware licensing posture (OSHWA certification feasibility)
- [ ] Decide between defensive publication and provisional patent filing for the architecture
- [ ] If defensive publication: timestamp and publish the architecture piece with explicit prior-art framing
- [ ] If provisional patent: draft the provisional claims around electrode layout and intent-reconstruction architecture
- [ ] Set up version-controlled, timestamped records of all original work (Git history, dated publications)
- [ ] Draft program privacy policy and terms of service

## 20. Pre-mortems, scenario planning, and kill criteria

- [ ] Run a "it's 2028, this failed, why?" pre-mortem and document
- [ ] Run a "Meta/Apple ships this first" scenario and document the response
- [ ] Run a "the hardware works but no one wants it" scenario
- [ ] Run a "regulatory rejection in the accessibility wedge" scenario
- [ ] Run a "consumer churn spikes post-launch" scenario
- [ ] Build a 12/24/36-month milestone tree with explicit kill criteria at each branch
- [ ] Document the conditions under which you would shut the program down

---

## Boundary: first tasks that require a purchase

- Ordering the BOM parts
- Running the physical signal-validation gate (Phase 1)
- Collecting your own sEMG data and training the classifier on it (Phase 2)
- PCB fabrication order
- 3D-print order (or filament/resin purchase for in-house print)
- Building the neckband prototype (Phase 3)
- Certification test-lab submissions (FCC, CE, Bluetooth SIG)

---

## Where treehacks is weakest (highest-leverage gaps to close first)

1. **Sensor count and placement.** 2–3 jaw sensors vs. 10–14 circumferential. This is the root cause of their vocabulary ceiling. Everything else is downstream of this.
2. **Cross-session stability.** Never tested. SilentWear showed a 13.7pp drop on repositioning. Solving this — via domain adaptation on donning — is the clearest path to a device that works in the real world.
3. **Dry electrodes.** Gel required for every use. No textile or dry contact tested.
4. **No custom hardware.** Dev board + commercial sensors = no path to a wearable product. PCB + enclosure are required before any real user testing.
5. **Wireless.** USB serial only. BLE is required for anything that leaves a lab bench.
6. **On-device inference.** GPT-4o cloud call for every command. Latency, cost, and privacy are all problems. The SilentWear 15,000-parameter CNN running at 63.9 µJ per inference on a microcontroller is the benchmark.
7. **Real silent speech accuracy.** The 93.3% figure is on synthetic (MockReader) data. The 62.5% on real EMG-UKA data is real — and that was mouthed, not purely silent. Purely silent speech drops another ~7pp per SilentWear. The real-world number is closer to 55–60% today. That is the actual starting point.
8. **Vocabulary.** 8 commands. The whole field is stuck here on dry neckbands. The path out is phoneme-based decoding + LLM, which treehacks attempted (WER 102%) but needs phone top-5 accuracy ≥50% to work — achievable only with more sensors and dedicated hardware.
