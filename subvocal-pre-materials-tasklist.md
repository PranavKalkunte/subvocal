# Subvocal venture: pre-materials task list

Two sections: product engineering and design, then everything supplementary (company, market, content, capital, legal). Within each, ordered by dependency and lead time. Long-lead-time items (lab access, researcher outreach, co-founder targeting) appear early even when downstream work uses them later. Items within a section are not strictly serial; many should run in parallel once their prerequisites land.

---

# Part 1: Product

## 1. Thesis and thresholds

- [ ] Draft the one-page thesis: structural shift, incumbent exposure, opening, why now, single product hypothesis
- [ ] Rewrite the thesis to the narrow, falsifiable version (remove "post-screen paradigm" framing)
- [ ] List the riskiest assumptions (technical and market)
- [ ] Write the go/no-go demand thresholds before any evidence is gathered
- [ ] List the risks that cannot be retired pre-product (capital, manufacturing, regulatory, hardware accuracy)
- [ ] Lock the thesis document, dated

## 2. Literature synthesis

- [ ] Obtain Kapur 2018 AlterEgo MIT thesis
- [ ] Obtain Jorgensen 2004 NASA Ames technical reports
- [ ] Obtain Meltzner et al. 2018 Delsys/NIH paper
- [ ] Obtain the 120-electrode high-density mapping study
- [ ] Obtain Wu et al. 2024 necklace form-factor paper
- [ ] Obtain SilentWear 2026 ETH Zurich paper
- [ ] Extract per paper: vocabulary size, accuracy/WER, electrode count, placement, form factor, dry vs gel, silent vs vocalized, calibration time, limitations
- [ ] Build the comparison table across all sources
- [ ] Mark points of convergence (anterior neck, infrahyoid SNR, neckband viability)
- [ ] Mark disagreements and gaps
- [ ] Draft the synthesis piece
- [ ] Verify every numeric claim against its source
- [ ] Add citations from Zotero
- [ ] Final edit synthesis

## 3. Architecture

- [ ] Write the neckband-vs-earbud reasoning from muscle anatomy
- [ ] Document the 5-zone electrode layout (suprahyoid/digastric, mylohyoid, thyrohyoid, larynx flanks, infrahyoid)
- [ ] Document the optional chin-extension rationale and omission condition
- [ ] Write the intent-reconstruction reframing (low-bandwidth intent channel, not silent mic)
- [ ] Create the signal-path diagram (subvocalize → sEMG → BLE → classifier → LLM → agent → bone conduction)
- [ ] Draft the architecture piece
- [ ] Edit architecture piece

## 4. Lab access and human-subjects clearance (kick off day one)

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

- [ ] Define the target command vocabulary (10–20 commands)
- [ ] Define the compressed-shorthand input format that simulates noisy subvocal output
- [ ] Choose stack (web or mobile)
- [ ] Set up the dev environment
- [ ] Obtain LLM API access and key
- [ ] Design the context schema (contacts, calendar, location, conversation history, app state)
- [ ] Create mock user context data
- [ ] Build the shorthand text input UI
- [ ] Write and iterate the intent-reconstruction prompt
- [ ] Build the agent execution layer with mock tool calls (send message, set timer, query info, control device)
- [ ] Add text-to-speech audio feedback output
- [ ] Wire the full pipeline end to end
- [ ] Assemble a test set of shorthand-to-intent examples
- [ ] Measure expansion accuracy on the test set
- [ ] Tune prompt against failures
- [ ] Record a demo video
- [ ] Write up Phase 0 method and results
- [ ] Push code to public repo with README

## 6. Bench experiment design and code (writable now, used post-purchase)

- [ ] Finalize the signal-validation protocol (3-pad triangle: each side of larynx + under chin; digits 0–9 vs rest)
- [ ] Specify sampling rate, ADC settings, window size
- [ ] Specify the 20–450 Hz bandpass filter parameters
- [ ] Write the raw-signal acquisition/serial-read code
- [ ] Write the real-time signal plotting code
- [ ] Write the data-logging format and schema
- [ ] Write the classifier training skeleton (1D CNN and GRU variants)
- [ ] Write the per-user calibration routine spec
- [ ] Finalize the ~$25 minimum BOM (gel pads + ESP32 + ADS1115)
- [ ] Finalize the ~$227 full BOM
- [ ] Resolve exact part numbers for every BOM line
- [ ] Resolve primary and backup source per part
- [ ] Build the final shopping list with quantities and totals (do not order)

## 7. Public-data replication and benchmarking

- [ ] Inventory public sEMG datasets (Ninapro, PutEMG, CSL-HDEMG, silent-speech-specific releases)
- [ ] Download and document each dataset's schema, sampling rate, electrode layout, subject count
- [ ] Reproduce a published classification result on a public dataset end to end
- [ ] Document the reproduction (matched, partial, failed) and publish notebook
- [ ] Benchmark architectures on the same data (SVM, 1D CNN, GRU, Transformer, feature-based)
- [ ] Benchmark cross-subject generalization on public data
- [ ] Benchmark calibration-time-to-accuracy curves on public data
- [ ] Build a synthetic sEMG signal generator with known ground-truth commands
- [ ] Stress-test the full classifier pipeline on synthetic data with controllable noise
- [ ] Publish the benchmark code as an open repo
- [ ] Write the public-data benchmark piece for the corpus

## 8. ML pipeline, on-device inference, and digital twin

- [ ] Build a software-only digital twin of the full system (synthetic signal → processing → classifier → LLM → action) runnable with no hardware
- [ ] Set up the model training pipeline with reproducible config
- [ ] Set up the model export pipeline (PyTorch → ONNX → Core ML and TFLite)
- [ ] Run int8 quantization on a candidate model and measure accuracy drop on public data
- [ ] Build a per-user calibration/fine-tuning routine on synthetic data
- [ ] Build an active-learning loop spec for ongoing model improvement
- [ ] Define the data schema for signal traces, labels, calibration sessions
- [ ] Build the inference benchmarking harness (latency, memory, energy estimates) against target hardware specs
- [ ] Build a Colab notebook demonstrating the full pipeline on public data
- [ ] Publish the ML pipeline repo

## 9. CAD: enclosure, mechanical, and fit

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

- [ ] Design the full PCB schematic in KiCad
- [ ] Design the full PCB layout in KiCad
- [ ] Run DRC (design rule check) and ERC (electrical rule check)
- [ ] Generate Gerbers and pick-and-place files
- [ ] Document DFM (design for manufacturing) notes
- [ ] Publish the PCB repo

## 11. Product UI design

- [ ] Build Figma mockups for the Phase 0 app
- [ ] Build Figma mockups for the companion calibration/pairing app
- [ ] Build Figma mockups for the onboarding and calibration wizard
- [ ] Build Figma mockups for the warehouse-ops admin/fleet view
- [ ] Build a clickable Figma prototype of the full app flow

## 12. Firmware on simulator

- [ ] Set up the ESP32 toolchain (ESP-IDF or Arduino)
- [ ] Develop the firmware against Wokwi or QEMU ESP32 simulator
- [ ] Write the ADS1115 driver and unit tests
- [ ] Write the INA128 gain/offset handling
- [ ] Implement the sampling loop with target rate and timing verification
- [ ] Implement the on-device bandpass filter
- [ ] Define the BLE GATT service and characteristic schema
- [ ] Write a mock BLE peer and verify end-to-end packet flow in simulation
- [ ] Design the on-device buffering and overflow handling
- [ ] Implement OTA firmware update mechanism with rollback
- [ ] Define and document the wire protocol between firmware and app
- [ ] Add power-management states (active, idle, sleep)
- [ ] Write firmware unit tests runnable in CI
- [ ] Publish the firmware repo with README and protocol spec

## 13. Data and security architecture

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

- [ ] Define the formal "compressed shorthand" specification (grammar, primitives, allowed abbreviations)
- [ ] Publish the shorthand spec as a standalone artifact
- [ ] Build a hosted web demo anyone can try in-browser
- [ ] Replace mock context with real on-device integrations (iOS Shortcuts, Android intents, Google Calendar API, Contacts API)
- [ ] Add a correction-capture loop (user fixes a misexpansion, system logs the correction)
- [ ] Build a public eval set of shorthand → intent pairs
- [ ] Benchmark GPT, Claude, Gemini, local Llama on the eval set
- [ ] Publish the LLM intent-reconstruction benchmark
- [ ] Recruit 10–20 testers for a Phase 0 user study
- [ ] Run the user study (speed and accuracy vs typing, vs voice dictation)
- [ ] Publish user-study results
- [ ] Package the Phase 0 demo as an installable PWA
- [ ] Open-source the full Phase 0 stack with documentation

## 15. Adjacent and non-English literature

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

- [ ] Build the power budget spreadsheet (per-subsystem current draw, expected battery life, charging time, thermal estimate) from datasheets
- [ ] Build the latency budget (per-stage worst-case timing from electrode to audio response)
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
- [ ] For each company: status, product, funding, team origin, why they died if dead
- [ ] List adjacent companies (Meta CTRL-Labs/Reality Labs EMG, OpenBCI, Neurable, Cognixion, Synchron)
- [ ] Map warehouse voice-tech competitor set (Honeywell Vocollect, Zebra, Lucas Systems, Körber, Ivanti Wavelink)
- [ ] Pull Vocollect and Honeywell case studies, pricing pages, partner directories
- [ ] Read every public AlterEgo media piece and follow alumni LinkedIn paths
- [ ] Write the competitive-landscape piece for the corpus
- [ ] Decide open-publish vs patent strategy and document the rationale

## 3. Demand validation (warehouse lead)

- [ ] Confirm warehouse/logistics as the lead wedge
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
- [ ] Write the warehouse-wedge demand write-up
- [ ] Write the verdict against thresholds (go or no-go), dated

## 4. Wedge-deeper validation (other wedges)

- [ ] Reverse-engineer the actual command vocabulary used by Vocollect VoiceClient from public training videos and manuals
- [ ] Watch warehouse voice-picking shift footage on YouTube; log command frequency, friction events, worker workarounds
- [ ] Interview former Vocollect/Honeywell employees via LinkedIn
- [ ] Identify and read warehouse-worker complaint threads (Reddit, Indeed reviews)
- [ ] Quantify the warehouse market (sites, seats, current spend on voice-picking)
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

- [ ] Write personas for warehouse picker, operations manager, IT buyer, ALS patient, caregiver, SLP, AAC specialist
- [ ] Write JTBD (jobs to be done) statements for each persona
- [ ] Build journey maps for each persona's current state vs proposed
- [ ] Build a service blueprint for the warehouse pilot end to end
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
- [ ] Produce comparative renders next to incumbents (Vocollect headset) for sales material

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
- [ ] Build the warehouse-pilot pricing model (per-seat, per-site, hardware vs subscription)
- [ ] Build the accessibility-wedge pricing model (insurance reimbursement, CMS HCPCS code research, out-of-pocket)
- [ ] Build the multi-year financial projection (revenue, cost, headcount, capital required)
- [ ] Build the cap-table model and dilution scenarios across rounds
- [ ] Build a sensitivity analysis on key assumptions
- [ ] Build the warehouse-pilot ROI calculator (publishable as a lead-gen tool)
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

- [ ] Draft the pitch deck (one version for investors, one for warehouse buyers)
- [ ] Draft the one-pager
- [ ] Draft the demo script
- [ ] Draft the buyer FAQ for the warehouse wedge
- [ ] Draft the investor FAQ
- [ ] Build the data room outline
- [ ] List target investors with thesis-fit notes
- [ ] Draft the warehouse pilot agreement template
- [ ] Draft the MSA, NDA, and data-processing-agreement templates
- [ ] Draft the customer advisory board structure and target member list
- [ ] Identify channel partners (warehouse system integrators, voice-tech consultants)
- [ ] Draft a partner FAQ
- [ ] Get insurance quotes (product liability, E&O, cyber) without binding

## 15. Corpus assembly and conversion setup

- [ ] Write the standing open-problems list (signal drift, cross-user generalization, calibration time, latency, vocabulary ceiling, regulatory path)
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
- [ ] Run a "warehouse pilot fails to renew" scenario
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
