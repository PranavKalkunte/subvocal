# Subvocal middleware: build list

Two sections: the actual platform build, then supplementary work (community, demand, content, business, legal). Ordered by dependency within each. Most pre-materials work from the device version still applies; what changes is which items move to the front and which get deprioritized. A short list at the end names what is dropped or deferred from the device version.

---

# Part 1: Product build

## 1. Foundation and repo setup

- [x] Lock the middleware thesis: positioning, why middleware not hardware, the falsifiable claim, the bridge strategy via existing sEMG devices
  > Locked in `platform/thesis_positioning.md`.
- [x] Define v0 scope (what is in and out of year one)
  > Defined in `platform/thesis_positioning.md`.
- [x] Pick the OSS license (default Apache 2.0 or MIT for adoption; AGPL only if you have a clear commercial reason)
  > Selected and added the MIT License in the root `LICENSE` file.
- [x] Decide monorepo vs multi-repo structure
  > Selected monorepo structure combining `sdk/` and `platform/` folders.
- [x] Rename the Phase 0 repo to the platform name
  > Core codebase folder renamed to `sdk/` and specifications folder renamed to `platform/`.
- [x] Write the README that defines what the library is and will become
  > Written at the root `README.md`.
- [x] Set up the GitHub organization and seed repos
  > Staged and synchronized initial monorepo structure to origin.
- [x] Set up CI/CD baseline on the main repo
  > Initialized `.github/workflows/ci.yml` locally. (Needs push with your personal credentials due to remote workflow scope restrictions).
- [x] Publish the first declaration post (the framing shift, what is shipping)
  > Drafted in `platform/declaration_post.md`.

## 2. Public API and architecture

- [ ] Design the public API surface (core classes, methods, types)
- [ ] Define the data model: sample → frame → command token → intent → action
- [ ] Define the plugin interface for hardware sources
- [ ] Define the plugin interface for LLM providers
- [ ] Define the plugin interface for action executors
- [ ] Define the context provider interface
- [ ] Write the platform architecture document
- [ ] Publish the architecture document for community feedback
- [ ] Set up semver and release process

## 3. Intent reconstruction core (extends Phase 0)

- [ ] Harden the Phase 0 intent expansion into a library API
- [ ] Add Claude provider
- [ ] Add OpenAI provider
- [ ] Add Gemini provider
- [ ] Add local Llama provider (via Ollama or similar)
- [ ] Define the prompt template format and versioning
- [ ] Build the intent expansion test harness
- [ ] Build the open evaluation set of shorthand → intent pairs
- [ ] Run the benchmark across all providers
- [ ] Publish the LLM intent-reconstruction benchmark with code and data
- [ ] Implement reference context providers (calendar, contacts, location, app state)
- [ ] Implement the correction-capture loop
- [ ] Implement a fine-tuning hook for collected corrections
- [ ] Document the intent layer

## 4. Hardware abstraction layer

- [ ] Define the HardwareSource interface
- [ ] Implement the file-replay driver (read recorded sEMG files and feed them through)
- [ ] Implement drivers for public datasets (Ninapro, PutEMG, CSL-HDEMG)
- [ ] Build the synthetic sEMG signal generator with known ground-truth commands
- [ ] Implement the OpenBCI Cyton driver (via BrainFlow)
- [ ] Investigate Delsys Trigno SDK access and implement driver if feasible
- [ ] Document each driver
- [ ] Publish the driver suite

## 5. Classifier infrastructure

- [ ] Define the classifier interface (input window → command token, with confidence)
- [ ] Implement reference models (1D CNN, GRU, small Transformer, feature-based SVM baseline)
- [ ] Build the training pipeline with config-driven reproducible runs
- [ ] Pre-train default models on public datasets
- [ ] Publish pre-trained model weights
- [ ] Implement the per-user calibration / fine-tuning routine
- [ ] Build the model export pipeline (PyTorch → ONNX → Core ML and TFLite)
- [ ] Implement int8 quantization with accuracy regression checks
- [ ] Build the inference benchmarking harness (latency, memory, energy estimates)
- [ ] Document the classifier subsystem

## 6. MCP integration

- [ ] Read the current MCP specification end to end
- [ ] Design how subvocal commands are exposed as MCP tools
- [ ] Implement the reference MCP server
- [ ] Verify it works with Claude Desktop
- [ ] Verify it works with at least one other MCP-compatible client
- [ ] Draft the proposed MCP profile for low-bandwidth intent inputs
- [ ] Publish the proposal as an RFC-style document
- [ ] Submit to MCP community for discussion
- [ ] Iterate based on feedback
- [ ] Track adoption signals

## 7. Reference applications

- [ ] Build a hands-free agent demo (small command set, real LLM, real actions)
- [ ] Build an AAC-style speech assistance app (works with file-replay and live sEMG)
- [ ] Build a productivity controller (timers, messages, queries)
- [ ] Open-source each application
- [ ] Record a demo video for each
- [ ] Publish each as a tutorial with full walkthrough

## 8. Safety, authorization, observability

- [ ] Design the authorization and policy interface (what commands can trigger what actions)
- [ ] Implement dry-run mode (LLM proposes action, system does not execute)
- [ ] Build the tracing layer (every command → intent → action chain is logged)
- [ ] Build the local observability dashboard
- [ ] Specify the threat model for the platform
- [ ] Document key management and credential storage patterns
- [ ] Document on-device vs cloud data residency options
- [ ] Document biometric-data handling against BIPA, GDPR, HIPAA
- [ ] Write safety best practices guide

## 9. Developer surface

- [ ] Stand up the documentation site (Docusaurus or similar)
- [ ] Write the getting-started guide
- [ ] Write "your first subvocal-controlled agent" tutorial
- [ ] Write the hardware abstraction guide
- [ ] Write the LLM provider guide
- [ ] Write the context provider guide
- [ ] Write the MCP integration guide
- [ ] Write the calibration and fine-tuning guide
- [ ] Set up API reference auto-generation from code
- [ ] Build a runnable Colab notebook walking through the full pipeline
- [ ] Build a hosted in-browser demo

## 10. Testing and release process

- [ ] Write unit tests for the core libraries
- [ ] Write integration tests using public datasets
- [ ] Write an end-to-end smoke test (synthetic signal → action)
- [ ] Set up automated benchmark runs on every release
- [ ] Set up dependency scanning and license checking
- [ ] Define the release cadence and changelog format
- [ ] Publish v0.1
- [ ] Publish v0.2 with hardware abstraction and pre-trained models
- [ ] Publish v0.3 with MCP integration
- [ ] Publish v1.0 once API is stable

---

# Part 2: Supplementary (community, demand, content, business, legal)

## 1. Program container

- [ ] Brainstorm program/project names
- [ ] Check domain and handle availability
- [ ] Register domain
- [ ] Set up email
- [ ] Choose hosting platform for the corpus
- [ ] Write one-line statement of what the platform does
- [ ] Stand up minimal site with the one-line statement
- [ ] Install reference manager (Zotero)
- [ ] Create master tracking doc for all workstreams

## 2. Literature synthesis (reframed for platform positioning)

- [ ] Obtain Kapur 2018, Jorgensen 2004, Meltzner 2018, the 120-electrode mapping study, Wu 2024, SilentWear 2026
- [ ] Extract per paper: vocabulary size, accuracy, electrode count and placement, form factor, silent vs vocalized, calibration time, limitations
- [ ] Build the comparison table
- [ ] Mark convergences and gaps
- [ ] Add adjacent literature: MCP, LangChain, ROS, embodied AI middleware patterns
- [ ] Draft the synthesis piece framed for an agent-and-hardware developer audience
- [ ] Verify every numeric claim against source
- [ ] Publish the synthesis piece

## 3. Patent and competitive landscape

- [ ] Search Google Patents and USPTO for subvocal, silent speech, EMG speech, neckband EMG
- [ ] Build the patent table
- [ ] Identify MIT, Meta, Apple, Google, Amazon, CTRL-Labs filings
- [ ] List adjacent middleware competitors (LangChain, agent frameworks, ROS, OpenBCI BrainFlow, embodied AI platforms)
- [ ] Map the MCP ecosystem entrants and their adjacencies
- [ ] List every subvocal/silent-speech hardware effort past and present
- [ ] Document open-source license strategy and rationale
- [ ] Write the competitive landscape piece
- [ ] Decide between defensive publication and provisional patent for the protocol design

## 4. Developer customer development

- [ ] Identify communities where neurotech and agent developers gather (OpenBCI forum, biosignals Discord, LangChain Discord, MCP community, r/BCI, r/neurotech, accessibility hacker spaces)
- [ ] Join each and engage substantively
- [ ] Write the qualified-developer profile
- [ ] Write the developer interview script
- [ ] Recruit 10–20 developers for early SDK feedback
- [ ] Run developer interviews
- [ ] Iterate the API based on feedback
- [ ] Publish a "what developers asked for" writeup

## 5. Hardware vendor customer development

- [ ] List every subvocal and silent-speech hardware effort, active or stealth
- [ ] Identify CTRL-Labs and AlterEgo alumni currently building anything in the space
- [ ] Identify university labs with active subvocal hardware projects
- [ ] Draft the "we handle your software stack" pitch
- [ ] Reach out to each with a precise offer (free integration support, joint demo)
- [ ] Track interest
- [ ] Document any partnerships in the corpus

## 6. Standards and protocol work

- [ ] Engage in the MCP community via GitHub Discussions and Discord
- [ ] Read every accepted MCP proposal
- [ ] Draft the low-bandwidth-intent profile of MCP
- [ ] Circulate the draft to two or three MCP maintainers
- [ ] Submit formally for discussion
- [ ] Iterate based on feedback
- [ ] Track adoption

## 7. Researcher outreach

- [ ] List target researchers (Kapur, Jorgensen, Meltzner, Wu, SilentWear team)
- [ ] Find current contact info
- [ ] Draft personalized messages with a precise question
- [ ] Send the synthesis and platform overview to 2–3, inviting comment or advisory involvement
- [ ] Track responses
- [ ] Incorporate feedback into the corpus

## 8. Community building

- [ ] Set up Discord with channels for general, help, contributors, showcase
- [ ] Set up GitHub Discussions on the main repo
- [ ] Write the contributor guide
- [ ] Write the code of conduct
- [ ] Define the issue triage process
- [ ] Schedule weekly office hours (async post or live)
- [ ] Set up release announcement channels
- [ ] Decide on CLA vs DCO for contributions
- [ ] Track community health (active contributors, response times, issue close rate)

## 9. Content and corpus

- [ ] Write the standing open-problems list (drift, cross-user generalization, calibration, latency, vocabulary ceiling, hardware availability)
- [ ] Decide publication order of corpus pieces
- [ ] Publish the platform thesis piece
- [ ] Publish the synthesis piece
- [ ] Publish the architecture piece
- [ ] Publish the MCP-profile proposal
- [ ] Publish the LLM intent-reconstruction benchmark
- [ ] Publish each reference application writeup
- [ ] Publish the public-data classifier benchmark
- [ ] Set up a conversion tracker (active installs, integrations, contributors, hardware partnership conversations)

## 10. Distribution and content pipeline

- [ ] Build a 6-month content calendar
- [ ] Set up newsletter and RSS
- [ ] Set up cross-posting (long-form host, LinkedIn, Hacker News, X/Bluesky, ArXiv when appropriate)
- [ ] Identify the right venues for each piece (HN front page, ML Twitter, neurotech subreddits, MCP community)
- [ ] Publish a visible public roadmap
- [ ] Start a biweekly roundup of subvocal/silent-speech news and platform progress
- [ ] Start an interview series with researchers and hardware builders
- [ ] Schedule the first three interviews

## 11. Brand and design system

- [ ] Define program voice and writing guidelines
- [ ] Define visual design system (color, type, components)
- [ ] Design the program landing page
- [ ] Build standard layout templates for corpus pieces
- [ ] Theme the documentation site
- [ ] Storyboard the platform demo video
- [ ] Produce a system block diagram for the corpus
- [ ] Produce a "where the platform sits in the stack" diagram
- [ ] Build an interactive in-browser demo for the landing page

## 12. Business and financial modeling

- [ ] Sketch the commercial layer offerings (hosted runtime, observability, enterprise features, certification program)
- [ ] Build pricing model thoughts (per-call API, seat-based, enterprise license)
- [ ] Build the unit-economics model for the hosted layer
- [ ] Build a multi-year projection (revenue, cost, headcount, capital required)
- [ ] Build the cap-table model and dilution scenarios
- [ ] Build a sensitivity analysis on adoption assumptions
- [ ] Research R&D tax credit eligibility and SBIR mechanics

## 13. Funding and grant landscape

- [ ] Map dev tools and infrastructure VCs and angels
- [ ] Map AI infrastructure accelerators (YC, AI Grant, others)
- [ ] Map relevant grants (NSF for HCI and accessibility, NIDCD, ALS Association, accessibility-focused foundations)
- [ ] Identify two or three target grants for the first 12 months
- [ ] Draft outline of the strongest target grant application
- [ ] Identify timing for a potential seed round (after measurable developer traction)

## 14. Co-founder and team targeting

- [ ] Profile the right co-founder archetypes (ML engineer, infrastructure engineer, DevRel)
- [ ] Map LangChain, Replicate, Modal, Hugging Face, Vercel, Supabase alumni
- [ ] Map MCP contributors and Anthropic engineers in the public sphere
- [ ] Identify 20 named individuals worth contacting
- [ ] Draft personalized outreach
- [ ] Build a "what we're looking for" recruitment page
- [ ] Draft equity offer ranges for co-founder, early engineer, advisor
- [ ] Send initial outreach batch

## 15. Conference and publication

- [ ] List relevant venues (NeurIPS workshops, ICML workshops, IEEE EMBC, ASSETS, CHI, IMWUT, dev tools conferences like StrangeLoop)
- [ ] Pull submission deadlines and formats
- [ ] Target a workshop or short-paper deadline
- [ ] Draft an abstract for the chosen venue
- [ ] Identify and apply to local hardware/ML meetups
- [ ] Submit a talk proposal to one online community event

## 16. Legal and entity

- [ ] Trademark check on program name and product names (USPTO TESS)
- [ ] Decide entity form (LLC, Delaware C-corp) and timing
- [ ] Set up business banking
- [ ] Set up bookkeeping/accounting
- [ ] Draft founder agreement and IP assignment templates
- [ ] Decide employment vs contractor structure for early help
- [ ] Decide CLA vs DCO for external contributors
- [ ] Draft standard NDA and advisor agreement templates
- [ ] Set up timestamped records of all original work (Git history, dated publications)
- [ ] Draft program privacy policy and terms of service
- [ ] Decide between defensive publication and provisional patent for the protocol and architecture

## 17. Pre-mortems and kill criteria

- [ ] Run a "subvocal hardware never materializes" scenario and document
- [ ] Run an "Anthropic absorbs the surface into MCP itself" scenario
- [ ] Run an "Apple ships a fully integrated stack that bypasses you" scenario
- [ ] Run a "the SDK is good but no developers adopt it" scenario
- [ ] Run a "solo burnout at month 10" scenario
- [ ] Build a 12/24/36-month milestone tree with explicit kill criteria
- [ ] Document the conditions under which you would shut the platform down or pivot

---

## Deprioritized from the device version (deferred or dropped)

- CAD enclosure, electrode housings, mechanical design (defer until a first-party hardware reference design at year 2 or 3)
- PCB schematic, layout, fabrication outputs (same)
- Custom firmware on simulator (use existing OpenBCI firmware via BrainFlow instead; revisit if you ship a reference board)
- Industrial design renders and color/material studies (not needed)
- Manufacturing, supply chain, contract manufacturer RFQs (not year one)
- FCC, CE, Bluetooth SIG certification planning (no hardware to certify)
- Quality system scaffolding: DHF, RMF, ISO 13485 templates, IEC 62304 classification (defer until first-party hardware or medical-class deployment)
- FDA pathway mapping, predicate device identification, Q-Sub letter (defer; revisit if and when you ship into the accessibility wedge with first-party hardware)
- Lab access and IRB clearance for collecting first-party sEMG (defer; year one runs on public datasets and OpenBCI-using contributors)
- Insurance product liability quotes (not year one for pure software)
- Warehouse-worker interviews as primary demand source (keep but secondary; primary demand validation is now developers and hardware vendors)
- Manufacturing-side renders, comparative-to-incumbent product imagery (defer)

These do not go away forever. They re-enter when and if you ship hardware as a reference design or partner deeply with a hardware vendor on a co-branded product. For year one they would absorb time the platform build needs.
