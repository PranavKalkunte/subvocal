# The middleware playbook: agentic AI ↔ subvocal hardware

A version of the venture that fits your skill set and constraints more cleanly than either building the hardware yourself or attracting a builder to it. The pivot is positional: instead of being the company that makes a neckband, become the software layer that every subvocal device has to talk to in order to reach an LLM-driven agent. Hardware vendors come and go; the middleware that defines how their signals become agent actions is the durable layer.

Most of the pre-materials work you have already planned still applies. What changes is the role each piece plays. The Phase 0 demo stops being a "proof that the software half works while we wait on hardware" and becomes the kernel of the actual product. The bench code, classifier training, and digital twin become the model infrastructure of the platform rather than scaffolding for a single device. The corpus stops being a builder-attractor and becomes the body of work that establishes you as the reference for how subvocal input should connect to agents.

---

## 1. What "middleware for subvocal" actually covers

Concretely, the layer sits between raw sensor data and the agent. Its surface includes:

- A vendor-agnostic abstraction over subvocal hardware (your own future neckband, fragments of CTRL-Labs / AlterEgo descendants, Wispr-class devices, research rigs like Delsys Trigno or OpenBCI, any future Apple/Meta wearable that exposes sEMG)
- A signal-processing and classifier infrastructure (training pipeline, calibration routines, on-device inference, model versioning, drift handling)
- The intent-reconstruction layer that turns a small noisy command set into structured intents (the LLM-mediated shorthand expansion you have already prototyped)
- Agent integration that exposes subvocal commands as callable tools, ideally as MCP servers so any MCP-compatible client can consume them
- Context management (user calendar, contacts, location, app state) made available to the intent layer
- A developer SDK so people building agents and applications can accept subvocal input without writing any of the above
- Observability, replay, and debugging tooling for what an agent did in response to which command
- Authorization, dry-run, and safety policy enforcement around what a subvocal command is allowed to trigger
- A registry or marketplace of subvocal-callable actions, eventually

You do not own the hardware, and you do not own the agent frameworks. You own the part neither side wants to build and both sides need.

## 2. Why this position over the hardware position

The hardware path required capital, manufacturing, regulatory clearance, and an engineering team you were never going to assemble alone. The middleware path requires software, content, and developer relationships, which match what you have. You can build the product yourself with AI assistance and bench data from public datasets; you do not need anyone's permission, capital, or supply chain.

Crucially, the middleware position benefits from every hardware company that gets built. You do not pick a winner. You are the rails. The neckband from MIT alumni, the Meta wearable two years from now, the OpenBCI hack rig in someone's apartment all need the same connective layer.

The cost of being wrong is also lower. If subvocal hardware takes another five years to ship at scale, the same middleware works for adjacent low-bandwidth intent channels: silent speech via lip cameras, throat mics, wrist EMG (which actually has shipping consumer products coming via Meta), accessibility AAC inputs, even high-error-rate voice in noisy environments. The intent-reconstruction core is general; the subvocal framing is the wedge.

## 3. The chicken-and-egg problem, named honestly

The hardest tension: if no subvocal hardware ships in volume, there are no users to support, no developers to court, and the middleware company is theoretical. If the hardware does ship at volume, big-tech bundlers (Apple, Meta, Google) will build their own integrated stack and skip third-party middleware, the way iOS does not depend on third-party HealthKit shims.

Three responses, in order of importance.

First, the bridge: anchor on hardware that already exists. OpenBCI Cyton boards, Delsys Trigno research kits, the Backyard Brains SpikerShield, off-label use of consumer EMG dev boards. These are already in the hands of researchers, hobbyists, and a few clinical labs. They are not subvocal-purpose devices, but they produce sEMG signals from neck and face electrodes that your middleware can ingest. This gives you users from day one, before any subvocal-specific hardware exists.

Second, the open-standard play: define and publish the protocol that subvocal hardware should expose to talk to agents, as an extension of MCP. Make it the obvious thing for any future hardware team to adopt because nothing else exists and the alternative is rolling their own. Anthropic does this with MCP for tools; you do the equivalent for subvocal input as a specific class of tool. Standards bodies, when one entrant defines a reasonable one early and gives it away, often consolidate around that entrant.

Third, the big-tech threat is real and mitigated only by being too embedded to replace. If you are the open-source reference implementation that every researcher, hobbyist, and pre-product hardware team has already integrated, even a vertically integrated Apple or Meta has to decide whether to support your protocol or build their own. They will probably do both. You lose the Apple/Meta surface and keep everything else.

## 4. The Stripe and HashiCorp pattern applied

Developer-facing infrastructure companies that became defaults followed a recognizable pattern: better DX than incumbents, narrow wedge first, open source where it makes sense, host the commercial layer, brand built on technical credibility. Stripe owned online payments for developers before expanding. HashiCorp owned Terraform's mindshare before commercializing Cloud. LangChain owned the default LLM-app library before monetizing observability.

For subvocal middleware the analogous shape:

The open-source SDK is the entry point. A developer with an OpenBCI board or a recorded sEMG file should be able to install your library, run a few lines of code, and get classified commands flowing into an LLM-driven action. That experience does not exist anywhere right now. Building it is a few weeks of work for someone with your stack.

The narrow first wedge is "any developer with an existing sEMG device who wants to control an agent with it." That is a small but real population: research groups, makers, accessibility hackers, the ALS community's tinkerers, neurotech students. Win them and you have the references and the documentation and the integration patterns ready when the broader population shows up.

The commercial layer comes later, once usage exists. Hosted training runs, managed model deployment to edge devices, observability dashboards, enterprise SSO and audit logs, certification programs for hardware vendors that want to claim compatibility, a marketplace of pre-built subvocal-callable actions. None of this is built in year one; year one is making the OSS irreplaceable.

## 5. The MCP relationship

Anthropic's Model Context Protocol is becoming the de facto standard for how agents talk to tools. The right posture is to extend it, not compete with it. Define a profile of MCP for "low-bandwidth intent inputs" or specifically for "subvocal command channels," publish it openly, and submit it for inclusion or reference in MCP's broader ecosystem. Your middleware becomes an MCP server (or a class of servers) that any MCP-compatible client can plug into.

This buys you two things: instant compatibility with Claude, with any other model that adopts MCP, and with the rapidly growing agent-framework ecosystem; and a position that is additive to Anthropic's roadmap rather than in conflict with it. Anthropic does not want to build subvocal-specific infrastructure. They want a thriving MCP ecosystem. You fit that.

If MCP loses momentum, the same logic applies to whatever replaces it. The point is to be on the protocol of the agent side, not to define a competing one.

## 6. The content and community flywheel

The thought-leadership work you were already planning still applies, with the audience shifted. The corpus is now aimed at agent and embodied-AI developers, MCP server builders, neurotech researchers, and the still-small population of subvocal-curious engineers, rather than at hardware co-founders.

The same restraint rules apply. Real bench data, replicable benchmarks, honest open problems, and direct engagement with the researchers you cite earn standing here. The Phase 0 demo is the single highest-conversion artifact because it is the thing developers can clone and run. Tutorials, video demos, conference talks, and a steady cadence of substantive writeups compound into the brand position you need.

A Discord and an active GitHub presence matter more than a newsletter. Developer middleware companies are evaluated on the responsiveness of their maintainers and the quality of their docs, not on follower counts. Plan a weekly office-hours or async support cadence from the first release.

## 7. What you actually build, sequenced

In rough order, year one:

The intent-reconstruction core, which you already have as Phase 0. Harden it into a library with a clean API, multi-model support (Claude, GPT, Gemini, local Llama), pluggable context providers, and a published evaluation set. Make this the public benchmark for compressed-intent expansion.

The hardware abstraction layer. A driver interface that any sEMG source plugs into, with reference implementations for OpenBCI Cyton, public sEMG datasets (Ninapro, PutEMG, CSL-HDEMG), and the synthetic signal generator. This is what lets developers without subvocal hardware build against your library today.

The classifier infrastructure. Training pipeline, calibration routines, model export to ONNX and Core ML and TFLite, on-device inference benchmarks. Pre-trained models on public data, with a clear path for users to fine-tune on their own data.

The MCP server. Subvocal commands exposed as MCP tools that any MCP-compatible client (Claude Desktop, agent frameworks, custom clients) can invoke. Examples and quickstart docs.

The reference applications. A few open-source apps built on the SDK to demonstrate what it can do: a hands-free agent for warehouse tasks, an AAC-style speech tool, a productivity controller. These are conversion assets, not products, and they show what is possible.

The developer surface around all of it. Docs site, examples repo, Discord, a couple of conference talks, a clear public roadmap, regular releases. None of this is glamorous and all of it is what separates infrastructure that gets adopted from infrastructure that does not.

## 8. Economic and capture logic

Pure open source with a hosted commercial layer is the realistic capture model. The OSS is free; you charge for hosted runtimes, managed model training, observability and tracing for production deployments, enterprise features (RBAC, audit, SSO), and eventually a certification program for hardware vendors claiming compatibility. This is the HashiCorp, Vercel, Supabase pattern.

Revenue is not near-term. The first 12 to 18 months are about adoption, with maybe small consulting or sponsorship income. The capital path is a seed round once developer traction is measurable (active installs, integrations, contributors). Hardware-adjacent dev tools companies have raised on hundreds of GitHub stars and a few dozen active users when the wedge was clearly hot; subvocal is not hot yet, which means the bar for fundraising is higher and the runway demand is longer.

Solo for year one is feasible. A small team (2 to 4 engineers) by year two if traction warrants. The middleware does not need a sales team for a long time, which is the structural reason this position is workable from a no-capital start.

## 9. Where defensibility actually comes from

The code itself is not the moat; it is open source by design. The moat is the things that compound: developer mindshare from being first and most credible, the integration count (hardware drivers, model adapters, framework plug-ins) that accumulates over time, the brand association where "subvocal input for agents" and your project's name become reflexively linked, the dataset and benchmarks you publish that everyone else has to compare against, and the personal relationships with the researchers and the few hardware teams that exist.

The category of "subvocal middleware" is small enough today that being the obvious one is achievable for a solo or small team. The category being small is the opportunity. If you wait until it is large, the position will be taken.

## 10. Failure modes specific to this version

The category never materializes. Subvocal hardware stays a research curiosity for another decade, no major hardware vendor ships, and the middleware is solving a problem that almost no one has. Mitigation is the adjacency hedge: the same library works for any low-bandwidth intent input, so even without subvocal-specific hardware the work is reusable for accessibility AAC, silent-speech via other modalities, error-prone voice in noisy environments, and wrist-EMG inputs that consumer products are starting to ship.

Big tech absorbs the layer. Apple or Meta ships a vertically integrated stack that does not use your protocol. You lose those device populations and keep the long tail, which is still a real business but not the dominant position. There is no clean defense against this; you can only stay ahead in correctness and breadth.

The protocol you define is not the one that wins. Someone else defines a better one, or Anthropic absorbs the surface into MCP itself in a way that obviates your layer. Mitigation: build close enough to MCP that your work is forward-compatible, and contribute upstream rather than defending a separate fiefdom.

Building the SDK well takes longer than you think. Developer infrastructure that is good enough to adopt is hard. The line between "it works in my demo" and "it works for someone else's project" is months of API design, error handling, documentation, and corner cases. Plan for that explicitly; do not under-resource the polish.

The user studies and demand work get neglected because the build is interesting. Middleware that no one uses is a hobby project. The corpus, the community work, the wedge customer development, and the demo applications matter as much as the SDK code. Budget time to all of it.

Solo-burnout. Year one alone is hard. The community work is energy-positive but the API design and bug-grinding is not. Plan for explicit recovery time and for bringing in a second person before you are desperate.

## 11. How this differs from the hardware playbook

You are the builder this time. The original playbook assumed someone else would build the product. Here, the product is your software, and the corpus is collateral and brand-building rather than a builder magnet. Most of the pre-materials task list applies, with the role of each piece changing.

Capital requirement is dramatically lower. Software runs on a laptop and a cloud bill in the low hundreds of dollars per month for a long time. The bench experiment, when you eventually run it, is now training-data acquisition for the platform rather than proof-of-concept for a single device.

Customer development changes shape. You are now developing two customer groups: hardware vendors (who you want to integrate your protocol) and application developers (who you want to build on your SDK). Both are reachable through technical content and community work rather than enterprise sales.

The regulatory and manufacturing burdens largely go away. They reappear if and when you build any first-party hardware reference design, which you might do at year two or three, but you do not need to plan for them in year one.

## 12. First six months

Months 1 to 2: Extract the Phase 0 demo into a clean, installable library. Publish v0.1 with a real API, basic docs, and one working example. Open the Discord. Write the first technical post: "What I want subvocal-input-for-agents to look like, and what I have built so far."

Months 2 to 4: Add the hardware abstraction layer. Make it work with OpenBCI Cyton and at least two public sEMG datasets. Publish the second post: "Running real sEMG through an LLM, end to end, with public data." Ship the synthetic signal generator as a separate package so people without any hardware can demo it.

Months 3 to 5: Draft the MCP-extension proposal for low-bandwidth intent inputs. Publish it openly. Send it to two or three of the MCP maintainers and to the researchers you would cite. Run a small user study (ten developers) on the SDK; publish the results and the API changes that came from it.

Months 4 to 6: Ship a reference application. Probably the warehouse-command demo, since it is concrete and you already have demand research backing it. Submit a workshop paper or short talk to a relevant venue (IEEE EMBC, ASSETS, IMWUT). Identify and reach out to any subvocal hardware team that exists or is forming, with the offer that you handle their software stack.

By month 6 the goal is not revenue. It is: a working OSS library with real installs, a published protocol that two or three people are arguing about, one reference application running end to end, the start of a small but real developer community, and at least one conversation with a hardware team that could plausibly use you.

## 13. First action this week

Rename the Phase 0 repository to reflect the platform framing rather than the demo framing, write a one-page README that describes what the library is for and what it will become, and publish a single short post under the program name explaining the shift. The shift is real and worth declaring: you are not waiting for hardware; you are building the layer that makes the hardware viable when it arrives, and useful to a smaller group of users today.

Everything you have already built and planned points at this. Most of the work survives the pivot. What changes is what you are calling it and who you are talking to.
