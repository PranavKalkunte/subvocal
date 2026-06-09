# Subvocal Middleware Thesis & Positioning

This document defines the core positioning, technical hypothesis, bridge strategy, and Year 1 scope of the **Subvocal SDK** platform.

---

## 1. Positioning Rationale: Software Middleware vs. Hardware

The central venture pivot is positional: rather than manufacturing and shipping a proprietary physical neckband, we focus entirely on the **connective software layer (middleware)** that sits between sEMG hardware and LLM-driven agents.

### Rationale:
1. **Capital and Supply Chain De-risking**: Hardware manufacturing requires extensive funding, assembly logistics, inventory costs, and a multi-disciplinary hardware engineering team. Software runs locally on standard setups and requires minimal overhead.
2. **Regulatory Bypass**: Medical-grade and assistive speech hardware devices are subject to complex FDA Class I/II and CE/MDR reviews. Middleware escapes immediate regulatory oversight in Year 1.
3. **Connective Rails Moat**: Hardware players will rise and fall, but they all need to connect to LLM agents. By becoming the standard interface, we capture value regardless of which physical hardware wins (e.g., consumer neckbands, smart earbuds, wristbands, or accessibility rigs).
4. **Adjacency Hedging**: If silent speech sEMG adoption is slow, the same intent-reconstruction core works for other low-bandwidth inputs: lip-camera silent speech, throat microphones, wrist sEMG (Meta), and acoustic speech in high-noise environments.

---

## 2. The Falsifiable Claim

> **Hypothesis**: *A low-bandwidth, physiological input channel (such as silent subvocalizations mapped to compressed articulatory phonetic shorthands) can be expanded using context-aware phonetic decoders to trigger complex, arbitrary agentic tasks with >90% intent resolution accuracy and <200ms latency, bypassing the whole-word vocabulary ceiling of silent speech interfaces.*

---

## 3. The Bridge Strategy: Target Incumbents

To resolve the chicken-and-egg hardware dependency, we target existing sEMG hardware and public datasets immediately:
* **Existing Hardware Adaptations**: Reference driver interfaces for **OpenBCI Cyton** (widely used in university biosignal research labs), **Backyard Brains SpikerShield** (hobbyist/educational communities), and general consumer developer boards.
* **Public Datasets**: Benchmarks built on continuous silent/audible corpora (EMG-UKA, CSL-HDEMG) to establish platform performance baseline.
* **Synthetic Generation**: A built-in mock channel simulator allowing developers to test agent integration fully offline without needing any physical sEMG hardware.

---

## 4. v0 Scope Definition (Year One)

### In-Scope (Year 1):
1. **Core Decoder SDK**: An installable Python library providing:
   - Physiological noise simulators.
   - Asymmetric Levenshtein distance matching.
   - Command-aware context filtering (Calendar, Contacts, App UI elements).
2. **Multi-Model LLM Adapters**: Modular REST connections to OpenAI GPT-4o, Anthropic Claude, Google Gemini, and local Llama (via Ollama).
3. **Hardware Abstraction Layer (HAL)**: Pluggable drivers for OpenBCI Cyton, SpikerShield, and synthetic signals.
4. **Classifier Infrastructure**: Local training skeletons for Random Forest, 1D CNN, and GRU models, with ONNX/CoreML model export workflows.
5. **Model Context Protocol (MCP)**: Exposing subvocal commands as MCP tools that Claude Desktop or any MCP-compliant client can invoke.
6. **Reference Application**: A hands-free agentic application (e.g. warehouse inventory controller).

### Out-of-Scope (Year 1):
1. **First-Party Neckband Manufacturing**: Designing or ordering custom mechanical enclosures or printed circuit boards.
2. **Clinical Regulatory Clearance**: FDA submissions or clinical human trials.
3. **Hosted Commercial Cloud Platform**: Managed SSO, enterprise RBAC, hosted database storage, or dashboard telemetry.
4. **Microcontroller Firmware Edge Inference**: Compiling models for low-power bare-metal MCUs.
