# treehacks-2026-main reference

Built at TreeHacks 2026 under the name **MindOS** (now reframed as Subvocal). sEMG silent speech interface: jaw/throat electrodes → ESP32 → Python DSP/ML → FastAPI → LLM agent → browser control via Playwright.

Demo video: https://www.youtube.com/watch?v=v-DJEQgKm-A  
GitHub: https://github.com/ansht3/treehacks-2026

---

## Top-level structure

```
treehacks-2026-main/
├── silentpilot/        # Main project (Python EMG core + TS agent + Next.js UI)
└── actions/            # Standalone desktop browser automation (Playwright + vision)
```

---

## silentpilot/

### Hardware used at hackathon
- 2× MyoWare 2.0 sensors on jaw (masseter), ENV output → Arduino Uno → PySerial
- Later planned: 3 sensors (jaw GPIO 32, submental GPIO 33, laryngeal GPIO 34) on ESP32
- Ag/AgCl gel electrodes; 250 Hz sample rate; 12-bit ADC

### Signal pipeline (Python)
`emg_core/` — all Python, installable via `pip install -e ".[dev]"`

| File | What it does |
|---|---|
| `config.py` | All constants: `SAMPLE_RATE=250`, `NUM_CHANNELS=4`, `BANDPASS_LOW=1.3`, `BANDPASS_HIGH=50.0`, `NOTCH_FREQ=60.0`, `SEGMENT_FIXED_LENGTH=150` (0.6s), `CONFIDENCE_THRESHOLD=0.75`, default commands list |
| `pipeline.py` | Async pipeline: reader → normalizer → segmenter → inference. `EMG_READER=mock` or `serial` env var. `Pipeline.classify_segment(segment_np)` → `Prediction` |
| `ingest/base_reader.py` | Abstract `BaseReader` interface |
| `ingest/mock_reader.py` | `MockReader` — synthetic signals, no hardware needed |
| `ingest/serial_reader.py` | `SerialReader` — reads from ESP32 via USB serial |
| `ingest/packet_parser.py` | Parses 19-byte binary packets: `0xAA 0x55` header + version + seq + timestamp + 4× uint16 channels + CRC16 |
| `dsp/filters.py` | `remove_dc()`, `bandpass()` (Butterworth), `notch_60hz()`, `smooth()`, `preprocess_multichannel()` |
| `dsp/features.py` | **Two pipelines:** (1) Legacy AlterEgo: 5 time-domain + MFCC per channel = 87 dims for 4ch. (2) **Default TD10**: double moving average decomposition → 5 features/frame/channel → ±10 frame context stacking → mean+std = **840 dims for 4ch**. Public API: `extract_features(segment_2d, sample_rate)` |
| `dsp/normalize.py` | `RollingNormalizer` — rolling z-score |
| `dsp/segment.py` | `PTTSegmenter` — push-to-talk segmentation, `start(label)` / `stop()` → segment |
| `ml/train.py` | `train_model(user_id)` → `TrainResponse`. Pipeline: `StandardScaler → LDA(32 components) → RandomForestClassifier(n_estimators=200)`. Saves to `models/{user_id}.pkl`. |
| `ml/infer.py` | `InferenceEngine(user_id)` — loads model, `predict(segment_np)` → `Prediction` with confidence thresholding |
| `ml/model_io.py` | `save_model()` / `load_model()` / `model_exists()` |
| `api/server.py` | FastAPI: `GET /status`, `POST /calib/start`, `POST /calib/stop`, `POST /calib/save`, `POST /train`, `POST /infer/start`, `POST /infer/stop`, `WS /ws/live`. WS sends `raw` and `prediction` messages; receives `ptt_start` / `ptt_stop` events. |
| `api/schemas.py` | Pydantic models: `RawSample`, `Prediction`, `TrainResponse`, `WSMessage`, `CalibStartRequest`, `CalibSaveRequest`, `TrainRequest`, `InferStartRequest` |

### Firmware
`firmware/emg_streamer/emg_streamer.ino`  
ESP32 Arduino firmware. 4 ADC channels (GPIO 32/33/34/35), 250 Hz, 115200 baud. 19-byte binary packets. To switch to 3 channels: change `NUM_CHANNELS 4→3` and remove GPIO 35. No BLE/OTA yet — USB serial only.

### Agent (TypeScript)
`agent/src/` — Node.js, `npm run dev`

| File | What it does |
|---|---|
| `index.ts` | Entry point; reads from EMG core WS, routes commands |
| `command_router.ts` | Maps EMG labels to `RoutedCommand` with mode `DIRECT` / `AGENT` / `CONTROL` |
| `orchestrator.ts` | `Orchestrator.execute(routed)`. DIRECT: calls MCP tool without LLM. AGENT: calls OpenAI Responses API with MCP browser tool, chains via `previous_response_id`. |
| `prompts.ts` | `SYSTEM_PROMPT` (8 EMG commands: OPEN/SEARCH/CLICK/SCROLL/TYPE/ENTER/CONFIRM/CANCEL) + `buildUserPrompt(state, command)` |
| `state.ts` | `AgentState`: goal, plan, currentStep, toolHistory, lastResponseId |

### MCP server (TypeScript)
`mcp_server/src/` — MCP server exposing browser tools to the agent via SSE at port 3333.  
`tools/browser.ts` — Playwright browser tools callable by the LLM.

### Frontend (Next.js)
`app_ui/pages/`

| Page | Purpose |
|---|---|
| `index.tsx` | Landing / mode select |
| `calibrate.tsx` | Calibration UI — PTT button, per-command sample counter, calls `/calib/*` and `/train` |
| `train.tsx` | Training trigger + accuracy display |
| `demo.tsx` | Live inference demo view |

`app_ui/components/`
- `SignalPlot.tsx` — real-time multi-channel EMG waveform
- `PTTButton.tsx` — push-to-talk with hold detection
- `ConfusionMatrix.tsx` — displays training confusion matrix
- `CommandOverlay.tsx` — predicted command display

`app_ui/lib/`
- `ws.ts` — WebSocket client connecting to `/ws/live`
- `api.ts` — HTTP client for REST endpoints

### Startup sequence
```bash
# 1. EMG core
cd silentpilot && uvicorn emg_core.api.server:app --reload --port 8000

# 2. MCP server
cd mcp_server && npm install && npx playwright install chromium && npm run dev

# 3. Agent
cd agent && npm install && npm run dev

# 4. Frontend
cd app_ui && npm install && npm run dev
```
`.env` vars: `EMG_READER=mock|serial`, `SERIAL_PORT`, `SAMPLE_RATE`, `NUM_CHANNELS`, `OPENAI_API_KEY`, `MCP_SERVER_URL`, `AGENT_MODEL`.

### Experimental scripts
`scripts/` — run against EMG-UKA corpus (not included in main branch; on `datasets` branch at `~/.cache/kagglehub/...`). 6 channels, 600 Hz.

| Script | What it does |
|---|---|
| `emguka_pipeline.py` | Whole-word classification from continuous speech. Result: 15.4% on 30 words — severe overfit. |
| `emguka_phones.py` | Phoneme/manner-of-articulation classification. Best: 35.3% on 6 manner classes (2.1× chance). |
| `emguka_arch_compare.py` | Benchmarks RF, Conformer, CNN+Attention, CNN, MAE, Transformer. **RF wins every experiment** at <10K samples. |
| `emguka_cnn.py` | Audible→silent transfer learning. Transfer doesn't help. |
| `emguka_commands.py` | **Best result: 62.5% on 5 real word commands (3.1× chance)** using RF. THE=90–100% recall; words ~300ms. |
| `emguka_synthetic.py` | Phone exemplar concatenation → synthetic word EMG. Syn→Real: 26.5–34.4% (2.1–2.8× chance). |
| `emguka_llm_decode.py` | RF phone lattice → GPT-4.1-nano sentence decoding. WER=102% — phone top-1 only 5.7%, needs ≥50% top-5 for viability. |
| `e2e_test.py` | 4 commands, MockReader: **100% accuracy** |
| `e2e_test_full.py` | 8 commands, MockReader: **93.3% accuracy** |

### Key numbers from experiments
| Metric | Value |
|---|---|
| Best real silent EMG accuracy | 62.5% on 5 commands (3.1× chance) |
| Predicted with PTT + 75 samples/command | 80–90% on 5 commands |
| RF vs CNN+Attention at <500 samples | RF wins by wide margin |
| Phone top-1 accuracy (silent, 41 classes) | 5.7% |
| Phone top-5 accuracy | 20.1% |
| LLM decode viability threshold | ≥50% phone top-5 |
| Manner classification (6 classes) | 35.3% (2.1× chance) |
| Calibration time target | ~15 min (50–75 samples × 5–8 commands) |
| Inference latency | <500ms end-to-end |
| Feature dimensions (4ch, TD10) | 840 → LDA → 32 |

### What's done vs. not done
**Done:** Full async pipeline (mock + serial), DSP filters, TD10 features, RF+LDA training, FastAPI server, WebSocket streaming, PTT calibration UI, agent orchestrator, MCP browser server, ESP32 firmware (USB serial), 7 experimental scripts with documented results.

**Not done:** BLE/OTA firmware, on-device ONNX inference, model export (ONNX/CoreML/TFLite), dry electrode support, IRB/human subjects, CAD enclosure, PCB, Figma mockups, cross-session adaptation, self-supervised pretraining.

---

## actions/

Standalone desktop browser automation — separate from `silentpilot`. Uses Playwright + GPT-4o vision. Can run independently or as a word-disambiguation layer on top of EMG output.

| File | What it does |
|---|---|
| `engine.py` | `ActionEngine` (interactive, shows DOM suggestions) + `AutonomousAgent` (LLM-driven goal completion, up to `MAX_AGENT_STEPS`) |
| `browser.py` | `BrowserController` — Playwright wrapper: `launch`, `goto`, `click_coords`, `page_type`, `scroll`, `go_back/forward`, `get_interactive_elements`, `get_page_text` |
| `vision.py` | `AgentPlanner` — GPT-4o decides next browser action from page context + element list |
| `cursor.py` | Custom cursor overlay on browser window |
| `overlay.py` | `Overlay` — displays suggestion list on top of browser |
| `word_finder.py` | Candidate word generation from partial EMG output |
| `word_disambiguate.py` | LLM-based word disambiguation from candidates |
| `config.py` | `OPENAI_API_KEY`, `MAX_AGENT_STEPS` |
| `main.py` | CLI entry point |
| `words_common.txt` | Common English word list for candidate generation |

---

## Relation to Subvocal task list

| Task list item | Status in treehacks |
|---|---|
| Phase 0 software demonstrator (Part 1.5) | Built — agent + PTT UI + browser control end-to-end |
| Bench experiment design and code (Part 1.6) | Built — filters, features, training skeleton, serial reader |
| Public-data replication and benchmarking (Part 1.7) | Done on EMG-UKA corpus; scripts in `scripts/`; results in `RESEARCH_REPORT.md` |
| ML pipeline / digital twin (Part 1.8) | Built — MockReader = software-only twin; training + inference pipeline complete |
| Firmware on simulator (Part 1.12) | Partial — ESP32 firmware done for USB serial; no BLE/OTA/simulator |
| Literature synthesis (Part 1.2) | Sources used: Kapur 2018 (AlterEgo), Jorgensen 2004 (NASA), EMG-UKA corpus |
| Architecture document (Part 1.3) | `HACKATHON_3SENSOR_PLAN.md` + `DATA_CAPTURE_PLAN.md` contain the reasoning |
| CAD, PCB, Figma mockups | Not started |
| All of Part 2 (company, market, legal) | Not started |
