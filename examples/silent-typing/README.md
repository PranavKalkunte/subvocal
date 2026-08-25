# Silent Typing Demo — Subvocal SDK

20-line end-to-end demo: synthetic sEMG → heuristic intent → MCP, no hardware or API keys.

```python
from subvocal import SubvocalPipeline
from subvocal.core.testing import MockActionExecutor, MockContextProvider, MockLLMProvider
from subvocal.hardware.drivers import SyntheticSignalGenerator
from subvocal.core.models import CommandToken
import time

hardware = SyntheticSignalGenerator(fs=1000.0, num_channels=8)

def classify(frame):
    arr = frame.to_numpy()
    if abs(arr).max() > 1.0:
        return CommandToken(text="gt", confidence=0.95, timestamp=time.time())
    return None

pipeline = SubvocalPipeline(
    hardware=hardware,
    classify_fn=classify,
    llm_provider=MockLLMProvider(),
    context_provider=MockContextProvider(),
    executor=MockActionExecutor(),
    phrase_timeout_seconds=0.5,
    on_action=lambda a,s: print("observed:", a.action_type, s),
)
# Thread-safe external injection (preferred for live sensors/MCP):
# pipeline.inject_token(CommandToken(text="clk", confidence=0.9, timestamp=time.time()))

hardware.start()
hardware.trigger_command("gt", duration_ms=120)
for _ in range(30):
    action = pipeline.step(window_ms=50)
    if action:
        print("Executed:", action.action_type, action.params)
        break
    time.sleep(0.05)
```

Swap `MockLLMProvider()` for `resolve_provider()` (Claude/Gemini/OpenAI) and `SyntheticSignalGenerator` for `OpenBCICytonDriver` / `DelsysTrignoDriver` / `FileReplayDriver` without changing pipeline code.

See `docs/content/configuration.md` for `SUBVOCAL_HARDWARE__SAMPLE_RATE` etc., and `benchmarks/eval_runner.py` for the 50-case 74% heuristic baseline.

