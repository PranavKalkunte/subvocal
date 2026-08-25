"""Minimal silent-typing demo — no hardware, no API keys."""
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
)

hardware.start()
hardware.trigger_command("gt", duration_ms=120)
for _ in range(30):
    action = pipeline.step(window_ms=50)
    if action:
        print("Executed:", action.action_type, action.params)
        break
    time.sleep(0.05)
else:
    print("No action — try increasing trigger_command duration or lowering classify threshold")

# Example thread-safe injection (e.g., from MCP server):
# pipeline.inject_token(CommandToken(text="clk", confidence=0.9, timestamp=time.time()))
# print(pipeline.process_phrase())
