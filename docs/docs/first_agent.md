# Your First Subvocal-Controlled Agent

In this tutorial, you will write a complete Python script to stand up a subvocal-controlled agent. The agent uses the `SyntheticSignalGenerator` to mimic raw biopotentials, filters them, feeds them to a mock classifier, and maps shorthand gesture strings to LLM-reconstructed system actions.

---

## 1. Complete Tutorial Code

Create a file named `my_first_agent.py` in the root of the repository:

```python
import time
from context.schema import UserContext, AppState
from core.models import Frame, CommandToken, Action
from core.interfaces import HardwareSource, LLMProvider, ActionExecutor, ContextProvider
from core.pipeline import SubvocalPipeline
from core.security import PolicyEngine, ConfidenceThresholdPolicy, CommandWhitelistPolicy
from hardware.drivers import SyntheticSignalGenerator

# 1. Define custom action executor to handle dispatched commands
class AgentExecutor(ActionExecutor):
    def execute(self, action: Action):
        print(f"\n[⚡ EXECUTION ACTIVE] Running action: {action.action_type}")
        print(f"Arguments: {action.params.get('arguments')}")
        print(f"Confidence score: {action.params.get('confidence')}")

    def can_execute(self, action: Action) -> bool:
        return action.action_type in ["click", "scroll", "type", "search"]

# 2. Define custom context provider representing user desktop state
class AgentContext(ContextProvider):
    def get_context(self) -> UserContext:
        return UserContext(
            app_state=AppState(current_app="Browser", page_title="Docusaurus Docs")
        )
    def get_provider_name(self) -> str:
        return "agent_context"

# 3. Define local intent decoder (simulated LLM provider)
class AgentLLM(LLMProvider):
    def reconstruct_intent(self, tokens: list, context: UserContext) -> "Intent":
        from core.models import Intent
        shorthand_phrase = " ".join([t.text for t in tokens])
        print(f"\n[🧠 DECODER] Shorthand sequence received: '{shorthand_phrase}'")
        
        # Simple rule-based resolver simulating the hybrid decoder
        if "clk" in shorthand_phrase:
            return Intent(
                command="CLICK",
                arguments=["link: Getting Started"],
                resolved_text="Click on the Getting Started link",
                raw_shorthand=shorthand_phrase,
                confidence=0.95,
                timestamp=time.time()
            )
        return Intent(
            command="SCROLL",
            arguments=["down"],
            resolved_text="Scroll down the page",
            raw_shorthand=shorthand_phrase,
            confidence=0.72,  # Trigger threshold limits in testing
            timestamp=time.time()
        )
    def get_provider_name(self) -> str:
        return "agent_llm"

def main():
    # A. Setup Hardware Signal Source
    hardware = SyntheticSignalGenerator()
    hardware.start()
    
    # B. Define classification rule: if signal amplitude on channel 0 bursts, emit "clk"
    def classify_step(frame: Frame) -> CommandToken:
        import numpy as np
        data = frame.to_numpy()
        if len(data) == 0:
            return None
        rms = np.sqrt(np.mean(data[:, 0] ** 2))
        if rms > 15.0:  # Voltages micro-volts
            print("[👄 GESTURE] Amplitude contraction detected on Channel 0!")
            return CommandToken(text="clk", confidence=0.98, timestamp=time.time())
        return None

    # C. Establish Security Gate Policy (e.g. reject confidence < 0.8)
    policy_engine = PolicyEngine([
        ConfidenceThresholdPolicy(threshold=0.8),
        CommandWhitelistPolicy(allowed_commands=["CLICK", "SCROLL"])
    ])

    # D. Initialize pipeline
    pipeline = SubvocalPipeline(
        hardware=hardware,
        classify_fn=classify_step,
        llm_provider=AgentLLM(),
        context_provider=AgentContext(),
        executor=AgentExecutor(),
        phrase_timeout_seconds=1.2,
        policy_engine=policy_engine
    )

    print("\n🚀 Subvocal Agent running! Press Ctrl+C to stop.")
    print("Action sequence: Triggers 'clk' gesture on contraction, then auto-decodes after 1.2s.")
    
    try:
        # Loop mimicking real-time processing
        while True:
            # Step A: Feed simulated bio-signals
            pipeline.step(window_ms=100)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping Agent...")
    finally:
        hardware.stop()

if __name__ == "__main__":
    main()
```

## 2. Running the Agent

Run the script:

```bash
PYTHONPATH=sdk python3 my_first_agent.py
```

To trigger a gesture, wait for the synthetic generator to inject contractions or force a muscle contraction. The pipeline will:
1. Ingest sEMG micro-signals.
2. Trigger the `classify_step` producing a `clk` token.
3. Wait for the `phrase_timeout_seconds` (1.2s of silence).
4. Extract the user context ("Browser").
5. Map tokens to an action ("CLICK").
6. Verify security policies (confidence 0.95 >= 0.8 threshold).
7. Dispatch the action to the terminal log.
