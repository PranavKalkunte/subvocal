"""Benchmark runner for Subvocal SDK intent-reconstruction.

Evaluates the 50-example test suite across Heuristic and concrete LLM providers.
"""

import os
import time

from subvocal.context.schema import AppState, CalendarEvent, Contact, UIElement, UserContext
from subvocal.core.llm_providers import ClaudeProvider, GeminiProvider, LlamaProvider, OpenAIProvider
from subvocal.core.models import CommandToken
from subvocal.shorthand.decoder import heuristic_decode_phrase
from subvocal.shorthand.eval_set import EVAL_SET
from subvocal.shorthand.spec import compress_word


def run_benchmark():
    print("=" * 80)
    print("SUBVOCAL SDK INTENT-RECONSTRUCTION BENCHMARK RUNNER")
    print("=" * 80)

    # 1. Identify which providers can run live
    live_providers = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        live_providers.append("claude")
    if os.environ.get("OPENAI_API_KEY"):
        live_providers.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        live_providers.append("gemini")
    
    # We only check Ollama if OLLAMA_HOST or OLLAMA_TEST is configured
    if os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_TEST"):
        live_providers.append("llama")

    print(f"Live LLM Providers Detected: {', '.join(live_providers) if live_providers else 'NONE (Running in Simulated mode)'}")
    print(f"Total Test Cases: {len(EVAL_SET)}")
    print("-" * 80)

    # Initialize stats

    # We evaluate Heuristic + all detected live providers.
    # If no live providers exist, we evaluate a simulated LLM provider (Gemini or Claude in mock mode) to verify execution.
    eval_targets = ["heuristic"]
    if live_providers:
        eval_targets.extend(live_providers)
    else:
        eval_targets.append("simulated-llm")

    target_stats = {
        tgt: {
            "correct": 0,
            "latency_ms_sum": 0.0,
            "failures": []
        }
        for tgt in eval_targets
    }

    # Run cases
    for idx, case in enumerate(EVAL_SET):
        noisy = case["noisy"]
        expected = case["expected"]
        ui_el = case["ui_elements"]
        contacts_names = case["contacts"]
        calendar_events = case["calendar"]

        # Clean/compress noisy shorthand input
        noisy_clean = " ".join([compress_word(w) for w in noisy.split()])

        # Build full Pydantic UserContext
        context = UserContext(
            contacts=[Contact(id=str(i), name=name, shorthand_name=compress_word(name)) for i, name in enumerate(contacts_names)],
            calendar=[CalendarEvent(id=str(i), title=title, start_time="2026-06-08T12:00:00Z", end_time="2026-06-08T13:00:00Z", shorthand_title=compress_word(title)) for i, title in enumerate(calendar_events)],
            app_state=AppState(
                current_app="System",
                visible_elements=[UIElement(element_id=str(i), element_type="button", label=label, shorthand_label=compress_word(label)) for i, label in enumerate(ui_el)]
            )
        )

        tokens = [CommandToken(text=t, confidence=0.9, timestamp=time.time()) for t in noisy_clean.split()]

        # 1. Evaluate Heuristic Decoder
        start_t = time.perf_counter()
        heur_decoded, _ = heuristic_decode_phrase(
            noisy_clean,
            ui_elements=ui_el,
            contacts=contacts_names,
            calendar_events=calendar_events
        )
        heur_latency = (time.perf_counter() - start_t) * 1000.0
        target_stats["heuristic"]["latency_ms_sum"] += heur_latency

        if heur_decoded.lower().strip() == expected.lower().strip():
            target_stats["heuristic"]["correct"] += 1
        else:
            target_stats["heuristic"]["failures"].append({
                "expected": expected,
                "got": heur_decoded,
                "noisy": noisy
            })

        # 2. Evaluate LLM providers
        for tgt in eval_targets:
            if tgt == "heuristic":
                continue

            # Instantiate corresponding provider
            provider = None
            if tgt == "claude":
                provider = ClaudeProvider(model_name="claude-3-5-haiku-20241022")
            elif tgt == "openai":
                provider = OpenAIProvider(model_name="gpt-4o-mini")
            elif tgt == "gemini":
                provider = GeminiProvider(model_name="gemini-1.5-flash")
            elif tgt == "llama":
                provider = LlamaProvider(model_name="llama3")
            elif tgt == "simulated-llm":
                # Mock a successful intent resolution for 96% of cases and incorrect for 4%
                mock_text = expected if (idx % 25 != 0) else "GOTO error.com"
                provider = GeminiProvider(api_key="mock", mock_response=mock_text)

            start_t = time.perf_counter()
            try:
                intent = provider.reconstruct_intent(tokens, context)
                got_phrase = intent.resolved_text
            except Exception as e:
                got_phrase = f"ERROR: {e}"

            latency = (time.perf_counter() - start_t) * 1000.0
            target_stats[tgt]["latency_ms_sum"] += latency

            if got_phrase.lower().strip() == expected.lower().strip():
                target_stats[tgt]["correct"] += 1
            else:
                target_stats[tgt]["failures"].append({
                    "expected": expected,
                    "got": got_phrase,
                    "noisy": noisy
                })

    # Print Report
    total = len(EVAL_SET)
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print(f"{'Decoder / Provider':<25} | {'Accuracy':<10} | {'Avg Latency':<15}")
    print("-" * 80)
    for tgt, stats in target_stats.items():
        acc = (stats["correct"] / total) * 100.0
        avg_lat = stats["latency_ms_sum"] / total
        name = tgt.upper()
        print(f"{name:<25} | {acc:>8.1f}% | {avg_lat:>10.2f} ms")

    # Failure Analysis
    print("\n" + "=" * 80)
    print("FAILURE PROFILE DETAILS")
    print("=" * 80)
    for tgt, stats in target_stats.items():
        failures_count = len(stats["failures"])
        print(f"Provider {tgt.upper()}: {failures_count} failures")
        if failures_count > 0:
            for f in stats["failures"][:5]:  # show up to 5
                print(f"  Noisy Input     : '{f['noisy']}'")
                print(f"  Expected Intent : '{f['expected']}'")
                print(f"  Received Output : '{f['got']}'")
                print()
            if failures_count > 5:
                print(f"  ... and {failures_count - 5} more failures.")
        print("-" * 80)


if __name__ == "__main__":
    run_benchmark()
