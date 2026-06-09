"""Evaluation runner for Subvocal Shorthand intent-reconstruction.

Executes the 50-example test suite across noise profiles and reports
accuracy, average latencies, category breakdowns, and trace failure analysis.
"""

import os
import time

from subvocal.shorthand.decoder import heuristic_decode_phrase, hybrid_decode
from subvocal.shorthand.eval_set import EVAL_SET
from subvocal.shorthand.spec import compress_word


def run_evaluation():
    print("=" * 80)
    print("SUBVOCAL SHORTHAND INTENT-RECONSTRUCTION EVALUATION RUNNER")
    print("=" * 80)
    
    has_llm = any(os.environ.get(k) for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    llm_info = "LLM Decoder: ACTIVE" if has_llm else "LLM Decoder: INACTIVE (No API keys found in env)"
    print(llm_info)
    print(f"Total Test Cases: {len(EVAL_SET)}")
    print("-" * 80)
    
    # Categories tracking
    categories = ["GOTO", "CLICK", "SCROLL", "SEARCH", "TYPE", "CLOSE", "CONFIRM", "CANCEL", "UNDO", "COPY", "PASTE"]
    cat_stats = {cat: {"total": 0, "heur_correct": 0, "hybrid_correct": 0} for cat in categories}
    cat_stats["OTHER"] = {"total": 0, "heur_correct": 0, "hybrid_correct": 0}
    
    heur_correct_count = 0
    hybrid_correct_count = 0
    
    total_heur_time = 0.0
    total_hybrid_time = 0.0
    
    failures = []
    
    # Process test cases
    for idx, case in enumerate(EVAL_SET):
        noisy = case["noisy"]
        # Ensure the noisy input from the test set is properly compressed first
        noisy_parts = [compress_word(w) for w in noisy.split()]
        noisy_clean = " ".join(noisy_parts)
        
        expected = case["expected"]
        ui_el = case["ui_elements"]
        contacts = case["contacts"]
        calendar = case["calendar"]
        
        # Categorize
        cmd = expected.split()[0].upper()
        category = cmd if cmd in cat_stats else "OTHER"
        cat_stats[category]["total"] += 1
        
        # 1. Run Heuristic decode
        start_t = time.perf_counter()
        heur_decoded, heur_conf = heuristic_decode_phrase(
            noisy_clean,
            ui_elements=ui_el,
            contacts=contacts,
            calendar_events=calendar
        )
        heur_time = (time.perf_counter() - start_t) * 1000.0  # ms
        total_heur_time += heur_time
        
        # 2. Run Hybrid decode
        start_t = time.perf_counter()
        hybrid_decoded, hybrid_conf, method = hybrid_decode(
            noisy_clean,
            ui_elements=ui_el,
            contacts=contacts,
            calendar_events=calendar
        )
        hybrid_time = (time.perf_counter() - start_t) * 1000.0  # ms
        total_hybrid_time += hybrid_time
        
        # Check correctness
        is_heur_correct = heur_decoded.lower().strip() == expected.lower().strip()
        is_hybrid_correct = hybrid_decoded.lower().strip() == expected.lower().strip()
        
        if is_heur_correct:
            heur_correct_count += 1
            cat_stats[category]["heur_correct"] += 1
            
        if is_hybrid_correct:
            hybrid_correct_count += 1
            cat_stats[category]["hybrid_correct"] += 1
        else:
            failures.append({
                "index": idx + 1,
                "description": case["description"],
                "noisy": noisy,
                "expected": expected,
                "heur_decoded": heur_decoded,
                "hybrid_decoded": hybrid_decoded,
                "method_used": method
            })
            
    # Compute aggregates
    total = len(EVAL_SET)
    heur_acc = (heur_correct_count / total) * 100
    hybrid_acc = (hybrid_correct_count / total) * 100
    avg_heur_latency = total_heur_time / total
    avg_hybrid_latency = total_hybrid_time / total
    
    # Print high-level summary
    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    print(f"Heuristic Decoder Accuracy: {heur_acc:.1f}% ({heur_correct_count}/{total})")
    print(f"Heuristic Avg Latency     : {avg_heur_latency:.2f} ms")
    if has_llm:
        print(f"Hybrid LLM Accuracy       : {hybrid_acc:.1f}% ({hybrid_correct_count}/{total})")
        print(f"Hybrid Avg Latency        : {avg_hybrid_latency:.2f} ms")
    print("-" * 80)
    
    # Print category breakdown table
    print("\nCATEGORY BREAKDOWN:")
    print("-" * 80)
    if has_llm:
        print(f"{'Category':<15} | {'Count':<5} | {'Heuristic Accuracy':<20} | {'Hybrid Accuracy':<20}")
        print("-" * 80)
        for cat, stats in cat_stats.items():
            if stats["total"] > 0:
                h_acc = (stats["heur_correct"] / stats["total"]) * 100
                hy_acc = (stats["hybrid_correct"] / stats["total"]) * 100
                print(f"{cat:<15} | {stats['total']:<5} | {h_acc:>18.1f}% | {hy_acc:>18.1f}%")
    else:
        print(f"{'Category':<15} | {'Count':<5} | {'Heuristic Accuracy':<20}")
        print("-" * 47)
        for cat, stats in cat_stats.items():
            if stats["total"] > 0:
                h_acc = (stats["heur_correct"] / stats["total"]) * 100
                print(f"{cat:<15} | {stats['total']:<5} | {h_acc:>18.1f}%")
                
    # Print failures analysis if any
    if failures:
        print("\n" + "=" * 80)
        print(f"FAILURE ANALYSIS ({len(failures)} failures)")
        print("=" * 80)
        for fail in failures:
            print(f"Case {fail['index']}: {fail['description']}")
            print(f"  Noisy Shorthand : '{fail['noisy']}'")
            print(f"  Expected Intent : '{fail['expected']}'")
            print(f"  Heuristic Output: '{fail['heur_decoded']}'")
            if has_llm:
                print(f"  Hybrid Output   : '{fail['hybrid_decoded']}' (via {fail['method_used']})")
            print()
    else:
        print("\n" + "=" * 80)
        print("ALL 50 TEST CASES PASSED SUCCESSFULLY!")
        print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
