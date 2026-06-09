"""Test suite and benchmark harness for Subvocal Compressed Shorthand.

Evaluates intent-reconstruction accuracy of the custom phonetic alignment
and hybrid LLM decoder under different simulated sEMG noise levels.
"""

import os
import sys
from typing import Dict, List, Any

# Ensure parent directory is in search path

from subvocal.shorthand.vocab import get_command_list
from subvocal.shorthand.spec import compress_word
from subvocal.shorthand.simulator import phrase_to_noisy_shorthand
from subvocal.shorthand.decoder import heuristic_decode_phrase, hybrid_decode

# Define evaluation dataset
EVAL_DATASET = [
    {
        "target": "GOTO google.com",
        "web_context": ["google.com", "search", "gmail", "images"],
        "calendar": [],
        "contacts": [],
        "description": "Simple navigation with domain name"
    },
    {
        "target": "SEARCH neural networks",
        "web_context": ["neural", "networks", "deep", "learning", "models"],
        "calendar": [],
        "contacts": [],
        "description": "Search query with multi-word argument"
    },
    {
        "target": "CLICK submit",
        "web_context": ["username", "password", "remember", "submit", "login"],
        "calendar": [],
        "contacts": [],
        "description": "Button click action"
    },
    {
        "target": "CONFIRM",
        "web_context": ["are", "you", "sure", "ok", "cancel"],
        "calendar": [],
        "contacts": [],
        "description": "Single word confirmation"
    },
    {
        "target": "CANCEL",
        "web_context": ["discard", "changes", "save", "exit"],
        "calendar": [],
        "contacts": [],
        "description": "Single word cancel/abort"
    },
    {
        "target": "TYPE hello world",
        "web_context": [],
        "calendar": [],
        "contacts": [],
        "description": "Typing command with arguments"
    },
    {
        "target": "WAIT",
        "web_context": ["loading", "please", "wait"],
        "calendar": [],
        "contacts": [],
        "description": "Single word wait state"
    },
    {
        "target": "ZOOM",
        "web_context": ["in", "out", "reset"],
        "calendar": [],
        "contacts": [],
        "description": "Single word zoom command"
    },
    {
        "target": "CLOSE",
        "web_context": ["tab", "window", "close"],
        "calendar": [],
        "contacts": [],
        "description": "Window close action"
    },
    {
        "target": "UNDO",
        "web_context": ["edit", "history", "undo"],
        "calendar": [],
        "contacts": [],
        "description": "Single word undo"
    },
    {
        "target": "SCROLL",
        "web_context": ["up", "down", "left", "right"],
        "calendar": [],
        "contacts": [],
        "description": "Viewport scroll action"
    },
    {
        "target": "COPY",
        "web_context": ["select", "text", "copy"],
        "calendar": [],
        "contacts": [],
        "description": "Clipboard copy action"
    },
    {
        "target": "PASTE",
        "web_context": ["input", "paste"],
        "calendar": [],
        "contacts": [],
        "description": "Clipboard paste action"
    },
    {
        "target": "BACK",
        "web_context": ["previous", "back"],
        "calendar": [],
        "contacts": [],
        "description": "History navigation back"
    },
    {
        "target": "REFRESH",
        "web_context": ["reload", "refresh"],
        "calendar": [],
        "contacts": [],
        "description": "Reload browser viewport"
    },
    {
        "target": "FORWARD",
        "web_context": ["next", "forward"],
        "calendar": [],
        "contacts": [],
        "description": "History navigation forward"
    },
    {
        "target": "GOTO mail.google.com",
        "web_context": ["inbox", "sent", "drafts", "mail.google.com"],
        "calendar": ["meeting with team", "lunch"],
        "contacts": ["Alice Smith", "Bob Jones"],
        "description": "Complex navigation with sub-domain and contacts context"
    }
]

# Noise configurations
NOISE_LEVELS = {
    "0% Clean": {"sub_rate": 0.0, "del_rate": 0.0, "ins_rate": 0.0},
    "10% Light Noise": {"sub_rate": 0.08, "del_rate": 0.02, "ins_rate": 0.02},
    "20% Moderate Noise": {"sub_rate": 0.16, "del_rate": 0.04, "ins_rate": 0.04},
    "30% Severe Noise": {"sub_rate": 0.24, "del_rate": 0.08, "ins_rate": 0.08}
}


def run_benchmark():
    """Run full evaluation suite over all noise profiles and print diagnostic statistics."""
    print("=" * 80)
    print("SUBVOCAL COMPRESSED SHORTHAND RECONSTRUCTION BENCHMARK")
    print("=" * 80)
    
    # Check LLM availability
    has_llm = any(os.environ.get(k) for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    llm_info = "LLM Decoder: ACTIVE" if has_llm else "LLM Decoder: INACTIVE (No API key found in env)"
    print(llm_info)
    print(f"Target Vocabulary Size: {len(get_command_list())} commands")
    print(f"Test cases: {len(EVAL_DATASET)}")
    print("-" * 80)
    
    # Track metrics
    results: Dict[str, Dict[str, Any]] = {}
    
    # Run test cases
    for noise_name, params in NOISE_LEVELS.items():
        print(f"\nEvaluating profile: {noise_name} ...")
        
        heur_correct = 0
        hybrid_correct = 0
        total = len(EVAL_DATASET)
        
        detail_logs = []
        
        for item in EVAL_DATASET:
            target = item["target"]
            # Convert to compressed and then apply simulated noise
            clean_sh, noisy_sh = phrase_to_noisy_shorthand(
                target,
                sub_rate=params["sub_rate"],
                del_rate=params["del_rate"],
                ins_rate=params["ins_rate"]
            )
            
            # Heuristic decode
            heur_decoded, heur_conf = heuristic_decode_phrase(
                noisy_sh,
                web_context_words=item["web_context"],
                calendar_words=item["calendar"],
                contacts_words=item["contacts"]
            )
            
            # Hybrid decode (uses LLM if available, otherwise falls back to heuristic)
            hybrid_decoded, hybrid_conf, method = hybrid_decode(
                noisy_sh,
                web_context_words=item["web_context"],
                calendar_words=item["calendar"],
                contacts_words=item["contacts"]
            )
            
            # Check correctness (case-insensitive)
            is_heur_correct = heur_decoded.lower().strip() == target.lower().strip()
            is_hybrid_correct = hybrid_decoded.lower().strip() == target.lower().strip()
            
            if is_heur_correct:
                heur_correct += 1
            if is_hybrid_correct:
                hybrid_correct += 1
                
            detail_logs.append({
                "target": target,
                "clean_sh": clean_sh,
                "noisy_sh": noisy_sh,
                "heur_decoded": heur_decoded,
                "is_heur_correct": is_heur_correct,
                "hybrid_decoded": hybrid_decoded,
                "is_hybrid_correct": is_hybrid_correct,
                "method_used": method
            })
            
        # Log summary statistics
        heur_acc = (heur_correct / total) * 100
        hybrid_acc = (hybrid_correct / total) * 100
        
        results[noise_name] = {
            "heur_acc": heur_acc,
            "hybrid_acc": hybrid_acc,
            "logs": detail_logs
        }
        
        print(f"  Heuristic Accuracy: {heur_acc:.1f}% ({heur_correct}/{total})")
        if has_llm:
            print(f"  Hybrid LLM Accuracy: {hybrid_acc:.1f}% ({hybrid_correct}/{total})")
            
    # Print clean benchmark summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    
    if has_llm:
        print(f"{'Noise Profile':<25} | {'Heuristic Accuracy':<20} | {'Hybrid LLM Accuracy':<20}")
        print("-" * 80)
        for name, metrics in results.items():
            print(f"{name:<25} | {metrics['heur_acc']:>18.1f}% | {metrics['hybrid_acc']:>18.1f}%")
    else:
        print(f"{'Noise Profile':<25} | {'Heuristic Accuracy':<20}")
        print("-" * 55)
        for name, metrics in results.items():
            print(f"{name:<25} | {metrics['heur_acc']:>18.1f}%")
            
    print("=" * 80)
    
    # Print sample debug log for Moderate Noise to demonstrate pipeline
    print("\nSAMPLE RECONSTRUCTION TRACE (Moderate Noise):")
    print("-" * 80)
    mod_logs = results["20% Moderate Noise"]["logs"]
    for i, log in enumerate(mod_logs[:5]):
        print(f"Test case {i+1}: '{log['target']}'")
        print(f"  Clean shorthand: '{log['clean_sh']}'")
        print(f"  Noisy shorthand: '{log['noisy_sh']}'")
        print(f"  Heuristic decode: '{log['heur_decoded']}' ({'PASS' if log['is_heur_correct'] else 'FAIL'})")
        if has_llm:
            print(f"  Hybrid decode   : '{log['hybrid_decoded']}' ({'PASS' if log['is_hybrid_correct'] else 'FAIL'} via {log['method_used']})")
        print()
    print("-" * 80)


if __name__ == "__main__":
    run_benchmark()
