"""Integration test for Context-Aware Subvocal Shorthand Decoding.

Tests how the ContextManager extracts plain-text vocabulary from active app state,
contacts, and calendar, feeding it to the articulatory decoder to resolve noisy shorthand.
"""

import os
import sys

# Ensure parent directory is in search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from context.mock_data import generate_mock_context
from context.manager import ContextManager
from shorthand.decoder import heuristic_decode_phrase, hybrid_decode

# Test cases representing noisy user inputs that require contextual vocabulary
CONTEXT_TEST_CASES = [
    {
        "noisy": "clk sbt",
        "expected": "CLICK submit",
        "description": "Button click with alveolar substitution error ('sbt' for 'submit')"
    },
    {
        "noisy": "clk sgn n",
        "expected": "CLICK Sign In",
        "description": "Multi-word button click on visible link ('sgn n' for 'Sign In')"
    },
    {
        "noisy": "gt mail.google.com",
        "expected": "GOTO mail.google.com",
        "description": "Navigation with exact URL matching browser state"
    },
    {
        "noisy": "srch bsnl",
        "expected": "SEARCH BioSignals",
        "description": "Search for calendar event abbreviation ('bsnl' for 'BioSignals')"
    },
    {
        "noisy": "typ alc smth",
        "expected": "TYPE Alice Smith",
        "description": "Typing a contact name from address book ('alc smth' for 'Alice Smith')"
    }
]


def run_context_tests():
    print("=" * 80)
    print("SUBVOCAL CONTEXT-AWARE DECODING TEST")
    print("=" * 80)
    
    # 1. Initialize mock context
    print("Generating mock user context...")
    context = generate_mock_context()
    manager = ContextManager(context)
    
    # 2. Extract vocabulary seeds from context
    context_words = manager.get_all_context_words()
    print(f"Extracted {len(context_words)} candidate words from active context:")
    print(f"  {context_words}")
    print("-" * 80)
    
    # Check LLM key
    has_llm = any(os.environ.get(k) for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    
    # 3. Execute test cases
    passed = 0
    total = len(CONTEXT_TEST_CASES)
    
    for i, case in enumerate(CONTEXT_TEST_CASES):
        noisy = case["noisy"]
        expected = case["expected"]
        print(f"Test Case {i+1}: {case['description']}")
        print(f"  Noisy input: '{noisy}'")
        
        # Extract structured contexts
        ui_elements = manager.get_visible_element_labels()
        contacts = manager.get_contact_names()
        calendar_events = manager.get_calendar_titles()
        # Add special URLs for GOTO commands
        navigation_urls = ["mail.google.com", "google.com"]

        # Heuristic decode with context seeds
        heur_decoded, heur_conf = heuristic_decode_phrase(
            noisy,
            web_context_words=navigation_urls,
            ui_elements=ui_elements,
            contacts=contacts,
            calendar_events=calendar_events
        )
        
        # Hybrid decode
        hybrid_decoded, hybrid_conf, method = hybrid_decode(
            noisy,
            web_context_words=navigation_urls,
            ui_elements=ui_elements,
            contacts=contacts,
            calendar_events=calendar_events
        )
        
        # Evaluate
        is_correct = hybrid_decoded.lower().strip() == expected.lower().strip()
        if is_correct:
            passed += 1
            status = "PASS"
        else:
            status = f"FAIL (Expected: '{expected}')"
            
        print(f"  Heuristic Decode: '{heur_decoded}' (conf: {heur_conf:.2f})")
        if has_llm:
            print(f"  Hybrid Decode   : '{hybrid_decoded}' (via {method})")
        print(f"  Verdict         : {status}")
        print()
        
    print("-" * 80)
    print(f"Context-Aware Decoding Accuracy: {(passed / total) * 100:.1f}% ({passed}/{total})")
    print("=" * 80)


if __name__ == "__main__":
    run_context_tests()
