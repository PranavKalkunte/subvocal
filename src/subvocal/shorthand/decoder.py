"""Hybrid decoder for subvocal compressed shorthand.

Combines an articulatory-distance heuristic alignment algorithm (dynamic programming)
with a model-agnostic LLM reconstruction layer.
"""

import json
import logging
import os
import urllib.error
import urllib.request

from subvocal.shorthand.spec import (
    ABBREVIATION_TO_COMMAND,
    LETTER_TO_GROUP,
    compress_word,
)
from subvocal.shorthand.vocab import COMMANDS

logger = logging.getLogger(__name__)

COMMON_WORDS_DICT = [
    "google", "search", "click", "scroll", "type", "enter", "confirm", "cancel",
    "back", "forward", "refresh", "zoom", "close", "copy", "paste", "undo", "wait",
    "the", "and", "for", "you", "to", "are", "your", "with", "this", "that", "from",
    "have", "not", "but", "what", "about", "people", "computer", "network", "github",
    "reddit", "youtube", "facebook", "login", "submit", "password", "username", "email",
    "inbox", "next", "previous", "settings", "profile", "logout", "ok", "yes", "no",
    "hello", "world", "chat", "agent", "music", "play", "pause", "stop",
    "volume", "weather", "calendar", "contacts", "today", "tomorrow", "reminder",
    "timer", "minutes", "seconds", "alarm", "call", "message", "send", "text", "mail",
    # Added directional, zoom, UI, and common website terms
    "up", "down", "left", "right", "in", "out", "tab", "window", "cart",
    "google.com", "github.com", "mail.google.com", "reddit.com", "youtube.com",
    "facebook.com", "wikipedia.org", "treehacks.org", "localhost:3000", "news.com"
]


DOMAIN_EXTENSIONS = {
    "cm": "com",
    "com": "com",
    "rg": "org",
    "org": "org",
    "nt": "net",
    "net": "net",
    "dv": "dev",
    "dev": "dev",
    "edu": "edu",
    "gov": "gov"
}



def articulatory_distance(word1: str, word2: str) -> float:
    """Calculates Levenshtein distance customized for sEMG articulatory phonetic groups.
    
    Substitutions between letters in the same articulatory group cost 0.25 (highly likely sEMG error).
    Substitutions across different articulatory groups cost 1.0.
    Deletion in candidate (extra character in query) costs 1.0.
    Insertion in candidate (skipped character in query shorthand) costs 0.4.
    """
    m, n = len(word1), len(word2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = float(i) * 1.0  # Deletion in candidate (query has extra characters)
    for j in range(n + 1):
        dp[0][j] = float(j) * 0.4  # Insertion in candidate (query omitted characters)
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1 = word1[i - 1].lower()
            c2 = word2[j - 1].lower()
            
            if c1 == c2:
                cost = 0.0
            elif c1 in LETTER_TO_GROUP and c2 in LETTER_TO_GROUP:
                if LETTER_TO_GROUP[c1] == LETTER_TO_GROUP[c2]:
                    cost = 0.25  # Cheap substitution (same articulatory cluster)
                else:
                    cost = 1.0   # Expensive substitution
            else:
                cost = 1.0  # Non-alphabet substitution cost
                
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,        # Deletion in candidate (query has extra char)
                dp[i][j - 1] + 0.4,        # Insertion in candidate (query omitted char)
                dp[i - 1][j - 1] + cost      # Substitution
            )
            
    return dp[m][n]


def find_best_shorthand_match(noisy_token: str, candidate_words: list[str]) -> list[tuple[str, float]]:
    """Scores a noisy shorthand token against a list of candidate words.
    
    Compresses candidates to shorthand and computes articulatory distance.
    Returns a list of tuples (candidate, distance) sorted by distance.
    """
    results = []
    seen = set()
    for candidate in candidate_words:
        cand_lower = candidate.lower()
        if cand_lower in seen:
            continue
        seen.add(cand_lower)
        
        cand_sh = compress_word(cand_lower)
        dist = articulatory_distance(noisy_token, cand_sh)
        results.append((candidate, dist))
        
    # Sort by distance (lower is better), and resolve ties by length similarity
    results.sort(key=lambda x: (x[1], abs(len(noisy_token) - len(compress_word(x[0])))))
    return results


def heuristic_decode_phrase(
    noisy_phrase: str,
    web_context_words: list[str] | None = None,
    calendar_words: list[str] | None = None,
    contacts_words: list[str] | None = None,
    # Advanced structured contexts
    ui_elements: list[str] | None = None,
    contacts: list[str] | None = None,
    calendar_events: list[str] | None = None
) -> tuple[str, float]:
    """Decodes a full noisy shorthand phrase using articulatory heuristics.
    
    Uses Command-Aware Context Prioritization and Phrase-Level Matching.
    If structured contexts are provided, they are prioritized based on the command.
    Otherwise, falls back to flat context words.
    """
    tokens = noisy_phrase.strip().split()
    if not tokens:
        return "", 0.0
        
    # 1. Decode Command Token (first word)
    cmd_token = tokens[0]
    cmd_candidates = list(ABBREVIATION_TO_COMMAND.values())
    cmd_shorthand_map = {compress_word(cmd): cmd for cmd in cmd_candidates}
    
    best_cmd_matches = []
    for sh, cmd in cmd_shorthand_map.items():
        dist = articulatory_distance(cmd_token, sh)
        best_cmd_matches.append((cmd, dist))
    best_cmd_matches.sort(key=lambda x: x[1])
    
    best_cmd, cmd_dist = best_cmd_matches[0]
    
    if len(tokens) == 1:
        raw_len = len(cmd_token)
        confidence = max(0.0, 1.0 - (cmd_dist / max(1.0, float(raw_len))))
        return best_cmd, confidence
        
    # 2. Extract arguments
    arg_tokens = tokens[1:]
    noisy_args_shorthand = "".join(arg_tokens).lower()
    
    # 3. Command-Aware Prioritization
    def expand_pool(pool_list: list[str]) -> list[str]:
        expanded = []
        for item in pool_list:
            expanded.append(item)
            words = item.split()
            if len(words) > 1:
                for w in words:
                    if len(w) > 2:  # only keep substantial words
                        expanded.append(w)
        return expanded

    prioritized_pool = []
    if best_cmd == "CLICK" and ui_elements:
        prioritized_pool = expand_pool(ui_elements)
    elif best_cmd == "SCROLL":
        prioritized_pool = ["up", "down", "left", "right"]
    elif best_cmd == "ZOOM":
        prioritized_pool = ["in", "out"]
    elif best_cmd == "GOTO":
        prioritized_pool = ["google.com", "github.com", "mail.google.com", "reddit.com", "youtube.com", "facebook.com", "wikipedia.org", "treehacks.org", "localhost:3000", "news.com"]
        if ui_elements:
            prioritized_pool.extend(expand_pool(ui_elements))
    elif best_cmd == "TYPE" and contacts:
        prioritized_pool = expand_pool(contacts)
    elif best_cmd == "SEARCH":
        if calendar_events:
            prioritized_pool.extend(expand_pool(calendar_events))
        if contacts:
            prioritized_pool.extend(expand_pool(contacts))
            
    # Helper for phrase-level matching
    def match_phrase_pool(pool: list[str], max_dist_per_char: float = 0.35) -> tuple[str, float] | None:
        best_match = None
        min_dist = 999.0
        for item in pool:
            # Compress entire item phrase by joining compressed words
            compressed_parts = [compress_word(w) for w in item.split() if w]
            compressed_item = "".join(compressed_parts).lower()
            
            dist = articulatory_distance(noisy_args_shorthand, compressed_item)
            if dist < min_dist:
                min_dist = dist
                best_match = item
                
        # Only accept if distance is reasonable relative to the shorthand length
        if best_match and min_dist <= max(1.5, len(noisy_args_shorthand) * max_dist_per_char):
            # Calculate confidence
            conf = max(0.0, 1.0 - (min_dist / max(1.0, len(noisy_args_shorthand))))
            return best_match, conf
        return None

    # Try matching in the prioritized pool
    if prioritized_pool:
        res = match_phrase_pool(prioritized_pool)
        if res:
            best_match, arg_conf = res
            combined_conf = (max(0.0, 1.0 - (cmd_dist / max(1.0, len(cmd_token)))) + arg_conf) / 2.0
            return f"{best_cmd} {best_match}", combined_conf

    # Try matching in all provided structured pools combined
    # Skip all_structured combined search for SEARCH and TYPE to avoid false positive UI/contact clicks
    if best_cmd not in ("SEARCH", "TYPE"):
        all_structured = []
        if ui_elements:
            all_structured.extend(expand_pool(ui_elements))
        if contacts:
            all_structured.extend(expand_pool(contacts))
        if calendar_events:
            all_structured.extend(expand_pool(calendar_events))
            
        if all_structured:
            res = match_phrase_pool(all_structured)
            if res:
                best_match, arg_conf = res
                combined_conf = (max(0.0, 1.0 - (cmd_dist / max(1.0, len(cmd_token)))) + arg_conf) / 2.0
                return f"{best_cmd} {best_match}", combined_conf

    # 4. Fallback to Flat Token-by-Token Decoding (Original logic)
    decoded_words = [best_cmd]
    total_distance = cmd_dist
    
    arg_candidates = COMMON_WORDS_DICT.copy()
    if web_context_words:
        arg_candidates.extend(web_context_words)
    if calendar_words:
        arg_candidates.extend(calendar_words)
    if contacts_words:
        arg_candidates.extend(contacts_words)
        
    # Dynamically inject individual words from active structured contexts
    if ui_elements:
        for item in ui_elements:
            arg_candidates.extend(item.split())
    if contacts:
        for item in contacts:
            arg_candidates.extend(item.split())
    if calendar_events:
        for item in calendar_events:
            arg_candidates.extend(item.split())
            
    for arg_token in arg_tokens:
        # Check if this is a dot-separated domain name
        if "." in arg_token and not arg_token.replace(".", "").isdigit():
            parts = arg_token.split(".")
            decoded_parts = []
            for part in parts:
                if part.lower() in DOMAIN_EXTENSIONS:
                    decoded_parts.append(DOMAIN_EXTENSIONS[part.lower()])
                else:
                    matches = find_best_shorthand_match(part, arg_candidates)
                    if matches:
                        best_part, dist = matches[0]
                        decoded_parts.append(best_part)
                        total_distance += dist
                    else:
                        decoded_parts.append(part)
                        total_distance += len(part)
            decoded_words.append(".".join(decoded_parts))
        else:
            matches = find_best_shorthand_match(arg_token, arg_candidates)
            if matches:
                best_match, dist = matches[0]
                decoded_words.append(best_match)
                total_distance += dist
            else:
                decoded_words.append(arg_token)
                total_distance += len(arg_token)
                
    raw_len = len(noisy_phrase)
    confidence = max(0.0, 1.0 - (total_distance / max(1.0, float(raw_len))))
    return " ".join(decoded_words), confidence


def call_llm_api(provider: str, api_key: str, model: str | None, prompt: str) -> str | None:
    """Helper to perform model-agnostic LLM calls via urllib without external SDK dependencies."""
    try:
        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = json.dumps({
                "model": model or "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 100
            }).encode("utf-8")
            
        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = json.dumps({
                "model": model or "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }).encode("utf-8")
            
        elif provider == "gemini":
            model_name = model or "gemini-1.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            data = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0}
            }).encode("utf-8")
        else:
            return None

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            
            if provider == "openai":
                return res_json["choices"][0]["message"]["content"].strip()
            elif provider == "anthropic":
                return res_json["content"][0]["text"].strip()
            elif provider == "gemini":
                return res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                
    except urllib.error.URLError as e:
        logger.error("LLM decoder HTTP call failed for %s: %s", provider, e)
    except Exception as e:
        logger.error("LLM decoder general failure for %s: %s", provider, e)
        
    return None


def reconstruct_intent_llm(
    noisy_input: str,
    heuristic_candidate: str,
    web_context: str | None = None,
    calendar: str | None = None,
    contacts: str | None = None,
    history: str | None = None
) -> str | None:
    """Reconstructs the target command phrase from noisy shorthand using a model-agnostic LLM.
    
    Discovers API keys dynamically from the environment.
    """
    # 1. Detect provider and credentials
    provider, api_key, model = None, None, None
    
    if os.environ.get("GEMINI_API_KEY"):
        provider = "gemini"
        api_key = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("AGENT_MODEL") or "gemini-1.5-flash"
    elif os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("AGENT_MODEL") or "gpt-4o"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        model = os.environ.get("AGENT_MODEL") or "claude-3-5-sonnet-20241022"
        
    if not provider or not api_key:
        # No API key found, unable to run LLM reconstruction step
        return None
        
    from subvocal.core.prompts import PromptManager
    prompt = PromptManager().format_prompt(
        noisy_input=noisy_input,
        heuristic_recommendation=heuristic_candidate,
        web_context=web_context or "(none)",
        calendar=calendar or "(none)",
        contacts=contacts or "(none)",
        history=history or "(none)",
    )
    result = call_llm_api(provider, api_key, model, prompt)
    if result:
        # Strip any quotes or whitespace the LLM might have returned
        result = result.replace('"', '').replace("'", "").strip()
    return result


def hybrid_decode(
    noisy_phrase: str,
    web_context_words: list[str] | None = None,
    calendar_words: list[str] | None = None,
    contacts_words: list[str] | None = None,
    history: str | None = None,
    # Advanced structured contexts
    ui_elements: list[str] | None = None,
    contacts: list[str] | None = None,
    calendar_events: list[str] | None = None
) -> tuple[str, float, str]:
    """Combines heuristic alignment and LLM disambiguation for premium accuracy.
    
    Returns:
        A tuple of (decoded_phrase, confidence, method_used)
        Method used is either "heuristic" or "llm".
    """
    # 1. Run heuristic decode
    heur_phrase, heur_conf = heuristic_decode_phrase(
        noisy_phrase,
        web_context_words=web_context_words,
        calendar_words=calendar_words,
        contacts_words=contacts_words,
        ui_elements=ui_elements,
        contacts=contacts,
        calendar_events=calendar_events
    )
    
    # 2. Check if LLM is available and should be used
    # We trigger the LLM if confidence is low (< 0.90) or if we want to run a complete reconstruction
    # Let's check if any API key is configured
    has_key = any(os.environ.get(k) for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
    
    if has_key and heur_conf < 0.9:
        flat_web = web_context_words
        if not flat_web and ui_elements:
            flat_web = ui_elements
            
        flat_cal = calendar_words
        if not flat_cal and calendar_events:
            flat_cal = calendar_events
            
        flat_con = contacts_words
        if not flat_con and contacts:
            flat_con = contacts
            
        web_context_str = ", ".join(flat_web) if flat_web else None
        calendar_str = ", ".join(flat_cal) if flat_cal else None
        contacts_str = ", ".join(flat_con) if flat_con else None
        
        llm_phrase = reconstruct_intent_llm(
            noisy_phrase,
            heur_phrase,
            web_context=web_context_str,
            calendar=calendar_str,
            contacts=contacts_str,
            history=history
        )
        if llm_phrase:
            # Check if LLM output starts with a valid command
            cmd_part = llm_phrase.split()[0].upper() if llm_phrase.split() else ""
            if cmd_part in COMMANDS:
                # LLM successfully reconstructed a valid command!
                return llm_phrase, 0.95, "llm"
                
    # Fallback to heuristic
    return heur_phrase, heur_conf, "heuristic"
