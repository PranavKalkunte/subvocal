"""Noisy channel simulator for subvocal compressed shorthand.

Converts target command phrases to compressed shorthand and applies realistic
articulatory sEMG noise (substitutions, insertions, and deletions).
"""

import random
from typing import List, Tuple
from shorthand.spec import LETTER_TO_GROUP, compress_word

# Group reverse mappings for substitution candidate lookup
GROUP_TO_LETTERS: dict[str, List[str]] = {}
for letter, grp in LETTER_TO_GROUP.items():
    GROUP_TO_LETTERS.setdefault(grp, []).append(letter)


def get_articulatory_substitution(char: str) -> str:
    """Get a random alternative character from the same articulatory group to simulate sEMG confusion."""
    c = char.lower()
    if c not in LETTER_TO_GROUP:
        return char  # Keep numbers/punctuation unchanged
    
    group = LETTER_TO_GROUP[c]
    candidates = [letter for letter in GROUP_TO_LETTERS[group] if letter != c]
    
    if not candidates:
        return char
    
    sub = random.choice(candidates)
    return sub.upper() if char.isupper() else sub


def simulate_emg_noise(
    shorthand_word: str,
    sub_rate: float = 0.10,
    del_rate: float = 0.05,
    ins_rate: float = 0.05
) -> str:
    """Applies deletion, insertion, and substitution noise to a shorthand word.
    
    - Substitution: Replaces char with another from the same articulatory group (sEMG confusion).
    - Deletion: Drops the character (packet drop / weak signal).
    - Insertion: Inserts a random character (muscle twitch / swallow artifact).
    """
    noisy_chars = []
    
    for char in shorthand_word:
        # 1. Check for Deletion
        if random.random() < del_rate:
            continue
        
        # 2. Check for Insertion before this char
        if random.random() < ins_rate:
            # Insert a random phonetic letter
            rand_letter = random.choice(list(LETTER_TO_GROUP.keys()))
            if char.isupper():
                rand_letter = rand_letter.upper()
            noisy_chars.append(rand_letter)
            
        # 3. Check for Substitution
        if random.random() < sub_rate:
            noisy_chars.append(get_articulatory_substitution(char))
        else:
            noisy_chars.append(char)
            
    # Handle edge case where entire word got deleted
    if not noisy_chars and shorthand_word:
        # Re-run simulator with zero deletion to ensure we have something
        return simulate_emg_noise(shorthand_word, sub_rate, 0.0, ins_rate)
        
    return "".join(noisy_chars)


def phrase_to_noisy_shorthand(
    phrase: str,
    sub_rate: float = 0.10,
    del_rate: float = 0.05,
    ins_rate: float = 0.05
) -> Tuple[str, str]:
    """Convert a full phrase to clean shorthand and simulated noisy shorthand.
    
    Returns:
        A tuple of (clean_shorthand_phrase, noisy_shorthand_phrase)
    """
    words = phrase.split()
    clean_words = []
    noisy_words = []
    
    for word in words:
        clean_sh = compress_word(word)
        noisy_sh = simulate_emg_noise(clean_sh, sub_rate, del_rate, ins_rate)
        clean_words.append(clean_sh)
        noisy_words.append(noisy_sh)
        
    return " ".join(clean_words), " ".join(noisy_words)
