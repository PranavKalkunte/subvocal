"""Shorthand and phonetic specifications for the subvocal interface.

Defines the articulatory phonetic groups representing sEMG jaw/throat
activation clusters, and compression rules to construct the shorthand representation.
"""

from typing import Dict, List, Optional

# Articulatory groupings for simulated throat/jaw sEMG activation.
# G: Glottal / Vowels / Glides (Open mouth, vocal chord vibration, throat)
# L: Labials / Labiodentals (Lips, jaw front)
# A: Alveolars / Dentals (Tongue tip, teeth)
# V: Velars / Palatals (Tongue back, soft palate)
# R: Rhotics / Retroflex (Tongue curl)
LETTER_TO_GROUP: Dict[str, str] = {
    "a": "G", "e": "G", "i": "G", "o": "G", "u": "G", "y": "G", "h": "G",
    "b": "L", "p": "L", "m": "L", "w": "L", "f": "L", "v": "L",
    "t": "A", "d": "A", "n": "A", "s": "A", "z": "A", "l": "A",
    "k": "V", "g": "V", "j": "V", "c": "V", "q": "V", "x": "V",
    "r": "R"
}

ARTICULATORY_GROUPS: List[str] = ["G", "L", "A", "V", "R"]

# Command abbreviations for rapid entry
COMMAND_ABBREVIATIONS: Dict[str, str] = {
    "GOTO": "gt",
    "SEARCH": "srch",
    "CLICK": "clk",
    "SCROLL": "scrl",
    "TYPE": "typ",
    "ENTER": "ent",
    "CONFIRM": "cnfm",
    "CANCEL": "cncl",
    "BACK": "bk",
    "FORWARD": "fwd",
    "REFRESH": "rfsh",
    "ZOOM": "zm",
    "CLOSE": "cls",
    "COPY": "cpy",
    "PASTE": "pst",
    "UNDO": "und",
    "WAIT": "wt"
}

# Inverse command abbreviations
ABBREVIATION_TO_COMMAND: Dict[str, str] = {v: k for k, v in COMMAND_ABBREVIATIONS.items()}

# Common english word abbreviations used in subvocal shortcuts
COMMON_ABBREVIATIONS: Dict[str, str] = {
    "the": "th",
    "and": "nd",
    "for": "fr",
    "you": "u",
    "to": "t",
    "are": "r",
    "your": "ur",
    "with": "wth",
    "this": "ths",
    "that": "tht",
    "from": "frm",
    "have": "hv",
    "not": "nt",
    "but": "bt",
    "what": "wht",
    "about": "abt",
    "people": "ppl",
    "computer": "cmptr",
    "network": "ntwrk",
    "google": "ggl",
    "reddit": "rddt",
    "github": "gthb"
}


def compress_word(word: str) -> str:
    """Compress a single English word into its skeletal shorthand format.
    
    Rules:
    1. Lowercase the word.
    2. Check if the word has a common abbreviation.
    3. Otherwise:
       a. Keep the first letter.
       b. Remove subsequent vowels (a, e, i, o, u, y) and 'h'.
       c. Deduplicate consecutive identical consonants (e.g. 'gg' -> 'g').
    """
    w = word.strip().lower()
    if not w:
        return ""
    
    # Check common abbreviations
    if w in COMMON_ABBREVIATIONS:
        return COMMON_ABBREVIATIONS[w]
    
    # Check command names to see if they should be abbreviated
    w_upper = w.upper()
    if w_upper in COMMAND_ABBREVIATIONS:
        return COMMAND_ABBREVIATIONS[w_upper]
    
    # Compress standard word
    result = [w[0]]
    vowels = {"a", "e", "i", "o", "u", "y", "h"}
    
    for char in w[1:]:
        # Keep alphanumeric characters that are not vowels
        if char.isalnum():
            if char not in vowels:
                # Deduplicate consecutive characters
                if not result or result[-1] != char:
                    result.append(char)
        else:
            # Keep punctuation as separators (e.g. dots in domain names)
            result.append(char)
            
    return "".join(result)


def word_to_articulatory_sequence(shorthand_word: str) -> List[str]:
    """Convert a shorthand word to its articulatory group sequence.
    
    Non-alphabet characters (like numbers and punctuation) are preserved as-is.
    """
    seq = []
    for char in shorthand_word.lower():
        if char in LETTER_TO_GROUP:
            seq.append(LETTER_TO_GROUP[char])
        else:
            seq.append(char)
    return seq
