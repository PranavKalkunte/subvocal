"""Target command vocabulary for the Subvocal interface.

Defines a phonetically diverse set of 17 commands optimized for throat/jaw
surface Electromyography (sEMG) classification.
"""

from typing import Any

# Target vocabulary definition
COMMANDS: dict[str, dict[str, Any]] = {
    "GOTO": {
        "phonetic_ipa": "ˈɡoʊ.tuː",
        "articulatory_features": "Velar plosive + Glottal vowel + Dental/Alveolar plosive + Glottal vowel",
        "category": "Navigation",
        "description": "Navigate to a specific URL, app screen, or input field.",
    },
    "SEARCH": {
        "phonetic_ipa": "sɜːrtʃ",
        "articulatory_features": "Dental/Alveolar fricative + Glottal vowel + Rhotic + Velar/Palatal affricate",
        "category": "Navigation",
        "description": "Initiate a search query on the current web page or search engine.",
    },
    "CLICK": {
        "phonetic_ipa": "klɪk",
        "articulatory_features": "Velar plosive + Dental/Alveolar lateral + Glottal vowel + Velar plosive",
        "category": "Interaction",
        "description": "Click/select the targeted or currently highlighted user interface element.",
    },
    "SCROLL": {
        "phonetic_ipa": "skroʊl",
        "articulatory_features": "Dental/Alveolar fricative + Velar plosive + Rhotic + Glottal vowel + Dental/Alveolar lateral",
        "category": "Interaction",
        "description": "Scroll the active viewport (can take directions like UP, DOWN, LEFT, RIGHT).",
    },
    "TYPE": {
        "phonetic_ipa": "taɪp",
        "articulatory_features": "Dental/Alveolar plosive + Glottal vowel (diphthong) + Labial plosive",
        "category": "Input",
        "description": "Enter the subsequent simulated text or characters into the active input field.",
    },
    "ENTER": {
        "phonetic_ipa": "ˈɛntər",
        "articulatory_features": "Glottal vowel + Dental/Alveolar nasal + Dental/Alveolar plosive + Rhotic",
        "category": "Action",
        "description": "Submit a form, select a focused option, or press the virtual enter key.",
    },
    "CONFIRM": {
        "phonetic_ipa": "kənˈfɜːrm",
        "articulatory_features": "Velar plosive + Glottal vowel + Dental/Alveolar nasal + Glottal vowel + Rhotic + Labial nasal",
        "category": "Confirmation",
        "description": "Accept a dialog, approve a suggested action, or say yes.",
    },
    "CANCEL": {
        "phonetic_ipa": "ˈkænsəl",
        "articulatory_features": "Velar plosive + Glottal vowel + Dental/Alveolar nasal + Dental/Alveolar fricative + Dental/Alveolar lateral",
        "category": "Abort",
        "description": "Dismiss a dialog, abort a running agent process, or say no.",
    },
    "BACK": {
        "phonetic_ipa": "bæk",
        "articulatory_features": "Labial plosive + Glottal vowel + Velar plosive",
        "category": "Navigation",
        "description": "Go back to the previous page in the browser history or previous app state.",
    },
    "FORWARD": {
        "phonetic_ipa": "ˈfɔːrwərd",
        "articulatory_features": "Labial fricative + Glottal vowel + Rhotic + Labial semivowel + Glottal vowel + Dental/Alveolar plosive",
        "category": "Navigation",
        "description": "Go forward in the browser history or next app state.",
    },
    "REFRESH": {
        "phonetic_ipa": "rɪˈfrɛʃ",
        "articulatory_features": "Rhotic + Glottal vowel + Labial fricative + Rhotic + Glottal vowel + Velar/Palatal fricative",
        "category": "Control",
        "description": "Reload the current web page or refresh context data.",
    },
    "ZOOM": {
        "phonetic_ipa": "zuːm",
        "articulatory_features": "Dental/Alveolar fricative + Glottal vowel + Labial nasal",
        "category": "Control",
        "description": "Zoom in/out or adjust scale of elements (often takes IN or OUT).",
    },
    "CLOSE": {
        "phonetic_ipa": "kloʊz",
        "articulatory_features": "Velar plosive + Dental/Alveolar lateral + Glottal vowel + Dental/Alveolar fricative",
        "category": "Control",
        "description": "Close the active tab, window, or modal dialog.",
    },
    "COPY": {
        "phonetic_ipa": "ˈkɑːpi",
        "articulatory_features": "Velar plosive + Glottal vowel + Labial plosive + Glottal vowel",
        "category": "Clipboard",
        "description": "Copy selected text or UI element to the clipboard.",
    },
    "PASTE": {
        "phonetic_ipa": "peɪst",
        "articulatory_features": "Labial plosive + Glottal vowel + Dental/Alveolar fricative + Dental/Alveolar plosive",
        "category": "Clipboard",
        "description": "Paste the content of the clipboard into the current focus.",
    },
    "UNDO": {
        "phonetic_ipa": "ʌnˈduː",
        "articulatory_features": "Glottal vowel + Dental/Alveolar nasal + Dental/Alveolar plosive + Glottal vowel",
        "category": "Control",
        "description": "Revert the last command or typing action.",
    },
    "WAIT": {
        "phonetic_ipa": "weɪt",
        "articulatory_features": "Labial semivowel + Glottal vowel (diphthong) + Dental/Alveolar plosive",
        "category": "Control",
        "description": "Temporarily pause execution or wait for dynamic elements to load.",
    }
}


def get_command_list() -> list[str]:
    """Return a list of all command names in the vocabulary."""
    return list(COMMANDS.keys())


def get_command_details(name: str) -> dict[str, Any]:
    """Retrieve details for a specific command."""
    return COMMANDS.get(name.upper(), {})
