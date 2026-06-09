---
title: shorthand.simulator
sidebar_label: simulator
---

Noisy channel simulator for subvocal compressed shorthand.

Converts target command phrases to compressed shorthand and applies realistic
articulatory sEMG noise (substitutions, insertions, and deletions).

## Functions

### `get_articulatory_substitution`

```python
def get_articulatory_substitution(char: str) -> str
```

Get a random alternative character from the same articulatory group to simulate sEMG confusion.

### `simulate_emg_noise`

```python
def simulate_emg_noise(shorthand_word: str, sub_rate: float = 0.1, del_rate: float = 0.05, ins_rate: float = 0.05) -> str
```

Applies deletion, insertion, and substitution noise to a shorthand word.

- Substitution: Replaces char with another from the same articulatory group (sEMG confusion).
- Deletion: Drops the character (packet drop / weak signal).
- Insertion: Inserts a random character (muscle twitch / swallow artifact).

### `phrase_to_noisy_shorthand`

```python
def phrase_to_noisy_shorthand(phrase: str, sub_rate: float = 0.1, del_rate: float = 0.05, ins_rate: float = 0.05) -> tuple[str, str]
```

Convert a full phrase to clean shorthand and simulated noisy shorthand.


**Returns:**

- A tuple of (clean_shorthand_phrase, noisy_shorthand_phrase)
