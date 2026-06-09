---
title: shorthand.spec
sidebar_label: spec
---

Shorthand and phonetic specifications for the subvocal interface.

Defines the articulatory phonetic groups representing sEMG jaw/throat
activation clusters, and compression rules to construct the shorthand representation.

## Functions

### `compress_word`

```python
def compress_word(word: str) -> str
```

Compress a single English word into its skeletal shorthand format.

Rules:
1. Lowercase the word.
2. Check if the word has a common abbreviation.
3. Otherwise:
   a. Keep the first letter.
   b. Remove subsequent vowels (a, e, i, o, u, y) and 'h'.
   c. Deduplicate consecutive identical consonants (e.g. 'gg' -> 'g').

### `word_to_articulatory_sequence`

```python
def word_to_articulatory_sequence(shorthand_word: str) -> list[str]
```

Convert a shorthand word to its articulatory group sequence.

Non-alphabet characters (like numbers and punctuation) are preserved as-is.
