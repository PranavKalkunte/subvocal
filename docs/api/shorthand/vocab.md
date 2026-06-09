---
title: shorthand.vocab
sidebar_label: vocab
---

Target command vocabulary for the Subvocal interface.

Defines a phonetically diverse set of 17 commands optimized for throat/jaw
surface Electromyography (sEMG) classification.

## Functions

### `get_command_list`

```python
def get_command_list() -> list[str]
```

Return a list of all command names in the vocabulary.

### `get_command_details`

```python
def get_command_details(name: str) -> dict[str, Any]
```

Retrieve details for a specific command.
