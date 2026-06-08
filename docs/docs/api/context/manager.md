---
title: context.manager
sidebar_label: manager
---

Context Manager for retrieving, filtering, and searching user context.

Integrates context querying with custom articulatory phonetic distance matching
to enable context-aware silent speech decoding.

## Classes

### `class ContextManager`

Manages active UserContext and provides phonetic searching utilities.

#### Methods

##### `__init__`

```python
def __init__(self, context: UserContext)
```

No description.

##### `update_context`

```python
def update_context(self, new_context: UserContext)
```

Update the active user context state.

##### `get_contact_names`

```python
def get_contact_names(self) -> List[str]
```

Get list of all contact names in plain text.

##### `get_contact_shorthands`

```python
def get_contact_shorthands(self) -> List[str]
```

Get list of all pre-computed contact name shorthands.

##### `get_calendar_titles`

```python
def get_calendar_titles(self) -> List[str]
```

Get list of all calendar event titles in plain text.

##### `get_calendar_shorthands`

```python
def get_calendar_shorthands(self) -> List[str]
```

Get list of all calendar event title shorthands.

##### `get_visible_element_labels`

```python
def get_visible_element_labels(self) -> List[str]
```

Get list of all visible UI element labels in plain text.

##### `get_visible_element_shorthands`

```python
def get_visible_element_shorthands(self) -> List[str]
```

Get list of all visible UI element shorthands.

##### `get_all_context_words`

```python
def get_all_context_words(self) -> List[str]
```

Compile a list of all potential word tokens present in the active context.

Useful for seeding the dictionary candidate generator in the decoder.

##### `search_contacts`

```python
def search_contacts(self, noisy_shorthand: str) -> List[Tuple[Contact, float]]
```

Search contacts by comparing noisy shorthand against pre-computed contact shorthands.


**Returns:**

- List of tuples (Contact, articulatory_distance) sorted by distance.

##### `search_elements`

```python
def search_elements(self, noisy_shorthand: str) -> List[Tuple[UIElement, float]]
```

Search visible UI elements using articulatory distance matching.


**Returns:**

- List of tuples (UIElement, articulatory_distance) sorted by distance.
