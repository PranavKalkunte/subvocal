---
title: context.mock_data
sidebar_label: mock_data
---

Mock user context generator for testing and demonstration.

Generates structured mock data for contacts, calendar events, location,
conversation history, and active browser app state, automatically pre-computing
compressed phonetic shorthands for all text labels.

## Functions

### `generate_mock_context`

```python
def generate_mock_context() -> UserContext
```

Generate a high-fidelity UserContext populated with mock data.
