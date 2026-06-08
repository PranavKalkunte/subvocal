---
title: emg_core.ml.model_io
sidebar_label: model_io
---

Model I/O utilities for saving and loading classifiers.

## Functions

### `get_model_path`

```python
def get_model_path(user_id: str, model_type: str = 'rf') -> str
```

Get the path to a user's model file.

Saves RF models as joblib, and CNN/GRU models as .pth.

### `save_model`

```python
def save_model(model_data: Dict[str, Any], user_id: str, model_type: str = 'rf') -> str
```

Save a model and its associated metadata to disk.

### `load_model`

```python
def load_model(user_id: str, model_type: str = 'rf') -> Dict[str, Any]
```

Load a model and its metadata from disk.

### `model_exists`

```python
def model_exists(user_id: str, model_type: str = 'rf') -> bool
```

Check if a model exists for a user.
