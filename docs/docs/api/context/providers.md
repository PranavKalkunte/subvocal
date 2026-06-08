---
title: context.providers
sidebar_label: providers
---

Modular ContextProvider implementations for the Subvocal SDK.

## Classes

### `class StaticContextProvider(ContextProvider)`

Simple provider wrapping a static UserContext instance (useful for tests/mock runs).

#### Methods

##### `__init__`

```python
def __init__(self, context: UserContext, name: str = 'static_context')
```

No description.

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

### `class CalendarContextProvider(ContextProvider)`

Modular provider for retrieving calendar events.

#### Methods

##### `__init__`

```python
def __init__(self, events: Optional[List[CalendarEvent]] = None)
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.

### `class ContactsContextProvider(ContextProvider)`

Modular provider for retrieving user contact directory.

#### Methods

##### `__init__`

```python
def __init__(self, contacts: Optional[List[Contact]] = None)
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.

### `class LocationContextProvider(ContextProvider)`

Modular provider for retrieving current location details.

#### Methods

##### `__init__`

```python
def __init__(self, location: Optional[LocationInfo] = None)
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.

### `class AppStateContextProvider(ContextProvider)`

Modular provider for retrieving active app state and visible elements.

#### Methods

##### `__init__`

```python
def __init__(self, app_state: Optional[AppState] = None)
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.

### `class CompositeContextProvider(ContextProvider)`

Aggregates multiple individual ContextProviders into a single UserContext.

#### Methods

##### `__init__`

```python
def __init__(self, providers: List[ContextProvider])
```

No description.

##### `get_provider_name`

```python
def get_provider_name(self) -> str
```

No description.

##### `get_context`

```python
def get_context(self) -> UserContext
```

No description.
