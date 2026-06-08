# Context Provider Guide

To translate compressed shorthand tokens into precise commands (e.g. mapping `gt g` to "go to George" instead of "go to Google"), the SDK leverages environmental context snapshots.

---

## 1. Context Schema

The data model for user context is defined as Pydantic schemas under `sdk/context/schema.py`. The root container is `UserContext`:

```python
from typing import List, Optional
from pydantic import BaseModel

class UserContext(BaseModel):
    contacts: List[Contact]          # User address book contacts
    calendar: List[CalendarEvent]    # Calendar appointments
    location: Optional[LocationInfo] # GPS/Coordinates/Address
    conversation_history: List[Message] # Dialogue turns
    app_state: AppState              # Active desktop application name and visible UI controls
```

---

## 2. Pluggable Context Providers

The SDK structures context retrieval using the `ContextProvider` abstract interface (defined in `sdk/core/interfaces.py`). Individual providers focus on querying specific environmental parameters:

*   **`CalendarContextProvider`**: Reads upcoming appointments.
*   **`ContactsContextProvider`**: Queries local contacts list and maps compressed shorthand names.
*   **`LocationContextProvider`**: Obtains latitude/longitude coordinates.
*   **`AppStateContextProvider`**: Ingests active application metadata (e.g. Browser or Terminal window names) and visible interactive button controls.

---

## 3. Aggregating Context with the `CompositeContextProvider`

In real-world pipelines, you aggregate multiple providers using the `CompositeContextProvider`. It queries each provider asynchronously or sequentially and merges the outputs into a single `UserContext` snapshot:

```python
from context.providers import (
    CalendarContextProvider,
    ContactsContextProvider,
    LocationContextProvider,
    AppStateContextProvider,
    CompositeContextProvider
)

# 1. Initialize modular providers
calendar_p = CalendarContextProvider()
contacts_p = ContactsContextProvider()
location_p = LocationContextProvider()
app_state_p = AppStateContextProvider()

# 2. Combine into a single aggregator
context_provider = CompositeContextProvider([
    calendar_p,
    contacts_p,
    location_p,
    app_state_p
])

# 3. Fetch consolidated context
user_context = context_provider.get_context()
```

---

## 4. Writing a Custom Context Provider

Below is an example showing how to write a custom context provider that fetches clipboard contents:

```python
import pyperclip
from core.interfaces import ContextProvider
from context.schema import UserContext, AppState

class ClipboardContextProvider(ContextProvider):
    def get_context(self) -> UserContext:
        try:
            clip_text = pyperclip.paste()
        except Exception:
            clip_text = ""
            
        # Wrap into standard UserContext container
        return UserContext(
            app_state=AppState(
                current_app="Clipboard",
                page_title="System Clipboard Buffer"
            ),
            conversation_history=[],
            contacts=[]
        )

    def get_provider_name(self) -> str:
        return "system_clipboard"
```
