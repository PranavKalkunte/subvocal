---
title: core.dashboard
sidebar_label: dashboard
---

Local observability dashboard server for Subvocal Middleware.

## Classes

### `class ObservabilityDashboardHandler(BaseHTTPRequestHandler)`

Observability Dashboard API and HTML endpoints.

#### Methods

##### `log_message`

```python
def log_message(self, format, *args)
```

No description.

##### `do_GET`

```python
def do_GET(self)
```

No description.

## Functions

### `start_dashboard`

```python
def start_dashboard(port: int = 8000)
```

No description.
