---
title: hardware.drivers
sidebar_label: drivers
---

Concrete implementations of HardwareSource drivers for the Subvocal SDK.

## Classes

### `class FileReplayDriver(HardwareSource)`

Replays raw sEMG data from a CSV file, simulating real-time acquisition.

#### Methods

##### `__init__`

```python
def __init__(self, file_path: str, fs: float = 1000.0, loop: bool = True)
```

Initializes the file replay driver.


**Arguments:**

- ` file_path `: Path to the recorded CSV file.
- ` fs `: Sample rate of the recording.
- ` loop `: Boolean representing whether to loop the data when EOF is reached.

##### `start`

```python
def start(self) -> None
```

No description.

##### `stop`

```python
def stop(self) -> None
```

No description.

##### `is_connected`

```python
def is_connected(self) -> bool
```

No description.

##### `read_frame`

```python
def read_frame(self, window_ms: int) -> Frame
```

No description.

### `class SyntheticSignalGenerator(HardwareSource)`

Generates continuous physiological sEMG noise and injects command triggers.

#### Methods

##### `__init__`

```python
def __init__(self, fs: float = 1000.0, num_channels: int = 8)
```

Initializes the synthetic signal generator.


**Arguments:**

- ` fs `: Sample frequency in Hz.
- ` num_channels `: Number of channels to generate (maps to 5-zone layouts).

##### `start`

```python
def start(self) -> None
```

No description.

##### `stop`

```python
def stop(self) -> None
```

No description.

##### `is_connected`

```python
def is_connected(self) -> bool
```

No description.

##### `trigger_command`

```python
def trigger_command(self, command: str, duration_ms: int = 300) -> None
```

Injects a high-amplitude transient signal profile on selected channels.


**Arguments:**

- ` command `: Command name (e.g., 'gt', 'clk', 'srch', 'typ').
- ` duration_ms `: Burst duration in milliseconds.

##### `read_frame`

```python
def read_frame(self, window_ms: int) -> Frame
```

No description.

### `class OpenBCICytonDriver(HardwareSource)`

Acquires raw biosignals from OpenBCI Cyton boards via BrainFlow.

#### Methods

##### `__init__`

```python
def __init__(self, port: str | None = None, simulated: bool = True)
```

Initializes the Cyton driver.

Loads BrainFlow dynamically.

##### `start`

```python
def start(self) -> None
```

No description.

##### `stop`

```python
def stop(self) -> None
```

No description.

##### `is_connected`

```python
def is_connected(self) -> bool
```

No description.

##### `read_frame`

```python
def read_frame(self, window_ms: int) -> Frame
```

No description.

### `class DelsysTrignoDriver(HardwareSource)`

Direct TCP Socket driver connecting to the Delsys Trigno Control Utility.

#### Methods

##### `__init__`

```python
def __init__(self, host: str = 'localhost', data_port: int = 50043, cmd_port: int = 50040, num_channels: int = 8, fs: float = 2000.0)
```

Initializes the Delsys socket driver.

##### `start`

```python
def start(self) -> None
```

No description.

##### `stop`

```python
def stop(self) -> None
```

No description.

##### `is_connected`

```python
def is_connected(self) -> bool
```

No description.

##### `read_frame`

```python
def read_frame(self, window_ms: int) -> Frame
```

No description.
