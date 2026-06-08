---
title: hardware.datasets
sidebar_label: datasets
---

Dataset drivers for public electromyography databases (Ninapro, PutEMG, CSL-HDEMG).

## Classes

### `class NinaproDriver(HardwareSource)`

Streams electromyography signals from Ninapro MATLAB (.mat) files.

#### Methods

##### `__init__`

```python
def __init__(self, file_path: str, fs: float = 2000.0, loop: bool = True)
```

Initializes the Ninapro driver.


**Arguments:**

- ` file_path `: Path to the downloaded Ninapro subject MATLAB .mat file.
- ` fs `: Sampling frequency (default: 2000.0 Hz for DB2/DB3, DB1 is 100 Hz).
- ` loop `: Boolean representing whether to loop data when EOF is reached.

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

### `class PutEMGDriver(HardwareSource)`

Streams electromyography signals from PutEMG HDF5 (.h5) files.

#### Methods

##### `__init__`

```python
def __init__(self, file_path: str, fs: float = 5120.0, loop: bool = True)
```

Initializes the PutEMG driver.

Loads h5py dynamically.

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

### `class CSLHDEMGDriver(HardwareSource)`

Streams electromyography signals from CSL-HDEMG binary or NumPy (.npy) files.

#### Methods

##### `__init__`

```python
def __init__(self, file_path: str, fs: float = 2000.0, loop: bool = True, num_channels: int = 8)
```

Initializes the CSL-HDEMG driver.


**Arguments:**

- ` file_path `: Path to the NumPy (.npy) or raw binary float data file.
- ` fs `: Sample rate of the recording.
- ` loop `: Boolean representing whether to loop the data when EOF is reached.
- ` num_channels `: Number of channels in the raw binary file (ignored for .npy, which encodes shape).

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
