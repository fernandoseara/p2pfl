# ⚠️ Common Errors

This page documents common errors you may encounter when using P2PFL and how to resolve them.

## TensorFlow Data Pipeline Hangs When Used with Ray

### Problem

When using TensorFlow's `tf.data.Dataset` (via HuggingFace's `to_tf_dataset()`) with p2pfl, the program hangs at:

```python
sample = next(iter(tf_dataset))
```

### Root Cause

**Import order conflict**: Ray is initialized before TensorFlow is imported, causing a threading deadlock.

1. Importing `p2pfl.management.logger` triggers `ray.init()` at module level
2. TensorFlow is imported afterwards
3. TensorFlow's data pipeline threads deadlock due to Ray's modified threading environment

```
p2pfl/management/logger/__init__.py:30 -> ray_installed() -> ray.init()
```

**This fails (hangs):**

```python
from p2pfl.management.logger import logger  # Ray initialized here
import tensorflow as tf  # Too late
from datasets import Dataset

dataset = Dataset.from_dict({"x": [[1]*784], "y": [0]})
tf_dataset = dataset.to_tf_dataset(batch_size=1, columns=["x"], label_cols=["y"])
next(iter(tf_dataset))  # Hangs forever
```

**This works:**

```python
import tensorflow as tf  # TensorFlow first
from p2pfl.management.logger import logger  # Ray after
from datasets import Dataset

dataset = Dataset.from_dict({"x": [[1]*784], "y": [0]})
tf_dataset = dataset.to_tf_dataset(batch_size=1, columns=["x"], label_cols=["y"])
next(iter(tf_dataset))  # Works
```

### Solutions

**Option 1: Import TensorFlow first**

```python
import tensorflow as tf  # FIRST
from p2pfl.management.logger import logger  # After TensorFlow
```

**Option 2: Disable Ray**

```python
from p2pfl.settings import Settings
Settings.general.DISABLE_RAY = True
# Then import other p2pfl modules
```

### Environment

- TensorFlow 2.20.0
- Ray 2.53.0
- Python 3.12
- macOS (Darwin)

### Status

**Open Issue** - Requires lazy Ray initialization or import order changes.

Tracked at: [ray-project/ray#59661](https://github.com/ray-project/ray/issues/59661)
