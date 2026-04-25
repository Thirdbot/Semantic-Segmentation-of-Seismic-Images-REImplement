# RockML Library Documentation

**Version:** 0.20.0  
**Python:** ≥ 3.10  
**License:** IBM Internal

RockML is a machine-learning framework for seismic and well-log data. It provides a pipeline from raw SEGY / LAS / horizon files through preprocessing, model training, inference, and horizon export.

---

## Table of Contents

1. [Installation](#installation)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
   - [Datum](#datum)
   - [Adapter](#adapter)
   - [Transformation & Composer](#transformation--composer)
   - [Estimator](#estimator)
4. [Data Adapters](#data-adapters)
   - [PostStackAdapter2D](#poststackadapter2d)
   - [CDPGatherAdapter](#cdpgatheradapter)
   - [HorizonAdapter](#horizonadapter)
   - [LASDataAdapter](#lasdataadapter)
5. [Transformations](#transformations)
   - [Image Transformations](#image-transformations)
   - [Filters](#filters)
   - [Horizon Transformations](#horizon-transformations)
   - [Gather Transformations](#gather-transformations)
   - [Lambda & Composer](#lambda--composer)
6. [Data Utilities](#data-utilities)
   - [Pipeline (Parallel Processing)](#pipeline-parallel-processing)
   - [Sampling](#sampling)
   - [Array Operations](#array-operations)
7. [Learning](#learning)
   - [PostStackEstimator](#poststackestimator)
   - [Model Zoo](#model-zoo)
   - [Metrics & Callbacks](#metrics--callbacks)
   - [Data Loaders](#data-loaders)
   - [VelocityAdjustment Estimator](#velocityadjustment-estimator)
8. [Visualization](#visualization)
9. [Complete Workflows](#complete-workflows)
   - [Semantic Segmentation](#semantic-segmentation-workflow)
   - [Classification](#classification-workflow)
   - [Velocity Analysis](#velocity-analysis-workflow)
   - [Well Log Processing](#well-log-processing-workflow)
10. [API Reference Quick-Look](#api-reference-quick-look)

---

## Installation

```bash
pip install -e ./rockml
```

**Dependencies:** `tensorflow`, `numpy`, `pandas`, `scipy`, `scikit-image`, `h5py`, `pillow`, `lasio`, `matplotlib`, `pyyaml`, `murmurhash`, `graphviz`, `pydot`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          RockML                                 │
│                                                                 │
│  Raw Files (SEGY / Horizon / LAS / Velocity)                    │
│       │                                                         │
│       ▼                                                         │
│  Data Adapters  ──►  Datum objects  ──►  Transformations        │
│  (read & wrap)       (typed units)        (Composer chain)      │
│                            │                                    │
│                            ▼                                    │
│                     HDF5 / tf.data  ──►  Estimator              │
│                     (saved dataset)      (fit / apply)          │
│                            │                                    │
│                            ▼                                    │
│                     Post-processing  ──►  Horizon Export        │
│                     (ReconstructFromWindows, ConvertHorizon)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Datum

A `Datum` is the fundamental data unit. Every adapter produces `Datum` objects, and every transformation consumes and returns them. Each subclass bundles both the array data and the provenance metadata needed to reconstruct spatial context.

| Datum subclass | Key fields |
|---|---|
| `PostStackDatum` | `features` (H×W×C ndarray), `label` (mask or scalar), `direction`, `line_number`, `pixel_depth`, `column` |
| `CDPGatherDatum` | `features` (samples×traces), `offsets`, `label`, `inline`, `crossline`, `pixel_depth`, `coherence`, `velocities` |
| `HorizonDatum` | `point_map` (DataFrame indexed by inline/crossline), `horizon_name` |
| `WellDatum` | `df` (DataFrame), `well_name`, `numerical_logs`, `categorical_logs` |

### Adapter

Adapters wrap raw file formats and expose a uniform Python sequence interface, so they work like lists:

```python
adapter = PostStackAdapter2D(...)
len(adapter)          # number of seismic lines
adapter[0]            # first line as PostStackDatum
adapter[0:10]         # slice → list of PostStackDatum
for datum in adapter: # iterate
    ...
```

### Transformation & Composer

A `Transformation` is a callable that takes a datum (or list) and returns a transformed datum (or list, or `None` to drop it). Chain them into a `Composer`:

```python
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image

composer = Composer([
    image.Crop2D(crop_left=0, crop_right=0, crop_top=75, crop_bottom=0),
    image.ScaleIntensity(gray_levels=256, percentile=1.0),
    image.FillSegmentationMask(),
    image.ViewAsWindows(tile_shape=(64, 64), stride_shape=(32, 32)),
])

tiles = composer.apply(list(adapter))  # list of PostStackDatum tiles
```

`Composer` rules:
- A transformation returning **`None`** → datum is dropped from the list.
- A transformation returning a **`list`** → results are flattened into the dataset.
- A transformation returning a **`Datum`** → replaces the current datum.

### Estimator

All models share the same interface:

```python
estimator.fit(epochs, train_set, valid_set, callbacks)
results = estimator.apply(datum_list, batch_size=32)
estimator.save_model(path)
estimator = EstimatorClass.load_model(path)
```

---

## Data Adapters

### PostStackAdapter2D

Reads a post-stack SEGY file together with one or more horizon files.

```python
from rockml.data.adapter.seismic.segy.poststack import PostStackAdapter2D, Direction

adapter = PostStackAdapter2D(
    segy_path='data/netherlands.sgy',
    horizons_path_list=[            # depth-sorted, shallowest first
        'horizons/North_Sea.xyz',
        'horizons/SSN.xyz',
        'horizons/Germanic.xyz',
    ],
    data_dict={                     # None = all lines
        'inline':    [[400, 600]],  # inclusive ranges
        'crossline': [[300, 500]],
    },
    inline_byte=189,                # SEGY trace header byte positions
    crossline_byte=193,
    x_byte=181,
    y_byte=185,
)
```

**Scan before use (required for multiprocessing):**

```python
segy_info = adapter.initial_scan()
# Returns dict with keys:
# 'num_inlines', 'num_crosslines', 'num_time_depth',
# 'inline_resolution', 'crossline_resolution',
# 'time_depth_resolution', 'initial_time', ...
```

**Access patterns:**

```python
# Single line
datum = adapter[0]                          # PostStackDatum
datum = adapter.get_line(Direction.INLINE, 500)

# Slice or list
datums = adapter[0:10]
datums = adapter[[0, 5, 10]]

# Iterate
for datum in adapter:
    process(datum)
```

**`PostStackDatum` fields:**

| Field | Type | Description |
|---|---|---|
| `features` | `np.ndarray` (H, W, C) | Seismic amplitude image |
| `label` | `np.ndarray` (H, W) | Horizon label mask (0 = no horizon, 1…N = horizon index) |
| `direction` | `Direction` | `INLINE` or `CROSSLINE` |
| `line_number` | `int` | Inline or crossline number |
| `pixel_depth` | `int` | Row offset in the full cube |
| `column` | `int` | Column offset in the full cube |

---

### CDPGatherAdapter

Reads pre-stack CDP gathers from a SEGY file, optionally loading reference velocity functions.

```python
from rockml.data.adapter.seismic.segy.prestack import CDPGatherAdapter

adapter = CDPGatherAdapter(
    segy_path='data/prestack.sgy',
    gather_list=[[100, 200], [101, 200]],   # [[inline, crossline], ...]
    velocity_file_path='velocities.dat',    # optional VFUNC file
    inline_byte=189,
    crossline_byte=193,
    x_byte=181,
    y_byte=185,
    source_byte=17,
    recx_byte=81,
    recy_byte=85,
)
```

Each element is a `CDPGatherDatum`:

| Field | Type | Description |
|---|---|---|
| `features` | `np.ndarray` (samples, traces) | Gather amplitudes |
| `offsets` | `np.ndarray` | Offset value per trace |
| `label` | `np.ndarray` or `None` | Velocity function `[[time_ms, v], ...]` |
| `inline` | `int` | Inline number |
| `crossline` | `int` | Crossline number |
| `coherence` | `np.ndarray` or `None` | Semblance panel |
| `velocities` | `list` or `None` | Derived velocity function |

---

### HorizonAdapter

Reads one or more horizon ASCII files (column-format XYZ files).

```python
from rockml.data.adapter.seismic.horizon import HorizonAdapter

adapter = HorizonAdapter(
    horizons_path_list=['horizon_A.xyz', 'horizon_B.xyz'],
    time_depth_resolution=4.0,   # ms per sample (from SEGY)
    initial_time=0,              # ms at sample 0
    column_dict=None,            # optional dict of column_name → dtype
    separator=r'\s+',
)

datum = adapter[0]               # HorizonDatum
print(datum.horizon_name)
print(datum.point_map.head())    # DataFrame with columns: inline, crossline, pixel_depth
```

**Export horizon back to file:**

```python
from rockml.data.adapter.seismic.horizon import HorizonDataDumper

HorizonDataDumper.to_text_file(
    datum_list=[datum_a, datum_b],
    path='output/horizons.xyz',
    segy_path='data/netherlands.sgy',
    include_xy=True,
)
```

---

### LASDataAdapter

Reads all `.las` well log files from a directory.

```python
from rockml.data.adapter.well.las import LASDataAdapter

adapter = LASDataAdapter(
    dir_path='data/wells/',
    numerical_logs=['GR', 'RHOB', 'NPHI', 'DT'],
    categorical_logs=['LITH'],
    depth_unit='m',              # 'm' or 'ft'
    lat_long=('SLAT', 'SLON'),  # optional: LAS header fields for coordinates
)

well = adapter[0]               # WellDatum
print(well.well_name)
print(well.df.head())
```

**Save well data:**

```python
from rockml.data.adapter.well import WellDataDumper

WellDataDumper.to_hdf([well_a, well_b], path='wells.h5')
df = WellDataDumper.concatenate([well_a, well_b])  # combined DataFrame
```

---

## Transformations

All transformations are imported from `rockml.data.transformations`.

### Image Transformations

```python
from rockml.data.transformations.seismic import image
```

#### `Crop2D`

Removes rows and columns from the edges of a seismic slice.

```python
image.Crop2D(
    crop_left=0,
    crop_right=0,
    crop_top=75,     # removes top 75 samples (often muted zone)
    crop_bottom=0,
    ignore_label=False,  # if True, only features are cropped
)
```

Also adjusts `datum.column` and `datum.pixel_depth` to maintain absolute position tracking.

---

#### `ScaleIntensity`

Clips amplitude outliers by percentile then quantizes to `uint8`.

```python
image.ScaleIntensity(
    gray_levels=256,   # number of quantization levels
    percentile=1.0,    # clip values below this percentile and above (100 - percentile)
)
```

---

#### `FillSegmentationMask`

Converts a horizon mask (sparse lines) to a filled segmentation mask. Each column is filled downward from each horizon line so regions between horizons have unique class labels.

```python
image.FillSegmentationMask()
```

Before:
```
0 0 1 0 0 0 2 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
```
After:
```
0 0 1 0 0 0 2 0 0
0 0 1 0 0 0 2 0 0
0 0 1 0 0 0 2 0 0
```

---

#### `BinarizeMask`

Converts all non-zero values in the label mask to 1 (foreground/background segmentation).

```python
image.BinarizeMask()
```

---

#### `ThickenLinesMask`

Expands horizon lines in the label mask by `n_points` pixels above and below, creating thicker targets for training.

```python
image.ThickenLinesMask(n_points=3)
```

---

#### `ViewAsWindows`

Splits a datum into a list of fixed-size tiles using a sliding window. This is the primary way to generate training samples from large seismic sections.

```python
image.ViewAsWindows(
    tile_shape=(64, 64),      # (height, width) of each tile in pixels
    stride_shape=(32, 32),    # (vertical, horizontal) stride
    auto_pad=False,           # pad image so all tiles have equal size
    filters=None,             # list of Transformation objects applied per tile
                              # (e.g. [MinimumTextureFilter(0.9)])
)
```

Returns a `List[PostStackDatum]`. Each tile's `pixel_depth` and `column` reflect its absolute position in the original line.

---

#### `ReconstructFromWindows`

Inverse of `ViewAsWindows`. Takes a list of predicted tiles and stitches them back into full seismic lines.

```python
image.ReconstructFromWindows(
    inline_shape=(num_samples, num_crosslines),
    crossline_shape=(num_samples, num_inlines),
    strides=(64, 64),
    overlapping_fn=None,  # e.g. array_ops.gauss_weight_map for weighted blending
)
```

Input is a flat list of tiles from multiple lines. Output is one `PostStackDatum` per reconstructed line.

---

#### `ScaleByteFeatures`

Divides feature values by 255.0 and casts to `float32`. Apply after `ScaleIntensity`.

```python
image.ScaleByteFeatures()
```

---

#### `EqualizeHistogram`

Applies histogram equalization to feature channels.

```python
image.EqualizeHistogram(n_bins=256)
```

---

#### `Resize2D` / `Interpolate2D` / `Resize`

Spatial resizing using PIL or scipy interpolation.

```python
image.Resize2D(height=128, width=128, mode='linear', ignore_label=False)
image.Interpolate2D(height_amp_factor=2.0, width_amp_factor=2.0, mode='spline')
image.Resize(height=256, width=256, mode='linear')
```

`mode` options: `'linear'` (bilinear / linear), `'spline'` (bicubic / cubic).

---

#### `PostStackLabelArgMax`

Collapses a one-hot encoded label (last axis) to a class index array via `np.argmax`.

```python
image.PostStackLabelArgMax()
```

---

### Filters

Filters are transformations that return `None` to drop a tile from the dataset.

```python
from rockml.data.transformations.seismic.filters import MinimumTextureFilter, ClassificationFilter
```

#### `MinimumTextureFilter`

Drops tiles that are mostly empty or have insufficient texture variation (e.g., zero-filled padding tiles).

```python
MinimumTextureFilter(
    min_texture_in_features=0.9  # fraction of rows/cols that must have non-zero std
)
```

A value of `0.9` means 90% of rows and 90% of columns must contain amplitude variation.

---

#### `ClassificationFilter`

For **classification** (not segmentation) tasks. Determines the dominant class in a tile's label mask. If the non-dominant-class fraction exceeds `noise`, the tile is dropped. Otherwise replaces the mask label with the dominant scalar class.

```python
ClassificationFilter(
    noise=0.3   # max allowed fraction of non-dominant-class pixels
)
```

After this filter, `datum.label` is a scalar integer (the class index).

---

### Horizon Transformations

```python
from rockml.data.transformations.seismic import horizon
```

#### `ConvertHorizon`

Converts a segmentation mask prediction into `HorizonDatum` objects by extracting class boundary positions.

```python
horizon.ConvertHorizon(
    horizon_names=['horizon_A', 'horizon_B', 'horizon_C'],
    crop_top=75,              # must match Crop2D crop_top
    crop_left=0,
    inline_resolution=1,      # from segy_info
    crossline_resolution=1,
    correction=10,            # search window in samples for snapping to boundary
)
```

Input: `PostStackDatum` with segmentation mask.  
Output: `List[HorizonDatum]`, one per horizon name.

---

#### `ConcatenateHorizon`

Merges multiple `HorizonDatum` objects sharing the same `horizon_name` (e.g., results from inline and crossline predictions) into one datum per horizon.

```python
horizon.ConcatenateHorizon()
```

Input/output: `List[HorizonDatum]`.

---

#### `RemoveMutedTraces`

Removes (inline, crossline) points from a horizon's point map where the seismic trace is muted (zero-filled).

```python
horizon.RemoveMutedTraces(
    valid_mask=mask_array,   # 2D bool array: True = valid trace
    segy_info=segy_info,
)
```

---

#### `PhaseCorrection`

Snaps horizon picks to the nearest seismic amplitude maximum or minimum within a search window.

```python
from rockml.data.adapter.seismic.segy.poststack import Phase

horizon.PhaseCorrection(
    segy_info=segy_info,
    seismic_lines=lines_dict,
    amp_factor=10,
    mode=Phase.MAX,          # Phase.MIN, Phase.MAX, Phase.CROSSUP, Phase.CROSSDOWN
)
```

---

### Gather Transformations

```python
from rockml.data.transformations.seismic import gather
```

#### `ComputeCoherence`

Computes semblance panels for velocity analysis on `CDPGatherDatum` objects.

```python
gather.ComputeCoherence(
    time_gate_ms=40.0,
    time_range_ms=[0.0, 3000.0],
    sample_rate_ms=4.0,
    velocity_range=[1400, 4500],
    velocity_step=50,
)
```

Populates `datum.coherence` (shape: `[num_time_windows, num_velocities]`).

---

#### `ComputeVelocityFunction`

Derives a smooth velocity function from the semblance panel.

```python
gather.ComputeVelocityFunction(
    time_gate_ms=40.0,
    time_range_ms=[0.0, 3000.0],
    sample_rate_ms=4.0,
    velocity_range=[1400, 4500],
    velocity_step=50,
    initial_velocity=1500,
    initial_analysis_time_ms=None,
    handle_sample_zero=True,
    spl_order=1,
    spl_smooth=0.1,
    spl_ext=0,
    savgol_window=51,
    savgol_order=1,
)
```

Populates `datum.velocities` as `[[time_ms, velocity_m_s], ...]`.

---

#### `GenerateGatherWindows`

Creates 3-channel image tiles from gathers for the velocity adjustment deep learning task.

```python
gather.GenerateGatherWindows(
    time_range_ms=[0.0, 3000.0],
    sample_rate_ms=4.0,
    num_samples=64,
    velocity_range=[1400, 4500],
    window_size=64,
    stride=4,
    velocity_deltas=[-200, 0, 200],   # augmentation: generate v ± delta variants
    ignore_velocites=False,
)
```

---

### Lambda & Composer

#### `Lambda`

Wraps any function as a reusable `Transformation`.

```python
from rockml.data.transformations import Lambda

def add_channel(datum):
    datum.features = np.concatenate([datum.features, datum.features[..., :1]], axis=-1)
    return datum

t = Lambda(add_channel)
```

---

#### `Composer`

Chains transformations and applies them sequentially.

```python
from rockml.data.transformations import Composer

composer = Composer([t1, t2, t3])

# Apply to a list of datums
result = composer.apply(datum_list)

# Apply to a single datum
result = composer.apply(datum)

# Serialize / deserialize
composer.dump('pipeline.pkl')
loaded = Composer.load('pipeline.pkl')
```

---

## Data Utilities

### Pipeline (Parallel Processing)

`Pipeline` parallelizes `Composer.apply` across multiple CPU cores by splitting the adapter into blocks.

```python
from rockml.data.pipeline import Pipeline

pipeline = Pipeline(composer=composer)

tiles = pipeline.build_dataset(
    data_adapter=adapter,
    num_blocks=8,    # split adapter into 8 chunks
    cores=4,         # use 4 worker processes
)
```

> **Important:** Call `adapter.initial_scan()` **before** creating the `Pipeline`. SEGY file handles cannot be pickled for `multiprocessing.Pool`, so the adapter uses lazy initialization on first access in each worker.

---

### Sampling

```python
from rockml.data.sampling import RandomSampler, ClassBalanceSampler, split_dataset
```

#### `split_dataset`

Randomly splits a dataset into training and validation sets.

```python
train, valid = split_dataset(tiles, valid_ratio=0.1)
```

#### `RandomSampler`

Randomly samples without replacement.

```python
sampler = RandomSampler(num_examples=5000)
subset = sampler.sample(tiles)
```

If `num_examples` is `None` or larger than the dataset, the full dataset is shuffled and returned.

#### `ClassBalanceSampler`

Balances dataset by sampling equally from each class (based on `datum.label`).

```python
sampler = ClassBalanceSampler(num_examples_per_class=500)
balanced = sampler.sample(tiles)
```

---

### Array Operations

Low-level NumPy utilities available in `rockml.data.array_ops`:

```python
from rockml.data import array_ops

# Crop
cropped = array_ops.crop_2d(image, crop_top=10, crop_bottom=5, crop_left=0, crop_right=0)

# Sliding window tiling
windows = array_ops.view_as_windows(array, window_shape=(64, 64), stride=(32, 32))
indexes = array_ops.get_tiles_indexes(shape=(512, 512), stride=(32, 32))

# Reconstruct from tiles
full = array_ops.reconstruct_from_windows(tile_array, final_shape=(512, 512), stride=(32, 32))

# Rescaling
scaled = array_ops.scale_intensity(image, gray_levels=256, percentile=1.0)
normed = array_ops.scale_minmax(array, feature_range=(0, 1))

# Padding
padded, pad_amounts = array_ops.exact_pad(image, window_shape=(64, 64), stride=(32, 32))

# Gaussian weight map for smooth overlap reconstruction
weights = array_ops.gauss_weight_map(shape=(64, 64), sigma_shape=(32, 32))
```

---

## Learning

### PostStackEstimator

```python
from rockml.learning.zoo.poststack import PostStackEstimator, danet2fcn
from rockml.learning.keras import SparseMeanIoU
import tensorflow as tf

model = danet2fcn(input_shape=(64, 64, 1), output_channels=5)

estimator = PostStackEstimator(
    model=model,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    train_metrics=[SparseMeanIoU(num_classes=5)],
)
```

**Training:**

```python
estimator.fit(
    epochs=50,
    train_set=train_tf_dataset,
    valid_set=valid_tf_dataset,
    callbacks=[...],            # optional list of tf.keras.callbacks
)
```

**Inference:**

```python
predicted_datums = estimator.apply(datum_list, batch_size=32)
# Each datum.label is now a predicted segmentation mask
```

**Save and load:**

```python
estimator.save_model('models/my_run/')
estimator = PostStackEstimator.load_model('models/my_run/')
estimator = PostStackEstimator.load_model('models/my_run/', is_best=True)  # best checkpoint
```

---

### Model Zoo

All factory functions return a compiled-ready `tf.keras.Model`.

```python
from rockml.learning.zoo.poststack import danet2, danet3, unet, danet2fcn, danet3fcn
```

| Function | Task | Architecture |
|---|---|---|
| `danet2(input_shape, num_classes)` | Classification | 7 Conv2D + BatchNorm + 3 Dense + Dropout |
| `danet3(input_shape, num_classes)` | Classification | Conv2D + 6 residual blocks + Dense |
| `unet(input_shape, output_channels)` | Segmentation | 4-level encoder/decoder with skip connections |
| `danet2fcn(input_shape, output_channels)` | Segmentation | Fully convolutional version of danet2 |
| `danet3fcn(input_shape, output_channels)` | Segmentation | Fully convolutional version of danet3 |

```python
# Segmentation
model = unet(input_shape=(64, 64, 1), output_channels=5)
model = danet2fcn(input_shape=(64, 64, 1), output_channels=5)
model = danet3fcn(input_shape=(64, 64, 1), output_channels=5)

# Classification
model = danet2(input_shape=(50, 50, 1), num_classes=4)
model = danet3(input_shape=(50, 50, 1), num_classes=4)
```

---

### Metrics & Callbacks

```python
from rockml.learning.keras import SparseMeanIoU
from rockml.learning.keras.callbaks import EarlyStoppingAtMinLoss, CSVLogger
```

#### `SparseMeanIoU`

Mean Intersection-over-Union for sparse (class-index) labels. Drop-in replacement for `tf.keras.metrics.MeanIoU` when labels are not one-hot.

```python
SparseMeanIoU(num_classes=5, name='sparse_mean_iou')
```

#### `EarlyStoppingAtMinLoss`

Stops training when validation loss stops improving; automatically restores best weights.

```python
EarlyStoppingAtMinLoss(patience=10)
```

#### `CSVLogger`

Extended CSV logger that handles iterable metric values.

```python
CSVLogger(filename='training_log.csv', append=True)
```

---

### Data Loaders

Convert saved HDF5 datasets (produced by `PostStackDataDumper.to_hdf`) into TensorFlow datasets for training.

```python
from rockml.learning.keras.data_loaders import hdf_2_tfdataset

train_ds = hdf_2_tfdataset(
    path='datasets/train.h5',
    features_name='features',
    labels_name='label',
)

# Chain standard tf.data operations
train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

Feature values are automatically normalized to `[0.0, 1.0]` (divided by 255).

---

### VelocityAdjustment Estimator

Fine-tunes velocity picks using a deep learning model trained on gather windows.

```python
from rockml.learning.zoo.va_estimator import VelocityAdjustment

va = VelocityAdjustment(
    input_shape=(64, 64, 3),
    model=None,                       # defaults to Xception backbone + Dense(1)
    best_valid_model_path='models/va_best/',
    learning_rate=0.001,
    early_stop_patience=50,
)

va.fit(epochs=100, train_set=train_ds, valid_set=valid_ds)

refined_datums = va.apply(
    datum_list=gather_datums,
    batch_size=16,
    velocity_range=[1400, 4500],      # optional: clip predicted velocities
)
```

---

## Visualization

```python
from rockml.data.visualization.seismic.gather import (
    plot_boxplot,
    plot_scatter_hist,
    plot_difference,
)
```

All plotting functions accept a `scores` array of shape `[N, 3]` where each row is `[inline, crossline, score]`.

```python
# Boxplot of score distribution
plot_boxplot(scores, score_name='Semblance', output_path='plots/boxplot.png')

# Spatial scatter plot + histogram
plot_scatter_hist(scores, output_path='plots/qa_map.png', plot_title='Velocity QC')

# Difference map (e.g. before vs. after velocity adjustment)
plot_difference(diff_values, output_path='plots/diff_map.png', plot_title='Velocity Delta', vmax=200)
```

---

## Complete Workflows

### Semantic Segmentation Workflow

Full pipeline: SEGY + horizons → trained model → horizon export.

```python
import tensorflow as tf
from rockml.data.adapter.seismic.segy.poststack import PostStackAdapter2D
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image, horizon
from rockml.data.transformations.seismic.filters import MinimumTextureFilter
from rockml.data.pipeline import Pipeline
from rockml.data.sampling import split_dataset
from rockml.data.adapter import PostStackDataDumper
from rockml.learning.keras.data_loaders import hdf_2_tfdataset
from rockml.learning.keras import SparseMeanIoU
from rockml.learning.keras.callbaks import EarlyStoppingAtMinLoss
from rockml.learning.zoo.poststack import danet3fcn, PostStackEstimator
from rockml.data.adapter.seismic.horizon import HorizonDataDumper

# --- 1. Define adapter ---
adapter = PostStackAdapter2D(
    segy_path='netherlands.sgy',
    horizons_path_list=['North_Sea.xyz', 'SSN.xyz', 'Germanic.xyz'],
    data_dict={'inline': [[400, 600]]},
)
segy_info = adapter.initial_scan()

TILE = (64, 64)
STRIDE = (32, 32)
NUM_CLASSES = 4  # background + 3 horizons
NUM_CHANNELS = 1

# --- 2. Pre-processing composer ---
pre_proc = Composer([
    image.Crop2D(crop_left=0, crop_right=0, crop_top=75, crop_bottom=0),
    image.ScaleIntensity(gray_levels=256, percentile=1.0),
    image.FillSegmentationMask(),
    image.ViewAsWindows(tile_shape=TILE, stride_shape=STRIDE, auto_pad=True),
    MinimumTextureFilter(min_texture_in_features=0.9),
    image.ScaleByteFeatures(),
])

# --- 3. Build dataset in parallel ---
pipeline = Pipeline(composer=pre_proc)
tiles = pipeline.build_dataset(adapter, num_blocks=8, cores=4)

train_tiles, valid_tiles = split_dataset(tiles, valid_ratio=0.1)

# --- 4. Save to HDF5 ---
PostStackDataDumper.to_hdf(train_tiles, 'train.h5')
PostStackDataDumper.to_hdf(valid_tiles, 'valid.h5')

# --- 5. Build tf.data pipelines ---
train_ds = hdf_2_tfdataset('train.h5', 'features', 'label').shuffle(2000).batch(32).prefetch(2)
valid_ds = hdf_2_tfdataset('valid.h5', 'features', 'label').batch(32).prefetch(2)

# --- 6. Train ---
model = danet3fcn(input_shape=(*TILE, NUM_CHANNELS), output_channels=NUM_CLASSES)
estimator = PostStackEstimator(
    model=model,
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    train_metrics=[SparseMeanIoU(num_classes=NUM_CLASSES)],
)
estimator.fit(
    epochs=100,
    train_set=train_ds,
    valid_set=valid_ds,
    callbacks=[EarlyStoppingAtMinLoss(patience=15)],
)
estimator.save_model('models/segmentation/')

# --- 7. Inference on new lines ---
test_adapter = PostStackAdapter2D(
    segy_path='netherlands.sgy',
    horizons_path_list=[],
    data_dict={'inline': [[600, 700]]},
)
inf_pre_proc = Composer([
    image.Crop2D(crop_left=0, crop_right=0, crop_top=75, crop_bottom=0),
    image.ScaleIntensity(gray_levels=256, percentile=1.0),
    image.ViewAsWindows(tile_shape=TILE, stride_shape=STRIDE, auto_pad=True),
    image.ScaleByteFeatures(),
])
tiles_for_pred = inf_pre_proc.apply(list(test_adapter))
predicted_tiles = estimator.apply(tiles_for_pred, batch_size=64)

# --- 8. Post-processing: reconstruct lines + extract horizons ---
num_xl = segy_info['num_crosslines']
num_il = segy_info['num_inlines']
num_td = segy_info['num_time_depth']

post_proc = Composer([
    image.ReconstructFromWindows(
        inline_shape=(num_td - 75, num_xl),
        crossline_shape=(num_td - 75, num_il),
        strides=STRIDE,
    ),
    horizon.ConvertHorizon(
        horizon_names=['North_Sea', 'SSN', 'Germanic'],
        crop_top=75,
        crop_left=0,
        inline_resolution=segy_info['inline_resolution'],
        crossline_resolution=segy_info['crossline_resolution'],
    ),
])
horizon_datums = post_proc.apply([predicted_tiles])
horizon_datums = horizon.ConcatenateHorizon()(horizon_datums)

# --- 9. Export ---
HorizonDataDumper.to_text_file(horizon_datums, 'output/horizons.xyz', 'netherlands.sgy')
```

---

### Classification Workflow

Classify tiles into stratigraphic facies.

```python
from rockml.data.adapter.seismic.segy.poststack import PostStackAdapter2D
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image
from rockml.data.transformations.seismic.filters import ClassificationFilter
from rockml.data.sampling import RandomSampler, split_dataset
from rockml.data.adapter import PostStackDataDumper
from rockml.learning.keras.data_loaders import hdf_2_tfdataset
from rockml.learning.zoo.poststack import danet3, PostStackEstimator
import tensorflow as tf

adapter = PostStackAdapter2D(
    segy_path='netherlands.sgy',
    horizons_path_list=['horizon_A.xyz', 'horizon_B.xyz', 'horizon_C.xyz'],
    data_dict={'inline': [[500, 550]]},
)

composer = Composer([
    image.Crop2D(crop_left=0, crop_right=0, crop_top=75, crop_bottom=0),
    image.ScaleIntensity(gray_levels=256, percentile=1.0),
    image.FillSegmentationMask(),
    image.ViewAsWindows(tile_shape=(50, 50), stride_shape=(25, 25)),
    ClassificationFilter(noise=0.3),   # assigns scalar label, drops ambiguous tiles
    image.ScaleByteFeatures(),
])

tiles = composer.apply(list(adapter))
train_tiles, valid_tiles = split_dataset(tiles, valid_ratio=0.1)
train_tiles = RandomSampler(num_examples=5000).sample(train_tiles)

PostStackDataDumper.to_hdf(train_tiles, 'cls_train.h5')
PostStackDataDumper.to_hdf(valid_tiles, 'cls_valid.h5')

train_ds = hdf_2_tfdataset('cls_train.h5', 'features', 'label').shuffle(1000).batch(32)
valid_ds = hdf_2_tfdataset('cls_valid.h5', 'features', 'label').batch(32)

NUM_CLASSES = 4
model = danet3(input_shape=(50, 50, 1), num_classes=NUM_CLASSES)
estimator = PostStackEstimator(
    model=model,
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    train_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
estimator.fit(epochs=50, train_set=train_ds, valid_set=valid_ds)
```

---

### Velocity Analysis Workflow

Compute semblance panels, derive velocity functions, and refine with deep learning.

```python
from rockml.data.adapter.seismic.segy.prestack import CDPGatherAdapter, CDPGatherDataDumper
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic.gather import (
    ComputeCoherence, ComputeVelocityFunction, GenerateGatherWindows
)

adapter = CDPGatherAdapter(
    segy_path='prestack.sgy',
    gather_list=None,              # None = all gathers
    velocity_file_path=None,
)

SAMPLE_RATE = 4.0        # ms
TIME_RANGE  = [0, 3000]  # ms
VEL_RANGE   = [1400, 4500]
VEL_STEP    = 50

# Automatic velocity picking
auto_pick = Composer([
    ComputeCoherence(
        time_gate_ms=40.0,
        time_range_ms=TIME_RANGE,
        sample_rate_ms=SAMPLE_RATE,
        velocity_range=VEL_RANGE,
        velocity_step=VEL_STEP,
    ),
    ComputeVelocityFunction(
        time_gate_ms=40.0,
        time_range_ms=TIME_RANGE,
        sample_rate_ms=SAMPLE_RATE,
        velocity_range=VEL_RANGE,
        velocity_step=VEL_STEP,
        initial_velocity=1500,
        savgol_window=51,
    ),
])

gathers = auto_pick.apply(list(adapter))

# Save velocity functions
CDPGatherDataDumper.to_velocity_file(gathers, 'velocities_auto.dat')

# Generate training windows for velocity adjustment network
window_gen = Composer([
    GenerateGatherWindows(
        time_range_ms=TIME_RANGE,
        sample_rate_ms=SAMPLE_RATE,
        num_samples=64,
        velocity_range=VEL_RANGE,
        window_size=64,
        stride=4,
        velocity_deltas=[-200, 0, 200],
    ),
])
windows = window_gen.apply(gathers)
CDPGatherDataDumper.to_hdf(windows, 'va_dataset.h5')
```

---

### Well Log Processing Workflow

Read, resample, and save LAS well logs.

```python
from rockml.data.adapter.well.las import LASDataAdapter
from rockml.data.adapter.well import WellDataDumper
from rockml.data import df_ops
import numpy as np

adapter = LASDataAdapter(
    dir_path='data/wells/',
    numerical_logs=['GR', 'RHOB', 'NPHI', 'DT'],
    categorical_logs=['LITH'],
    depth_unit='m',
)

wells = list(adapter)

# Resample logs to a uniform depth grid
for well in wells:
    old_depth = well.df.index.values
    new_depth = np.arange(old_depth.min(), old_depth.max(), 0.5)  # 0.5 m grid
    well.df = well.df.reindex(new_depth)
    for col in well.numerical_logs:
        well.df[col] = df_ops.interpolate_numeric_series(
            series=well.df[col], depth_old=old_depth, depth_new=new_depth
        )

# Save
WellDataDumper.to_hdf(wells, 'wells_resampled.h5')
combined_df = WellDataDumper.concatenate(wells)
combined_df.to_csv('wells_combined.csv')
```

---

## API Reference Quick-Look

### Constants

```python
from rockml.data.adapter import FEATURE_NAME, LABEL_NAME
# FEATURE_NAME = 'features'
# LABEL_NAME   = 'label'
```

### Direction Enum

```python
from rockml.data.adapter.seismic.segy.poststack import Direction
Direction.INLINE      # 'inline'
Direction.CROSSLINE   # 'crossline'
Direction.BOTH        # 'both'
```

### Phase Enum

```python
from rockml.data.adapter.seismic.segy.poststack import Phase
Phase.MIN        # amplitude minimum
Phase.MAX        # amplitude maximum
Phase.CROSSUP    # zero crossing (upward)
Phase.CROSSDOWN  # zero crossing (downward)
```

### PostStackDataDumper

```python
from rockml.data.adapter.seismic.segy.poststack import PostStackDataDumper

PostStackDataDumper.to_hdf(datum_list, path='dataset.h5')
d = PostStackDataDumper.to_dict(datum_list)
```

### CDPGatherDataDumper

```python
from rockml.data.adapter.seismic.segy.prestack import CDPGatherDataDumper

CDPGatherDataDumper.to_hdf(datum_list, path='gathers.h5', save_list=['features', 'label'])
CDPGatherDataDumper.to_velocity_file(datum_list, path='vfunc.dat')
```

### Keras Neural Network Blocks

```python
from rockml.learning.keras.nn_ops import (
    residual_block,
    residual_bottleneck,
    residual_block_transposed,
    crop_border,
)

x = residual_block(input_tensor, filters=64, strides=1)
x = residual_bottleneck(input_tensor, filters=64)
x = residual_block_transposed(input_tensor, filters=32)
x = crop_border(x, shape=target_shape)
```
