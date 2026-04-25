from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from rockml.data.adapter.seismic.segy.poststack import PostStackDatum, Direction, PostStackDataDumper
from rockml.data.sampling import split_dataset
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image
from rockml.learning.keras.data_loaders import hdf_2_tfdataset
from rockml.learning.keras.metrics import SparseMeanIoU
from rockml.learning.keras.callbaks import EarlyStoppingAtMinLoss
from rockml.learning.zoo.poststack import danet3fcn, PostStackEstimator

print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

TILE_SHAPE = (256, 256)
STRIDE_SHAPE = (128, 128)
BATCH_SIZE = 16
NUM_CLASSES = 8
EPOCHS = 100

root = Path(__file__).parent.absolute()
penobscot_data = root / "Penobscot Interpretation Dataset" / "dataset.h5"
train_data = root / "train.h5"
validate_data = root / "validate.h5"
save_path = root / "model"

compose = Composer([
    image.ScaleIntensity(gray_levels=256, percentile=1.0),
    image.FillSegmentationMask(),
    image.ViewAsWindows(tile_shape=TILE_SHAPE, stride_shape=STRIDE_SHAPE, auto_pad=False, filters=None),
])

# Batch-read all arrays from HDF5 in one shot — avoids per-sample I/O overhead
with h5py.File(penobscot_data, 'r') as f:
    features   = f['features'][:]
    labels     = f['label'][:]
    directions = f['direction'][:]
    line_nums  = f['line_number'][:]
    pix_depths = f['pixel_depth'][:]
    columns    = f['column'][:]

tiles = [
    PostStackDatum(
        features[i], labels[i],
        Direction.INLINE if directions[i] in b'inline' else Direction.CROSSLINE,
        line_nums[i], pix_depths[i], columns[i],
    )
    for i in range(len(columns))
]

transform_tiles = compose.apply(dataset=tiles)

train_dataset, validation_dataset = split_dataset(transform_tiles, 0.3)

PostStackDataDumper.to_hdf(train_dataset, train_data.as_posix())
PostStackDataDumper.to_hdf(validation_dataset, validate_data.as_posix())


def plot_random_samples(tile_list, num_samples=3):
    indices = random.sample(range(len(tile_list)), num_samples)
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    for i, idx in enumerate(indices):
        tile = tile_list[idx]
        feature = np.squeeze(tile.features)
        label = np.squeeze(tile.label)
        axes[i, 0].imshow(feature, cmap='bone')
        axes[i, 0].set_title(f"Sample {idx} - Feature (Line {tile.line_number})")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(label, cmap='jet')
        axes[i, 1].set_title(f"Sample {idx} - Label")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


# plot_random_samples(tiles, num_samples=3)

train_tf = (
    hdf_2_tfdataset(train_data.as_posix(), 'features', 'label')
    .shuffle(buffer_size=2000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
validate_tf = (
    hdf_2_tfdataset(validate_data.as_posix(), 'features', 'label')
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
train_metrics = [SparseMeanIoU(num_classes=NUM_CLASSES)]
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

model = danet3fcn(train_tf.element_spec[0].shape[1:], NUM_CLASSES)

e1 = PostStackEstimator(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_metrics=train_metrics,
)
e1.fit(
    epochs=EPOCHS,
    train_set=train_tf,
    valid_set=validate_tf,
    callbacks=[EarlyStoppingAtMinLoss(patience=15)],
)
e1.save_model(is_best=True, path=save_path.as_posix())