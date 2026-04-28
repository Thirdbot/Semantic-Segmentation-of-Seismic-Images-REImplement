from rockml.data.adapter.seismic.segy.poststack import PostStackDatum, Direction, PostStackDataDumper
from rockml.data.sampling import split_dataset
from rockml.data.transformations import Composer
from rockml.data.transformations.seismic import image
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm

root = Path(__file__).parent.absolute()
penobscot_data = root / "Penobscot Interpretation Dataset" / "dataset.h5"
train_data = root / "train.h5"
validate_data = root / "validate.h5"

TILE_SHAPE = (128, 128)
STRIDE_SHAPE = (64, 64)

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

if __name__ == '__main__':
    compose = Composer([
        # gray scale
        image.ScaleIntensity(gray_levels=256, percentile=1.0),
        # modify for segmentation mask
        image.FillSegmentationMask(),
        # slide window patch
        image.ViewAsWindows(tile_shape=TILE_SHAPE, stride_shape=STRIDE_SHAPE, auto_pad=False, filters=None),
    ])


    # Batch-read all arrays from HDF5 in one shot — avoids per-sample I/O overhead
    with h5py.File(penobscot_data, 'r') as f:
        columns    = len(f['column'])
        all_tiles = []
        for i in tqdm(range(columns)):

            tile = PostStackDatum(
                f['features'][i], f['label'][i],
                Direction.INLINE if f['direction'][i] == b'inline' else Direction.CROSSLINE,
                f['line_number'][i], f['pixel_depth'][i], f['column'][i],
            )
            # transformations
            all_tiles.extend(compose.apply(dataset=tile))


    # split train,validation
    train_dataset, validation_dataset = split_dataset(all_tiles, 0.3)

    PostStackDataDumper.to_hdf(train_dataset, train_data.as_posix())
    PostStackDataDumper.to_hdf(validation_dataset, validate_data.as_posix())

    plot_random_samples(all_tiles,3)
