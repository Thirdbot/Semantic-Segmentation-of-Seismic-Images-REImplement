from pathlib import Path
import os

import h5py
import numpy as np
import tensorflow as tf

root = Path(__file__).parent.absolute()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from rockml.learning.zoo.poststack import danet3fcn


BATCH_SIZE = 4
NUM_SAMPLES = 3
NUM_CLASSES = 8
TILE_SHAPE = (128, 128, 1)

model_path = root / "model" / "best.h5"
validate_data = root / "validate.h5"
output_path = root / "test_graph.png"


def load_validation_samples(path: Path, num_samples: int = NUM_SAMPLES):
    with h5py.File(path, "r") as data:
        total_samples = data["features"].shape[0]
        indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)

        features = data["features"][indices].astype("float32")
        labels = data["label"][indices].astype("uint8")

    return indices, features, labels


def plot_predictions(indices, features, labels, predictions, save_path: Path):
    rows = len(indices)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample_index in enumerate(indices):
        axes[row, 0].imshow(np.squeeze(features[row]), cmap="bone")
        axes[row, 0].set_title(f"Sample {sample_index} - Seismic")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(labels[row], cmap="tab20", vmin=0, vmax=7)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(predictions[row], cmap="tab20", vmin=0, vmax=7)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=plt.cm.tab20(i / 7), label=f"Class {i}")
        for i in range(8)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    model = danet3fcn(TILE_SHAPE, NUM_CLASSES)
    model.load_weights(model_path.as_posix())

    sample_indices, sample_features, sample_labels = load_validation_samples(validate_data)

    probabilities = model.predict(sample_features, batch_size=BATCH_SIZE)
    sample_predictions = np.argmax(probabilities, axis=-1)

    plot_predictions(
        sample_indices,
        sample_features,
        sample_labels,
        sample_predictions,
        output_path,
    )
    print(f"Saved test graph to {output_path}")
