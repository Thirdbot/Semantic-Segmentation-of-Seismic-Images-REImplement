from pathlib import Path
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU')
print("Physical GPUs:", gpu_devices)
if gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from rockml.learning.keras.data_loaders import hdf_2_tfdataset
from rockml.learning.keras.metrics import SparseMeanIoU
from rockml.learning.keras.callbaks import EarlyStoppingAtMinLoss
from rockml.learning.zoo.poststack import danet3fcn, PostStackEstimator

BATCH_SIZE = 1
NUM_CLASSES = 8
EPOCHS = 10

root = Path(__file__).parent.absolute()
penobscot_data = root / "Penobscot Interpretation Dataset" / "dataset.h5"
train_data = root / "train.h5"
validate_data = root / "validate.h5"
save_path = root / "model"



if __name__ == '__main__':

    # train tensor
    train_tf = (
        hdf_2_tfdataset(train_data.as_posix(), 'features', 'label')
        .shuffle(buffer_size=1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    #validate tensor
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