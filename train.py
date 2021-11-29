import cv2
import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from myModel import build_attention_unet
from data import loadData, tf_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

if __name__ == "__main__":
    datasetPath = r"D:\paper implementation\Refined image segmentation for calorie estimation of multiple dish food item\images\models\dataset"
    (train_x, train_y), (test_x, test_y) = loadData(datasetPath)
    input_shape = (256, 256, 3)
    batch_size = 2
    epochs = 30
    lr = 1e-4
    model_path = "Aunet.h5"
    csv_path = "data.csv"

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    val_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    model = build_attention_unet(input_shape)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(), 'accuracy'
        ]

    )

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
        CSVLogger(csv_path),
        # ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5)
    ]

    train_step = len(train_x) // batch_size
    if len(train_x) % batch_size != 0:
        train_step += 1

    val_step = len(test_x) // batch_size
    if len(test_x) % batch_size != 0:
        val_step += 1

    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=epochs,
              batch_size=batch_size,
              steps_per_epoch=train_step,
              validation_steps=val_step,
              callbacks=callbacks)

    model.save("myModelAUnet/A-unet_model.h5", save_format='tf')
    # model.summary()
