import cv2
import os
from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def loadData(datasetPath):
    images = sorted(glob(os.path.join(datasetPath, "train/*")))
    mask = sorted(glob(os.path.join(datasetPath, "mask/*")))

    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(mask, test_size=0.2, random_state=42)

    return (train_x, train_y), (test_x, test_y)


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        x = read_image(image_path)
        y = read_mask(mask_path)

        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 1])

    return image, mask


def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    datasetPath = r"D:\paper implementation\Refined image segmentation for calorie estimation of multiple dish food item\images\models\dataset"
    (train_x, train_y), (test_x, test_y) = loadData(datasetPath)

    train_dataset = tf_dataset(train_x, train_y, 6)

    # for image, mask in train_dataset:
    #     print(image.shape, mask.shape)
    batch = 8
    train_steps = len(train_x) // batch
    if len(train_x) % batch != 0:
        train_steps += 1


