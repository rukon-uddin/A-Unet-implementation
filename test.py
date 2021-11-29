import cv2
import os
from glob import glob
import numpy as np
import tensorflow
from tensorflow.keras import models
from tqdm import tqdm


if __name__ == "__main__":
    test_images = glob("test_images/*")
    model = models.load_model("myModelAUnet/A-unet_model.h5")
    model.summary()
    k = 1;
    for path in tqdm(test_images, total=len(test_images)):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        original_image = x
        h, w, _ = x.shape

        x = cv2.resize(x, (256,256))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        pred_mask = model.predict(x)

        pred_mask = pred_mask[0]

        pred_mask = np.concatenate(
            [
                pred_mask, pred_mask, pred_mask
            ], axis=2
        )

        pred_mask[pred_mask > 0.5] = 255
        pred_mask = pred_mask.astype(np.float32)

        pred_mask = cv2.resize(pred_mask, (w, h))
        original_image = original_image.astype(np.float32)
        alpha = 0.5
        cv2.addWeighted(pred_mask, alpha, original_image, 1-alpha, 0, original_image)

        cv2.imwrite(f"mask/{k}.png", original_image)
        cv2.imwrite(f"mask/{k+10}.png", pred_mask)

        k += 1

