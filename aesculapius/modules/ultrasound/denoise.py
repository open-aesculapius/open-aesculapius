import cv2
import numpy as np
import tensorflow as tf

from aesculapius.modules.core.estimate_noise import (
    estimate_noise, classify_noise)


class Denoiser:
    _IMAGE_SCALE = (1024, 1024)
    _SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, self._IMAGE_SCALE) / 255.0
        resized_image = np.expand_dims(resized_image, axis=(0, -1))
        return resized_image

    def postprocess_image(self, predicted_image: np.ndarray) -> np.ndarray:
        predicted_image_resized = (
            predicted_image[0].reshape(
                self._IMAGE_SCALE[0], self._IMAGE_SCALE[1]) * 255
        )
        return predicted_image_resized.astype(float)

    def resize_to_original(
        self, image: np.ndarray, us_image_shape: tuple[int, ...]
    ) -> np.ndarray:
        return cv2.resize(image, (us_image_shape[1], us_image_shape[0]))

    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(np.float32(image), -1,
                            self._SHARPEN_KERNEL).astype(float)

    def __call__(
        self, us_image_obj
    ):
        image = us_image_obj
        processed_image = self.preprocess_image(image)
        pred = self.model.predict(processed_image, verbose=0)
        pred_image = self.postprocess_image(pred)
        sharpened_image = self.sharpen_image(pred_image)
        original_image = (processed_image[0] * 255).astype(float)
        sharpened_image_resized = self.resize_to_original(
            sharpened_image, image.shape)
        original_image = np.squeeze(original_image)
        absolute_metrics = estimate_noise(original_image, pred_image)
        classified_metrics = classify_noise(absolute_metrics)
        res_image = sharpened_image_resized
        us_image_obj = res_image
        return us_image_obj, classified_metrics
