import cv2
import numpy as np
import tensorflow as tf


class ThyroidSegmentation:
    _INPUT_SHAPE = (256, 256)

    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self._INPUT_SHAPE)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=(0, -1))

    def _postprocess_mask(
        self, predicted: np.ndarray, original_shape: tuple[int, int]
    ) -> np.ndarray:
        predicted = predicted[0, :, :, 0]
        predicted_resized = cv2.resize(
            predicted, (original_shape[1], original_shape[0])
        )
        binary_mask = (predicted_resized > 0.5).astype(np.uint8) * 255
        return self._refine_mask(binary_mask)

    def _refine_mask(self, mask: np.ndarray, min_area=500,
                     kernel_size=5) -> np.ndarray:
        smoothed = cv2.medianBlur(mask, 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        refined = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                refined[labels == i] = 255
        return refined

    def draw_contours(self, original: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_rgb, contours, -1, (255, 0, 0), 2)
        return image_rgb

    def overlay_mask(
            self,
            original: np.ndarray,
            mask: np.ndarray,
            color: tuple = (255, 0, 0),
            alpha: float = 0.4
    ) -> np.ndarray:
        image_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        mask_rgb = np.zeros_like(image_rgb)
        mask_rgb[mask > 0] = color
        return cv2.addWeighted(image_rgb, 1 - alpha, mask_rgb, alpha, 0)

    def __call__(
            self,
            us_image_obj
    ):
        original = us_image_obj
        input_tensor = self._preprocess_image(original)
        prediction = self.model.predict(input_tensor, verbose=0)
        mask = self._postprocess_mask(prediction, original.shape)
        overlay = self.overlay_mask(original, mask)
        contour = self.draw_contours(original, mask)
        return (mask, overlay), contour
