import cv2
import numpy as np
import torch

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class AnomalyDetector:
    def __init__(
        self,
        weights_path,
        device: str = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.weights_path = weights_path
        self._init_model()

    def _init_model(self):
        self.model = load_model(self.weights_path, compile=False)

        initial_lr = 0.001
        lr_schedule = ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True,
        )

        self.model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def _preprocess_image(self):
        target_size = (256, 256)

        image = self._src_image
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

    def _predict_image(self):
        image = self._preprocess_image()
        predicted_mask = self.model.predict(image, verbose=0)
        predicted_mask = np.squeeze(predicted_mask)

        original_image = self._src_image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        resized_mask = cv2.resize(
            predicted_mask, (original_image.shape[1], original_image.shape[0])
        )

        heatmap = cv2.applyColorMap(
            (resized_mask * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        blended = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)

        return blended

    def detect_anomalies(self):
        return self._predict_image()

    def __call__(self, image):
        self._src_image = image
        return self.detect_anomalies()
