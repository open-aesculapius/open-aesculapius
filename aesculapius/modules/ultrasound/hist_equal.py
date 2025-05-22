import cv2
import torch
import numpy as np

from aesculapius.modules.ultrasound.histogram_equalization.\
    architectures.unet_ultrasound import UNet


class HistogramEqualizer:
    _INPUT_SIZE = (256, 256)

    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        apply_gamma: bool = True,
        target_brightness: int = 128,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model(model_path)
        self.apply_gamma = apply_gamma
        self.target_brightness = target_brightness

    @staticmethod
    def _gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        gamma_table = (
            np.array([(x / 255.0) ** gamma * 255 for x in range(256)])
            .round()
            .astype(np.uint8)
        )
        return cv2.LUT(img, gamma_table)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = UNet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image, self._INPUT_SIZE,
                             interpolation=cv2.INTER_LANCZOS4)
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).unsqueeze(
            0).unsqueeze(0).to(self.device)
        return tensor

    @staticmethod
    def _postprocess_image(pred_tensor: torch.Tensor) -> np.ndarray:
        pred = pred_tensor.squeeze().cpu().numpy()
        pred = np.clip(pred, 0, 1)
        pred_uint8 = (pred * 255).astype(np.uint8)

        if pred_uint8.ndim == 3 and pred_uint8.shape[0] in [1, 3]:
            pred_uint8 = np.mean(pred_uint8, axis=0)

        return pred_uint8

    @staticmethod
    def _resize_to_original(image: np.ndarray,
                            original_shape: tuple) -> np.ndarray:
        return cv2.resize(
            image, (original_shape[1], original_shape[0]
                    ), interpolation=cv2.INTER_AREA
        )

    def _enhance(self, image_array: np.ndarray) -> np.ndarray:
        original_shape = image_array.shape
        input_tensor = self._preprocess_image(image_array)

        with torch.no_grad():
            output = self.model(input_tensor)
            output = output[1] if isinstance(output, tuple) else output

        pred_image = self._postprocess_image(output)

        if self.apply_gamma:
            pred_image = self._gamma_transform(pred_image, gamma=1.15)

        return self._resize_to_original(pred_image, original_shape)

    def __call__(self, us_image_obj):
        return self._enhance(us_image_obj)
