import torch
import cv2
import numpy as np


class ImageEnhancer:
    def __init__(self, weights: str):
        self.weights = weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model()

    def _init_model(self):
        self.model = torch.jit.load(self.weights, map_location=self.device)
        self.model.eval()

    def _preprocess_image(self):
        image = self._src_image
        self._original_size = (image.shape[1], image.shape[0])

        resized = cv2.resize(image, (512, 512))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = (torch.from_numpy(normalized).
                  permute(2, 0, 1).unsqueeze(0).to(self.device))
        return tensor

    def _predict_image(self):
        input_tensor = self._preprocess_image()
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.clamp(output, 0, 1)

        if output.dim() == 4 and output.shape[1] == 1:
            output_np = output.squeeze().cpu().numpy()
        elif output.dim() == 4 and output.shape[1] == 3:
            output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
            output_np = (output_np * 255.0).astype(np.uint8)
            output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
        elif output.dim() == 4:
            output_np = output.mean(dim=1).squeeze().cpu().numpy()
        else:
            raise ValueError(f"[ERROR] Ошибка выхода: {output.shape}")

        if output_np.max() <= 1.0:
            output_np = (output_np * 255).astype(np.uint8)
        else:
            output_np = output_np.astype(np.uint8)

        output_resized = cv2.resize(output_np, self._original_size)
        return output_resized

    def enhance_image(self):
        return self._predict_image()

    def __call__(self, image):
        self._src_image = image
        return self.enhance_image()