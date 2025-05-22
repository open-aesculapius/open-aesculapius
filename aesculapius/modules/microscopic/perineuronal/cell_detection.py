import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .architectures.rt_detr import TransformerVGG


class CellDetector:
    Z_SIZE = 24

    def __init__(
        self,
        weights_path,
        device: str = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        layers: list = None,
        threshold: float = 0.7,
    ):
        self.weights_path = weights_path
        self.device = device
        self.layers = layers
        self.image_size = layers[0].shape
        self.threshold = threshold

        self._init_model()
        self._init_transform()

    def _init_model(self):
        model = TransformerVGG
        self.model = model()
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def _init_transform(self):
        self.transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.5, 0.5, 0.5] * self.Z_SIZE,
                    std=[0.5, 0.5, 0.5] * self.Z_SIZE,
                ),
                ToTensorV2(),
            ]
        )

    def _preprocess_data(self):
        data = np.zeros(
            (self.image_size[0], self.image_size[1],
             self.image_size[2] * self.Z_SIZE)
        )
        i = 0
        for image in self.layers:
            data[:, :, i: i + 3] = np.array(image)
            i += 3
        data = (
            self.transform(image=data)["image"]
            .view(
                self.Z_SIZE,
                self.image_size[2], self.image_size[0], self.image_size[1]
            )
            .unsqueeze(0)
            .to(self.device)
        )
        return data

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + np.exp(-v))

    def _postprocess_data(self, data):
        data = data.detach().squeeze(0).squeeze(0).cpu().numpy()
        coordinates = []

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if self.sigmoid(data[y][x]) > 0.6:
                    coordinates.append(
                        [32 * x, 32 * y, 32 * (x + 1), 32 * (y + 1)])
        return coordinates

    def forward(self):
        image = self._preprocess_data()
        with torch.no_grad():
            result, _ = self.model(image)
        return self._postprocess_data(result)

    def __call__(self):
        return self.forward()
