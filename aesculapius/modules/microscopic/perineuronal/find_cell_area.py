import torch
import numpy as np
import cv2
from torchvision import transforms
from scipy import ndimage
from .architectures.find_cell_area import Unet


class AreaDetector:
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
        self.threshold = threshold

        self._init_model()
        self._init_transform()

    def _init_model(self):
        model = Unet
        self.model = model()
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def _init_transform(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def _modify_image(image, threshold):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] > threshold:
                    image[i][j] = 255
                else:
                    image[i][j] = 0
        return image

    def _postprocess_data(self, data):
        data = [
            self._modify_image(
                image.detach().squeeze(0).squeeze(0).cpu().data.numpy(),
                self.threshold
            )
            for image in data
        ]
        return data

    def _calc_metrics(self, mask: np.ndarray):
        kernel = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_dilation(
            mask, structure=kernel).astype(kernel.dtype)
        mask = mask * 255
        contours, _ = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        if (
            abs(contour[0][0][1] - contour[-1][0][1]) > 1
            or abs(contour[0][0][0] - contour[-1][0][0]) > 1
        ):
            contour = np.append(
                contour, [[[contour[0][0][0], contour[0][0][1] + 1]]], axis=0
            )
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        return area, perimeter

    def forward(self):
        images = [
            self.transform(image).to(self.device).unsqueeze(0)
            for image in self.layers
        ]
        with torch.no_grad():
            results = [self.model(image) for image in images]
        masks = self._postprocess_data(results)
        metrics = [[self._calc_metrics(masks[i])] for i in range(len(masks))]
        return masks, metrics

    def __call__(self, *args, **kwargs):
        return self.forward()
