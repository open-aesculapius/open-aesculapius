import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np


class AstrocytesClassificator:
    def __init__(
        self,
        weights_path: str,
        device: str = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        layers: list = None,
    ):
        self.weights_path = weights_path
        self.device = device
        self.layers = layers
        self.class_names = ["healthy", "sick"]

        self._init_model()
        self._init_transform()

    def _init_model(self):
        self.model = models.vgg19()
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Linear(num_features, 2)
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def _init_transform(self):
        self.data_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ]
        )

    def _preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        image = self.data_transform(image)
        image = image.unsqueeze(0)
        return image.to(self.device)

    def _model_inference(self, image):
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        predicted_class_idx = probabilities.argmax()
        predicted_class = self.class_names[predicted_class_idx]
        probability = probabilities[predicted_class_idx]

        return predicted_class, probability

    def classify(self):
        results = {}
        for i, layer in enumerate(self.layers):
            layer = self._preprocess_image(layer)
            predicted_class, probability = self._model_inference(layer)
            results[i] = [predicted_class, probability]
        return results
