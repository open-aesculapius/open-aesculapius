import cv2
import numpy as np
from keras import Model, Input
from keras.layers import TFSMLayer


class ImageEnhancer:
    def __init__(self, weights: str):
        self.weights = weights
        self._init_model()

    def _init_model(self):
        layer = TFSMLayer(self.weights, call_endpoint="serving_default")
        inputs = Input(shape=(640, 960, 3))
        outputs = layer(inputs)
        self.model = Model(inputs, outputs)

    def _preprocess_image(self):
        target_size = (960, 640)
        image = self._src_image
        original_size = (image.shape[1], image.shape[0])

        resized = cv2.resize(image, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        input_array = np.expand_dims(normalized, axis=0)

        self._original_size = original_size
        return input_array

    def _predict_image(self):
        input_array = self._preprocess_image()
        output_dict = self.model.predict(input_array, verbose=0)
        output = list(output_dict.values())[0]
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = cv2.resize(
            output[0], (self._original_size[0], self._original_size[1]))
        return output

    def enhance_image(self):
        return self._predict_image()

    def __call__(self, image):
        self._src_image = image
        return self.enhance_image()
