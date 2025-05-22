"""
MIT License

Copyright (c) 2024 Foat
Copyright (c) 2019 Ryan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import PIL

import numpy as np  # noqa: E402

from tensorflow.keras.applications.mobilenet import (
    MobileNet,
    preprocess_input,
)  # noqa: E402
from tensorflow.keras.preprocessing import image as process_image  # noqa: E402
from tensorflow.keras.layers import GlobalAveragePooling2D  # noqa: E402
from tensorflow.keras import Model  # noqa: E402


def resize_image(image, target_size=(224, 224)):
    img = PIL.Image.fromarray(image)
    if len(image.shape) < 3:
        img = img.convert("RGB")
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            resample = PIL.Image.Resampling.NEAREST
            img = img.resize(width_height_tuple, resample)
    return process_image.img_to_array(img)


class DeepModel:
    """MobileNet deep model."""

    def __init__(self):
        self._model = self._define_model()

    def predict(self, images):
        images = np.array([DeepModel.preprocess_image(img) for img in images])
        return self._model.predict(images, verbose=0)

    @staticmethod
    def _define_model(output_layer=-1):
        """Define a pre-trained MobileNet model.

        Args:
            output_layer: the number of layer that output.

        Returns:
            Class of keras model with weights.
        """
        base_model = MobileNet(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        output = base_model.layers[output_layer].output
        output = GlobalAveragePooling2D()(output)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    @staticmethod
    def preprocess_image(x):
        """Process an image to numpy array.

        Returns:
            Numpy array of the image.
        """
        x = resize_image(x, (224, 224))
        x = preprocess_input(x)
        return x

    @staticmethod
    def cosine_distance(input1, input2):
        """Calculating the distance of two inputs.

        The return values lies in [-1, 1]. `-1` denotes two features are
        the most unlike, `1` denotes they are the most similar.

        Args:
            input1, input2: two input numpy arrays.

        Returns:
            Element-wise cosine distances of two inputs.
        """
        return np.dot(input1, input2.T) / np.dot(
            np.linalg.norm(input1, axis=1, keepdims=True),
            np.linalg.norm(input2.T, axis=0, keepdims=True),
        )
