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

"""Image similarity using deep features.

Recommendation: the threshold of the `DeepModel.cosine_distance` can be set as
the following values.
    0.84 = greater matches amount
    0.845 = balance, default
    0.85 = better accuracy
"""
import numpy  # noqa: E402
import numpy as np  # noqa: E402

from .model_util import DeepModel  # noqa: E402


class ImageSimilarity:
    """Image similarity."""

    def __init__(self, images: list[numpy.ndarray], reference_images=None):
        self._batch_size = 64
        self._model = None
        self._images = images
        if reference_images is not None:
            self._reference_images = reference_images

    @property
    def batch_size(self):
        """Batch size of model prediction."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def single_iteration(self):
        """Calculate the cosine distance of two inputs, save the matched fields
         to `.csv` file.

        Args:
            save_header: header of the result `.csv` file.
            thresh: threshold of the similarity.
            title1, title2: Optional. If `save_data()` is not invoked, titles
            of two inputs should be passed.

        Returns:
            A matrix of element-wise cosine distance.

        Note:
            1. The threshold can be set as the following values.
                0.84 = greater matches amount
                0.845 = balance, default
                0.85 = better accuracy

            2. If the generated files are exist, set `title1` or `title2` as
            same as the title of their source files. For example, pass
            `benchmark.csv` to `save_data()` will generate `_test_feature.h5`
            and `_test_fields.csv` files, so set `title1` or `title2`
            to `benchmark`, and `save_data()` will not be required
            to invoke.
        """

        if self._model is None:
            self._model = DeepModel()

        # Prediction
        features = self._model.predict(self._images)

        distances = DeepModel.cosine_distance(features, features)
        return distances

    def double_iteration(self):
        """Calculate the cosine distance of two inputs, save the matched fields
         to `.csv` file.

        Args:
            save_header: header of the result `.csv` file.
            thresh: threshold of the similarity.
            title1, title2: Optional. If `save_data()` is not invoked, titles
            of two inputs should be passed.

        Returns:
            A matrix of element-wise cosine distance.

        Note:
            1. The threshold can be set as the following values.
                0.84 = greater matches amount
                0.845 = balance, default
                0.85 = better accuracy

            2. If the generated files are exist, set `title1` or `title2` as
            same as the title of their source files. For example, pass
            `benchmark.csv` to `save_data()` will generate `_test_feature.h5`
             and `_test_fields.csv` files, so set `title1` or `title2`
             to `benchmark`, and `save_data()` will not be required to invoke.
        """

        if self._model is None:
            self._model = DeepModel()

        # Prediction
        features_1 = self._model.predict(self._images)

        features_2 = self._model.predict(self._reference_images)

        distances = DeepModel.cosine_distance(features_2, features_1)
        return distances

    def get_similar_pairs(self, ignore_indexes=None):
        if ignore_indexes is None:
            ignore_indexes = []

        self.batch_size = len(self._images)
        result = self.single_iteration()

        for i in range(len(result)):
            for j in range(i + 1, len(result[i])):
                if [i, j] in ignore_indexes:
                    result[i][j] = 0

        for i in range(len(result)):
            for j in range(len(result[i])):
                if i > j:
                    result[i][j] = 0
                else:
                    result[i][j] = round(result[i][j], 3)

        result = np.array(result)
        result[result < 0.7] = 0
        result[result > 0.861] = 0

        max_indexes_i = []
        max_indexes_j = []

        while result.max() != 0:
            max_indexes = np.where(result == result.max())
            i, j = max_indexes[0][0], max_indexes[1][0]
            result[:, i] = 0
            result[:, j] = 0
            result[i, :] = 0
            result[j, :] = 0
            max_indexes_i.append(int(i))
            max_indexes_j.append(int(j))
        return [max_indexes_i, max_indexes_j]

    def get_similarity(self):
        self.batch_size = max(len(self._images), len(self._reference_images))
        result = self.double_iteration()

        return result
