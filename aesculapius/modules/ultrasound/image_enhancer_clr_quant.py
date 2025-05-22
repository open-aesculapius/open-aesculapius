import numpy as np
from sklearn.cluster import KMeans


class ImageEnhancerColorQuantizer:
    def __init__(self, n_colors: int, random_state: int = 42):
        self.n_colors = n_colors
        self.random_state = random_state

    def __call__(self, image) -> np.ndarray:
        img = image
        original_shape = img.shape

        if len(original_shape) == 2:
            data = img.reshape((-1, 1)).astype(np.float32)
            kmeans = KMeans(
                n_clusters=self.n_colors,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(data)
            centers = np.uint8(kmeans.cluster_centers_)
            quantized = centers[labels].reshape(original_shape)
        else:
            h, w, c = original_shape
            data = img.reshape((-1, 3)).astype(np.float32)
            kmeans = KMeans(
                n_clusters=self.n_colors,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(data)
            centers = np.uint8(kmeans.cluster_centers_)
            quantized = centers[labels].reshape((h, w, 3))

        return quantized
