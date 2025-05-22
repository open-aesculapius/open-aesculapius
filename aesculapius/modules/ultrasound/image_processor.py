import numpy as np
from aesculapius.modules.core.utils.image_preprocessing.filters import (
    inverse,
    blur,
    clahe,
    contours_simple,
    erode,
)


class ImageProcessor:
    _FILTERS = [
        [inverse, blur, clahe, contours_simple],
        [blur, clahe, contours_simple],
        [blur, clahe, erode, contours_simple],
    ]

    def __init__(self, image: np.ndarray):
        self.image = image
        self.last_processed_image = None

    def apply_multiple_filters(self) -> list[np.ndarray]:
        process_images = []
        for filter_sequence in self._FILTERS:
            self.last_processed_image = self.image
            for filter_callable in filter_sequence:
                self.last_processed_image = filter_callable(
                    self.last_processed_image)
            process_images.append(self.last_processed_image)
        return process_images
