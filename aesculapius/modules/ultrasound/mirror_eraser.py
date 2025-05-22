import numpy as np
import cv2 as cv

from .image_processor import ImageProcessor
from .find_candidates import FindCandidates
from aesculapius.modules.core.image_processing import (
    dist, get_normalized_lightness
)
from aesculapius.modules.core.utils.image_simiarity.image_similarity import (
    ImageSimilarity,
)
from aesculapius.modules.core.utils.image_infilling.image_infilling import (
    get_infilled_image,
)


class MirrorEraser:

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._candidates = []
        self._masks = []
        self._pairs = []

    def find_mirror(self):
        masks = self._masks
        ignore_indexes = []

        len_candidates = len(self._candidates)
        for i in range(len_candidates):
            first = self._candidates[i]
            for j in range(i + 1, len_candidates):
                second = self._candidates[j]
                min_radius = min(first.radius, second.radius)
                max_radius = max(first.radius, second.radius)
                center_dist = dist(*first.center, *second.center)
                if (
                    first.radius + second.radius >= center_dist
                    or max_radius / min_radius > 1.5
                    or center_dist >= 3 * min_radius + max_radius
                ):
                    ignore_indexes.append([i, j])

        image_similarity = ImageSimilarity(masks)
        key_indexes = image_similarity.get_similar_pairs(
            ignore_indexes=ignore_indexes)

        result_mask = np.zeros(self._image.shape, dtype=np.uint8)
        for first_index, second_index in zip(key_indexes[0], key_indexes[1]):
            first_lightness = round(
                get_normalized_lightness(self._masks[first_index], 10), 3
            )
            second_lightness = round(
                get_normalized_lightness(self._masks[second_index], 10), 3
            )
            if abs(first_lightness - second_lightness) < 0.071:
                continue
            if first_lightness < second_lightness:
                cv.drawContours(
                    result_mask,
                    [self._candidates[first_index].contour],
                    -1,
                    (255, 255, 255),
                    -1,
                    cv.LINE_AA,
                )
            else:
                cv.drawContours(
                    result_mask,
                    [self._candidates[second_index].contour],
                    -1,
                    (255, 255, 255),
                    -1,
                    cv.LINE_AA,
                )
        return result_mask

    def __call__(self, image) -> np.ndarray:
        self._image = image

        self._candidates.clear()
        self._masks.clear()
        self._pairs.clear()

        processor = ImageProcessor(self._image)
        analyzer = FindCandidates(self._image)

        filtered_images = processor.apply_multiple_filters()
        for filtered_image in filtered_images:
            candidates, masks = analyzer.find(filtered_image)

            self._candidates.extend(candidates)
            self._masks.extend(masks)
        mirror_mask = self.find_mirror()
        filled_image = get_infilled_image(
            image=self._image, contour=mirror_mask, model_path=self._model_path
        )
        return filled_image
