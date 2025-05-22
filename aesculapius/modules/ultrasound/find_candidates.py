import cv2 as cv
import numpy as np

from .candidate import Candidate
from aesculapius.modules.core.image_processing import dist


class FindCandidates:
    def __init__(self, original_image: np.ndarray):
        self._original_image = original_image
        self._filtered_image = None

        self._candidates = []
        self._crops = []

    def _build_candidates(self):
        ret, thresh = cv.threshold(
            self._filtered_image, 150, 200, cv.THRESH_BINARY)
        unsorted_contours, _ = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        contours = [cv.approxPolyDP(contour, 4, False)
                    for contour in unsorted_contours]
        for contour in contours:
            self._candidates.append(Candidate(contour))

    def find(self, filtered_image: np.ndarray):
        self._candidates.clear()
        self._crops.clear()

        self._filtered_image = filtered_image

        self._build_candidates()
        self._filter_candidates()
        self._make_crops()
        return self._candidates, self._crops

    def _filter_candidates(self):
        self._filter_candidates_by_radius()
        self._filter_candidates_by_intersections()

    def _filter_candidates_by_radius(
            self, min_radius: int = 20, max_radius: int = 70):
        self._candidates[:] = [
            x for x in self._candidates if min_radius < x.radius < max_radius
        ]

    def _filter_candidates_by_intersections(self):
        exclude_list = set()
        candidates_size = len(self._candidates)
        for i in range(candidates_size):
            first = self._candidates[i]
            for j in range(i + 1, candidates_size):
                second = self._candidates[j]
                max_radius = max(first.radius, second.radius)
                min_radius = max(first.radius, second.radius)
                if (
                    dist(*first.center, *second.center) <= 15
                    and (max_radius - min_radius) / max_radius <= 0.2
                ):
                    exclude_list.add(first if min_radius ==
                                     first.radius else second)
        self._candidates[:] = [
            x for x in self._candidates if x not in exclude_list]

    def _make_crops(self):
        for candidate in self._candidates:
            x = max(int(candidate.center[0]) - int(candidate.radius), 0)
            y = max(int(candidate.center[1]) - int(candidate.radius), 0)
            h = 2 * int(candidate.radius)
            w = 2 * int(candidate.radius)
            self._crops.append(self._original_image[y: y + h, x: x + w])
