import cv2
import math
import os.path
import numpy as np

from aesculapius.modules.core.us_img import UltrasoundImage
from aesculapius.modules.core.utils.image_simiarity.image_similarity import (
    ImageSimilarity,
)


class ArtifactDetector:
    def __init__(
        self,
        comet_tail_steps_x: int,
        comet_tail_steps_y: int,
        ring_down_steps_x: int,
        ring_down_steps_y: int,
        right_down_reference_dir: str,
        comet_tail_reference_dir: str,
    ):
        self._comet_tail_chunks = {"Chunks": [], "Positions": []}
        self._ring_down_chunks = {"Chunks": [], "Positions": []}

        self._comet_tail_references = []
        self._ring_down_references = []

        self._comet_tail_steps_x = comet_tail_steps_x
        self._comet_tail_steps_y = comet_tail_steps_y
        self._ring_down_steps_x = ring_down_steps_x
        self._ring_down_steps_y = ring_down_steps_y

        self._right_down_reference_dir = right_down_reference_dir
        self._comet_tail_reference_dir = comet_tail_reference_dir

    def _build_comet_tail_candidates(self):
        source_size_x = self._src_image.shape[0]
        source_size_y = self._src_image.shape[1]

        step_x = math.ceil(source_size_x / self._comet_tail_steps_x) + 1
        step_y = math.ceil(source_size_y / self._comet_tail_steps_y) + 1

        n_rows, n_cols = math.ceil(source_size_x / step_x), math.ceil(
            source_size_y / step_y
        )

        for r in range(0, source_size_x, step_x):
            for c in range(0, source_size_y, step_y):
                self._comet_tail_chunks["Chunks"].append(
                    self._src_image[r: r + step_x, c: c + step_y]
                )
                self._comet_tail_chunks["Positions"].append((r, c))

        for i in range(n_rows * n_cols - 1, n_rows * n_cols - 1 - n_cols, -1):
            self._comet_tail_chunks["Chunks"].pop(i)
            self._comet_tail_chunks["Positions"].pop(i)
        for i in range(1, n_rows - 1):
            self._comet_tail_chunks["Chunks"].pop((n_rows - i) * n_cols - 1)
            self._comet_tail_chunks["Chunks"].pop((n_rows - i - 1) * n_cols)
            self._comet_tail_chunks["Positions"].pop((n_rows - i) * n_cols - 1)
            self._comet_tail_chunks["Positions"].pop((n_rows - i - 1) * n_cols)
        for i in range(n_cols - 1, -1, -1):
            self._comet_tail_chunks["Chunks"].pop(i)
            self._comet_tail_chunks["Positions"].pop(i)

    def _build_ring_down_candidates(self):
        source_size_x = self._src_image.shape[0]
        source_size_y = self._src_image.shape[1]

        step_x = math.ceil(source_size_x / self._ring_down_steps_x) + 1
        step_y = math.ceil(source_size_y / self._ring_down_steps_y) + 1

        for r in range(0, source_size_x, step_x):
            for c in range(0, source_size_y, step_y):
                self._ring_down_chunks["Chunks"].append(
                    self._src_image[r: r + step_x, c: c + step_y]
                )
                self._ring_down_chunks["Positions"].append((r, c))

    def _build_comet_tail_references(self):
        for r, d, f in os.walk(self._comet_tail_reference_dir):
            for file in f:
                img = UltrasoundImage(os.path.join(r, file))
                self._comet_tail_references.append(img.us_image)

    def _build_ring_down_references(self):
        for r, d, f in os.walk(self._right_down_reference_dir):
            for file in f:
                img = UltrasoundImage(os.path.join(r, file))
                self._ring_down_references.append(img.us_image)

    def _detect_artifacts(self):
        self._build_comet_tail_references()
        mask_1 = self._remove_comet_tail()

        self._build_ring_down_references()
        mask_2 = self._remove_ring_down()

        return mask_1 | mask_2

    def _remove_comet_tail(self):
        step_x = math.ceil(
            self._src_image.shape[0] / self._comet_tail_steps_x) + 1
        step_y = math.ceil(
            self._src_image.shape[1] / self._comet_tail_steps_y) + 1

        s = ImageSimilarity(
            self._comet_tail_chunks["Chunks"], self._comet_tail_references
        )
        similarity_result = s.get_similarity()

        sums = []

        for i in range(len(similarity_result[0])):
            sums.append([np.round(sum(similarity_result[:, i]), 3), i])

        sums = sorted(sums, reverse=True)

        for i in range(len(self._comet_tail_chunks["Chunks"])):
            if i in [sums[j][1] for j in range(0, 4)]:
                chk = self._comet_tail_chunks["Chunks"][i]
                chk = cv2.merge([chk, chk, chk])
                size_x = chk.shape[0]
                size_y = chk.shape[1]
                cv2.line(
                    chk,
                    (int(size_x / 4), int(size_y / 5)),
                    (int(size_x), int(size_y / 5)),
                    (255, 0, 0),
                    1,
                )
                cv2.line(
                    chk,
                    (int(size_x / 4), int(size_y / 4)),
                    (int(size_x), int(size_y / 4)),
                    (255, 0, 0),
                    1,
                )

        masks_all = np.zeros(
            (self._src_image.shape[0],
             self._src_image.shape[1]), dtype=np.uint8
        )
        for i in range(3):
            x1 = self._comet_tail_chunks["Positions"][sums[i][1]][0]
            y1 = self._comet_tail_chunks["Positions"][sums[i][1]][1]
            masks_all[x1: x1 + step_x, y1: y1 + step_y] = 255

        return masks_all

    def _remove_ring_down(self):
        step_x = math.ceil(
            self._src_image.shape[0] / self._ring_down_steps_x) + 1
        step_y = math.ceil(
            self._src_image.shape[1] / self._ring_down_steps_y) + 1

        s = ImageSimilarity(
            self._ring_down_chunks["Chunks"], self._ring_down_references
        )
        similarity_result = s.get_similarity()

        sums = []

        for i in range(len(similarity_result[0])):
            sums.append([np.round(sum(similarity_result[:, i]), 3), i])

        sums = sorted(sums, reverse=True)

        indexes = [sums[i][1] for i in range(0, 4)]

        for i in range(len(self._ring_down_chunks["Chunks"])):
            if i in indexes:
                chk = self._ring_down_chunks["Chunks"][i]
                chk = cv2.merge([chk, chk, chk])
                cv2.line(
                    chk, (0, int(step_x / 5)), (step_x, int(step_y / 5)),
                    (255, 0, 0), 1
                )

        masks_all = np.zeros(
            (self._src_image.shape[0],
             self._src_image.shape[1]), dtype=np.uint8
        )
        for i in range(len(indexes)):
            x1 = self._ring_down_chunks["Positions"][sums[i][1]][0]
            y1 = self._ring_down_chunks["Positions"][sums[i][1]][1]
            masks_all[x1: x1 + step_x, y1: y1 + step_y] = 255

        return masks_all

    def __call__(self, image):
        self._src_image = image
        self._build_comet_tail_candidates()
        self._build_ring_down_candidates()
        return self._detect_artifacts()
