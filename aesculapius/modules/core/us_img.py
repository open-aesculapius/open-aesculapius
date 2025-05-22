import cv2
import numpy as np
from pathlib import Path
from typing import Union, List


class UltrasoundImage:
    def __init__(self, path):
        self._images = self._get_from_file(path)

    @property
    def us_image(self) -> Union[np.ndarray, List[np.ndarray]]:
        return self._images

    @us_image.setter
    def us_image(self, image: Union[np.ndarray, List[np.ndarray]]):
        self._images = image

    @staticmethod
    def _get_from_file(path: str) -> np.ndarray:
        path = Path(path)

        supported_exts = ["*.png", "*.jpg",
                          "*.jpeg", "*.tif",
                          "*.tiff", "*.bmp"]

        if path.is_dir():
            files = []
            for ext in supported_exts:
                files.extend(path.glob(ext))
            if not files:
                raise ValueError(f"No image files \
                 found in directory: {path.resolve()}")

            loaded_images = []
            for f in sorted(files):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    print(f"Warning: Skipping unreadable \
                     or empty file: {f}")
                    continue
                loaded_images.append(img)

            if not loaded_images:
                raise ValueError(
                    f"Error: Image could not be loaded \
                                from path: {path.resolve()}"
                )

            return loaded_images

        elif path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                raise ValueError(
                    f"Failed to load image from \
                    file: {path.resolve()}")
            return img

        else:
            raise ValueError(
                f"Invalid path: {path.resolve()}"
            )
