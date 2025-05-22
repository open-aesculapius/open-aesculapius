import math
import numpy as np
import cv2 as cv


def dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_normalized_lightness(image: np.ndarray, dim: int = 10):
    image = cv.resize(image, (dim, dim))
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    lightness, *_ = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
    normalized_lightness = np.mean(lightness / np.max(lightness))
    return normalized_lightness
