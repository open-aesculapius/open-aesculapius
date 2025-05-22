import cv2 as cv
import numpy as np


class Candidate:
    def __init__(self, contour: np.array):
        self.crop = None
        self.contour = contour
        circle = cv.minEnclosingCircle(contour)
        self.center = circle[0]
        self.radius = circle[1]

    def get_contour(self) -> np.array:
        return self.contour
