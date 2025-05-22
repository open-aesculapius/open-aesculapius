import cv2
import numpy as np
from PIL import ImageFilter as Filter
from PIL import Image as PilImage


def inverse(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(img)


def blur(img: np.ndarray, ksize: tuple = (3, 3)) -> np.ndarray:
    return cv2.blur(img, ksize)


def clahe(img: np.ndarray) -> np.ndarray:
    clahe_filter = cv2.createCLAHE()
    return clahe_filter.apply(img)


def edge_enhance(img: np.ndarray) -> np.ndarray:
    return np.array(PilImage.fromarray(img).filter(Filter.EDGE_ENHANCE))


def contours_simple(img: np.ndarray, threshold_value: int = 180) -> np.ndarray:
    ret, image_thresh = cv2.threshold(
        img, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = np.uint8(np.zeros((img.shape[0], img.shape[1])))
    cv2.drawContours(image_contours, contours, -1, (255, 255, 255), 1)
    return image_contours


def contours_canny(
    img: np.ndarray, threshold1: int = 100, threshold2: int = 200
) -> np.ndarray:
    return cv2.Canny(img, threshold1, threshold2)


def dilation(img: np.ndarray, kernel_size: tuple = (2, 2)) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel)


def erode(
    img: np.ndarray, kernel_size: tuple = (2, 2), iterations: int = 1
) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)
