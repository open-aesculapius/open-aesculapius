import cv2


class ImageEnhancerContrast:
    def __init__(self, contrast: float = 1.0):
        self.contrast = contrast

    def __call__(self, image):
        alpha = 1.0 + (self.contrast / 100.0)
        alpha = max(0.0, min(alpha, 3.0))
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
