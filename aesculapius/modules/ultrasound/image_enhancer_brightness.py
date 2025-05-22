import cv2


class ImageEnhancerBrightness:
    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, image):
        return cv2.convertScaleAbs(
            image, alpha=1.0, beta=self.brightness)
