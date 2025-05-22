import cv2
import numpy as np


def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.5):
    """Наложение маски на изображение с заданным цветом и прозрачностью."""
    mask = mask.astype(np.uint8) * 255
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color
    result = image.copy()
    result[mask == 255] = cv2.addWeighted(
        image[mask == 255],
        1 - alpha,
        colored_mask[mask == 255],
        alpha,
        0)
    return result
