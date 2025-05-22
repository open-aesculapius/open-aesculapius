from collections import Counter

from .noise_level import NoiseLevel
from sewar.full_ref import ergas, uqi
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
    mean_squared_error as mse,
)


_METRICS = {
    "PSNR": {
        "call": lambda a, b: psnr(a, b, data_range=255.0),
        NoiseLevel.LOW: 32,
        NoiseLevel.MEDIUM: 20,
        NoiseLevel.HIGH: float("-inf"),
    },
    "MSE": {
        "call": lambda a, b: mse(a, b),
        NoiseLevel.LOW: 50,
        NoiseLevel.MEDIUM: 120,
        NoiseLevel.HIGH: float("inf"),
    },
    "SSIM": {
        "call": lambda a, b: ssim(a, b, data_range=255.0),
        NoiseLevel.LOW: 0.7,
        NoiseLevel.MEDIUM: 0.4,
        NoiseLevel.HIGH: float("-inf"),
    },
    "UQI": {
        "call": lambda a, b: uqi(a, b),
        NoiseLevel.LOW: 0.98,
        NoiseLevel.MEDIUM: 0.94,
        NoiseLevel.HIGH: float("-inf"),
    },
    "ERGAS": {
        "call": lambda a, b: ergas(a, b),
        NoiseLevel.LOW: 2,
        NoiseLevel.MEDIUM: 5,
        NoiseLevel.HIGH: float("inf"),
    },
}


def estimate_noise(first_image, second_image):
    results = []
    if first_image.shape != second_image.shape:
        raise ValueError(
            "The sizes of the first image \
                         and the second image are different: "
            f"{first_image.shape} != {second_image.shape}"
        )
    for metric in _METRICS:
        result = _METRICS[metric]["call"](first_image, second_image)
        results.append({"name": metric, "value": result})
    return results


def classify_noise(metrics):

    classified_noise = {}
    for metric in metrics:
        metric_name = metric["name"]
        value = metric["value"]

        low = _METRICS[metric_name][NoiseLevel.LOW]
        mid = _METRICS[metric_name][NoiseLevel.MEDIUM]
        high = _METRICS[metric_name][NoiseLevel.HIGH]

        noise_level = NoiseLevel.UNKNOWN
        if min(mid, high) <= value <= max(mid, high):
            noise_level = NoiseLevel.HIGH
        elif min(low, mid) <= value <= max(low, mid):
            noise_level = NoiseLevel.MEDIUM
        else:
            noise_level = NoiseLevel.LOW

        classified_noise[metric_name] = noise_level
    result_counter = Counter(list(classified_noise.values()))
    return result_counter.most_common()[0][0]
