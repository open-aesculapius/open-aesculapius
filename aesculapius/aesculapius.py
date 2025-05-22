from .modules import DeepCalculator
from .modules import CellDetector
from .modules import AreaDetector

from .modules import Denoiser
from .modules import ArtifactDetector
from .modules import AnomalyDetector
from .modules import ArtifactsRemover
from .modules import MirrorEraser

from .modules import AstrocyteSegmenter
from .modules import AstrocytesClassificator

from .modules import ThyroidSegmentation
from .modules import ImageEnhancer
from .modules import ImageEnhancerContrast
from .modules import ImageEnhancerColorQuantizer
from .modules import ImageEnhancerBrightness
from .modules import HistogramEqualizer

from .modules.core.utils.download_weights import (
    download_file_from_hf,
    download_folder_from_hf,
)


def detect_perineuronal_cells(image, config):
    download_file_from_hf(config["weights"])
    detector = CellDetector(
        weights_path=config["weights"], layers=image
    )
    bounding_box = detector()
    return bounding_box


def find_perineuronal_cell_area(image, config):
    download_file_from_hf(config["weights"])
    detector = AreaDetector(
        weights_path=config["weights"], layers=image
    )
    masks, metrics = detector()
    return masks, metrics


def calculate_perineuronal_cell_depth(image, config):
    download_file_from_hf(config["weights"])
    depth_calculator = DeepCalculator(
        weights_path=config["weights"], layers=image
    )
    depth, coords = depth_calculator()
    return depth, coords


def detect_ultrasound_artifacts(image, config):
    detector = ArtifactDetector(
        comet_tail_steps_x=config["comet_tail_steps_x"],
        comet_tail_steps_y=config["comet_tail_steps_y"],
        ring_down_steps_x=config["ring_down_steps_x"],
        ring_down_steps_y=config["ring_down_steps_y"],
        right_down_reference_dir=config["right_down_reference_dir"],
        comet_tail_reference_dir=config["comet_tail_reference_dir"],
    )
    masks = detector(image=image)
    return masks


def remove_ultrasound_artifacts(image, config):
    download_file_from_hf(config["weights"])
    remover = ArtifactsRemover(config["weights"])
    mask = detect_ultrasound_artifacts(image, config)
    image = remover(image, mask)
    return image


def denoise_ultrasound_image(image, config):
    download_file_from_hf(config["weights"])
    denoiser = Denoiser(model_path=config["weights"])
    result, metrics = denoiser(image)

    return result, metrics


def detect_ultrasound_edges(image, config):
    download_file_from_hf(config["weights"])
    segmenter = ThyroidSegmentation(model_path=config["weights"])
    mask, contur = segmenter(image)
    return mask, contur


def remove_mirror_ultrasound_artifact(image, config):
    download_file_from_hf(config["weights"])
    eraser = MirrorEraser(model_path=config["weights"])
    clear_image = eraser(image)
    return clear_image


def detect_astrocyte_tips(image, config):
    download_file_from_hf(config["weights"])
    find_enpoints = AstrocyteSegmenter(
        weights_path=config["weights"],
        layers=image,
    )
    endpoints, _, _ = find_enpoints.get_astrocyte_info()
    return endpoints


def segment_gfap_microstructure(image, config):
    download_file_from_hf(config["weights"])
    sep_body_and_branches = AstrocyteSegmenter(
        weights_path=config["weights"],
        layers=image,
    )
    _, body_pixels, branch_pixels = sep_body_and_branches.get_astrocyte_info()
    return body_pixels, branch_pixels


def detect_ultrasound_anomalies(image, config):
    download_file_from_hf(config["weights"])
    detector = AnomalyDetector(weights_path=config["weights"])
    mask = detector(image=image)
    return mask


def classify_astrocytes(image, config):
    download_file_from_hf(config["weights_segmenter"])
    download_file_from_hf(config["weights_classifier"])
    segmenter = AstrocyteSegmenter(
        weights_path=config["weights_segmenter"], layers=image
    )
    astrocytes = segmenter.get_astrocytes()
    classifier = AstrocytesClassificator(
        weights_path=config["weights_classifier"], layers=astrocytes
    )
    return classifier.classify()


def apply_ultrasound_histogram_equalization(image, config: dict):
    download_file_from_hf(config["weights"])
    zeroes = HistogramEqualizer(model_path=config["weights"])
    result = zeroes(image)
    return result


def update_ultrasound_brightness_contrast(image, config):
    download_folder_from_hf(config["weights"])
    enhancer = ImageEnhancer(weights=config["weights"])
    return enhancer(image)


def update_ultrasound_contrast_manual(image, config):
    enhancer = ImageEnhancerContrast(contrast=config["contrast"])
    return enhancer(image)


def update_ultrasound_brightness_manual(image, config):
    enhancer = ImageEnhancerBrightness(brightness=config["brightness"])
    return enhancer(image)


def update_ultrasound_color_reduction_manual(image, config):
    reducer = ImageEnhancerColorQuantizer(n_colors=config["n_colors"])
    return reducer(image)
