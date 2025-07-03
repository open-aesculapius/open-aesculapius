import yaml
import time
import inspect
from pathlib import Path
import sys
sys.path.append('..')
from settings import CFG_ROOT, WEIGHTS_ROOT, DATA_ROOT
from aesculapius.modules.core.ms_img import MicroscopicImage
from aesculapius.modules.core.us_img import UltrasoundImage

from aesculapius.aesculapius import (
    detect_perineuronal_cells,
    find_perineuronal_cell_area,
    calculate_perineuronal_cell_depth,
    detect_ultrasound_artifacts,
    remove_ultrasound_artifacts,
    denoise_ultrasound_image,
    detect_ultrasound_edges,
    remove_mirror_ultrasound_artifact,
    detect_astrocyte_tips,
    segment_gfap_microstructure,
    detect_ultrasound_anomalies,
    classify_astrocytes,
    apply_ultrasound_histogram_equalization,
    update_ultrasound_brightness_contrast,
    update_ultrasound_contrast_manual,
    update_ultrasound_brightness_manual,
    update_ultrasound_color_reduction_manual,
)


def benchmark_multiple(name, fn, args=(), kwargs=None, repeats=5):
    kwargs = kwargs or {}
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - start)

    mean = sum(times) / repeats
    print(f"{name:<40}: avg {mean:.4f}s over {repeats} runs")
    return times


def resolve_paths_in_config(config: dict,
                            base: Path = Path(DATA_ROOT)) -> dict:
    def resolve(value, key_hint=""):
        if isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                if any(k in key_hint.lower() for k in ["weight", "model"]):
                    return (Path(WEIGHTS_ROOT) / p).resolve().as_posix()
                elif any(
                    k in key_hint.lower()
                    for k in [
                        "folder_path",
                        "out_path",
                        "weights_classifier",
                        "weights_segmenter",
                        "weights",
                    ]
                ):
                    return (base / p).resolve().as_posix()
        elif isinstance(value, dict):
            return {k: resolve(v, k) for k, v in value.items()}
        return value

    return {k: resolve(v, k) for k, v in config.items()}


def load_config(name: str):
    config_path = Path(CFG_ROOT) / f"{name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = resolve_paths_in_config(config)
    return config


def run_functional_module(name: str,
                          func,
                          config_name: str,
                          image_type: str):

    config = load_config(config_name)

    if image_type == "microscopic":
        MImage = MicroscopicImage(config["folder_path"])
        if name == 'AstrocyteSegmenter' or name == 'AstrocytesClassificator'\
                or name == 'AstrocyteGFAPSegmenter':
            image = MImage.get_layers(0)
        else:
            image = MImage.get_layers()
    elif image_type == "ultrasound":
        us_ = UltrasoundImage(config['folder_path'])
        image = us_.us_image[0]
    else:
        raise ValueError(f"Unknown image type: {image_type}")

    sig = inspect.signature(func)
    if len(sig.parameters) == 2:
        benchmark_multiple(name, func, args=(image, config))
    else:
        benchmark_multiple(name, func, args=(image,))


def main():
    microscopic_modules = [
        ("DeepCalculator",
         calculate_perineuronal_cell_depth,
         "calculate_perineuronal_cell_depth"),

        ("CellDetector",
         detect_perineuronal_cells,
         "detect_perineuronal_cells"),

        ("AreaDetector",
         find_perineuronal_cell_area,
         "find_perineuronal_cell_area"),

        ("AstrocyteSegmenter",
         detect_astrocyte_tips,
         "detect_astrocyte_tips"),

        ("AstrocyteGFAPSegmenter",
         segment_gfap_microstructure,
         "segment_gfap_microstructure"),

        ("AstrocytesClassificator",
         classify_astrocytes,
         "classify_astrocyte"),
    ]

    ultrasound_modules = [
        ("Denoiser",
         denoise_ultrasound_image,
         "denoise_ultrasound_image"),

        ("ArtifactDetector",
         detect_ultrasound_artifacts,
         "detect_ultrasound_artifacts"),

        ("AnomalyDetector",
         detect_ultrasound_anomalies,
         "detect_ultrasound_anomalies"),

        ("ArtifactsRemover",
         remove_ultrasound_artifacts,
         "remove_ultrasound_artifacts"),

        ("MirrorEraser",
         remove_mirror_ultrasound_artifact,
         "remove_mirror_ultrasound_artifact"),

        ("ThyroidSegmentation",
         detect_ultrasound_edges,
         "segment_thyroid_image"),

        ("ImageEnhancer",
         update_ultrasound_brightness_contrast,
         "update_ultrasound_brightness_contrast"),

        ("ImageEnhancerContrast",
         update_ultrasound_contrast_manual,
         "update_ultrasound_contrast_manual"),

        ("ImageEnhancerBrightness",
         update_ultrasound_brightness_manual,
         "update_ultrasound_brightness_manual"),

        ("ImageEnhancerColorQuantizer",
         update_ultrasound_color_reduction_manual,
         "update_ultrasound_color_count_manual"),

        ("HistogramEqualizer",
         apply_ultrasound_histogram_equalization,
         "apply_ultrasound_histogram_equalization"),
    ]

    print("\033[1;36mMicroscopic modules:\033[0m")
    for name, func, cfg in microscopic_modules:
        run_functional_module(name, func, config_name=cfg,
                              image_type="microscopic")

    print("\033[1;36mUltrasound modules:\033[0m")
    for name, func, cfg in ultrasound_modules:
        run_functional_module(name, func, config_name=cfg,
                              image_type="ultrasound")


if __name__ == "__main__":
    main()
