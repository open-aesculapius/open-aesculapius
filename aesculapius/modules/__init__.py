from aesculapius.modules import early_setup

from .microscopic.perineuronal.calc_deep_of_perinet import (
    DeepCalculator as DeepCalculator,
)
from .microscopic.perineuronal.cell_detection import (
    CellDetector as CellDetector)
from .microscopic.perineuronal.find_cell_area import (
    AreaDetector as AreaDetector)
from .microscopic.astrocyte.astrocyte_segmentation import (
    AstrocyteSegmenter as AstrocyteSegmenter,
)
from .microscopic.astrocyte.astrocyte_classificator import (
    AstrocytesClassificator as AstrocytesClassificator,
)
from .microscopic.perineuronal.architectures.rt_detr import (
    TransformerVGG as TransformerVGG,
    collate_fn as collate_fn,
)

from .ultrasound.denoise import Denoiser as Denoiser
from .ultrasound.artifacts_detection import (
    ArtifactDetector as ArtifactDetector)
from .ultrasound.remove_artifacts import (
    ArtifactsRemover as ArtifactsRemover)
from .ultrasound.mirror_eraser import MirrorEraser as MirrorEraser
from .ultrasound.hist_equal import HistogramEqualizer as HistogramEqualizer
from .ultrasound.find_thyroid import ThyroidSegmentation as ThyroidSegmentation
from .ultrasound.anomaly_detection import AnomalyDetector as AnomalyDetector
from .ultrasound.image_enhancer import ImageEnhancer as ImageEnhancer
from .ultrasound.image_enhancer_contrast import (
    ImageEnhancerContrast as ImageEnhancerContrast,
)
from .ultrasound.image_enhancer_brightness import (
    ImageEnhancerBrightness as ImageEnhancerBrightness,
)
from .ultrasound.image_enhancer_clr_quant import (
    ImageEnhancerColorQuantizer as ImageEnhancerColorQuantizer)
