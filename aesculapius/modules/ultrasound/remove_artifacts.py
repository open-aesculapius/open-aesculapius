from aesculapius.modules.core.utils.image_infilling.image_infilling import (
    infill)


class ArtifactsRemover:
    def __init__(self, model):
        self._model = model

    def __call__(self, image, mask):
        return infill(image, mask, model_path=self._model)
