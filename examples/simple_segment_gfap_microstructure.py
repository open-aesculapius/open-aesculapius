import os
import cv2
import yaml

from aesculapius import aesculapius
from aesculapius.modules.core.ms_img import MicroscopicImage
from aesculapius.modules.core.utils.display_tools import overlay_mask
from settings import CFG_ROOT, abs_paths


def simple_separate_astrocyte_body_and_branches():
    config_path = os.path.join(CFG_ROOT, "segment_gfap_microstructure.yaml")
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            config = abs_paths(config, base_path=os.path.dirname(config_path))
            out_path = config['out_path']
            os.makedirs(out_path, exist_ok=True)

            MImage = MicroscopicImage(config['folder_path'])
            layers = MImage.get_layers()
            body_pix, branch_pix = aesculapius.segment_gfap_microstructure(
                layers, config)
            for layer_index, _ in body_pix.items():
                img = layers[layer_index].copy()
                img = overlay_mask(img, body_pix[layer_index],
                                   color=(0, 255, 0))
                img = overlay_mask(img, branch_pix[layer_index],
                                   color=(0, 0, 255))
                output_filename = os.path.join(out_path,
                                               f"layer_{layer_index}.png")
                cv2.imwrite(output_filename, img)
            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    simple_separate_astrocyte_body_and_branches()
