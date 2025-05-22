import os
import yaml
import cv2
from pathlib import Path

from aesculapius import aesculapius
from aesculapius.modules.core.us_img import UltrasoundImage
from settings import CFG_ROOT, abs_paths


def simple_segmentation():
    config_path = os.path.join(CFG_ROOT, "segment_thyroid_image.yaml")
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            config = abs_paths(config, base_path=os.path.dirname(config_path))
            folder_path = config["folder_path"]
            out_path = config["out_path"]
            os.makedirs(out_path, exist_ok=True)

            us_ = UltrasoundImage(folder_path)
            images = us_.us_image

            image_list = images if isinstance(images, list) else [images]

            for idx, img in enumerate(image_list):
                msk, overlay = aesculapius.detect_ultrasound_edges(img, config)

                original_name = f"{Path(folder_path).stem}_{idx}"
                mask_name = f"mask_{original_name}.png"
                overlay_name = f"overlay_{original_name}.png"

                mask_path = Path(os.path.join(out_path, mask_name))
                overlay_path = Path(os.path.join(out_path, overlay_name))

                cv2.imwrite(str(mask_path), msk[1])
                cv2.imwrite(
                    str(overlay_path), cv2.cvtColor(
                        overlay, cv2.COLOR_RGB2BGR))

            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    simple_segmentation()
