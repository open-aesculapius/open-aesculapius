import os
import yaml
import cv2
from pathlib import Path

from aesculapius import aesculapius
from aesculapius.modules.core.us_img import UltrasoundImage
from settings import CFG_ROOT, abs_paths


def contrast_all_ultrasound_images():
    config_path = os.path.join(
        CFG_ROOT, "update_ultrasound_contrast_manual.yaml")
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
                res_contrast = aesculapius.update_ultrasound_contrast_manual(
                    img, config)
                original_name = f"{Path(folder_path).stem}_{idx}"
                output_name = f"enhanced_contrast_{original_name}.png"
                output_file = Path(os.path.join(out_path, output_name))

                cv2.imwrite(str(output_file), res_contrast)
            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    contrast_all_ultrasound_images()
