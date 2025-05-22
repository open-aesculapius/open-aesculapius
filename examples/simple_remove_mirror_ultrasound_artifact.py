import os
import yaml
import cv2
from pathlib import Path

from aesculapius import aesculapius
from aesculapius.modules.core.us_img import UltrasoundImage
from settings import CFG_ROOT, abs_paths


def simple_remove_mirror():
    config_path = os.path.join(
        CFG_ROOT, "remove_mirror_ultrasound_artifact.yaml")
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

                clear_image = aesculapius.remove_mirror_ultrasound_artifact(
                    img, config
                )

                original_name = f"{Path(folder_path).stem}_{idx}.png"
                output_file = Path(out_path) / original_name

                cv2.imwrite(str(output_file), clear_image)
            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    simple_remove_mirror()
