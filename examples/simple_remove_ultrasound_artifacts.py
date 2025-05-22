import os
import yaml
import cv2
from pathlib import Path

from aesculapius import aesculapius
from aesculapius.modules.core.us_img import UltrasoundImage
from settings import CFG_ROOT, abs_paths


def simple_remove_artifacts():
    config_path = os.path.join(CFG_ROOT, "remove_ultrasound_artifacts.yaml")
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

                infilled_image = aesculapius.remove_ultrasound_artifacts(
                    img, config
                )

                original_name = f"{Path(folder_path).stem}_{idx}"

                output_name = f"infilled_{original_name}.png"
                output_file = os.path.join(out_path, output_name)

                cv2.imwrite(output_file, infilled_image)
            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    simple_remove_artifacts()
