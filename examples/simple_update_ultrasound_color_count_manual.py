import os
import cv2
import yaml
from pathlib import Path

from settings import CFG_ROOT, abs_paths
from aesculapius import aesculapius
from aesculapius.modules.core.us_img import UltrasoundImage


def simple_color_reduction_batch():
    config_path = os.path.join(
        CFG_ROOT,
        "update_ultrasound_color_count_manual.yaml"
    )
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
                reduced = aesculapius.update_ultrasound_color_reduction_manual(
                    img,
                    config
                )
                original_name = f"{Path(folder_path).stem}_{idx}"
                output_name = f"reduced_colors_{original_name}.png"
                output_path = Path(os.path.join(out_path, output_name))
                cv2.imwrite(str(output_path), reduced)
            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    simple_color_reduction_batch()
