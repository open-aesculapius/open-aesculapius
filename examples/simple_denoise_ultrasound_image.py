import os
import yaml
import cv2
from pathlib import Path

from aesculapius import aesculapius
from aesculapius.modules.core.us_img import UltrasoundImage
from settings import CFG_ROOT, abs_paths


def simple_denoise():
    config_path = os.path.join(CFG_ROOT, "denoise_ultrasound_image.yaml")
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
                res_image, noise_level = aesculapius.denoise_ultrasound_image(
                    img, config)
                original_name = f"{Path(folder_path).stem}_{idx}"
                output_name = f"denoised_{original_name}.png"
                output_file = Path(os.path.join(out_path, output_name))

                cv2.imwrite(str(output_file), res_image)
                print(f"Image saved as: {output_file}")
                print(f"General type of noise: {noise_level.value}")

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    simple_denoise()
