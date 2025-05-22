import os
import yaml
import cv2

from aesculapius import aesculapius
from aesculapius.modules.core.ms_img import MicroscopicImage
from settings import CFG_ROOT, abs_paths


def simple_find_astrocyte_endpoints():
    config_path = os.path.join(CFG_ROOT, "detect_astrocyte_tips.yaml")
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            config = abs_paths(config, base_path=os.path.dirname(config_path))
            out_path = config["out_path"]
            os.makedirs(out_path, exist_ok=True)

            MImage = MicroscopicImage(config["folder_path"])
            layers = MImage.get_layers()

            endpoints = aesculapius.detect_astrocyte_tips(layers, config)

            for layer_index, layer_endpoint in endpoints.items():
                img = layers[layer_index].copy()
                for endpoint in layer_endpoint:
                    cv2.circle(
                        img, center=endpoint, radius=3, color=(
                            0, 0, 255), thickness=-1)

                output_filename = os.path.join(
                    out_path, f"layer_{layer_index}.png")
                cv2.imwrite(output_filename, img)
            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    simple_find_astrocyte_endpoints()
