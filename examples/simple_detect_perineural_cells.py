import os
import yaml
import cv2

from aesculapius import aesculapius
from aesculapius.modules.core.ms_img import MicroscopicImage
from settings import CFG_ROOT, abs_paths


def simple_detect_cells():
    config_path = os.path.join(CFG_ROOT, "detect_perineuronal_cells.yaml")
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            config = abs_paths(config, base_path=os.path.dirname(config_path))
            out_path = config["out_path"]
            os.makedirs(out_path, exist_ok=True)

            MImage = MicroscopicImage(config["folder_path"])
            img = MImage.get_layers()

            if img is not None:
                if img[0].shape != (256, 256, 3):
                    raise Warning("Images need to be in 256x256x3 shape")
                if len(img) > 24:
                    raise Warning("The maximum value of the layers is 24")

                result = aesculapius.detect_perineuronal_cells(img, config)

                img1copy = img[0].copy()
                for coords in result:
                    print(
                        f"x_min: {coords[0]}, y_min: {coords[1]}, "
                        f"x_max: {coords[2]}, y_max: {coords[3]}"
                    )
                    cv2.rectangle(
                        img1copy,
                        (coords[0], coords[1]),
                        (coords[2], coords[3]),
                        (0, 255, 0),
                        2,
                    )
                cv2.imwrite(f"{out_path}/simple_detect_cells.png", img1copy)
            print(f"Результаты сохранены в: {out_path}")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    simple_detect_cells()
