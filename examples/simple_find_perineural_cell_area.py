import os
import yaml
import cv2
import numpy as np

from aesculapius import aesculapius
from aesculapius.modules.core.ms_img import MicroscopicImage
from settings import CFG_ROOT, abs_paths


def simple_find_cell_area():
    config_path = os.path.join(CFG_ROOT, "find_perineuronal_cell_area.yaml")
    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            config = abs_paths(config, base_path=os.path.dirname(config_path))
            out_path = config["out_path"]
            os.makedirs(out_path, exist_ok=True)

            MImage = MicroscopicImage(config["folder_path"])
            img = MImage.get_layers()

            if img is not None:
                if img[0].shape != (32, 32, 3):
                    raise Warning("Images of size (32, 32, 3) are required")
                if len(img) > 24:
                    raise Warning("The maximum value of the layers is 24")

                masks, metrics = aesculapius.find_perineuronal_cell_area(
                    img, config)

                results_path = os.path.join(out_path, "results.txt")
                with open(results_path, "w") as results_file:
                    for i, metric in enumerate(metrics):
                        level = MImage.__next__()
                        area = metric[0][0]
                        perimeter = metric[0][1]

                        line = f"level: {level}, area: {area}, \
                                            perimeter: {perimeter}"
                        print(line)
                        results_file.write(line + "\n")

                        black_background = np.zeros(masks[i].shape,
                                                    dtype=np.uint8)
                        black_background[masks[i] > 0] = 255
                        cv2.imwrite(
                            os.path.join(out_path, f"{level}.png"),
                            black_background
                        )
                print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    simple_find_cell_area()
