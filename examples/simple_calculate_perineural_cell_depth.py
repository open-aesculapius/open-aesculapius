import os
import yaml

from aesculapius import aesculapius
from aesculapius.modules.core.ms_img import MicroscopicImage
from settings import CFG_ROOT, abs_paths


def simple_calculate_depth():
    config_path = os.path.join(
        CFG_ROOT, "calculate_perineuronal_cell_depth.yaml")
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
                    raise Warning("Images need to be in 32x32x3 shape")
                if len(img) > 24:
                    raise Warning("The maximum value of the layers is 24")

                result = aesculapius.calculate_perineuronal_cell_depth(
                    img, config)

                print(f"thickness on the z axis: {result[0]}")
                print(f"Point Cloud {result[1]}")
                result[1].to_csv(out_path + "/deep.txt", sep=" ", mode="a")

            print(f"Результаты сохранены в: {out_path}")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    simple_calculate_depth()
