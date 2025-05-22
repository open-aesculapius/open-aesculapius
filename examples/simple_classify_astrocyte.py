import os
import yaml

from aesculapius import aesculapius
from aesculapius.modules.core.ms_img import MicroscopicImage
from settings import CFG_ROOT, abs_paths


def simple_classify_astrocyte():
    config_path = os.path.join(CFG_ROOT, "classify_astrocyte.yaml")

    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
            config = abs_paths(config, base_path=os.path.dirname(config_path))
            out_path = config["out_path"]
            os.makedirs(out_path, exist_ok=True)

            MImage = MicroscopicImage(config["folder_path"])
            layers = MImage.get_layers()

            results = aesculapius.classify_astrocytes(layers, config)

            results_file = os.path.join(out_path, "classification_results.txt")

            with open(results_file, "w") as f:
                for layer in results.keys():
                    predicted_class = results[layer][0]
                    probability = results[layer][1]
                    line = (
                        f"layer: {layer}: predicted class - {predicted_class},"
                        f" probability - {probability:.4f}\n")
                    print(line.strip())
                    f.write(line)

            print(f"\nРезультаты сохранены в: {out_path}")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    simple_classify_astrocyte()
