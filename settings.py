import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(APP_ROOT, "data")
REFERENCE_ROOT = os.path.join(DATA_ROOT, "reference")
WEIGHTS_ROOT = os.path.join(APP_ROOT, "weights")
CFG_ROOT = os.path.join(APP_ROOT, "cfg")


def abs_paths(config: dict, base_path: str) -> dict:
    """
    Рекурсивно приводит все относительные пути в config к абсолютным,
    относительно пути к YAML-файлу.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            abs_paths(value, base_path)
        elif isinstance(value, str):
            if value.startswith("../") or value.startswith("./"):
                config[key] = os.path.abspath(os.path.join(base_path, value))
    return config
