# Aesculapius [![ru](https://img.shields.io/badge/ru-en-green.svg)](README.ru.md)

Aesculapius is open source library for ultrasound and microscopic image processing.

# Table of content
- [Installation](#installation)
    - [Installation from github](#installation_from_github)
    - [Package installation and build](#installation_from_package)
- [Git hooks configuration](#git-hooks)
    - [MacOS or Linux](#hooks-unix)
    - [Windows](#hooks-windows)
- [Using Library functions](#how-to-use)
- [Examples](#examples)
- [License](#license)

# <a name="installation">📝 Installation</a>
## <a name="installation_from_github">Installation from github</a>
1) Download and install Python 3.10+
2) Download and install Cuda 12.0+
3) Download repository
```shellscript
git clone git@github.com:egorchevanton/open_aesculapius_private.git
```
4) Create virtualenv
```shellscript
pip install virtualenv
virtualenv venv3 -p python3
```
5) Activate virtualenv

Windows:
```shellscript
venv3\Scripts\activate.bat
```
Linux:
```shellscript
source venv3/bin/activate
```
6) Install dependencies
```shellscript
pip install -r requirements.txt
```
7) Download weights
```shellsctopy
python download_weights.py
```

## <a name="installation_from_package">Package installation and build</a>
1) Download and install Python 3.10+
2) Download and install Cuda 12.0+
3) Assemble the whl package. In the build folder:
- For Windows: 
```shellsctopy
build.bat
```
- For Ubuntu: 
```shellsctopy
./build.sh
```
4) Install package
```shellscript
pip install open_aesculapius.whl
```
# <a name="git-hooks">📝 Git hooks configuration</a>
## Git hooks check commit message to match '\<type>[scope]: \<description>' template. In this section we configurate git hooks.
## <a name="hooks-unix">MacOS or Linux</a>
1. Make .githooks/commit-msg executable
```shellscript
chmod +X .githooks/commit-msg
```
2. Run .githooks/hooks-init.sh
```shellscript
. .githooks/commit-msg
```
## <a name="hooks-windows">Windows</a>
1. Run PowerShell .githooks/hooks-init.sh
```shellscript
. .githooks/commit-msg
```
# <a name="how-to-use">📝 Using Library functions</a>
The library provides two main classes for working with images: 
- Microscopic image - microscopic image processing 
- Ultrasound imaging - ultrasound image processing 

```shellscript
from aesculapius.modules.core.ms_img import MicroscopicImage
from aesculapius.modules.core.us_img import UltrasoundImage

m_image = MicroscopicImage("путь_к_папке_с_изображениями")
layers = m_image.get_layers()

us_image = UltrasoundImage("путь_к_папке_с_изображениями/или_к_изображению")
us_layer = us_image.us_image
```

Example of calling functions from a library: 

```shellscript
from aesculapius.aesculapius import (
    detect_perineuronal_cells,
    denoise_ultrasound_image,
    segment_gfap_microstructure,
)

cells = detect_perineuronal_cells(layers, config)

denoised = denoise_ultrasound_image(us_layer, config)

segmented = segment_gfap_microstructure(layers, config)
```

The functions use YAML configs to configure the parameters.
Configuration example (calculate_perineuronal_cell_depth.yaml):

```shellscript
folder_path: ../data/microscopic/perineuronal/cells_stack
out_path: ../data/out/calculate_perineuronal_cell_depth
weights: ../weights/cell_model.pkl
threshold: 0.7
```

# <a name="examples">📝 Examples</a>
There are various [examples](examples) to demonstrate functions of the Aesculapius library. The list of all examples is provided in the [examples.md](examples/examples.md) document
# <a name="license">📝 License</a>
This project is licensed under GPL-3.0 the license
