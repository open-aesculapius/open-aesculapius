# Aesculapius [![ru](https://img.shields.io/badge/ru-en-green.svg)](README.ru.md)

Aesculapius is open source library for ultrasound and microscopic image processing.

# Table of content
- [Installation](#installation)
- [Git hooks configuration](#git-hooks)
    - [MacOS or Linux](#hooks-unix)
    - [Windows](#hooks-windows)
- [Using Library functions](#how-to-use)
- [Examples of new code use cases](#examples)
- [List of areas of applied use](#list)
- [Minimum system requirements](#requirements)
- [License](#license)

# <a name="installation">📝 Installation</a>
1) Download and install Python 3.10
2) Download and install Cuda 12.0+
3) Download repository
```shellscript
git clone git@github.com:open-aesculapius/open-aesculapius.git

cd open-aesculapius
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

By default, the Windows version of Pytorch is installed for CPU. To run modules on GPU, visit the following [website](https://pytorch.org/get-started/previous-versions/) and install the desired version.

7) Building and installing the whl package (optional)

```shellscript
cd build 
```

- For Windows: 
```shellsctopy
build.bat
```
- For Ubuntu: 
```shellsctopy
./build.sh
```

- Installing the compiled package (the package will appear in the dist directory)
```shellscript
pip install aesculapius-1.0.0-py3-none-any.whl
```

8) To test the functionality of all modules in the test directory:
```shellscript 
cd test
python module_testing.py
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

# <a name="examples">📝 Examples of new code use cases</a>
There are various [examples](examples) to demonstrate functions of the Aesculapius library. The list of all examples is provided in the [examples.md](examples/examples.md) document

Examples can be run using the following command (from the root directory):

```shellscript
python -m examples.<module_name>
```
Example:
```shellscript
python -m examples.simple_apply_ultrasound_histogram_equalization 
```

# <a name="list">📝 List of areas of applied use:</a>
The library can be applied in the following application areas:

• Medical imaging – to improve the quality of ultrasound images, including increasing the contrast and noise suppression, as well as to automatically select diagnostically significant areas.

• Microscopic image analysis – the library includes modules for processing and analyzing microscopic images containing astrocytes or perineural network cells.

• Scientific and educational tasks — for reproducing the results of scientific research, as well as for training specialists in digital medicine and bioinformatics.


# <a name="requirements">📝 Minimum system requirements:: </a>

• Intel or AMD processor 2.3 GHz x10 cores; RAM: 8 GB; Nvidia RTX 4060 graphics card; 512 GB SSD; Windows 10 or Ubuntu 22.04 operating system;

• Python interpreter version 3.10;

• installed libraries from the requirements.txt file.

# <a name="license">📝 License</a>
This project is licensed under GPL-3.0 the license
