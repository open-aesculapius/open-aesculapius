# Detection of Astrocyte Tip Projections on Confocal Microscopic Images [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/detect_astrocyte_tips.md)
This file contains the description of the module for detecting the tips of astrocyte projections on confocal microscopic images.

# Basic Idea
The module for detecting the tips of astrocyte projections on confocal microscopic images is designed to automatically find the tips of astrocyte projections in each of the layers of an overall astrocyte microscopic image. The algorithm uses a combination of classical computer vision methods and deep learning techniques to accurately segment and detect the terminal points of the projections. The hybrid approach allows for compensating the weaknesses of individual methods, achieving high accuracy even in the case of complex entanglements of projections.

# Demonstration
To demonstrate the functionality of the module, the example `simple_detect_astrocyte_tips.py` was used, located at the following path:

```
<project_root>/examples/simple_detect_astrocyte_tips.py
```


Below is the raw image used as input data:

![raw detect_astrocyte_tips](/doc/assets/raw_detect_astrocyte_tips.png)

The result of the module's operation — the image with detected astrocyte tip projections:

![result detect_astrocyte_tips](/doc/assets/result_detect_astrocyte_tips.png)
