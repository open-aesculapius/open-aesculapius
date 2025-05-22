# Remove mirror ultrasound artifact [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/remove_mirror_ultrasound_artifact.ru.md)
This file contains description of method to remove mirror ultrasound artifact.

# Basic idea
The algorithm takes an input image with a duplication artifact and starts by identifying candidates for removal using edge detection, convolutions, and threshold filters. Then, features are extracted from these candidate images using a neural network. If the distance between these features is sufficiently small, it is determined that the two candidates are duplicates. In this case, the brighter candidate is filled with a mask and further refined using a generative adversarial neural network, which considers the image content. The final output of the algorithm is an image without the duplication artifact.

# Demonstration
To demonstrate how the module works, use the example `simple_remove_mirror_ultrasound_artifact.py`, located at the following path:
```
<project_root>/examples/simple_remove_mirror_ultrasound_artifact.py
```
Below is the original image, which was used as input. The green areas contain the original and dual areas:

![raw mirror ultrasound](/doc/assets/raw_mirror_ultrasound.png)

The result of the module:

![result mirror ultrasound](/doc/assets/result_mirror_ultrasound.png)