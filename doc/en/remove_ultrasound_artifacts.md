# Remove ultrasound artifacts [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/remove_ultrasound_artifacts.ru.md)
This file contains description of method to remove ultrasound artifacts.

# Basic idea
The algorithm takes an image and a mask as input. Using a generative adversarial neural network, it fills the masked area, considering the original content of the image. The output of the algorithm is the final image with the masked areas filled.

# Demonstration
To demonstrate how the module works, use the example `simple_remove_ultrasound_artifacts.py`, located at the following path:
```
<project_root>/examples/simple_remove_ultrasound_artifacts.py
```
Below is the original image that was used as input:

![raw remove artifact ultrasound](/doc/assets/raw_remove_artifact_ultrasound.png)

Below is the artifact mask that was used as input:

![raw remove artifact ultrasound](/doc/assets/raw_mask_remove_artifact_ultrasound.png)

The result of the module:

![result remove artifact ultrasound](/doc/assets/result_remove_artifact_ultrasound.png)