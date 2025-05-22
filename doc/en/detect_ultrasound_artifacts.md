# Detect ultrasound artifacts [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/detect_ultrasound_artifacts.ru.md)
This file contains description of method to detect ultrasound artifacts.

# Basic idea
The algorithm divides the original image into a grid of small fragments, which are then compared to reference images. Utilizing neural networks, ultrasound frames are matched with reference images of "comet tail" and "ring down" artifacts. If a high degree of similarity between a fragment and a reference artifact is detected, the corresponding areas are included in the final mask and presented as the algorithm's output.

# Demonstration
To demonstrate how the module works, use the example `simple_detect_ultrsound_artifacts.py`, located in the following path:
```
<project_root>/examples/simple_detect_ultrsound_artifacts.py
```
Below is the original image that was used as input:

![raw detect artifact ultrasound](/doc/assets/raw_detect_artifact_ultrasound.png)    

Result of the module:

![result detect artifact ultrasound](/doc/assets/result_detect_artifact_ultrasound.png)