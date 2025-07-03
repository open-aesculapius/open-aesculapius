# Automatic Enhancement of Ultrasound Images by Brightness and Contrast [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/update_ultrasound_brightness_contrast.ru.md)

This file contains a description of the module for automatic brightness and contrast enhancement of ultrasound images using the MIRNet neural network.

# Core Idea

The module is designed for automatic correction of brightness, contrast, and sharpness in ultrasound images, including noise suppression and detail recovery.  
The module solves the following tasks:
- automatic enhancement of brightness and contrast;
- restoration of structural details;
- noise reduction.

The algorithm is based on the MIRNet (Multi-Scale Residual Network) neural network, developed to improve image quality under low-light conditions.  
The network takes an RGB image of fixed size (960×640), processes it using residual connections and attention mechanisms, and then reconstructs the image with enhanced features.

# Demonstration

The module was demonstrated using the example `simple_update_ultrasound_brightness_contrast.py`, located along the following path:

```
<project_root>/examples/simple_update_ultrasound_brightness_contrast.py.py
```

Input image before enhancement:

![raw ultrasound](/doc/assets/raw_update_ultrasound_brightness_contrast.png)

Enhanced result:

![enhanced ultrasound](/doc/assets/result_update_ultrasound_brightness_contrast.png)
