# Manual Contrast Enhancement of Ultrasound Images [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/update_ultrasound_contrast_manual.ru.md)

This file describes the module for **manual** contrast enhancement of ultrasound images using the `aesculapius` library.

# Main Idea

The module is designed for manual adjustment of ultrasound image contrast based on parameters specified in a YAML configuration file.  
It performs the following tasks:
- enhances image contrast using user-defined settings;
- saves the resulting image with a new name;
- processes a single image at a time.

Unlike automatic neural network enhancement, this method allows precise control over contrast levels through configuration.

# Demonstration

To demonstrate the module, the example `simple_update_ultrasound_contrast_manual.py` was used, located at:
```
<project_root>/examples/simple_update_ultrasound_contrast_manual.py
```
Original ultrasound image:

![raw ultrasound](/doc/assets/raw_update_ultrasound_contrast_manual.png)

Result after contrast enhancement:

![enhanced ultrasound](/doc/assets/result_update_ultrasound_contrast_manual.png)
