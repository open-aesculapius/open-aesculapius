# Manual Brightness Enhancement of Ultrasound Images [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/update_ultrasound_brightness_manual.ru.md)

This file describes the module for **manual** brightness enhancement of ultrasound images using the `aesculapius` library.

# Main Idea

The module is designed for manual adjustment of ultrasound image brightness based on parameters specified in a configuration file.  
It performs the following tasks:
- increases image brightness using user-defined settings;
- saves enhanced images with a new name;
- batch processes all images in the specified folder.

This approach allows direct control over the brightness parameters via a YAML configuration file.

# Demonstration

To demonstrate the module, the example `simple_update_ultrasound_brightness_manual.py` was used, located at:
```
<project_root>/examples/simple_update_ultrasound_brightness_manual.py
```
Original ultrasound image:

![raw ultrasound](/doc/assets/raw_update_ultrasound_brightness_contrast.png)

Result after brightness enhancement:

![enhanced ultrasound](/doc/assets/result_update_ultrasound_brightness_manual.png)