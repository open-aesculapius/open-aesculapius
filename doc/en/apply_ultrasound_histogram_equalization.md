# equalization histogram ultrasound image [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/apply_ultrasound_histogram_equalization.ru.md)
This file contains description of method to equalization histogram ultrasound image.

# Basic idea
The histogram equalization algorithm takes an ultrasound image as input and processes it through a deep neural network based on a UNet-like architecture with an enhancement block. Unlike traditional histogram filters, the model is trained on a dataset of ultrasound images with varying levels of illumination and contrast.
The network learns to predict adaptive correction parameters to locally improve contrast and brightness without distorting anatomical structures. The result is an ultrasound image with enhanced contrast and improved visibility of fine details.

# Demonstration
To demonstrate how the module works, use the example `simple_apply_ultrasound_histogram_equalization.py`, located in the following path:
```
<project_root>/examples/simple_apply_ultrasound_histogram_equalization.py
```

Below is the original image that was used as input:

![raw histogram_equalization ultrasound](/doc/assets/raw_histogram_equalization_ultrasound.png)  

The result of the module is an ultrasound image using an equalization histogram:

![result histogram_equalization  ultrasound](/doc/assets/result_histogram_equalization_ultrasound.png)
