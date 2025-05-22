# Denoise ultrasound image [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/denoise_ultrasound_image.ru.md)
This file contains description of method to denoise ultrasound image.

# Basic idea
The noise reduction algorithm takes an image as input and passes it through an autoencoder-type neural network. This neural network has been pre-trained on a dataset of artificially noised ultrasound images. The result of the algorithm is an image cleansed of noise.

# Demonstration
To demonstrate how the module works, use the example `simple_denoise_ultrasound_image.py`, located in the following path:
```
<project_root>/examples/simple_denoise_ultrasound_image.py
```

Below is the original image that was used as input:

![raw denoise ultrasound](/doc/assets/raw_denoise_ultrasound.png)    

The result of the module is an ultrasound image with a reduced amount of noise:

![result denoise ultrasound](/doc/assets/result_denoise_ultrasound.png)

Output to the console for determining the noise level:

![result denoise level ultrasound](/doc/assets/result_denoise_level_ultrasound.png)