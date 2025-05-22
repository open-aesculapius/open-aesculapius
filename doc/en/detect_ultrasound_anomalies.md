# Detect ultrasound anomalies [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/detect_ultrasound_anomalies.ru.md)
This file contains description of method to detect ultrasound anomalies.

# Basic idea
The core idea of the algorithm is the automatic segmentation of tumor formations on breast ultrasound images using a convolutional neural network based on the U-Net architecture. The network takes an ultrasound image as input, processes it through an encoder-decoder structure with skip connections to preserve spatial information, and outputs a binary mask highlighting tumor pixels.

# Demonstration
To demonstrate how the module works, use the example `simple_detect_ultrsound_anomalies.py`, located in the following path:
```
<project_root>/examples/simple_detect_ultrsound_anomalies.py
```
Below is the original image that was used as input:

![raw detect anomaly ultrasound](/doc/assets/raw_detect_anomaly_ultrasound.png)    

Result of the module:

![result detect anomaly ultrasound](/doc/assets/result_detect_anomaly_ultrasound.png)