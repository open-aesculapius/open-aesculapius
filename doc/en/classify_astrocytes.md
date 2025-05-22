# Classify astrocytes on confocal microscope images [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/classify_astrocytes.ru.md)

This file contains a description of the module for classifying previously detected astrocytes as healthy or diseased on confocal microscopy images.

# Basic idea
The module processes an input mask highlighting the astrocyte region and extracts morphological features. Classification is performed by a VGG19 convolutional neural network pre-trained on ImageNet, modified for binary output and fine-tuned on an augmented dataset of healthy and diseased astrocytes. Class imbalance is addressed via weighted loss, and final predictions are obtained by applying softmax to the two outputs and selecting the index of the maximum probability.

# Demonstration
To run the module, use the example script:
```
<project_root>/examples/simple_classify_astrocyte.py
```

Below is a sample raw confocal microscopy image of an astrocyte:

![raw classify_astrocytes](/doc/assets/raw_classify_astrocytes.png)

Result of the module:

The module outputs a label along with its probability:
```
layer: 0: predicted class - healthy, probability - 0.7299
layer: 1: predicted class - healthy, probability - 0.8539
layer: 2: predicted class - healthy, probability - 0.5224
layer: 3: predicted class - healthy, probability - 0.8440
layer: 4: predicted class - healthy, probability - 0.9012
layer: 5: predicted class - healthy, probability - 0.7491
layer: 6: predicted class - healthy, probability - 0.9363
layer: 7: predicted class - healthy, probability - 0.9133
layer: 8: predicted class - healthy, probability - 0.9193
layer: 9: predicted class - healthy, probability - 0.7508
layer: 10: predicted class - healthy, probability - 0.8715
```
