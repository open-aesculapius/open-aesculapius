# Find perineuronal cell area [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/find_perineuronal_cell_area.ru.md)
This file contains description of method to find perineuronal cell area.

# Basic idea
Module operation algorithm:

Import necessary libraries:

Torch – a module for implementing deep learning algorithms.

Torchvisions.transform – a module for image transformation, normalization, and conversion to tensors.

OpenCV – for image processing.

NumPy, SciPy – for performing mathematical and morphological operations on images.

Read the configuration settings. The function find_perineuronal_cells(), intended for demonstrating the module, loads operational parameters from a dictionary and input images. Parameters include the device used (CPU or GPU) and the path to the trained model weights in .pkl format.

Initialize the model. The core of the algorithm is a neural network based on the U-Net architecture. The model weights are loaded from a pre-trained weights file specified in the configuration file. The model is transferred to the specified device (CPU or GPU), and image transformations are initialized. Transformations include conversion to tensors and normalization with a mean of 0.5 and standard deviation of 0.5 by default.

Preprocess data. The method _preprocess_data() performs the following steps:

Reading images.

Converting images to tensors using the torchvision.transform library. Specified transformations (normalization and conversion to tensors) are applied.

Process images using the model. After preprocessing, images are input into the model, where segmentation occurs. For each image, the network predicts a mask, highlighting the area of interest. Predictions are performed in no-gradient mode using torch.no_grad(), optimizing computations and reducing device load.

Post-process results. The method _postprocess_data() applies a threshold to the segmented images, converting the mask to a binary image where pixels above the threshold are white, and those below the threshold are black.

Calculate metrics. For each binary image (mask), the following metrics are calculated:

Area of the object, calculated using the cv2.contourArea() method.

Perimeter, calculated as the length of the object's contour using cv2.arcLength().

# Demonstration
To demonstrate how the module works, use the example `simple_find_perineural_cell_area`, located at the following path:
```
<project_root>/examples/simple_find_perineural_cell_area.py
```
The demo dataset is located at the following path:
```
<project_root>/data/microscopic/perineural/cells_stack
```
Below is one of the original input microscopic images.

![raw find perineuronal cell](/doc/assets/raw_find_perineuronal_cell.png)

The result of the module is the cell outline in the form of coordinates - white pixels:

![result find perineuronal cell](/doc/assets/result_find_perineuronal_cell.png)

The results of calculating the area and perimeter of cells for each layer:

![result find perineuronal cell](/doc/assets/result_find_perineuronal_cell_area.png)