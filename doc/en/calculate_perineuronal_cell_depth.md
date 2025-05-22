# Calculate perineuronal cell depth [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/calculate_perineuronal_cell_depth.ru.md)
This file contains description of method to calculate perineuronal cell depth.

# Basic idea
Module operation algorithm:

Import necessary libraries:

Torch, torchvision – for working with neural networks and image transformations.

OpenCV – for image processing.

SciPy, NumPy – for numerical and statistical computations, and working with multidimensional arrays.

Scikit-learn – for clustering, distance calculations, and other statistical methods.

Pandas – for working with tables and saving results.

Read the configuration settings. The function calculate_perineuronal_cell_depth(), intended for demonstrating the module, loads operational parameters from a dictionary and input images. Parameters include the device used (CPU or GPU) and the path to the trained model weights in .pkl format.

Initialize the model. The U-Net model. The model weights are loaded from a pre-trained weights file specified in the configuration file. The model is transferred to the specified device (CPU or GPU), and image transformations are initialized. Transformations include conversion to tensors and normalization with a mean of 0.5 and standard deviation of 0.5 by default.

The U-Net model is applied for predicting object masks on each image.

The connected component method (scipy.ndimage.label) is applied for identifying and filtering objects based on their size.

The GaussianMixture algorithm is applied for identifying objects based on their shape.

Clustering is performed using the DBSCAN algorithm for segmenting objects.

# Demonstration

To demonstrate how the module works, use the example `simple_calculate_perineural_cell_depth.py`, located in the following path:
````
<project_root>/examples/simple_calculate_perineural_cell_depth.py
````
Below is the original image stack that is used as input:

![raw perineuronal cell depth](/doc/assets/raw_perineuronal_cell_depth.png)    

The result of calculating the cell depth (depth is the number of Z points on the XY projection):

![result perineuronal cell depth](/doc/assets/result_perineuronal_cell_depth.png)