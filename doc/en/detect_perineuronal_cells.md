# Detect perineuronal cells [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/detect_perineuronal_cells.ru.md)
This file contains description of method to detect perineuronal cells.

# Basic idea
Module operation algorithm:

1. Import necessary libraries:

- Torch, torchvision – for neural networks and image transformations.

- OpenCV – for image processing.

- NumPy – for numerical and statistical computations, and for working with multidimensional arrays.

- Albumentations – for data transformations.

2. Read the configuration settings. The function calculate_perineuronal_cell_depth(), intended for demonstrating the module, loads operational parameters from a dictionary and input images. Parameters include the device used (CPU or GPU) and the path to the trained model weights in .pkl format.

3. Initialize the TransformerVGG model. The model weights are loaded from a pre-trained weights file specified in the configuration file. The model is transferred to the specified device (CPU or GPU), and image transformations are initialized. Transformations include conversion to tensors and normalization with a mean of 0.5 and standard deviation of 0.5 by default.

4. Preprocess data. The method _preprocess_data() combines image layers into a single volume, applies transformations, and prepares data for inference.

5. Process images using the TransformerVGG model. After preprocessing, images are input into the model, where probability maps are generated. For each i, j position of the stack, the model's confidence in the existence of a cell is calculated. Predictions are performed in no-gradient mode using torch.no_grad(), optimizing computations and reducing device load.

6. Post-process results. The method _postprocess_data() is applied to the model output to extract the coordinates of detected cells.

# Demonstration
To demonstrate how the module works, use the example `simple_detect_perineural_cells.py`, located in the following path:
```
<project_root>/examples/simple_detect_perineural_cells.py
```
The demo dataset is located in the following path: 
```
<project_root>/data/microscopic/perineural/stack_layers
```

Below is the original image stack that was used as input:

![raw detect perineuronal cells](/doc/assets/raw_detect_perineuronal_cells.png)    

Result of the module:

![result detect perineuronal cells](/doc/assets/result_detect_perineuronal_cells.png)