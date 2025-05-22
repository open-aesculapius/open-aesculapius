# Segmentation of GFAP Cytoskeleton Microstructure in Microscopic Images [![en](https://img.shields.io/badge/en-ru-green.svg)](../en/segment_gfap_microstructure.md)
This file describes the method for segmenting the GFAP cytoskeleton microstructure in microscopic images using the U-Net neural network.

# Basic Idea
The GFAP cytoskeleton microstructure segmentation module is designed to isolate structural elements of the glial cell cytoskeleton for further analysis of fiber morphology and spatial organization. The module solves the following tasks:
- Segmentation of GFAP cytoskeleton microstructure;
- Highlighting structural elements with different colors.

The segmentation algorithm uses a U-Net neural network model trained on the task of astrocyte segmentation. The model takes RGB images scaled to a fixed size and predicts a probabilistic map, which is then binarized using a threshold. This approach allows high sensitivity to fine branching structures.

After building the binary mask, morphological post-processing is performed using the Region Growing algorithm and skeletonization via the Lee algorithm. The structure is then divided into the body and processes. Each branch is analyzed to determine if it belongs to the astrocyte body or is a true process.

# Demonstration
To demonstrate the module, the example `segment_gfap_microstructure.py` was used, located at the following path:

```
<project_root>/examples/segment_gfap_microstructure.py
```


Below is the original image used as input:

![raw gfap microstructure](/doc/assets/raw_segment_gfap_microstructure.png)

The result of the module's operation — an image with the segmented body and processes:

![result segment gfap microstructure](/doc/assets/result_segment_gfap_microstructure.png)
