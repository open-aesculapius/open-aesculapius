# Thyroid Gland Segmentation on Ultrasound Images [![ru](https://img.shields.io/badge/ru-en-green.svg)](../ru/thyroid_segmentation.ru.md)

This document describes a method for **automatic segmentation of the thyroid gland** on ultrasound images.

---

## Basic idea
The algorithm receives a grayscale ultrasound image, resizes it to **256 × 256**, normalises pixel values, and feeds it into a convolutional neural network **T‑Net** (a modified U‑Net with channel attention).  
The network outputs a probability map that is converted into the final mask through:

1. **Binarisation** (threshold 0.5)  
2. **Morphological cleaning** (median‑blur → open → close)  
3. **Area filtering** (removal of connected components smaller than 500 px²)

The module returns a binary mask of the thyroid gland and an image with the contour or a semi‑transparent overlay.

---

## Demonstration
Run the example:

```
<project_root>/examples/simple_segment_thyroid_image.py
```

**Input image**

![raw thyroid ultrasound](/doc/assets/raw_thyroid_segmentation_ultrasound.png)

**Binary mask**

![mask thyroid ultrasound](/doc/assets/result_thyroid_segmentation_mask.png)

**Overlay mask**

![overlay thyroid ultrasound](/doc/assets/result_thyroid_segmentation_overlay.png)
