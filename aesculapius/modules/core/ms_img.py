import glob
import cv2
import os
import numpy as np
import re


class MicroscopicImage:

    def __init__(self, path):
        self.layers = []
        self._layer_list = None
        self.current_index = 0

        images_path = (
            glob.glob(os.path.join(path, "*.png"))
            + glob.glob(os.path.join(path, "*.tif"))
            + glob.glob(os.path.join(path, "*.tiff"))
        )

        def natural_sort_key(s):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("([0-9]+)", os.path.basename(s))
            ]

        images_path = sorted(images_path, key=natural_sort_key)

        loaded_images = []
        for image_path in images_path:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    loaded_images.append(img)
                else:
                    print(f"Warning: Could not load image {image_path}")
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")

        self.layers.append(loaded_images)

    def get_layers(self, layer=-1):
        if isinstance(layer, int) and layer >= 0:
            self._layer_list = [layer]
            return [self.layers[0][layer]]

        self.current_index = 0

        if isinstance(layer, slice):
            start = layer.start if layer.start is not None else 0
            stop = layer.stop if layer.stop is not None else len(
                self.layers[0])
            self._layer_list = np.arange(start, stop)
            return self.layers[0][layer]
        else:
            self._layer_list = np.arange(0, len(self.layers[0]))

        return self.layers[layer]

    def __next__(self):
        if self._layer_list is None:
            raise StopIteration("No layers to iterate.")

        if self.current_index >= len(self._layer_list):
            self.current_index = 0
            raise StopIteration("End of layers.")

        layer_index = self._layer_list[self.current_index]
        self.current_index += 1
        return layer_index

    def __iter__(self):
        return self
