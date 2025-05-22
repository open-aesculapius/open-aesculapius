import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.cluster import DBSCAN
from scipy.ndimage import label
from sklearn.mixture import GaussianMixture
import sklearn.metrics.pairwise
import scipy.stats

from .architectures.find_cell_area import Unet


class DeepCalculator:
    def __init__(
        self,
        weights_path,
        device: str = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"),
        layers: list = None,
        threshold: float = 0.7,
    ):
        self.weights_path = weights_path
        self.device = device
        self.layers = layers
        self.threshold = threshold

        self._init_model()
        self._init_transform()

    def _init_model(self):
        model = Unet()
        model.load_state_dict(torch.load(
            self.weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model

    def _init_transform(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def modify_image(image):
        image = np.where(image > 0.7, 255, 0)
        return image.astype(np.uint8)

    def __call__(self):
        stack = self.layers
        self.model.to(self.device)
        masks_array, layers_array, masks_dict = self._process_images(stack)
        masks_3d, contours_calc = self._compute_contours(masks_array)
        keys_w_cells = self._filter_contours(contours_calc, masks_dict)
        masks_to_dist = self._prepare_masks_for_distance(
            masks_array, keys_w_cells)
        distances = self._compute_distances(masks_to_dist)
        mean, std = self._fit_distance_distribution(distances)
        labels = self._cluster_cells(masks_to_dist, mean, std)
        list_of_cell_ids = self._extract_cell_ids(labels, keys_w_cells)
        first_layer_num, last_layer_num = self._determine_layer_bounds(
            list_of_cell_ids)
        inter_masks, layers_show = self._process_intermediate_masks(
            masks_3d, layers_array, first_layer_num, last_layer_num
        )
        X, Y, Z, convexhull = self._collect_convex_hull(
            inter_masks, masks_3d, first_layer_num, last_layer_num
        )
        thickness, df = self._compute_thickness_and_dataframe(masks_3d)
        return thickness, df

    def _process_images(self, stack):
        masks_array = []
        layers_array = []
        masks_dict = {}
        z = 0

        for image in stack:
            transformed_image = self.transform(
                image).to(self.device).unsqueeze(0)
            with torch.no_grad():
                mask = self.model(transformed_image)
            mask_np = mask.cpu().data.numpy()

            mask_np = mask_np[0]
            mask_np = mask_np.transpose(1, 2, 0)
            mask_np = np.squeeze(mask_np, axis=2)

            mask_np = self.modify_image(mask_np)
            mask_np = self._filter_small_labels(mask_np)
            normalized_image = self._normalize_image(transformed_image)
            layers_array.append(normalized_image[:, :, 0].astype("uint8"))
            masks_dict[z] = mask_np.astype("uint8")
            masks_array.append(mask_np.astype("uint8"))
            z += 1

        return masks_array, layers_array, masks_dict

    def _filter_small_labels(self, mask):
        labeled_mask, num_features = label(mask)
        original_len = np.sum(mask == 255)

        for label_id in range(1, num_features + 1):  # Метки начинаются с 1
            label_size = np.sum(labeled_mask == label_id)
            if label_size < 0.1 * original_len:
                mask[labeled_mask == label_id] = 0

        return mask

    def _normalize_image(self, image):
        image_np = image[0].cpu().data.numpy().transpose(1, 2, 0)
        image_np = image_np - np.min(image_np)
        image_np = image_np / np.max(image_np) * 255
        return image_np

    def _compute_contours(self, masks_array):
        contours_calc = [
            mask[mask == 255].shape[0]
            for mask in masks_array
            if mask.shape != (0, 0)
        ]
        if not contours_calc:
            raise ValueError("No cell detected.")
        contours_calc = np.array(contours_calc).reshape(-1, 1)
        masks_3d = masks_array
        return masks_3d, contours_calc

    def _filter_contours(self, contours_calc, masks_dict):
        gaussian_mix = GaussianMixture(n_components=2, random_state=0).fit(
            contours_calc
        )
        lowest_mean = gaussian_mix.means_.reshape(-1)[-1]
        lowest_std = np.sqrt(gaussian_mix.covariances_.reshape(-1)[-1])
        keys_w_cells = [
            key
            for key in masks_dict.keys()
            if contours_calc[key] > lowest_mean + lowest_std
        ]
        if not keys_w_cells:
            raise ValueError("No cell detected after filtering contours.")
        return keys_w_cells

    def _prepare_masks_for_distance(self, masks_array, keys_w_cells):
        masks_to_dist = [masks_array[key].flatten() for key in keys_w_cells]
        return np.array(masks_to_dist)

    def _compute_distances(self, masks_to_dist):
        distances_triu = np.triu(
            sklearn.metrics.pairwise.euclidean_distances(masks_to_dist)
        )
        distances = distances_triu[distances_triu > 0]
        if distances.size == 0:
            raise ValueError("No cell detected after computing distances.")
        return distances

    def _fit_distance_distribution(self, distances):
        mean, std = scipy.stats.norm.fit(distances)
        if np.isnan(mean) or np.isnan(std):
            raise ValueError("Invalid distance distribution parameters.")
        return mean, std

    def _cluster_cells(self, masks_to_dist, mean, std):
        db = DBSCAN(eps=mean - std, min_samples=2).fit(masks_to_dist)
        return db.labels_

    def _extract_cell_ids(self, labels, keys_w_cells):
        list_of_cell_ids = [
            keys_w_cells[idx]
            for idx in range(len(labels))
            if labels[idx] != -1
        ]
        if not list_of_cell_ids:
            raise ValueError("No cell detected after clustering.")
        return list_of_cell_ids

    def _determine_layer_bounds(self, list_of_cell_ids):
        first_layer_num = min(list_of_cell_ids)
        last_layer_num = max(list_of_cell_ids)
        return first_layer_num, last_layer_num

    def _process_intermediate_masks(
        self, masks_3d, layers_array, first_layer_num, last_layer_num
    ):
        inter_masks = {}
        layers_show = []

        inter_mask = masks_3d[first_layer_num].copy()
        for i in reversed(range(first_layer_num)):
            layer_show, inter = self._process_single_layer(
                masks_3d, layers_array, i, inter_mask, direction="down"
            )
            inter_mask = inter.copy()
            inter_masks[i] = inter_mask
            layers_show.append(layer_show)

        inter_mask = masks_3d[last_layer_num].copy()
        for i in range(last_layer_num, len(masks_3d)):
            layer_show, inter = self._process_single_layer(
                masks_3d, layers_array, i, inter_mask, direction="up"
            )
            inter_mask = inter.copy()
            inter_masks[i] = inter_mask

        return inter_masks, layers_show

    def _process_single_layer(
        self, masks_3d, layers_array, layer_idx, inter_mask, direction="down"
    ):
        layer_show = []
        inter = np.zeros_like(masks_3d[layer_idx])

        if direction == "down":
            adjacent_layer_idx = layer_idx + 1
        else:
            adjacent_layer_idx = layer_idx - 1

        if direction == "down" and adjacent_layer_idx >= len(layers_array):
            adjacent_layer = np.zeros_like(layers_array[layer_idx])
        elif direction == "up" and adjacent_layer_idx < 0:
            adjacent_layer = np.zeros_like(layers_array[layer_idx])
        else:
            adjacent_layer = layers_array[adjacent_layer_idx]

        for j in range(masks_3d[layer_idx].shape[0]):
            for k in range(masks_3d[layer_idx].shape[1]):
                if masks_3d[layer_idx][j][k] == 255:
                    layer_show.append(layers_array[layer_idx][j][k])
                    if (
                        inter_mask[j][k] == 255
                        and masks_3d[layer_idx][j][k] == 255
                    ) and layers_array[layer_idx][j][k] > np.median(
                        (layers_array[layer_idx] + adjacent_layer) / 2
                    ) + 0.15 * np.median(
                        layers_array[layer_idx]
                    ):
                        inter[j][k] = 255

        return layer_show, inter

    def _collect_convex_hull(
        self, inter_masks, masks_3d, first_layer_num, last_layer_num
    ):
        X, Y, Z, convexhull = [], [], [], []

        for i, mask in inter_masks.items():
            for j in range(mask.shape[0]):
                for k in range(mask.shape[1]):
                    if mask[j][k] == 255:
                        X.append(k)
                        Y.append(j)
                        Z.append(i)
                        convexhull.append([k, j, i])

        for i in range(first_layer_num, last_layer_num + 1):
            for j in range(masks_3d[i].shape[0]):
                for k in range(masks_3d[i].shape[1]):
                    if masks_3d[i][j][k] == 255:
                        X.append(k)
                        Y.append(j)
                        Z.append(i)
                        convexhull.append([k, j, i])

        return X, Y, Z, convexhull

    def _compute_thickness_and_dataframe(self, masks_3d):
        null_mask_print = np.zeros_like(masks_3d[0])
        for mask in masks_3d:
            null_mask_print = (
                np.logical_or(null_mask_print, mask ==
                              255).astype(np.uint8) * 255
            )

        x, y, z = [], [], []
        for i in range(null_mask_print.shape[0]):
            for j in range(null_mask_print.shape[1]):
                if null_mask_print[i, j] == 255:
                    z_in = [k for k in range(
                        len(masks_3d)) if masks_3d[k][i][j] == 255]
                    x.append(j)
                    y.append(i)
                    z.append(z_in)

        max_viewability_lvl = []
        for levels in z:
            if len(levels) > len(max_viewability_lvl):
                max_viewability_lvl = levels

        if not max_viewability_lvl:
            raise ValueError("No cell detected in max viewability levels.")

        z_min = np.min(max_viewability_lvl)
        z_max = np.max(max_viewability_lvl)
        thickness = z_max - z_min

        df = pd.DataFrame({"X": x, "Y": y, "Z": z})
        return thickness, df
