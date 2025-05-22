from collections import deque

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.ndimage import gaussian_filter1d, label
from skimage import morphology
from skimage.segmentation import flood_fill
from sklearn.cluster import KMeans

from .architectures.unet_astrocyte import UNET


class AstrocyteSegmenter:
    def __init__(
            self,
            weights_path,
            device: str = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'),
            layers: list = None,
            model_threshold: float = 0.2,
            tolerance: int = 50,
            plaque_filtration: bool = True
    ):
        self.weights_path = weights_path
        self.device = device
        self.layers = layers
        self.model_threshold = model_threshold
        self.tolerance = tolerance
        self.plaque_filtration = plaque_filtration

        self._init_model()
        self._init_transforms()

    def _init_model(self):
        """Инициализирует модель UNET и загружает веса."""
        self.model = UNET(in_channels=3, out_channels=1).to(self.device)
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)[
                'state_dict'])
        self.model.eval()

    def _init_transforms(self):
        """Инициализирует преобразования для изображений."""
        self.transforms = A.Compose([
            A.Resize(height=720, width=720),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def _find_endpoints(self, skeleton):
        """Находит конечные точки на скелете."""
        endpoints = []
        height, width = skeleton.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skeleton[y, x] == 255:
                    neighbors = sum(
                        skeleton[y + dy, x + dx] == 255 for dy in [-1, 0, 1]
                        for dx in [-1, 0, 1]
                        if not (dy == 0 and dx == 0))
                    if neighbors == 1:
                        endpoints.append((x, y))
        return endpoints

    def _ray_cast_from_center(self, mask, center_coord, distance_map,
                              n_rays=360, max_radius=1500):
        """
        Проходит от центра по лучам и собирает перепады карты расстояний.
        """
        height, width = mask.shape
        cy, cx = center_coord

        angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        distances_along_rays = []
        profiles = []

        for angle in angles:
            ray_profile = []
            for r in range(1, max_radius):
                x = int(round(cx + r * np.cos(angle)))
                y = int(round(cy + r * np.sin(angle)))

                if x < 0 or x >= width or y < 0 or y >= height:
                    break

                if mask[y, x] == 0:
                    distances_along_rays.append(r)
                    profiles.append(ray_profile)
                    break

                ray_profile.append(distance_map[y, x])
            else:
                distances_along_rays.append(max_radius)
                profiles.append(ray_profile)

        return distances_along_rays, profiles

    def _find_branch_regions(self, distances, iqr_multiplier=0.15,
                             min_region_width=1):
        """
        Находит интервалы отростков по пороговому разрезанию профиля distances.
        """
        distances = np.array(distances)
        smoothed = gaussian_filter1d(distances, sigma=2)

        q1 = np.percentile(smoothed, 35)
        q3 = np.percentile(smoothed, 75)
        iqr = q3 - q1
        median = np.median(smoothed)

        threshold = median + iqr_multiplier * iqr

        is_branch = smoothed > threshold
        labeled, num_features = label(is_branch)

        regions = []
        for region_idx in range(1, num_features + 1):
            region_mask = labeled == region_idx
            indices = np.where(region_mask)[0]
            if len(indices) >= min_region_width:
                regions.append((indices[0], indices[-1]))

        return regions

    def _split_mask_by_branch_regions(self, refined_mask, center_coord,
                                      branch_regions, distances, num_rays=360):
        """
        Делит маску на тело и отростки по угловым регионам
        и длине отростков (distances).
        """
        height, width = refined_mask.shape
        cy, cx = center_coord
        distances = np.array(distances)

        Y, X = np.indices(refined_mask.shape)
        dy = Y - cy
        dx = X - cx
        angles = np.arctan2(dy, dx)
        angles_deg = np.degrees(angles) % 360
        radius = np.sqrt(dx ** 2 + dy ** 2)

        ray_indices = np.floor(angles_deg / 360 * num_rays).astype(int)
        ray_indices = np.clip(ray_indices, 0, num_rays - 1)

        branch_mask = np.zeros_like(refined_mask, dtype=bool)
        for start_idx, end_idx in branch_regions:
            if start_idx <= end_idx:
                valid_indices = np.arange(start_idx, end_idx + 1)
            else:
                valid_indices = np.concatenate([np.arange(start_idx, num_rays),
                                                np.arange(0, end_idx + 1)])
            in_region = np.isin(ray_indices, valid_indices)
            distance_threshold = distances[ray_indices]
            in_radius = radius >= 0.9 * distance_threshold
            branch_mask |= (in_region & in_radius)

        branch_mask = self._directional_propagation(refined_mask, branch_mask,
                                                    center_coord)
        branch_mask = branch_mask & (refined_mask > 0)
        body_mask = (refined_mask > 0) & (~branch_mask)

        body_mask_uint8 = (body_mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(body_mask_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        branch_mask_dop = np.zeros_like(body_mask_uint8)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            for contour in contours:
                if not np.array_equal(contour, max_contour):
                    cv2.drawContours(
                        branch_mask_dop,
                        [contour],
                        -1,
                        255,
                        thickness=cv2.FILLED)

        branch_mask_dop_bool = branch_mask_dop > 0
        body_mask = body_mask & (~branch_mask_dop_bool)
        branch_mask = branch_mask | branch_mask_dop_bool

        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        colored_mask[body_mask] = (0, 255, 0)  # зелёный — тело
        colored_mask[branch_mask] = (255, 0, 0)  # красный — отростки
        return colored_mask, body_mask, branch_mask

    def _directional_propagation(self, refined_mask, branch_seed_mask,
                                 center_coord, unlock_steps=150):
        """
        Распространяет отростки внутри refined_mask
        """
        height, width = refined_mask.shape
        cy, cx = center_coord

        Y, X = np.indices((height, width))
        distance_from_center = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

        visited = np.zeros_like(refined_mask, dtype=bool)
        output_mask = np.zeros_like(refined_mask, dtype=bool)
        queue = deque()

        seed_points = np.argwhere(branch_seed_mask)
        for y, x in seed_points:
            queue.append((y, x, 0))
            visited[y, x] = True
            output_mask[y, x] = True

        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1), (0, 1),
                     (1, -1), (1, 0), (1, 1)]

        while queue:
            y, x, steps = queue.popleft()
            dist_curr = distance_from_center[y, x]
            steps = 0
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if visited[ny, nx]:
                        continue
                    if not refined_mask[ny, nx]:
                        continue

                    dist_next = distance_from_center[ny, nx]

                    if steps < unlock_steps:
                        if dist_next < dist_curr:
                            continue

                    visited[ny, nx] = True
                    output_mask[ny, nx] = True
                    queue.append((ny, nx, steps + 1))

        return output_mask

    def _get_center_point(self, refined_mask):
        """Находит центральную точку маски."""
        refined_mask = refined_mask.astype(np.uint8)
        refined_mask = refined_mask.astype(bool)
        refined_mask = binary_fill_holes(refined_mask)
        refined_mask = refined_mask.astype(np.uint8)

        distance_map = distance_transform_edt(refined_mask)
        inverted_distance = np.max(distance_map) - distance_map
        local_min = (minimum_filter(inverted_distance,
                                    size=5) == inverted_distance)
        local_min = local_min & (refined_mask > 0)
        min_coords = np.argwhere(local_min)

        if len(min_coords) > 0:
            distances_at_minima = distance_map[
                min_coords[:, 0], min_coords[:, 1]
            ]
            center_idx = np.argmax(distances_at_minima)
            center_coord = min_coords[center_idx]
        else:
            center_coord = None

        return center_coord, refined_mask, distance_map

    def _find_body_and_appendages(self, refined_mask):
        """Находит маски тела и отростков."""
        center_coord, refined_mask, distance_map = self._get_center_point(
            refined_mask)
        distances, profiles = self._ray_cast_from_center(refined_mask,
                                                         center_coord,
                                                         distance_map)
        region = self._find_branch_regions(distances)
        colored_mask, body_mask, branch_mask = (
            self._split_mask_by_branch_regions(
                refined_mask, center_coord, branch_regions=region,
                distances=distances, num_rays=len(distances)
            ))
        return colored_mask, body_mask, branch_mask

    def _overlay_mask(self, image, mask, color=(0, 0, 255), alpha=0.5):
        """
        Наложение маски на изображение с заданным цветом и прозрачностью.
        """
        mask = mask.astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 255] = color
        result = image.copy()
        result[mask == 255] = cv2.addWeighted(image[mask == 255], 1 - alpha,
                                              colored_mask[mask == 255], alpha,
                                              0)
        return result

    def _find_brightest_point(self, image, mask):
        """Находит самую яркую точку внутри маски."""
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        _, max_val, _, max_loc = cv2.minMaxLoc(gray_image)
        return max_loc

    def _region_growing(self, image, mask, tolerance=50):
        """
        Выполняет Region Growing, начиная с самой яркой точки внутри маски.
        """
        brightest_point = self._find_brightest_point(image, mask)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filled_image = flood_fill(
            gray_image,
            (brightest_point[1],
             brightest_point[0]),
            new_value=255,
            tolerance=tolerance)
        region_mask = (filled_image == 255).astype(np.uint8) * 255
        return region_mask

    def _preprocess_image(self, image_np):
        """
        Предобработка изображения: контрастирование, эквализация гистограммы,
        сглаживание, гамма-коррекция.
        """
        image_np = cv2.normalize(image_np, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        channels = list(cv2.split(image_np))
        channels[0] = cv2.equalizeHist(channels[0])
        image_np = cv2.merge(channels)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_YCrCb2RGB)
        image_np = cv2.GaussianBlur(image_np, (5, 5), sigmaX=1.5)
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in
                          np.arange(0, 256)]).astype("uint8")
        image_np = cv2.LUT(image_np, table)
        return image_np

    def _postprocess_mask(self, mask):
        """
        Постобработка маски: морфологические операции и фильтрация контуров.
        """
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(
                filtered_mask,
                [max_contour],
                -1,
                255,
                thickness=cv2.FILLED
            )
        return filtered_mask

    def _filter_largest_component(self, mask):
        """
        Оставляет только самый большой компонент связности на бинарной маске,
        сохраняя её оригинальную форму (без заливки контура).
        """
        mask = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        output_mask = (labels == largest_label).astype(np.uint8) * 255
        return output_mask

    def _refine_mask(self, image_np, filtered_mask, tolerance=50):
        """Уточнение маски через Region Growing."""
        refined = self._region_growing(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), filtered_mask,
            tolerance=tolerance)
        contours, hierarchy = cv2.findContours(refined, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        max_contour_mask = np.zeros_like(refined)
        if contours:
            max_contour_index = -1
            max_area = 0
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_contour_index = i
            max_contour = contours[max_contour_index]
            cv2.drawContours(
                max_contour_mask,
                [max_contour],
                -1,
                255,
                thickness=cv2.FILLED
            )
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] == max_contour_index:
                    cv2.drawContours(
                        max_contour_mask,
                        [cnt],
                        -1,
                        0,
                        thickness=cv2.FILLED
                    )
        refined_mask = max_contour_mask
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel,
                                        iterations=3)
        return refined_mask

    def model_inference(self, image_tensor, original_size):
        """
        Выполняет инференс модели и возвращает обработанную маску.
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)
            output = (output > self.model_threshold).float()

        output_image = output.squeeze().cpu().numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.resize(output_image, original_size,
                                  interpolation=cv2.INTER_NEAREST)
        return output_image

    def _masked_kmeans(self, image, mask, n_clusters=2):
        """
        Выполняет K-Means кластеризацию на области изображения,
        определённой маской.
        """
        binary_mask = (mask > 0)
        if image.ndim == 3:
            pixels = image[binary_mask]
        else:
            pixels = image[binary_mask][:, np.newaxis]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels)
        label_map = np.zeros(mask.shape, dtype=np.uint8)
        label_map[binary_mask] = labels + 1
        return label_map

    def _filter_largest_cluster_and_contour(self, clustered_mask):
        """
        Оставляет в маске только самый крупный кластер и внутри него
        — только самый большой контур.
        """
        labels, counts = np.unique(clustered_mask[clustered_mask > 0],
                                   return_counts=True)
        if len(counts) == 0:
            return np.zeros_like(clustered_mask, dtype=np.uint8)
        largest_label = labels[np.argmax(counts)]
        binary_mask = (clustered_mask == largest_label).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(clustered_mask, dtype=np.uint8)
        largest_contour = max(contours, key=cv2.contourArea)
        output_mask = np.zeros_like(clustered_mask, dtype=np.uint8)
        cv2.drawContours(
            output_mask,
            contours=[largest_contour],
            contourIdx=-1,
            color=1,
            thickness=cv2.FILLED
        )

        return output_mask

    def _filter_plaques_by_shape(self, binary_mask, min_area, min_circularity):
        """
        Фильтрует объекты по площади и округлости,
        возвращая маску предполагаемых бляшек.
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        result_mask = np.zeros_like(binary_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if area >= min_area and circularity >= min_circularity:
                cv2.drawContours(
                    result_mask,
                    [cnt],
                    -1,
                    255,
                    thickness=cv2.FILLED)
        return result_mask

    def _split_branches_by_plaques(self, branch_mask, plaques_mask):
        """
        Вычитает бляшки из отростков.
        Если отросток разваливается — сохраняет только самый большой кусок,
        а оставшиеся отправляет в бляшки.
        """
        branch_mask = branch_mask.astype(np.uint8)
        plaques_mask = plaques_mask.astype(np.uint8)
        final_branch = np.zeros_like(branch_mask, dtype=np.uint8)
        updated_plaques = plaques_mask.copy()
        contours, _ = cv2.findContours(branch_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            component_mask = np.zeros_like(branch_mask, dtype=np.uint8)
            cv2.drawContours(
                component_mask,
                [cnt],
                -1,
                1,
                thickness=cv2.FILLED
            )
            cut = cv2.bitwise_and(component_mask,
                                  cv2.bitwise_not(plaques_mask))
            sub_contours, _ = cv2.findContours(cut, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

            if not sub_contours:
                continue

            sub_contours = sorted(sub_contours, key=cv2.contourArea,
                                  reverse=True)
            cv2.drawContours(
                final_branch,
                [sub_contours[0]],
                -1,
                1,
                thickness=cv2.FILLED
            )
            for cnt_other in sub_contours[1:]:
                cv2.drawContours(
                    updated_plaques,
                    [cnt_other],
                    -1,
                    1,
                    thickness=cv2.FILLED
                )
        return final_branch, updated_plaques

    def _process_image(self, stack):
        """Основной метод обработки изображения."""
        endpoints_dict = {}
        body_pixels_dict = {}
        branch_pixels_dict = {}
        z = 0
        for image in stack:
            original_size = image.shape[:2]

            image = self._preprocess_image(image)

            transformed = self.transforms(image=image)
            image_tensor = transformed["image"].unsqueeze(0).to(self.device)

            output_image = self.model_inference(image_tensor, original_size)

            filtered_mask = self._postprocess_mask(output_image)

            refined_mask = self._refine_mask(image, filtered_mask,
                                             self.tolerance)

            _, body_mask, branch_mask = self._find_body_and_appendages(
                refined_mask)

            body_mask = self._region_growing(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                body_mask.astype(np.uint8) * 255, tolerance=50)
            body_mask = np.logical_and(body_mask, np.logical_not(branch_mask))
            body_mask = self._filter_largest_component(
                body_mask.astype(np.uint8) * 255).astype(bool)

            if self.plaque_filtration:
                clustering_astrocyte = self._masked_kmeans(image, refined_mask,
                                                           n_clusters=2)

                filt_clustering_astrocyte = (
                        self._filter_largest_cluster_and_contour(
                            clustering_astrocyte) * 255
                )

                suspected_plaques = refined_mask - filt_clustering_astrocyte

                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (9, 9)
                )
                suspected_plaques = cv2.morphologyEx(suspected_plaques,
                                                     cv2.MORPH_OPEN, kernel,
                                                     iterations=1)

                plaques = self._filter_plaques_by_shape(suspected_plaques,
                                                        min_area=5000.0,
                                                        min_circularity=0.2)

                branch_mask, plaques_bool = self._split_branches_by_plaques(
                    branch_mask, plaques.astype(bool))

            refined_mask = body_mask.astype(
                np.uint8) * 255 + branch_mask.astype(np.uint8) * 255

            skeleton = morphology.skeletonize(refined_mask // 255,
                                              method='lee')
            skeleton = (skeleton * 255).astype(np.uint8)

            endpoints = self._find_endpoints(skeleton)

            endpoints_dict[z] = endpoints
            body_pixels_dict[z] = body_mask
            branch_pixels_dict[z] = branch_mask
            z += 1

        return endpoints_dict, body_pixels_dict, branch_pixels_dict

    def get_astrocyte_info(self):
        """Возвращает информацию об астроците"""
        endpoints, body_pixels, branch_pixels = self._process_image(
            self.layers)
        return endpoints, body_pixels, branch_pixels

    def get_astrocytes(self):
        """
        Возвращает список изображений с вырезанными астроцитами
         на черном фоне для каждого слоя
         """
        astrocyte_images = []
        for image in self.layers:
            original_size = image.shape[:2]
            image_preprocessed = self._preprocess_image(image)
            transformed = self.transforms(image=image_preprocessed)
            image_tensor = transformed["image"].unsqueeze(0).to(self.device)
            output_image = self.model_inference(image_tensor, original_size)
            filtered_mask = self._postprocess_mask(output_image)
            refined_mask = self._refine_mask(image_preprocessed, filtered_mask,
                                             self.tolerance)
            astrocyte = cv2.bitwise_and(image, image, mask=refined_mask)
            astrocyte_images.append(astrocyte)
        return astrocyte_images
