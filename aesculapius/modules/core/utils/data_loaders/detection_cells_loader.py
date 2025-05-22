from torch.utils.data import Dataset
import numpy as np
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
import os
import glob
import albumentations as A
import torch


class StackBatchLoader(Dataset):
    def __init__(self):

        self.dataset_dir = "data_cells_detection"
        self.list_of_paths = os.listdir(self.dataset_dir)

        self.transform = A.Compose(
            [
                A.RandomCrop(width=256, height=256),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5] * 24,
                            std=[0.5, 0.5, 0.5] * 24),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"]),
        )

    def __len__(self):
        return len(self.list_of_paths) * 20

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx % (len(self.list_of_paths) - 1)
        bboxes_df = pd.read_csv(
            self.dataset_dir + "/" + self.list_of_paths[idx] + "/target.txt",
            sep=","
        )
        x = bboxes_df.X.values / 1024
        x[x < 0] = 1e-4
        y = bboxes_df.Y.values / 1024
        y[y < 0] = 1e-4
        w = bboxes_df.W.values / 512
        w[w < 0] = 1e-4
        h = bboxes_df.H.values / 512
        h[h < 0] = 1e-4
        bboxes = []

        for i in range(len(x)):
            assert x[i] >= 0 and y[i] >= 0 and h[i] >= 0 and w[i] >= 0
            bboxes.append([x[i], y[i], w[i], h[i]])
        image = np.zeros((1024, 1024, 72))
        i = 0
        self.bbox = bboxes
        for image_pth in glob.glob(
            self.dataset_dir + "/" + self.list_of_paths[idx] + "/*.png"
        ):
            image[:, :, i: i + 3] = np.array(cv2.imread(image_pth))
            # cv2.imwrite(f"{i}.png", cv2.imread(image_pth))
            i += 3
        transformed = self.transform(
            image=image, bboxes=bboxes, class_labels=[1] * len(bboxes)
        )
        image = transformed["image"].view(24, 3, 256, 256)
        return image, transformed["bboxes"], transformed["class_labels"]
