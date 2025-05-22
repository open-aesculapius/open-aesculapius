import glob
import cv2
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DataStructure(Dataset):
    def __init__(self, train):
        self.status = train

        self.dataset_dir = f"DataSetPath/{self.status}/"
        self.list_of_0_images = None
        self.list_of_images = None
        self.make_list()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5)),
            ]
        )
        self.transform_mask = transforms.Compose([transforms.ToTensor()])

    def make_list(self):
        self.list_of_images = glob.glob(
            os.path.join(self.dataset_dir, "*_gt.png"), recursive=False
        )
        self.list_of_0_images = glob.glob(
            os.path.join(self.dataset_dir, "*_layer.png"), recursive=False
        )
        for i in range(len(self.list_of_images)):
            self.list_of_images[i] = self.list_of_images[i][:-7] + ".png"
        self.list_of_images = self.list_of_images + self.list_of_0_images

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_pth = self.list_of_images[idx]
        if input_image_pth[-9:] == "layer.png":
            gt_image = np.zeros((32, 32))
        else:
            gt_image_pth = input_image_pth[:-4] + "_gt.png"
            gt_image = np.array(cv2.imread(gt_image_pth))

        input_image = np.array(cv2.imread(input_image_pth))

        if [gt_image.shape[0], gt_image.shape[1]] != [32, 32]:
            gt_image = cv2.resize(gt_image, (32, 32))
        if [input_image.shape[0], input_image.shape[1]] != [32, 32]:
            input_image = cv2.resize(input_image, (32, 32))
        gt_image_null = np.zeros((gt_image.shape[:2]))
        for i in range(gt_image.shape[0]):
            for j in range(gt_image.shape[1]):
                if gt_image[i][j].tolist() == [0, 0, 255]:
                    gt_image_null[i][j] = 1
        out_table = {
            "dt": self.transform(input_image),
            "gt": self.transform_mask(gt_image_null),
        }
        return out_table
