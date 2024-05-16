#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   08 February 2019

from __future__ import absolute_import, print_function

import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset


class COCO(_BaseDataset):
    """
    COCO 2014 Segmentation dataset
    """

    def __init__(self, year=14, **kwargs):
        super(COCO, self).__init__(**kwargs)

    def _set_files(self):
        self.root = osp.join(self.root, "coco")

        if "train" in self.split:
            self.image_dir_path = osp.join(self.root, "train2014")
            self.label_dir_path = osp.join(self.root, self.label_dir)
        else:
            self.image_dir_path = osp.join(self.root, "val2014")
            self.label_dir_path = osp.join(self.root, "mask")

        self.datalist_file = osp.join("./data/datasets/coco/", self.split + ".txt")
        print(self.datalist_file)
        self.image_ids, self.cls_labels = self.read_labeled_image_list(self.root, self.datalist_file)

    def _load_data(self, index):
        image_id = self.image_ids[index]
        image_path = osp.join(self.image_dir_path, image_id + ".jpg")
        label_path = osp.join(self.label_dir_path, image_id + ".png")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        cls_label = self.cls_labels[index]

        return image_id, image, label, cls_label

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, "r") as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []

        for line in lines:
            fields = line.strip().split()

            labels = np.zeros((81,), dtype=np.float32)
            labels[0] = 1.0  # background

            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index + 1] = 1.0

            img_name_list.append(fields[0])
            img_labels.append(labels)

        return img_name_list, img_labels
