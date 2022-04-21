import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms.functional as tf
import pandas as pd
import torch.nn as nn
from models import resnet32,UNet
from tqdm import tqdm

np.random.seed(100)


train_path = "./datasets/Nyu_v2/train"
test_path = "./datasets/Nyu_v2/test"

train_data = {"images":[], "depths":[], "labels":[]}
for file in os.listdir(train_path + "/train_images"):
        train_data["images"].append(os.path.join(train_path + "/train_images", file))
        train_data["depths"].append(os.path.join(train_path + "/train_depths", file))
        train_data["labels"].append(os.path.join(train_path + "/train_labels", file))

test_data = {"images": [], "depths": [], "labels": []}
for file in os.listdir(test_path + "/test_images"):
    test_data["images"].append(os.path.join(test_path + "/test_images", file))
    test_data["depths"].append(os.path.join(test_path + "/test_depths", file))
    test_data["labels"].append(os.path.join(test_path + "/test_labels", file))



class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.number = len(self.data["images"])
        self.length = int(np.ceil(len(self.data["images"]) / self.batch_size))
        self.index = [i for i in range(self.__len__())]
        if self.shuffle:
            np.random.shuffle(self.index)

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        # return int(np.ceil(len(self.data) / self.batch_size))
        return self.length


    def __getitem__(self, id):
        index = self.index[id]
        batch_item_number = self.batch_size
        if (index + 1) * self.batch_size > self.length:
            batch_item_number = self.number - index * self.batch_size
        x = np.zeros([batch_item_number,3,480, 640])
        y = np.zeros([batch_item_number, 1,480, 640])

        for i in range(batch_item_number ):
            x[i,], y[i,] = self.load(
                self.data["images"][index * self.batch_size + i],
                self.data["depths"][index * self.batch_size + i],
                self.data["labels"][index * self.batch_size + i],
            )
        return x,y

    def load(self, image_dir, depth_dir, label_dir):
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2,0,1)/255
        return image, 1

train_loader = DataGenerator(train_data)
test_loader = DataGenerator(test_data)




