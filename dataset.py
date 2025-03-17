import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):

        self.image_paths = []
        self.labels = []
        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

        data_path = os.path.join(root, 'animals')

        if train:
            data_path = os.path.join(data_path, 'train')
        else:
            data_path = os.path.join(data_path, 'test')

        for (i, category) in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            # print(data_files)
            for image_path in os.listdir(data_files):
                path = os.path.join(data_files, image_path)
                self.image_paths.append(path)
                self.labels.append(i)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # using opencv
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (224, 224))
        # image = np.transpose(image, (2, 0, 1))
        # image /= 255

        label = self.labels[idx]
        return image, label
if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    dataset = AnimalDataset(root='D:/DL4CV/my_dataset', train=True, transform=transform)
    # image, label = dataset[17]
    # image.show()
    # print(label)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    for images, labels in dataloader:
        print(images.shape, labels.shape)