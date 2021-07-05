import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms

DEFAULT_PATH = "/home/louishemadou/data/maml-data/safran_raw/"


class SafranDataset(Dataset):
    def __init__(self, data, path=DEFAULT_PATH):

        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img = Image.open(img_name)

        img = self.transform(img)
        return img, label


def safran_datasets(n_train, n_test, path=DEFAULT_PATH):

    train_data = []
    test_data = []
    class_to_int = {"background": 0, "civil": 1, "military": 2}

    classes = os.listdir(path)

    for cls in classes:
        cls_path = os.path.join(path, cls)
        label = class_to_int[cls]
        cls_img = [(os.path.join(cls_path, img), label) for img in os.listdir(cls_path)]
        random.shuffle(cls_img)
        train_data += cls_img[:n_train]
        test_data += cls_img[n_train:n_train+n_test]

    random.shuffle(train_data)
    random.shuffle(test_data)

    trainset = SafranDataset(train_data, path)
    testset = SafranDataset(test_data, path)

    return trainset, testset
