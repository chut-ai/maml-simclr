import os
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def preprocess(path, domain, transform, n_max):
    visda_path = os.path.join(path, "visda_raw")
    domain_path = os.path.join(visda_path, domain)

    all_imgs = os.listdir(domain_path)

    for i, img_name in enumerate(all_imgs):
        print("{}/{}  ".format(i+1, len(all_imgs)), end="\r")
        cls = img_name[-14: -11]
        new_img_folder = os.path.join(path, "VisdaCrop", domain, cls)
        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)
        n_img = len(os.listdir(new_img_folder))
        if n_img < n_max:
            img = Image.open(os.path.join(domain_path, img_name))
            img = transform(img)
            new_img_path = os.path.join(new_img_folder, img_name)
            img.save(new_img_path)


n_max = 2000
transform = transforms.RandomResizedCrop(224, (1, 1), (1, 1))
print("real")
preprocess("/home/louishemadou/data/maml-data/", "real", transform, n_max)
print("\nquickdraw")
preprocess("/home/louishemadou/data/maml-data/", "quickdraw", transform, n_max)
print("\npainting")
preprocess("/home/louishemadou/data/maml-data/", "painting", transform, n_max)
print("\nclipart")
preprocess("/home/louishemadou/data/maml-data/", "clipart", transform, n_max)
print("\nsketch")
preprocess("/home/louishemadou/data/maml-data/", "sketch", transform, n_max)
print("\ninfograph")
preprocess("/home/louishemadou/data/maml-data/", "infograph", transform, n_max)
