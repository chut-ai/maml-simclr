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
from data.utils import squeeze

DEFAULT_PATH = "/home/louishemadou/data/"


def get_all_data(data_root, domains, transform):

    all_data = {}

    for domain in domains:
        domain_data = {idx: [] for idx in range(1, 346)}
        data_path = os.path.join(data_root, domain)
        for i, cls in enumerate(os.listdir(data_path)):
            message = "Loading Visda {} images, {}/345".format(domain, i+1)
            print(message, end="\r")
            cls_path = os.path.join(data_path, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                domain_data[int(cls)].append(img_path)
        all_data[domain] = domain_data
        print("\nDone !")

    return all_data


class VisdaTask:
    """Class containing methods to easily create tasks."""

    def __init__(self, n_class, n_qry, n_spt, domains,
                 root, n_train_class=200, train_class=None):

        self.n_qry = n_qry
        self.n_spt = n_spt
        self.n_class = n_class
        data_root = os.path.join(root, "VisdaCrop")
        self.classes = range(1, 346)
        self.domains = domains

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.data = get_all_data(data_root, domains, self.transform)

        if train_class is not None:
            self.train_class = train_class
        else:
            self.train_class = list(np.random.choice(
                self.classes, n_train_class, False))

        self.test_class = [
            x for x in self.classes if x not in self.train_class]

    def task(self, mode, source, target):
    
        if mode == "train":
            task_classes = self.train_class
        elif mode == "test":
            task_classes = self.test_class
        else:
            raise "WrongModeError"

        chosen_classes = list(np.random.choice(
            task_classes, self.n_class, False))
        
        spt_instances = self.data[source]
        qry_instances = self.data[target]

        spt_data = []
        qry_data = []

        for cls in chosen_classes:
            spt_cls_instances = spt_instances[cls]
            spt_n_img = len(spt_cls_instances)
            spt_indexes = np.random.choice(spt_n_img, self.n_spt, False)
            random.shuffle(spt_indexes)
            for spt_index in spt_indexes:
                spt_instance = Image.open(spt_cls_instances[spt_index])
                spt_instance = self.transform(spt_instance)
                spt_data.append([spt_instance, cls])
            qry_cls_instances = qry_instances[cls]
            qry_n_img = len(qry_cls_instances)
            qry_indexes = np.random.choice(qry_n_img, self.n_qry, False)
            random.shuffle(qry_indexes)
            for qry_index in qry_indexes:
                qry_instance = Image.open(qry_cls_instances[qry_index])
                qry_instance = self.transform(qry_instance)
                qry_data.append([qry_instance, cls])
        random.shuffle(spt_data)
        random.shuffle(qry_data)

        instances_spt = [elem[0] for elem in spt_data]
        labels_spt = squeeze([elem[1] for elem in spt_data])

        instances_qry = [elem[0] for elem in qry_data]
        labels_qry = squeeze([elem[1] for elem in qry_data])

        x_spt = torch.stack(instances_spt, 0)
        x_qry = torch.stack(instances_qry, 0)
        y_spt = torch.Tensor(labels_spt).type(torch.int64)
        y_qry = torch.Tensor(labels_qry).type(torch.int64)

        return x_spt, x_qry, y_spt, y_qry

    def task_batch(self, task_bsize, mode, source, target):

        tasks = [self.task(mode, source, target) for _ in range(task_bsize)]
        return tasks
