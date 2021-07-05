"""Module containing Safran task class."""

import os
import random
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from data.utils import int_to_class, squeeze


class SafranTask:
    """Class containing methods to easily create tasks. SafranTask only
    provides test tasks."""

    def __init__(self, n_qry, n_spt, root):
        """Loads Safran images in computer RAM.

        Args:
            n_qry (int) : number of test images.
            n_spt (int) : number of train images.
            root (str) : path to safran_raw folder.
        """
        self.n_qry = n_qry
        self.n_spt = n_spt
        data_root = os.path.join(root, "safran_raw")
        self.classes = range(3)
        relation_dict = int_to_class(data_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.all_instances = {cls_int: [] for cls_int in self.classes}

        for i, cls_int in enumerate(self.classes):
            message = "Loading Safran images, {}/3".format(i+1)
            print(message, end="\r")
            cls = relation_dict[cls_int]
            for img_name in os.listdir(os.path.join(data_root, cls)):
                img_path = os.path.join(data_root, cls, img_name)
                img = Image.open(img_path).convert("RGB")
                self.all_instances[cls_int].append(img)
        print("\nDone !")

    def task(self):
        """Create a single task. A task is a 3-way classification problem
        composed of n_spt training images per class and n_qry test images per
        class.

        Returns:
            (list) : - x_spt, x_qry : train and test images (torch.Tensor).
                     - y_spt, y_qry : train and test labels (torch.Tensor).
                     - z_spt, z_qry : train and test rotation label. As we do
                                      not rotate Safran images, label is always
                                      0.
        """

        chosen_classes = self.classes

        spt_data = []
        qry_data = []
        for cls in chosen_classes:
            cls_instances = self.all_instances[cls]
            n_img = len(cls_instances)
            indexes = np.random.choice(n_img, self.n_spt+self.n_qry, False)
            random.shuffle(indexes)
            for spt_index in indexes[:self.n_spt]:
                spt_instance = self.transform(cls_instances[spt_index])
                spt_data.append([spt_instance, cls, 0])
            for qry_index in indexes[self.n_spt:]:
                qry_instance = self.transform(cls_instances[qry_index])
                qry_data.append([qry_instance, cls, 0])
        random.shuffle(spt_data)
        random.shuffle(qry_data)

        instances_spt = [elem[0] for elem in spt_data]
        labels_spt = squeeze([elem[1] for elem in spt_data])
        rot_spt = [elem[2] for elem in spt_data]

        instances_qry = [elem[0] for elem in qry_data]
        labels_qry = squeeze([elem[1] for elem in qry_data])
        rot_qry = [elem[2] for elem in qry_data]

        x_spt = torch.stack(instances_spt, 0)
        x_qry = torch.stack(instances_qry, 0)
        y_spt = torch.Tensor(labels_spt).type(torch.int64)
        y_qry = torch.Tensor(labels_qry).type(torch.int64)
        z_spt = torch.Tensor(rot_spt).type(torch.int64)
        z_qry = torch.Tensor(rot_qry).type(torch.int64)

        return x_spt, x_qry, y_spt, y_qry, z_spt, z_qry

    def task_batch(self, task_bsize):
        """Create a batch of tasks.

        Args:
            task_bsize (int) : number of task in a batch.
        Returns:
            list of tasks.
        """

        tasks = [self.task() for _ in range(task_bsize)]

        return tasks
