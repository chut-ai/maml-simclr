"""Module containing coco task class."""

import os
import random
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from data.utils import int_to_class, squeeze


class CocoTask:
    """Class containing methods to easily create tasks."""

    def __init__(self, n_qry, n_spt, root, n_train_class=50, train_class=None):
        """Loads coco preprocessed images in computer RAM. Also, split classes
        for meta train and meta test.

        Args:
            n_qry (int) : number of query images for meta optimization.
            n_spt (int) : number of support images for inner optimization.
            root (str) : path to coco_preprocessed folder.
            n_train_class (str) : number of train classes.
            train_class (list) : list of train classes.
        """

        self.n_qry = n_qry
        self.n_spt = n_spt
        data_root = os.path.join(root, "coco_preprocessed")
        self.classes = range(80)
        relation_dict = int_to_class(data_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.all_instances = {cls_int: [] for cls_int in self.classes}

        for i, cls_int in enumerate(self.classes):
            message = "Loading Coco images, {}/80".format(i+1)
            print(message, end="\r")
            cls = relation_dict[cls_int]
            for img_name in os.listdir(os.path.join(data_root, cls)):
                img_path = os.path.join(data_root, cls, img_name)
                img = Image.open(img_path).convert("RGB")
                self.all_instances[cls_int].append(img)
        print("\nDone !")

        if train_class is not None:
            self.train_class = train_class
        else:
            self.train_class = list(np.random.choice(
                self.classes, n_train_class, False))

        self.test_class = [
            x for x in self.classes if x not in self.train_class]

    def task(self, task_classes):
        """Create a single task. A task is a 3-way classification problem
        composed of n_spt training images per class and n_qry test images per
        class. Task classes can be meta train classes or meta test classes.

        Args:
            task_classes (list) : classes to use to make task. Can be
                                  self.test_classes or self.train_classes.
        Returns:
            (list) : - x_spt, x_qry : train and test images.
                     - y_spt, y_qry : train and test labels).
                     - z_spt, z_qry : train and test rotation label.
        """

        chosen_classes = np.random.choice(task_classes, 3, False)

        spt_data = []
        qry_data = []
        for cls in chosen_classes:
            cls_instances = self.all_instances[cls]
            n_img = len(cls_instances)
            indexes = np.random.choice(n_img, self.n_spt+self.n_qry, False)
            random.shuffle(indexes)
            for spt_index in indexes[:self.n_spt]:
                rot = np.random.randint(0, 4)
                rotation = transforms.RandomRotation([90*rot, 90*rot])
                spt_instance = self.transform(
                    rotation(cls_instances[spt_index]))
                spt_data.append([spt_instance, cls, rot])
            for qry_index in indexes[self.n_spt:]:
                rot = np.random.randint(0, 4)
                rotation = transforms.RandomRotation([90*rot, 90*rot])
                qry_instance = self.transform(
                    rotation(cls_instances[qry_index]))
                qry_data.append([qry_instance, cls, rot])
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

    def task_batch(self, task_bsize, mode):
        """Create a batch of tasks.

        Args:
            task_bsize (int) : number of task in a batch.
            mode (str) : 'train' or 'test'.
        Returns:
            list of tasks.
        """

        if mode == "train":
            task_classes = self.train_class
        elif mode == "test":
            task_classes = self.test_class
        else:
            raise "WrongModeError"

        tasks = [self.task(task_classes) for _ in range(task_bsize)]

        return tasks
