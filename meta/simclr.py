"""Module containing meta train & meta test methods."""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F


def positive_loss(v1, v2):
    """L2 loss between two tensors.

    Args:
        v1, v2 (torch.Tensor) : tensors.
    Returns:
        L2 loss.
    """
    loss = torch.sum(torch.pow(v1-v2, 2))/v1.size(0)
    return loss


def negative_loss(v1, v2, eps):
    """Loss to minimize similarity.

    Args:
        v1, v2 (torch.Tensor) : tensors.
    Returns:
        loss.
    """
    loss = F.relu(eps - torch.sum(torch.pow(v1-v2, 2)/v1.size(0)))
    return loss


def rotation():
    rot = 90*np.random.randint(1, 4)
    transform = transforms.RandomRotation([rot, rot])
    return transform


def flip():
    choice = np.random.randint(0, 2)
    if choice == 0:
        transform = transforms.RandomVerticalFlip(p=1.)
    else:
        transform = transforms.RandomHorizontalFlip(p=1.)
    return transform


def cutout():
    transform = transforms.RandomErasing(p=1., scale=(.05, .2), ratio=(.3, 2.))
    return transform


def crop():
    crop_size = np.random.randint(90, 150)
    _crop = transforms.CenterCrop(crop_size)
    resize = transforms.Resize(170)
    transform = transforms.Compose([_crop, resize])
    return transform


def blur():
    transform = transforms.GaussianBlur(3)
    return transform


class GaussianNoise:
    def __init__(self, std):
        self.std = std

    def __call__(self, tensor):
        noised_tensor = tensor + torch.randn(tensor.size()).cuda() * self.std
        return noised_tensor


def noise():
    std = np.random.uniform(.15, .35)
    transform = GaussianNoise(std)
    return transform


def simclr(x, net):
    """Performs SimCLR regularization.

    Args:
        x (torch.Tensor) : image batch.
    Returns:
        SimCLR loss.
    """
    n_img = x.size(0)
    pos_idx, neg_idx = list(np.random.choice(n_img, 2, replace=False))
    pos_img = x[pos_idx]
    neg_img = x[neg_idx]
    transform_dict = {
        0: rotation(),
        1: flip(),
        2: cutout(),
        3: crop(),
        4: blur(),
        5: noise()
    }
    tr1_idx, tr2_idx = list(np.random.choice(6, 2, replace=False))
    tr1, tr2 = transform_dict[tr1_idx], transform_dict[tr2_idx]
    aug_img = tr2(tr1(pos_img))
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.imshow(pos_img[0], cmap="gray")
    # ax1.set_title("Positive")
    # ax2.imshow(aug_img[0], cmap="gray")
    # ax2.set_title(tr1.__class__.__name__ + " + " + tr2.__class__.__name__)
    # ax3.imshow(neg_img[0], cmap="gray", label="Negative")
    # ax3.set_title("Negative")
    # plt.show()
    imgs = torch.stack([pos_img, neg_img, aug_img])
    _, reprs = net(imgs)
    pos_repr = reprs[0]
    neg_repr = reprs[1]
    aug_repr = reprs[2]
    pos_loss = positive_loss(pos_repr, aug_repr)
    neg_loss = negative_loss(pos_repr, neg_repr, 1)
    loss = pos_loss + neg_loss
    return loss
