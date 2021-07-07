"""Preprocess COCO raw images into 'safran-like' images."""

import json
import os
import random
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


def bounds(bbox, pad):
    x0, y0, w, h = bbox
    x0, y0 = x0+pad, y0+pad
    xm, xM, ym, yM = x0 - min(w, h), x0 + w + \
        min(w, h), y0 - min(w, h), y0 + h + min(w, h)
    xm, xM, ym, yM = int(xm), int(xM), int(ym), int(yM)
    return xm, xM, ym, yM


def preprocess(path, transform, p1, p2, n_max):
    """Preprocess COCO dataset. COCO is a detection dataset, a bounding box and
    a label is provided for each image.
        - Center and scale the object.
        - Apply the PyTorch transform.
        - Save the image.

    Args:
        - path (str) : path to coco_raw folder.
        - transform (torch.transforms) : transform to apply.
        - p1 (float) : lower bound for the ratio area of the box over area of
                       the image.
        - p2 (float) : upper bound.
        - n_max (int) : maximal number of images to be saved per class.

    """

    _file = open(os.path.join(
        path, "coco_raw/annotations2014/instances_val2014.json"), "r")
    json_file = json.load(_file)
    classes = json_file["categories"]
    annotations = json_file["annotations"]
    random.shuffle(annotations)

    for i, img_info in enumerate(annotations):

        print("{}/{}  ".format(i+1, len(annotations)), end="\r")
        for cls in classes:
            if cls["id"] == img_info["category_id"]:
                img_cls = str(cls["name"])
                break

        img_id = img_info["image_id"]
        bbox = img_info["bbox"]

        raw_img = str(img_id).zfill(12)
        raw_img_path = os.path.join(
            path, "coco_raw/val2014/COCO_val2014_{}.jpg".format(raw_img))

        new_img_folder = os.path.join(path, "CocoCrop", img_cls)

        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)

        n_img = len(os.listdir(new_img_folder))

        if n_img < n_max:

            img = cv2.imread(raw_img_path)
            box_area = bbox[2]*bbox[3]
            img_area = img.shape[0]*img.shape[1]
            ratio = box_area/img_area

            if p1 < ratio < p2:

                pad = int(min(bbox[2], bbox[3]))
                img2 = cv2.copyMakeBorder(
                    img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                xm, xM, ym, yM = bounds(bbox, pad)
                img2 = img2[ym:yM, xm:xM]
                img2 = cv2.resize(img2, (170, 170))
                img2 = Image.fromarray(np.uint8(img2)).convert("RGB")
                img2 = transform(img2).convert("LA")

                new_img_path = os.path.join(new_img_folder, raw_img + ".png")
                img2.save(new_img_path)


transform = transforms.Compose([
    transforms.ColorJitter([.60, 1.], [1., 3.]),
    transforms.GaussianBlur(3, sigma=[.40, .70]),
    transforms.Grayscale()
])

p1 = 0.002
p2 = 0.6
n_max = 2000

preprocess("/home/louishemadou/data/maml-data/", transform, p1, p2, n_max)
