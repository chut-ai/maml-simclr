"""Useful functions to create relationships between class id and class name"""

import os


def int_to_class(path):
    """Returns dict giving class id from class name.

    Args:
        path (str) : path to database class folders.
    Returns:
        class id to class name dictionnary
    """

    relation_dict = {}

    for i, cls in enumerate(sorted(os.listdir(path))):
        relation_dict[i] = cls
    return relation_dict


def class_to_int(path):
    """Returns dict giving class name from class id.

    Args:
        path (str) : path to database class folders.
    Returns:
        class name to class id dictionnary
    """

    relation_dict = {}

    for i, cls in enumerate(sorted(os.listdir(path))):
        relation_dict[cls] = i

    return relation_dict

def squeeze(labels):

    items = []
    for label in labels:
        if label not in items:
            items.append(label)

    items = sorted(items)

    squeezed = []
    for label in labels:
        squeezed.append(items.index(label))
    return squeezed
