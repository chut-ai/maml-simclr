import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_confusion_matrix(xp_name, classes=["background", "civil", "military"], normalize=True, title='Confusion matrix', cmap=plt.cm.Reds):
    cm = np.load(os.path.join("./", xp_name, "confusion_matrix.npy"))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()


plot_confusion_matrix("test2")
