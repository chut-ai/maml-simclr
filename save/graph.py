import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 15})

def plot_graph(xp_name, w, title):
        
    xp_path = os.path.join("./", xp_name)
    y_train = np.load(os.path.join(xp_path, "train_acc.npy"))
    y_train = np.convolve(y_train, np.ones(w), "valid")/w

    y_train_rot = np.load(os.path.join(xp_path, "train_sup_acc.npy"))
    y_train_rot = np.convolve(y_train_rot, np.ones(w), "valid")/w

    y_test_coco = np.load(os.path.join(xp_path, "test_acc_coco.npy"))
    y_test_coco = np.convolve(y_test_coco, np.ones(w), "valid")/w

    y_test_safran = np.load(os.path.join(xp_path, "test_acc_safran.npy"))
    y_test_safran = np.convolve(y_test_safran, np.ones(w), "valid")/w

    y_base = 0.6*np.ones(len(y_train))

    x = range(len(y_train))
    plt.figure()
    plt.scatter(x, y_train, color="k", label="meta train")
    plt.scatter(x, y_train_rot, color="grey", label="supervision acc")
    plt.scatter(x, y_test_coco, color="r", label="meta test - coco")
    plt.scatter(x, y_test_safran, color="g", label="meta test - safran")
    plt.plot(x, y_base, "b--", label="baseline")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.show()


title = "MAML 5 shots with auto supervision & dropout, meta train half ResNet and classifier on augmented COCO database, meta test on 3-way Safran problem."
w = 1
xp_name = "no_super_augment"

plot_graph(xp_name, w, title)
