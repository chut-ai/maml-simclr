import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 15})

def plot_graph(xp_name, w, title):
        
    xp_path = os.path.join("./", xp_name)
    y_train = np.load(os.path.join(xp_path, "train_acc.npy"))
    y_train = np.convolve(y_train, np.ones(w), "valid")/w

    y_test = np.load(os.path.join(xp_path, "test_acc.npy"))
    y_test = np.convolve(y_test, np.ones(w), "valid")/w
    print(max(y_test))

    y_acc_simclr_src = np.load(os.path.join(xp_path, "simclr_acc_src.npy"))
    y_acc_simclr_src = np.convolve(y_acc_simclr_src, np.ones(w), "valid")/w

    y_acc_simclr_tgt = np.load(os.path.join(xp_path, "simclr_acc_tgt.npy"))
    y_acc_simclr_tgt = np.convolve(y_acc_simclr_tgt, np.ones(w), "valid")/w

    x = range(len(y_train))
    fig, ax = plt.subplots()
    ax.set_ylabel("Precision")
    ax.set_xlabel("Tasks")
    ax.grid()
    ax.scatter(x, y_train, color="k", label="meta train")
    ax.scatter(x, y_test, color="r", label="meta test")
    ax.scatter(x, y_acc_simclr_src, color="lime", label="simclr source")
    ax.scatter(x, y_acc_simclr_tgt, color="seagreen", label="simclr target")
    fig.legend(loc="upper left")
    ax.set_title(title)
    plt.show()


title = "MAML 5 shots with SimCLR - meta auto supervision on every state of fast weights"
w = 2
xp_name = "simclr-1"

plot_graph(xp_name, w, title)
