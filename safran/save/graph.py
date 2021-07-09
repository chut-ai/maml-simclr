import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 15})

def plot_graph(xp_name, w, title):
        
    xp_path = os.path.join("./", xp_name)
    y_train = np.load(os.path.join(xp_path, "train_acc.npy"))
    y_train = np.convolve(y_train, np.ones(w), "valid")/w

    y_test_coco = np.load(os.path.join(xp_path, "test_acc_coco.npy"))
    y_test_coco = np.convolve(y_test_coco, np.ones(w), "valid")/w

    y_test_safran = np.load(os.path.join(xp_path, "test_acc_safran.npy"))
    y_test_safran = np.convolve(y_test_safran, np.ones(w), "valid")/w
    print(max(y_test_safran))

    y_loss_simclr = np.load(os.path.join(xp_path, "simclr_losses.npy"))
    y_loss_simclr = np.convolve(y_loss_simclr, np.ones(w), "valid")/w

    y_acc_simclr = np.load(os.path.join(xp_path, "simclr_accs.npy"))
    y_acc_simclr = np.convolve(y_acc_simclr, np.ones(w), "valid")/w

    y_base = 0.6*np.ones(len(y_train))

    x = range(len(y_train))
    fig, ax = plt.subplots()
    ax.set_ylabel("Precision")
    ax.set_xlabel("Tasks")
    ax.grid()
    # ax_loss = ax.twinx()
    # ax_loss.set_ylabel("SimCLR loss")
    # ax_loss.plot(x, y_loss_simclr, color="grey", label="SimCLR loss")
    ax.scatter(x, y_train, color="k", label="meta train")
    ax.scatter(x, y_test_coco, color="r", label="meta test - coco")
    ax.scatter(x, y_test_safran, color="g", label="meta test - safran")
    ax.scatter(x, y_acc_simclr, color="grey", label="SimCLR acc")
    ax.plot(x, y_base, "b--", label="baseline")
    fig.legend(loc="upper left")
    ax.set_title(title)
    plt.show()


title = "MAML 5 shots with SimCLR - meta auto supervision on every state of fast weights"
w = 20
xp_name = "simclr-2"

plot_graph(xp_name, w, title)
