import time
import os
import numpy as np
import torch
import torch.optim as optim
from meta.meta import Meta
from models import FullNet
from data.coco_task import CocoTask
from data.safran_task import SafranTask

xp_name = "simclr-1"
xp_path = os.path.join("./save/", xp_name)
if not os.path.exists(xp_path):
    os.makedirs(xp_path)

n_class = 3
n_train_class = 50
n_spt = 10
n_qry = 10
task_bsize = 20
n_batch = 50000
root = "/home/louishemadou/data/maml-data/"

coco = CocoTask(n_qry, n_spt, root, n_train_class)
safran = SafranTask(n_qry, n_spt, root)

inner_lr = 0.001
n_inner_loop = 10

net = FullNet()
meta_model = Meta(net, lamb=1)
meta_model.cuda()
meta_lr = 0.001

meta_opt = optim.Adam(meta_model.parameters(), meta_lr)
train_accs = []
test_accs_safran = []
test_accs_coco = []
simclr_losses = []

max_acc = 0

for i in range(0, n_batch):

    # Train COCO
    meta_opt.zero_grad()
    train_batch = coco.task_batch(task_bsize, "train")
    train_acc, simclr_loss = meta_model.train(
        train_batch, inner_lr, n_inner_loop)
    train_accs.append(train_acc)
    simclr_losses.append(simclr_loss)
    meta_opt.step()
    del train_batch

    # Test COCO
    test_batch_coco = coco.task_batch(task_bsize, "test")
    test_acc_coco = meta_model.test(test_batch_coco, inner_lr, n_inner_loop)
    test_accs_coco.append(test_acc_coco)
    del test_batch_coco

    # Test Safran
    test_batch_safran = safran.task_batch(task_bsize)
    test_acc_safran, matrix = meta_model.test(
        test_batch_safran, inner_lr, n_inner_loop, return_matrix=True)
    test_accs_safran.append(test_acc_safran)
    if test_acc_safran > max_acc:
        max_acc = test_acc_safran
        np.save(os.path.join(xp_path, "confusion_matrix.npy"), np.array(matrix))
        torch.save(meta_model, os.path.join(xp_path, "meta_model.pt"))
    del test_batch_safran

    np.save(os.path.join(xp_path, "train_acc.npy"), np.array(train_accs))
    np.save(os.path.join(xp_path, "test_acc_coco.npy"),
            np.array(test_accs_coco))
    np.save(os.path.join(xp_path, "simclr_losses.npy"), np.array(simclr_losses))
    np.save(os.path.join(xp_path, "test_acc_safran.npy"),
            np.array(test_accs_safran))

    message = "Task batch {}/{}, train {:.2f}%, test coco {:.2f}%, simclr loss {:.2f}, test safran {:.2f}%".format(
        i+1, n_batch, 100*train_acc, 100*test_acc_coco, simclr_loss, 100*test_acc_safran)

    print(message)
