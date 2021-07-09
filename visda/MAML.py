import time
import os
import numpy as np
import torch
import torch.optim as optim
from meta.meta import Meta
from models import FullNet
from data.visda import VisdaTask

xp_name = "simclr-1"
xp_path = os.path.join("./save/", xp_name)
if not os.path.exists(xp_path):
    os.makedirs(xp_path)

n_class = 10
n_train_class = 200
n_spt = 10
n_qry = 20
task_bsize = 10
n_batch = 50000
n_max = 75
root = "/home/louishemadou/data/maml-data/"

train_class = list(np.random.choice(range(1, 346), n_train_class, replace=False))
np.save(os.path.join(xp_path, "train_class.npy"), train_class)

source = "real"
target = "quickdraw"

domains = [source, target]

visda = VisdaTask(n_class, n_qry, n_spt, domains, n_max, root, train_class=train_class)

inner_lr = 0.001
n_inner_loop = 5

net = FullNet(n_class)
meta_model = Meta(net, lamb=1, eps=.5).cuda()
meta_lr = 0.001

meta_opt = optim.Adam(meta_model.parameters(), meta_lr)
train_accs = []
test_accs = []
simclr_accs_src = []
simclr_accs_tgt = []

max_acc = 0
    
for i in range(0, n_batch):

    # Train COCO
    meta_opt.zero_grad()
    train_batch = visda.task_batch(task_bsize, "train", source, target)
    train_acc, simclr_acc_src, simclr_acc_tgt = meta_model.train(
        train_batch, inner_lr, n_inner_loop)
    train_accs.append(train_acc)
    simclr_accs_src.append(simclr_acc_src)
    simclr_accs_tgt.append(simclr_acc_tgt)
    meta_opt.step()
    del train_batch

    # Test COCO
    test_batch = visda.task_batch(task_bsize, "test", source, target)
    test_acc = meta_model.test(test_batch, inner_lr, n_inner_loop)
    test_accs.append(test_acc)
    del test_batch

    np.save(os.path.join(xp_path, "train_acc.npy"), np.array(train_accs))
    np.save(os.path.join(xp_path, "test_acc.npy"), np.array(test_accs))
    np.save(os.path.join(xp_path, "simclr_acc_src.npy"), np.array(simclr_accs_src))
    np.save(os.path.join(xp_path, "simclr_acc_tgt.npy"), np.array(simclr_accs_tgt))

    message = "Task batch {}/{}, train {:.2f}%, test {:.2f}%, simclr src {:.2f}%, simclr tgt {:.2f}%".format(
        i+1, n_batch, 100*train_acc, 100*test_acc, 100*simclr_acc_src, 100*simclr_acc_tgt)

    print(message)
