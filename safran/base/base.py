from model import FullNet
from data import safran_datasets
from torch.utils.data import DataLoader
import numpy as np
import higher
import torch
import torch.nn as nn
import torch.optim as optim
import sys

n_train = 5
n_test = 60

n_monte_carlo = 20

accs = []

net = FullNet().cuda()
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for n in range(1, n_monte_carlo+1):

    trainset, testset = safran_datasets(n_train, n_test)
    trainloader = DataLoader(trainset, len(trainset), shuffle=True)
    testloader = DataLoader(testset, len(trainset), shuffle=True)

    n_epochs = 20

    with higher.innerloop_ctx(net, optimizer, track_higher_grads=False) as (fnet, diffopt):

        for epoch in range(1, n_epochs+1):
            train_acc = 0
            for x, y in trainloader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                y_hat = fnet(x)[0]
                loss = criterion(y_hat, y)
                diffopt.step(loss)
                train_acc += (y_hat.argmax(dim=1) == y).sum().item() / \
                    (y.size(0)*len(trainloader))
            test_acc = 0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.cuda(), y.cuda()
                    y_hat = fnet(x)[0]
                    test_acc += (y_hat.argmax(dim=1) == y).sum().item() / \
                        (y.size(0)*len(testloader))
            accs.append(test_acc)
            print("Run {}/{}, {:.0f}%  ".format(n,
                                                n_monte_carlo, 100*epoch/n_epochs), end="\r")

mean_acc = np.average(accs)
print("\n mean accuracy : {:.2f}".format(mean_acc))
