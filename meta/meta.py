"""Module containing meta train & meta test methods."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import higher
from meta.simclr import simclr


class Meta(nn.Module):
    """Object containing slow weights and methods to meta train and meta test.
    """

    def __init__(self, net, lamb=1):
        """
        Initialise meta model.

        Args:
            net (nn.Module) : meta model's slow weights.
            lamb (float) : weight given to auto supervision loss during
                           training.
        """
        super(Meta, self).__init__()
        self.net = net
        self.lamb = lamb

    def train(self, task_batch, inner_lr, n_inner_loop):
        """Meta train meta model. The meta model solves every tasks in
        task_batch, then slow weights are updated according to meta loss.

        Args:
            task_batch (list) : list of tasks to solve.
            inner_lr (float) : learning rate to use in inner optimization.
            n_inner_loop (float) : number of inner loops.
        Returns:
            (float, float) : average classification & auto supervision accuracy
                             of trained models on query images.
        """

        self.net.train()

        inner_opt = optim.Adam(self.net.parameters(), lr=inner_lr)

        qry_accs = []
        simclr_losses = []

        for task in task_batch:
            x_spt, x_qry, y_spt, y_qry = task
            x_spt, y_spt = x_spt.cuda(), y_spt.cuda()
            x_qry, y_qry = x_qry.cuda(), y_qry.cuda()

            with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                for _ in range(n_inner_loop):
                    spt_logits, _ = fnet(x_spt)
                    clsf_loss = F.cross_entropy(spt_logits, y_spt)
                    spt_loss = clsf_loss
                    diffopt.step(spt_loss)

                qry_logits, _ = fnet(x_qry)
                simclr_loss = simclr(x_qry, fnet)
                clsf_loss = F.cross_entropy(qry_logits, y_qry)
                qry_loss = clsf_loss + self.lamb*simclr_loss
                qry_loss.backward()

                qry_logits, _ = fnet(x_qry)
                qry_logits = qry_logits.detach()
                qry_acc = (qry_logits.argmax(dim=1) ==
                           y_qry).sum().item()/y_qry.size(0)
                qry_accs.append(qry_acc)
                simclr_losses.append(simclr_loss.item())

        mean_qry_acc = np.average(qry_accs)
        mean_simclr_loss = np.average(simclr_losses)
        return mean_qry_acc, mean_simclr_loss

    def test(self, task_batch, inner_lr, n_inner_loop, return_matrix=False):
        """Meta test meta model. The meta model solves every tasks in
        task_batch.

        Args:
            task_batch (list) : list of tasks to solve.
            inner_lr (float) : learning rate to use in inner optimization.
            n_inner_loop (float) : number of inner loops.
            return_matrix (bool) : return average confusion matrix.
        Returns:
            float : average classification accuracy of trained models on query
                    images.
            (optionnaly) np.array : confusion matrix
        """
        self.net.train()

        inner_opt = optim.Adam(self.net.parameters(), lr=inner_lr)

        qry_accs = []
        matrices = []

        for task in task_batch:
            x_spt, x_qry, y_spt, y_qry = task
            x_spt, y_spt = x_spt.cuda(), y_spt.cuda() 
            x_qry, y_qry = x_qry.cuda(), y_qry.cuda()

            with higher.innerloop_ctx(self.net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                for _ in range(n_inner_loop):
                    spt_logits, spt_super = fnet(x_spt)
                    clsf_loss = F.cross_entropy(spt_logits, y_spt)
                    spt_loss = clsf_loss
                    diffopt.step(spt_loss)

                qry_logits = fnet(x_qry)[0].detach()
                qry_acc = (qry_logits.argmax(dim=1) ==
                           y_qry).sum().item()/y_qry.size(0)
                matrix = confusion_matrix(
                    y_qry.cpu(), qry_logits.cpu().argmax(dim=1))
                qry_accs.append(qry_acc)
                matrices.append(matrix)

        if return_matrix:
            return np.average(qry_accs), np.average(matrices, axis=0)
        return np.average(qry_accs)
