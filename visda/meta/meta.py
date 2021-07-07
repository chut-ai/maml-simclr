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

    def __init__(self, net, lamb=1, eps=1):
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
        self.eps = eps

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
        simclr_accs_src = []
        simclr_accs_tgt = []

        for task in task_batch:
            x_spt, x_qry, y_spt, y_qry = task
            x_spt, y_spt = x_spt.cuda(), y_spt.cuda()
            x_qry, y_qry = x_qry.cuda(), y_qry.cuda()

            with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                super_meta_loss = 0
                super_acc_src = 0
                super_acc_tgt = 0
                for _ in range(n_inner_loop):
                    spt_logits, _ = fnet(x_spt)
                    super_loss_src, acc_src = simclr(x_spt, fnet, self.eps)
                    super_loss_tgt, acc_tgt = simclr(x_qry, fnet, self.eps)
                    super_meta_loss += (1/n_inner_loop)*(super_loss_src+super_loss_tgt)
                    super_acc_src += acc_src/n_inner_loop
                    super_acc_tgt += acc_tgt/n_inner_loop
                    clsf_loss = F.cross_entropy(spt_logits, y_spt)
                    diffopt.step(clsf_loss)

                qry_logits, _ = fnet(x_qry)
                clsf_loss = F.cross_entropy(qry_logits, y_qry)
                qry_loss = clsf_loss + self.lamb*super_meta_loss
                qry_loss.backward()

                qry_logits, _ = fnet(x_qry)
                qry_logits = qry_logits.detach()
                qry_acc = (qry_logits.argmax(dim=1) ==
                           y_qry).sum().item()/y_qry.size(0)
                qry_accs.append(qry_acc)
                simclr_accs_src.append(super_acc_src)
                simclr_accs_tgt.append(super_acc_tgt)

        mean_qry_acc = np.average(qry_accs)
        mean_simclr_acc_src = np.average(simclr_accs_src)
        mean_simclr_acc_tgt = np.average(simclr_accs_tgt)
        return mean_qry_acc, mean_simclr_acc_src, mean_simclr_acc_tgt

    def test(self, task_batch, inner_lr, n_inner_loop):
        """Meta test meta model. The meta model solves every tasks in
        task_batch.

        Args:
            task_batch (list) : list of tasks to solve.
            inner_lr (float) : learning rate to use in inner optimization.
            n_inner_loop (float) : number of inner loops.
        Returns:
            float : average classification accuracy of trained models on query
                    images.
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
                    spt_logits, _ = fnet(x_spt)
                    clsf_loss = F.cross_entropy(spt_logits, y_spt)
                    diffopt.step(clsf_loss)

                qry_logits = fnet(x_qry)[0].detach()
                qry_acc = (qry_logits.argmax(dim=1) ==
                           y_qry).sum().item()/y_qry.size(0)
                matrix = confusion_matrix(
                    y_qry.cpu(), qry_logits.cpu().argmax(dim=1))
                qry_accs.append(qry_acc)
                matrices.append(matrix)

        return np.average(qry_accs)
