"""Module containing meta train & meta test methods."""

import numpy as np
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import higher


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
        qry_rot_accs = []

        for task in task_batch:
            x_spt, x_qry, y_spt, y_qry, z_spt, z_qry = task
            x_spt, y_spt, z_spt = x_spt.cuda(), y_spt.cuda(), z_spt.cuda()
            x_qry, y_qry, z_qry = x_qry.cuda(), y_qry.cuda(), z_qry.cuda()

            with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                for _ in range(n_inner_loop):
                    spt_logits, spt_super = fnet(x_spt)
                    clsf_loss = F.cross_entropy(spt_logits, y_spt)
                    super_loss = F.cross_entropy(spt_super, z_spt)
                    spt_loss = clsf_loss + self.lamb*super_loss
                    diffopt.step(spt_loss)

                qry_logits, qry_super = fnet(x_qry)
                clsf_loss = F.cross_entropy(qry_logits, y_qry)
                super_loss = F.cross_entropy(qry_super, z_qry)
                qry_loss = clsf_loss + self.lamb*super_loss
                qry_loss.backward()

                qry_logits, qry_super = fnet(x_qry)
                qry_logits = qry_logits.detach()
                qry_super = qry_super.detach()
                qry_acc = (qry_logits.argmax(dim=1) ==
                           y_qry).sum().item()/y_qry.size(0)
                qry_rot_acc = (qry_super.argmax(dim=1) ==
                               z_qry).sum().item()/y_qry.size(0)
                qry_accs.append(qry_acc)
                qry_rot_accs.append(qry_rot_acc)

        mean_qry_acc = np.average(qry_accs)
        mean_qry_rot_acc = np.average(qry_rot_accs)
        return mean_qry_acc, mean_qry_rot_acc

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
            x_spt, x_qry, y_spt, y_qry, z_spt, z_qry = task
            x_spt, y_spt, z_spt = x_spt.cuda(), y_spt.cuda(), z_spt.cuda()
            x_qry, y_qry, z_qry = x_qry.cuda(), y_qry.cuda(), z_qry.cuda()

            with higher.innerloop_ctx(self.net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                for _ in range(n_inner_loop):
                    spt_logits, spt_super = fnet(x_spt)
                    clsf_loss = F.cross_entropy(spt_logits, y_spt)
                    super_loss = F.cross_entropy(spt_super, z_spt)
                    spt_loss = clsf_loss + self.lamb*super_loss
                    diffopt.step(spt_loss)

                qry_logits = fnet(x_qry)[0].detach()
                qry_acc = (qry_logits.argmax(dim=1) ==
                           y_qry).sum().cpu()/y_qry.size(0)
                matrix = confusion_matrix(
                    y_qry.cpu(), qry_logits.cpu().argmax(dim=1))
                qry_accs.append(qry_acc)
                matrices.append(matrix)

        if return_matrix:
            return np.average(qry_accs), np.average(matrices, axis=0)
        return np.average(qry_accs)
