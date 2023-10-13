# Copyright 2019 Indro Spinelli. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import dgl
import time
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils import *
from models import *

#torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class GSAVES(object):

    """
    Impute with GINN!
    Functions:
        __init__
        add_data()
        fit()
        fit_transform()
        transform()
    """

    def __init__(
        self,
        features,
        mask,
        nb_sensors,
        embedding_dim=512,
        activation=F.relu,
        dropout=True,
):
        """
            Build the graph-structure of the dataset based on the similarity
            Instantiate the network based on the graph using the dgl library
        """
        self.features = features.T
        self.mask = mask.T
        self.nb_sensors=nb_sensors
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.dropout = dropout


        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")#
        print("GSAVES is running on", self.device)

        maskT = torch.FloatTensor(self.mask).to(self.device)

        nxg = merl_dataset2nxg(self.nb_sensors)
        g = dgl.DGLGraph()
        g.set_e_initializer(dgl.init.zero_initializer)
        g.set_n_initializer(dgl.init.zero_initializer)

        g=dgl.from_networkx(nxg, edge_attrs=['weight'])
        g.edata['weight'] = g.edata['weight'].to(torch.float32)

        g=g.to(self.device)
        self.graph = g
        self.generator = gsaves_gcn(
            self.graph,
            maskT,
            self.features.shape[1],
            self.embedding_dim,
            self.activation,
            self.dropout,
        ).to(self.device)

        self.critic = gsaves_critic(self.features.shape[1], self.embedding_dim, 0.25).to(
            self.device
        )

        return

    def fit(
        self,
        epochs=10,
        batch_size=32,
        patience=10,
        auto_lr=5e-5,
        crit_lr=5e-5,
        crit_iter=5,
        weight_decay=0.1,
        adversarial=True,
        fine_tune=False,
        verbose=True,

    ):
        """
            Trains the network, if fine_tune=True uses the previous state of the optimizer 
            instantiated before.
        """

        self.optim_auto = torch.optim.Adam(self.generator.parameters(),
                                           lr=auto_lr, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=weight_decay, amsgrad=True)
        if adversarial:
            self.optim_crit = torch.optim.Adam(self.critic.parameters(),
                                               lr=crit_lr, betas=(0.9, 0.999),
                                               eps = 1e-08, weight_decay=weight_decay,amsgrad = True)

        if fine_tune:
            checkpoint = torch.load("ginn_2.pth")
            self.generator.load_state_dict(checkpoint["auto_state_dict"])
            self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])
            if adversarial:
                self.critic.load_state_dict(checkpoint["crit_state_dict"])
                self.optim_crit.load_state_dict(checkpoint["optim_crit_state_dict"])



        dur = []
        a_losses = []
        c_losses = []
        r_losses = []
        w_losses = []
        total_norm=[]
        total_norm_epoch=0
        t0 = time.time()

        featT = torch.FloatTensor(self.features).to(self.device)
        #maskT = torch.FloatTensor(self.mask).to(self.device)

        maskT = torch.BoolTensor(self.mask.astype(int)).to(self.device)
        mask_inv = (~self.mask.astype(bool)).astype(int)
        maskT_inv = torch.BoolTensor(mask_inv).to(self.device)
        #calculating bce loss weights
        total_ones=torch.count_nonzero(featT[maskT])
        total_zeros=torch.numel(featT[maskT])-total_ones
        weight_ones=total_zeros/total_ones
        bce_weight=torch.ones(torch.numel(featT[maskT])).to(self.device)
        bce_weight=torch.where(featT[maskT]==1,weight_ones,bce_weight)
        bse_criterion = nn.BCELoss(weight= bce_weight).to(self.device)
        for epoch in range(epochs + 1):

            self.generator.train()
            self.critic.train()
            iX = self.generator(featT)
            a_loss = bse_criterion(iX[maskT], featT[maskT])
            if adversarial:
                self.optim_auto.zero_grad()
                a_loss.backward()
                self.optim_auto.step()
                for _ in range(crit_iter):
                    iX = self.generator(featT)
                    x_real =torch.mul(featT, maskT)
                    x_fake = torch.mul(iX, maskT)
                    c_real = self.critic(x_real)
                    c_fake = self.critic(x_fake)
                    gp = hard_gradient_penalty(self.critic, x_real, x_fake, self.device)
                    w_loss = c_fake.mean() - c_real.mean()
                    c_loss = w_loss +10*gp
                    self.optim_crit.zero_grad()
                    c_loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                    self.optim_crit.step()

                iX = self.generator(featT)
                num_loss = bse_criterion(iX[maskT], featT[maskT])
                #ra_loss +=  num_loss
                x_fake =torch.mul(iX, maskT)
                rc_fake = self.critic(x_fake)
                print(rc_fake.mean())
                r_loss = -rc_fake.mean() +  num_loss
                self.optim_auto.zero_grad()
                r_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1)
                self.optim_auto.step()
                for p in self.generator.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm_epoch += param_norm.item() ** 2
                total_norm.append( total_norm_epoch ** 0.5)

            else:
                self.optim_auto.zero_grad()
                a_loss.backward()
                self.optim_auto.step()

            if epoch % 1 == 0:
                dur.append(time.time() - t0)

                a_losses.append(a_loss.detach().item())
                if adversarial:
                    c_losses.append(c_loss.detach().item())
                    r_losses.append(r_loss.detach().item())
                    w_losses.append(w_loss.detach().item())
                    if verbose:
                        print(
                            "EPOCH: %05d," % epoch,
                            "A_LOSS: %f," % a_loss.detach().item(),
                            "C_LOSS: %f," % c_loss.detach().item(),
                            "R_LOSS: %f," % r_loss.detach().item(),
                            "W_LOSS: %f " % w_loss.detach().item(),
                            "= (%f" % c_fake.mean().detach().item(),
                            "-(%f))" % c_real.mean().detach().item(),
                            "GP: %f" % gp.detach().item(),
                        )
                else:
                    if verbose:
                        print(
                            "EPOCH: %05d," % epoch,
                            "A_LOSS: %f" % a_loss.detach().item(),
                        )
        if adversarial:
            torch.save(
                {
                    "auto_state_dict": self.generator.state_dict(),
                    "optim_auto_state_dict": self.optim_auto.state_dict(),
                    "crit_state_dict": self.critic.state_dict(),
                    "optim_crit_state_dict": self.optim_crit.state_dict(),
                },
                "ginn_2.pth",
            )
        else:
            torch.save(
                {
                    "auto_state_dict": self.generator.state_dict(),
                    "optim_auto_state_dict": self.optim_auto.state_dict(),
                },
                "ginn_2.pth",
            )


        return  total_norm, a_losses, c_losses, r_losses, w_losses


    def transform(self, train=True):
        """
            Impute the missing values in the dataset
        """
        auto_lr = 5e-6
        crit_lr = 5e-6
        if not train:
            self.optim_auto = torch.optim.Adam(self.generator.parameters(),
                                               lr=auto_lr, betas=(0.9, 0.999),
                                               eps=1e-08, weight_decay=0, amsgrad=True)

            self.optim_crit = torch.optim.Adam(self.critic.parameters(),
                                               lr=crit_lr, betas=(0.9, 0.999),
                                               eps=1e-08, weight_decay=0, amsgrad=True)

            checkpoint = torch.load("ginn_2.pth")
            self.generator.load_state_dict(checkpoint["auto_state_dict"])
            self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])

            self.critic.load_state_dict(checkpoint["crit_state_dict"])
            self.optim_crit.load_state_dict(checkpoint["optim_crit_state_dict"])

        featT = torch.FloatTensor(self.features).to(self.device)
        imputed_data = imputation(self.generator, featT)
        filled_data = np.where(self.mask, featT.cpu(), imputed_data)
        return filled_data

    def transform_test(self, test_features, mask_te):
        """
            Impute the missing values in the test set
        """
        featT = torch.FloatTensor(test_features).to(self.device)
        maskT = torch.FloatTensor(mask_te).to(self.device)
        self.generator.mask = maskT
        imputed_data = imputation(self.generator, featT)
        filled_data = np.where(mask_te, test_features, imputed_data)

        return filled_data
