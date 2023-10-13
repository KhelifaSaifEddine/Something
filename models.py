import math
import torch
import numpy as np
import torch.nn as nn
from dgl import DGLGraph
import pandas as pd
from dgl.nn import GraphConv, EdgeWeightNorm
import dgl.function as fn
import torch.nn.functional as F
import torch.autograd as autograd
from utils import  batch_mask
from dgl.data import register_data_args, load_data
import warnings
import torch.onnx

warnings.filterwarnings("ignore")

class gsaves_gcn(nn.Module):
    def __init__(self, g, mask, in_feats, h_feats, activation, dropout):
        super(gsaves_gcn, self).__init__()
        self.g=g
        self.mask = mask
        edge_weight=g.edata['weight']
        edge_weight = edge_weight.to(torch.device('cuda'))
        norm = EdgeWeightNorm(norm='both')
        self.norm_edge_weight = norm(g, edge_weight)
        self.masked_gcn = GraphConv( in_feats, h_feats, norm='none', weight=True, bias=True, activation=activation)
        self.middle_gcn = GraphConv(h_feats, h_feats, norm='both', bias=True, activation=activation)
        self.output_gcn = GraphConv( h_feats, in_feats, norm='both',activation=torch.sigmoid)

    def forward(self, features):
        features = torch.mul(features, self.mask)
        h = self.masked_gcn(self.g,features,edge_weight=self.norm_edge_weight)
        h = self.middle_gcn(self.g, h)
        h = self.output_gcn(self.g,h)
        return h

class gcn_unweighted(nn.Module):
    def __init__(self, g, mask, in_feats, h_feats, activation, dropout):
        super(gcn_unweighted, self).__init__()
        self.g=g
        self.mask = mask
        self.masked_gcn = GraphConv( in_feats, h_feats, norm='both', weight=True, bias=True, activation=activation)

        self.output_gcn = GraphConv( h_feats, in_feats)

    def forward(self, features):
        features = torch.mul(features, self.mask)
        h = self.masked_gcn(self.g,features)
        h = self.output_gcn(self.g,h)
        return h

class gsaves_critic(nn.Module):
    def __init__(self, in_feats, h_feats, dropout):
        super(gsaves_critic, self).__init__()

        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, in_feats)
        self.linear3 = nn.Linear(in_feats, 1)
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.0

    def forward(self, features):
        h = self.relu(self.linear1(features))
        if self.dropout:
            h = self.dropout(h)
        h = self.relu(self.linear2(h))
        if self.dropout:
            h = self.dropout(h)
        h = self.linear3(h)
        return h


def gradient_penalty(net, real_data, fake_data, device):

    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    c_interpolates = net(interpolates)
    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(c_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def hard_gradient_penalty(net, real_data, fake_data, device):

    mask = torch.FloatTensor(real_data.shape).to(device).uniform_() > 0.5
    inv_mask = ~mask
    mask, inv_mask = mask.float(), inv_mask.float()

    interpolates = mask * real_data + inv_mask * fake_data
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    c_interpolates = net(interpolates)

    gradients = autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(c_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = (gradients.norm(2, dim=1) - 1).pow(2).mean()
    return gp


def imputation(model, features):
    model.eval()
    torch.onnx.export(
        model,  # Model to export
        features,    # Sample input tensor to determine the model's structure
        onnx_file_path,  # Path to save the ONNX file
        verbose=True,     # Print information about export
        input_names=["input"],  # Names for input tensor
        output_names=["output"],  # Names for output tensor
    )
    with torch.no_grad():
        imp = model(features)
        imp = imp.cpu()
    return imp


