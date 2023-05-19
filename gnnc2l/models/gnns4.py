import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
import torch_geometric
from torch_geometric.nn import MessagePassing, global_mean_pool, GATv2Conv, GCNConv
from torch_scatter import scatter
import pyro.distributions as dist

import collections
from typing import Iterable

import torch
from scvi.nn._utils import one_hot
from torch import nn as nn
from pyro.nn.module import PyroModule, PyroSample
from gnnc2l.models.gnns import *

# PyroGCN
class PyroGCNModel_4Layer(PyroModule):
    """PyroModule with sampled weights and biases"""
    def __init__(self, input_dim=11, output_dim=1, emb_dim=64, adj=None, activation_fn=None, bias=True):
        super(PyroGCNModel_4Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.adj = adj
        self.activation_fn = activation_fn
        self.bias = bias

        A = adj
        A_tilde = A + torch.eye(A.shape[0]).to(A.device)
        D_tilde = torch.diag(torch.sum(A_tilde, axis=1))
        D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
        D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
        A_tilde = A_tilde.to_sparse()
        D_tilde_inv_sqrt = D_tilde_inv_sqrt.to_sparse()
        self.adj_norm = torch.sparse.mm(torch.sparse.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)

        # + Simple linear transformation and non-linear activation
        self.linear = PyroModule[nn.Linear](input_dim, emb_dim)
        # self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([emb_dim, input_dim]).to_event(2))
        # self.linear.bias = PyroSample(dist.Normal(0., 1.).expand([emb_dim]).to_event(1))

        self.linear2 = PyroModule[nn.Linear](emb_dim, emb_dim)
        self.linear3 = PyroModule[nn.Linear](emb_dim, emb_dim)
        self.linear4 = PyroModule[nn.Linear](emb_dim, emb_dim)

        self.linearout = PyroModule[nn.Linear](emb_dim, output_dim)
        # self.linearout.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, emb_dim]).to_event(2))
        # self.linearout.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

    def forward(self, data):
        x = data.x
        
        # First Layer
        x = torch.sparse.mm(self.adj_norm, x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        # 2nd Layer
        x = torch.sparse.mm(self.adj_norm, x)
        x = self.linear2(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        # 3rdd Layer
        x = torch.sparse.mm(self.adj_norm, x)
        x = self.linear3(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        # 4th Layer
        x = torch.sparse.mm(self.adj_norm, x)
        x = self.linear4(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        # Output
        x = self.linearout(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

class GATAModel_4Layer(PyroModule):
    """PyroModule for GAT model that is part of sampling asf
    
       Uses the dynamic attention Gatv2Conv
    """
    def __init__(self, input_dim, output_dim, emb_dim=64, adj=None, edge_dim=None, activation_fn=None, bias=True):
        super(GATAModel_4Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.adj = adj
        self.activation_fn = activation_fn
        self.bias = bias
        self.edge_dim = edge_dim
        
        # initialise GAT Layer
        self.conv = PyroModule[GATv2Conv](in_channels=input_dim, out_channels=emb_dim, heads=4, concat=True, dropout=0.2, add_self_loops=True, edge_dim=edge_dim, bias=bias)

        # 2nd Layer
        self.conv2 = PyroModule[GATv2Conv](in_channels=4*emb_dim, out_channels=emb_dim, heads=4, concat=True, dropout=0.2, add_self_loops=True, edge_dim=edge_dim, bias=bias)

        # 3rd Layer
        self.conv3 = PyroModule[GATv2Conv](in_channels=4*emb_dim, out_channels=emb_dim, heads=4, concat=True, dropout=0.2, add_self_loops=True, edge_dim=edge_dim, bias=bias)

        # 3rd Layer
        self.conv4 = PyroModule[GATv2Conv](in_channels=4*emb_dim, out_channels=emb_dim, heads=4, concat=True, dropout=0.2, add_self_loops=True, edge_dim=edge_dim, bias=bias)


        # linear out
        self.linear = PyroModule[nn.Linear](4*emb_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x, attention_weights = self.conv(x, data.edge_index, return_attention_weights=True)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        x, attention_weights = self.conv2(x, data.edge_index, return_attention_weights=True)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        x, attention_weights = self.conv3(x, data.edge_index, return_attention_weights=True)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        x, attention_weights = self.conv4(x, data.edge_index, return_attention_weights=True)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        self.attention_weights = attention_weights

        return x
