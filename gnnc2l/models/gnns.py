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


###########################
#       GNN Layers        #
###########################
class GNNLayers(nn.Module):
    """
    A helper class to build GNN layers for a neural network.
    Dropout is performed on input rather than on output.
    Adapted with modifications from scvi-tools:
    Copyright (c) 2020 Romain Lopez, Adam Gayoso, Galen Xing, Yosef Lab
    All rights reserved.
    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        edge_index = None,
        edge_attr = None,
        pos = None,
        gnn_type = 'EquivariantMPNN',
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 32,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
        **gnn_kwargs
    ):
        super().__init__()
        
        # print('Using GNNLayers')
        # print('edge_index', edge_index)
        # print(gnn_kwargs)
        # print(n_layers, n_hidden)
        # print('Out dim: ', n_out)
        self.inject_covariates = inject_covariates
        assert edge_index is not None
        
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
            
        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                            GNNLayerWrapper(n_in + cat_dim * self.inject_into_layer(i),
                                            n_out,
                                            edge_index=edge_index,
                                            edge_attr=edge_attr,
                                            pos=pos, 
                                            gnn_type=gnn_type,
                                            bias=bias,
                                            **gnn_kwargs),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn() if use_activation else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )
        """
        self.fc_layers = PyroModule[nn.Sequential](
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        PyroModule[nn.Sequential](
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                            PyroModule[GNNLayerWrapper](n_in + cat_dim * self.inject_into_layer(i),
                                            n_out,
                                            edge_index=edge_index,
                                            edge_attr=edge_attr,
                                            pos=pos, 
                                            gnn_type=gnn_type,
                                            bias=bias,
                                            **gnn_kwargs),
                            # non-default params come from defaults in original Tensorflow implementation
                            PyroModule[nn.BatchNorm1d](n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            PyroModule[nn.LayerNorm](n_out, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn() if use_activation else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )
        """

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1))) for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


###########################
#   GNN Layer wrapper     #
###########################
class GNNLayersPyro(GNNLayers, PyroModule):
    pass

class GNNLayerWrapper(nn.Module):
    def __init__(self, n_in, n_out, edge_index, edge_attr=None, pos=None, gnn_type='EquivariantMPNN', bias=True, **gnn_kwargs):
        super().__init__()
        self.gnn_type = gnn_type
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        
        if gnn_type == 'GCN':
            adj = torch_geometric.utils.to_dense_adj(edge_index)[0]  # Function to_dense_adj returns single-batched dense adj
            # print('Adj shape: ', adj.shape)
            self.gnn_layer = PyroGCNModel(input_dim=n_in, output_dim=n_out, adj=adj)  # PyroModule[GCNLayer]
        elif gnn_type == 'InvariantMPNN':
            self.linear_layer = PyroModule[nn.Linear](n_in, n_out, bias=bias)  # Inserting linear layer to match layer interface
            self.gnn_layer = PyroModule[InvariantMPNNLayer](emb_dim=n_out, edge_dim=edge_attr.shape[1], bias=bias, **gnn_kwargs)
        elif gnn_type == 'EquivariantMPNN':
            self.linear_layer = PyroModule[nn.Linear](n_in, n_out, bias=bias) # Inserting linear layer to match layer interface
            self.gnn_layer = PyroModule[EquivariantMPNNLayer](emb_dim=n_out, edge_dim=edge_attr.shape[1], bias=bias, **gnn_kwargs)
        else:
            raise ValueError('GNN not recognised')
    
    def forward(self, x):
        # print('Forward of GNN!')
        # print('Input shape: ', x.shape)
        if self.gnn_type == 'GCN':
            return self.gnn_layer(x)
        else:
            h = self.linear_layer(x)
            return self.gnn_layer(h, self.pos.cuda(), self.edge_index.cuda(), self.edge_attr.cuda())


###########################
#       GNN Layers        #
###########################

# GCN
class GCNLayer(nn.Module):
    """GCN layer to be implemented by students of practical

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A, activation_fn=None, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A
        self.activation_fn = activation_fn

        A_tilde = A + torch.eye(A.shape[0]).to(A.device)
        D_tilde = torch.diag(torch.sum(A_tilde, axis=1))
        D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
        D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
        A_tilde = A_tilde.to_sparse()
        D_tilde_inv_sqrt = D_tilde_inv_sqrt.to_sparse()
        self.adj_norm = torch.sparse.mm(torch.sparse.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)

        # + Simple linear transformation and non-linear activation
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        x = torch.sparse.mm(self.adj_norm, x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


# MPNN    
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add', bias=True):
        """Message Passing Neural Network Layer

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim, bias=bias), BatchNorm1d(emb_dim), ReLU()
          )

    def forward(self, h, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        As our MPNNLayer class inherits from the PyG MessagePassing parent class,
        we simply need to call the `propagate()` function which starts the 
        message passing procedure: `message()` -> `aggregate()` -> `update()`.
        
        The MessagePassing class handles most of the logic for the implementation.
        To build custom GNNs, we only need to define our own `message()`, 
        `aggregate()`, and `update()` functions (defined subsequently).

        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """Step (1) Message

        The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take 
        any arguments that were initially passed to `propagate`. Additionally, 
        we can differentiate destination nodes and source nodes by appending 
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`. 
        
        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features
        
        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        """Step (2) Aggregate

        The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        """
        Step (3) Update

        The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        `update()` takes the first argument `aggr_out`, the result of `aggregate()`, 
        as well as any optional arguments that were initially passed to 
        `propagate()`. E.g. in this case, we additionally pass `h`.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


# Invariant MPNN
class InvariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add', bias=True):
        """Message Passing Neural Network Layer

        This layer is invariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # ============ YOUR CODE HERE ==============
        # MLP `\psi` for computing messages `m_ij`
        # dims: (???) -> d
        #
        # self.mlp_msg = Sequential(...)
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim + 1, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )
        # ==========================================
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # dims: 2d -> d
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim, bias=bias), BatchNorm1d(emb_dim), ReLU()
          )

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        # ============ YOUR CODE HERE ==============
        # Notice that the `forward()` function has a new argument 
        # `pos` denoting the initial node coordinates. Your task is
        # to update the `propagate()` function in order to pass `pos`
        # to the `message()` function along with the other arguments.
        #
        # out = self.propagate(...)
        # return out
        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out
        # ==========================================

    # ============ YOUR CODE HERE ==============
    # Write a custom `message()` function that takes as arguments the
    # source and destination node features, node coordiantes, and `edge_attr`.
    # Incorporate the coordinates `pos` into the message computation such
    # that the messages are invariant to rotations and translations.
    # This will ensure that the overall layer is also invariant.
    #
    # def message(self, ...):
    # """The `message()` function constructs messages from source nodes j 
    #    to destination nodes i for each edge (i, j) in `edge_index`.
    #
    #    Args:
    #        ...
    #    
    #    Returns:
    #        ...
    # """
    #   ...  
    #   msg = ...
    #   return self.mlp_msg(msg)
    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            pos_i: (e, 3) - destination node coordinates
            pos_j: (e, 3) - source node coordinates
            edge_attr: (e, d_e) - edge features
        
        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        dists = torch.norm(pos_i - pos_j, dim=-1).unsqueeze(1)
        msg = torch.cat([h_i, h_j, edge_attr, dists], dim=-1)
        return self.mlp_msg(msg)
    # ==========================================
    
    def aggregate(self, inputs, index):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        """The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


# Equivariant MPNN
class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add', bias=True):
        """Message Passing Neural Network Layer

        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # ============ YOUR CODE HERE ==============
        # Define the MLPs constituting your new layer.
        # At the least, you will need `\psi` and `\phi` 
        # (but their definitions may be different from what
        # we used previously).
        #
        # self.mlp_msg = ...  # MLP `\psi`
        # self.mlp_upd = ...  # MLP `\phi`
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim + 1, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )
        
        # MLP `\psi` for computing messages `m_ij`
        # dims: d -> 1
        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, 1)
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # dims: 2d -> d
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim, bias=bias), BatchNorm1d(emb_dim), ReLU()
          )
        # ===========================================

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        # ============ YOUR CODE HERE ==============
        # Notice that the `forward()` function has a new argument 
        # `pos` denoting the initial node coordinates. Your task is
        # to update the `propagate()` function in order to pass `pos`
        # to the `message()` function along with the other arguments.
        #
        # out = self.propagate(...)
        # return out
        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out
        # ==========================================

    # ============ YOUR CODE HERE ==============
    # Write custom `message()`, `aggregate()`, and `update()` functions
    # which ensure that the layer is 3D rotation and translation equivariant.
    #
    # def message(self, ...):
    #   ...  
    #
    # def aggregate(self, ...):
    #   ...
    #
    # def update(self, ...):
    #   ...
    #
    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            pos_i: (e, 3) - destination node coordinates
            pos_j: (e, 3) - source node coordinates
            edge_attr: (e, d_e) - edge features
        
        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        # Compute messages
        pos_diff = pos_i - pos_j
        radial = torch.sum(pos_diff**2, 1).unsqueeze(1)
        msg = torch.cat([h_i, h_j, radial, edge_attr], dim=-1)
        msg = self.mlp_msg(msg)

        # Compute pos updates
        pos_updates = pos_diff * self.mlp_pos(msg) # torch.clamp(updates, min=-100, max=100)

        return msg, pos_updates
    
    def aggregate(self, inputs, index):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: [(e, d),(e,3)] - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        msgs, pos_updates = inputs

        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)

        # Aggregate position updates
        pos_aggr = scatter(pos_updates, index, dim=self.node_dim, reduce='mean')

        return msg_aggr, pos_aggr
    
    def update(self, aggr_out, h, pos):
        """The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        Args:
            aggr_out: [(n, d),(n, 3)] - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        msg_aggr, pos_aggr = aggr_out
        upd_out = torch.cat([h, msg_aggr], dim=-1)
        pos_update = pos + pos_aggr
        return self.mlp_upd(upd_out), pos_update
    # ==========================================

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


###########################
#       GNN Models        #
###########################

# PyroMLP
class PyroMLP(PyroModule):
    """PyroMLP with sampled weights and biases"""
    def __init__(self, input_dim=11, output_dim=1, emb_dim=64, adj=None, activation_fn=None, bias=True):
        super(PyroMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.adj = adj # Technically unnecessary and will not be used, but easier for keeping API
        self.activation_fn = activation_fn
        self.bias = bias

        # + Simple linear transformation and non-linear activation
        self.linear = PyroModule[nn.Linear](input_dim, emb_dim)
        # self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([emb_dim, input_dim]).to_event(2))
        # self.linear.bias = PyroSample(dist.Normal(0., 1.).expand([emb_dim]).to_event(1))

        self.linearout = PyroModule[nn.Linear](emb_dim, output_dim)
        # self.linearout.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, emb_dim]).to_event(2))
        # self.linearout.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

    def forward(self, data):
        x = data.x
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.linearout(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x

# PyroGCN
class PyroGCNModel(PyroModule):
    """PyroModule with sampled weights and biases"""
    def __init__(self, input_dim=11, output_dim=1, emb_dim=64, adj=None, activation_fn=None, bias=True):
        super(PyroGCNModel, self).__init__()
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

        self.linearout = PyroModule[nn.Linear](emb_dim, output_dim)
        # self.linearout.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, emb_dim]).to_event(2))
        # self.linearout.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

    def forward(self, data):
        x = data.x
        x = torch.sparse.mm(self.adj_norm, x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.linearout(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x
class PyGPyroGCNModel(PyroModule):
    """PyroModule for GCN model to reimplement PyroGCNModel
    """
    def __init__(self, input_dim, output_dim, emb_dim=64, adj=None, activation_fn=None, bias=True):
        super(PyGPyroGCNModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.adj = adj
        self.activation_fn = activation_fn
        self.bias = bias
        
        # initialise GAT Layer
        self.conv = PyroModule[GCNConv](in_channels=input_dim, out_channels=emb_dim, add_self_loops=True, bias=bias)

        # linear out
        self.linear = PyroModule[nn.Linear](emb_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv(x, data.edge_index)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

class GATAModel(PyroModule):
    """PyroModule for GAT model that is part of sampling asf
    
       Uses the dynamic attention Gatv2Conv
    """
    def __init__(self, input_dim, output_dim, emb_dim=64, adj=None, edge_dim=None, activation_fn=None, bias=True):
        super(GATAModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.adj = adj
        self.activation_fn = activation_fn
        self.bias = bias
        self.edge_dim = edge_dim
        
        # initialise GAT Layer
        self.conv = PyroModule[GATv2Conv](in_channels=input_dim, out_channels=emb_dim, heads=4, concat=True, dropout=0.2, add_self_loops=True, edge_dim=edge_dim, bias=bias)

        # linear out
        self.linear = PyroModule[nn.Linear](4*emb_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x, attention_weights = self.conv(x, data.edge_index, return_attention_weights=True)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        self.attention_weights = attention_weights

        return x

class InvariantASFModel(PyroModule):
    """PyroModule for Invariant MPNN designed for modelling A_SF (or rather modulating it)

    """
    def __init__(self, input_dim, output_dim, edge_dim, emb_dim=64, activation_fn=None, bias=True, **kwargs):
        super(InvariantASFModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.emb_dim = emb_dim
        self.activation_fn = activation_fn
        self.bias = bias

        # linear in
        self.lin_in = PyroModule[nn.Linear](input_dim, emb_dim)

        # initialise the invariant mpnn layer
        self.conv = PyroModule[InvariantMPNNLayer](emb_dim, edge_dim, aggr='add')

        # linear out
        self.linear = PyroModule[nn.Linear](emb_dim, output_dim)

    def forward(self, data):
        x = data.x
        x = self.lin_in(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.conv(x, data.pos, data.edge_index, data.edge_attr)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

class EquivariantASFModel(PyroModule):
    """PyroModule for Equivariant MPNN designed for modelling A_SF (or rather modulating it)

    """
    def __init__(self, input_dim, output_dim, edge_dim, emb_dim=64, activation_fn=None, bias=True, **kwargs):
        super(EquivariantASFModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.emb_dim = emb_dim
        self.activation_fn = activation_fn
        self.bias = bias

        # linear in
        self.lin_in = PyroModule[nn.Linear](input_dim, emb_dim)

        # initialise the invariant mpnn layer
        self.conv = PyroModule[EquivariantMPNNLayer](emb_dim, edge_dim, aggr='add')

        # linear out
        self.linear = PyroModule[nn.Linear](emb_dim, output_dim)

    def forward(self, data):
        x = data.x
        x = self.lin_in(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x, pos_update = self.conv(x, data.pos, data.edge_index, data.edge_attr)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x

# GCN
class GCNModel(Module):
    def __init__(self, input_dim=11, output_dim=1, num_layers=2, emb_dim=64, adj=None, activation_fn=None, bias=True):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        self.activation_fn = activation_fn
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(input_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCNLayer(emb_dim, emb_dim, A=adj, activation_fn=torch.nn.ReLU()))
            
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool
        
        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, output_dim, bias=bias)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h) # (n, d) -> (n, d)
            # Note that we add a residual connection after each layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
        
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out.view(-1)

# MPNN
class MPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1, activation_fn=None, bias=True):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        self.activation_fn = activation_fn
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim, bias=bias)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
        
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out.view(-1)

# Invariant MPNN
class InvariantMPNNModel(MPNNModel):
    def __init__(self, num_layers=2, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1, activation_fn=None, bias=True, **kwargs):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations.

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        self.activation_fn = activation_fn
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of invariant MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(InvariantMPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim, bias=bias)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h, data.pos, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
              
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out.view(-1)

    
# Equivariant MPNN
class EquivariantMPNNModel(MPNNModel):
    def __init__(self, num_layers=2, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1, activation_fn=None, bias=True, **kwargs):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations (the constituent MPNN layers
        are equivariant to 3D rotations and translations).

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        self.activation_fn = activation_fn
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EquivariantMPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim, bias=bias)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        pos = data.pos
        
        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, data.edge_index, data.edge_attr)
            
            # Update node features
            h = h + h_update # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer
            
            # Update node coordinates
            pos = pos_update # (n, 3) -> (n, 3)

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
        
        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out.view(-1)
