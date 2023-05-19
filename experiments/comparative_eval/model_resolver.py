"""This module contains utilities for instantiating different
c2l-x model variants
"""
# For GNN models
import torch
import torch_geometric
import gnnc2l
import squidpy as sq
import numpy as np
from gnnc2l.models import Cell2location

def model_resolver(model_name, 
    adata_vis,
    cell_state_df,
    N_cells_per_location=9,
    detection_alpha=200,
    dataset='c2l'):
    """Utility to construct the intended C2L model variant

    Args:
        model_name (str): name of desired model
        adata_vis (anndata): spatial rna-seq data
        cell_state-df (pd.DataFrame): reference cell type signatures
        N_cells_per_location (int): C2L hyperparameter, very important to system
        detection_alpha (int): C2L hyperparameter
    """

    # prepare anndata for cell2location model
    if dataset == "c2l":
        adata_vis = adata_vis[adata_vis.obs['sample'] == 'exper0',:].copy()
        gnnc2l.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")
    if dataset == "mpoa":
        gnnc2l.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="Bregma")
    if dataset == "xenium":
        gnnc2l.models.Cell2location.setup_anndata(adata=adata_vis, batch_key=None)

    # Setup graph data (ignored by non_graph methods anyway)
    sq.gr.spatial_neighbors(adata_vis, spatial_key='X_spatial', n_neighs=8, coord_type='generic')
    adj = torch.tensor(adata_vis.obsp['spatial_connectivities'].toarray())
    edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
    edge_index, edge_attr = torch_geometric.utils.add_self_loops(edge_index, edge_attr)
    edge_attr = edge_attr[:, None]
    # dists = adata_vis.obsp['spatial_distances'].toarray()
    pos = adata_vis.obsm['X_spatial']
    pos_norm = pos / np.max(pos, axis=0)

    ##################
    # MODEL RESOLVER #
    ##################
    if model_name == "Cell2Location":
        mod = Cell2location(
            adata_vis, cell_state_df=cell_state_df,
            # the expected average cell abundance: tissue-dependent
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection (using default here):
            detection_alpha=detection_alpha
        )
        cell_abundance_site = "w_sf"
        nameprefix = "cell_abundance"

    elif model_name == "MLP_ASF":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroMLP', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "MLP_ASF_2Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroMLP_2Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "MLP_ASF_3Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroMLP_3Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GCNModel_ASF":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroGCNModel', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GCNModel_ASF_2Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroGCNModel_2Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"


    elif model_name == "GCNModel_ASF_3Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroGCNModel_3Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GCNModel_ASF_4Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroGCNModel_4Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GCNModel_ASF_5Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroGCNModel_5Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GCNModel_ASF_6Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'PyroGCNModel_6Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        )
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GAT_ASF_PyG":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'GATAModel', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GAT_ASF_PyG_2Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'GATAModel_2Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GAT_ASF_PyG_3Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'GATAModel_3Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GAT_ASF_PyG_4Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'GATAModel_4Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GAT_ASF_PyG_5Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'GATAModel_5Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "GAT_ASF_PyG_6Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'GATAModel_6Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "INV_ASF":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'InvariantMPNN', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "INV_ASF_2Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'InvariantMPNN_2Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "INV_ASF_3Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'InvariantMPNN_3Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "EQUI_ASF":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'EquivariantMPNN', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "EQUI_ASF_2Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'EquivariantMPNN_2Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    elif model_name == "EQUI_ASF_3Layer":
        mod = gnnc2l.models.Cell2location(
            adata_vis, cell_state_df=cell_state_df, 
            model_class = gnnc2l.models._cell2location_GNNDirect_module.DirectGNNLocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
            # the expected average cell abundance: tissue-dependent 
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=N_cells_per_location,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=detection_alpha,
            # Adjacency matrix for GNN
            gnn_type = 'EquivariantMPNN_3Layer', # 'EquivariantMPNN',
            edge_index=torch.tensor(edge_index).cuda(),
            edge_attr=torch.tensor(edge_attr).float().cuda(),
            pos=torch.tensor(pos_norm).float().cuda(),
            # Dropout to regularise
        #     dropout_p=None
        ) 
        cell_abundance_site = "a_sf"
        nameprefix = "gnn_cell_abundance"

    # """
    # elif model == "YOUR MODEL HERE":
    #     mod = gnnc2l.models.yourmodel
    #     cell_abundance_site = "w_sf", "a_sf", or your own latent variable representing cellabundance
    #     nameprefix = "cell_abundance"
    # """
    
    else:
        raise ValueError(f"Model name: {model_name} not implemented")

    return mod, adata_vis, cell_abundance_site, nameprefix