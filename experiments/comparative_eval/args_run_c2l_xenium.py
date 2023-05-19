import sys, ast, os
from pathlib import Path
import time
import pickle
import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import os
from plotnine import *
import matplotlib.pyplot as plt
import matplotlib
import scipy
import seaborn as sns
import itertools
import scvi
import scanpy as sc
import squidpy as sq
import argparse

#######
####### Get settings from argparse
parser = argparse.ArgumentParser(description='Script for running C2LModels on Xenium')
parser.add_argument('model_name', type=str, help='Model name')
args = parser.parse_args()
print('Argument values:')
print(args.model_name)

if __name__ == '__main__':
    data_type = 'float32'

    import gnnc2l

    # scanpy prints a lot of warnings
    import warnings
    warnings.filterwarnings('ignore')

    # For GNN models
    import torch
    import torch_geometric

    # Load our utilities
    from data import preprocess_data
    from celltype_signatures import get_reference_celltype_signatures
    from model_resolver import model_resolver
    from metrics_mpoa import compute_metrics

    ## Setup paths
    sp_data_folder = 'assets_for_quantitative_results/xenium/'
    results_folder = 'results'
    metrics_folder = os.path.join(results_folder, 'metrics_xenium')
    # create paths and names to results folders for reference regression and cell2location models
    ref_run_name = os.path.join(results_folder, "reference_signatures")
    run_name = os.path.join(results_folder, "cell2location_map")
    Path(ref_run_name).mkdir(parents=True, exist_ok=True)
    Path(run_name).mkdir(parents=True, exist_ok=True)
    Path(metrics_folder).mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    num_c2l_epochs = 25000

    for random_seed in range(1, 6):

        # Results file name
        results_fh = f"{model_name}-epochs-{num_c2l_epochs}-{random_seed}.pkl"
        results_fh = os.path.join(metrics_folder, results_fh)
        #Check that results for this haven't been computed already
        if os.path.exists(results_fh):
            print(f"Learned this file already: {results_fh}")
            continue

        # Step 1
        # Read in Synthetic dataset and snrna dataset and do standard preprocessing1
        sp_data_folder = 'assets_for_quantitative_results/xenium/'
        adata_xenium = anndata.read(f'{sp_data_folder}xenium_convolved.h5ad')
        adata_xenium.obsm['X_spatial'] = adata_xenium.obsm['spatial'][:,:2]
        adata_snrna_raw = anndata.read(f'{sp_data_folder}xenium_processed.h5ad')

        # Step 2
        # Obtain reference cell type signatures
        adata_xenium.uns['ct_names'] = ["B cells", "Breast cancer", "Breast grandular cells", "Breast myoepithelial cells", "Endothelial cells", "Fibroblasts", "Macrophages", "T cells"]
        adata_xenium.uns['gexp_signatures'] = pd.DataFrame(adata_xenium.varm['mean_counts'], columns = adata_xenium.uns['ct_names'])
        inf_aver_xenium = adata_xenium.uns['gexp_signatures']

        # Step 3 
        # Combine/intersect sc reference data and spatial data for input into c2l-x model
        # find shared genes and subset both anndata and reference signatures
        inf_aver_xenium.index = adata_xenium.var_names.array
        intersect = np.intersect1d(adata_xenium.var_names, inf_aver_xenium.index)
        adata_vis = adata_xenium[:, intersect].copy()
        inf_aver_xenium = inf_aver_xenium.loc[intersect, :].copy()
        inf_aver_xenium # This should now be a pd with shape gene x celltypes (12078 x 49)

        # # Step 4 Instantiate and train C2L-X model
        mod, adata_vis, cell_abundance_site, nameprefix = model_resolver(model_name, 
                                                            adata_vis=adata_vis, 
                                                            cell_state_df=inf_aver_xenium,
                                                            N_cells_per_location=40,
                                                            detection_alpha=200,
                                                            dataset = "xenium")
        mod.train(
            max_epochs=num_c2l_epochs,
            # train using full data (batch_size=None)
            batch_size=None,
            # use all data points in training because
            # we need to estimate cell abundance at all locations
            train_size=1,
        )

        # Step 5 Compute all metrics # need new metrics for MPOA and Xenium 
        # (due to no ULCA, RHCA, etc. in the adata.uns['design'])
        metrics = compute_metrics(model=mod, 
            adata_vis=adata_vis,
            sitename=cell_abundance_site,
            nameprefix=nameprefix)
        metrics["cell_abundance_site"] = cell_abundance_site
        metrics["nameprefix"] = nameprefix
        metrics["adata_vis"] = adata_vis # Optionally save the adata_vis, note that this is several gigs

        # Save the metrics 
        ####
        with open(results_fh, 'wb') as f:
            pickle.dump(metrics, f)