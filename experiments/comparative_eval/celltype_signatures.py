"""This module contains the code for obtaining the celltype reference signatures
this can happen either by 
- learning over the scrna data and inferring the signatures,
- loading the most recently learned ones
- passing in a specific file to load and process 
"""
import scipy
import numpy as np
import scanpy as sc
import gnnc2l
from gnnc2l.models import RegressionModel

def get_reference_celltype_signatures(ref_run_name=None, learn=False, file_handle=None, save=False, adata_snrna_raw=None, batch_key="sample", labels_key='annotation_1'):
    """Return the reference celltype signatures either by
    - learning over the scrna data and inferring the signatures,
    - loading the most recently learned ones
    - passing in a specific file to load and process

    Args:
        ref_run_name (str, os.Path): Path to ref_run_name
        learn (bool: False): Decides whether to train a new negative binomial regressor
        file_handle (str, os.Path): Path to SC cell type reference anndata
        save (bool: False): Whether to save learned model or not, only affects learn=True
    """
    if file_handle:
        # Lets load the cell type signatures already trained
        adata_file = file_handle
        adata_ref = sc.read_h5ad(adata_file)
        print(f"## Loaded user specified cell type signatures: {adata_file}")
    elif learn:
        print("## Training negative binomial regressor to predict reference cell type signatures")
        # prepare anndata for the regression model
        gnnc2l.models.RegressionModel.setup_anndata(adata=adata_snrna_raw, 
                                # 10X reaction / sample / batch
                                batch_key=batch_key, # sample
                                # cell type, covariate used for constructing signatures
                                labels_key=labels_key, # celltype
                                # multiplicative technical effects (platform, 3' vs 5', donor effect)
        #                         categorical_covariate_keys=['Method'] # method for obtaining
                               )
        mod = RegressionModel(adata_snrna_raw) 

        # # view anndata_setup as a sanity check
        # mod.view_anndata_setup()
        # Lets train the negative binomial regression model on the snrna data
        mod.train(max_epochs=500, use_gpu=True)

        # Export the posterior
        adata_ref = mod.export_posterior(
            adata_snrna_raw, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
        )

        if save:
            # Save model
            mod.save(f"{ref_run_name}", overwrite=True)
            # Save anndata object with results
            adata_file = f"{ref_run_name}/sc.h5ad"
            adata_ref.write(adata_file)
            adata_file

    else:
        # Lets load the cell type signatures already trained
        adata_file = f"{ref_run_name}/sc.h5ad"
        adata_ref = sc.read_h5ad(adata_file)
        print(f"## Loaded previously inferred cell type signatures: {adata_file}")

    # export estimated expression in each cluster
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}' 
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}' 
                                        for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    return adata_ref, inf_aver, adata_file