"""This file contains code for computing the various metrics we want to compute for
comparative analysis.
"""
from re import sub
import numpy as np
from scipy import interpolate
from metric_utils import *

# New code to obtain summaries of posteriors for specific latent variables 
def add_posterior_df(model, adata, sitename, nameprefix):
    """Use after export posterior has been run for 
    nUMI and other values of interest
    
    Args:
        sitename (str): refers to the registered name of the latent variable in the pyro backend
        nameprefix (str): refers to an additional string denoter used for reading as in the original
            mod.export posterior. It was "cell_abundance" for w_sf
    """
    add_to_obsm = ['means', 'stds', 'q05', 'q95']
    for k in add_to_obsm:
        sample_df = model.sample2df_obs(
            model.samples,
            site_name=sitename,
            summary_name=k,
            name_prefix=nameprefix,
        )
        try:
            adata.obsm[f"{k}_{nameprefix}_{sitename}"] = sample_df.loc[adata.obs.index, :]
        except ValueError:
            # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
            adata.obs[sample_df.columns] = sample_df.loc[adata.obs.index, :]

    return adata


def compute_metrics(model, adata_vis, sitename, nameprefix):
    """Compute all the metrics
    """
    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_vis = model.export_posterior(
        adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': model.adata.n_obs, 'use_gpu': True} #1
    )

    if sitename == "w_sf":
        adata_vis_res = adata_vis.copy()
        cell_count = adata_vis_res.obs.loc[:, ['cell_abundances_' in i for i in adata_vis_res.obs.columns]]
        cell_count.columns =  [sub('cell_abundances_', '', i) for i in cell_count.columns]
        cell_count_columns = cell_count.columns
        umi_count = adata_vis_res.obs.loc[:, ['UMI_count_' in i for i in adata_vis_res.obs.columns]]
        umi_count.columns =  [sub('UMI_count_', '', i) for i in umi_count.columns]

        # Extract w_sf factor (spot specific abundances) rename columns via regex (edited to new data)
        spot_factors = adata_vis_res.obsm['means_cell_abundance_w_sf']
        spot_factors.columns =  [sub('meanscell_abundance_w_sf_', '', i) for i in spot_factors.columns]

        spot_factors_sd = adata_vis_res.obsm['stds_cell_abundance_w_sf']
        spot_factors_sd.columns =  [sub('stdsspot_factors_', '', i) for i in spot_factors_sd.columns]

    elif sitename == "a_sf":
        # Export the summaries of the posterior for a_sf
        adata_vis = add_posterior_df(model, adata_vis, sitename, nameprefix)

        adata_vis_res = adata_vis.copy()
        cell_count = adata_vis_res.obs.loc[:, ['cell_abundances_' in i for i in adata_vis_res.obs.columns]]
        cell_count.columns =  [sub('cell_abundances_', '', i) for i in cell_count.columns]
        cell_count_columns = cell_count.columns
        umi_count = adata_vis_res.obs.loc[:, ['UMI_count_' in i for i in adata_vis_res.obs.columns]]
        umi_count.columns =  [sub('UMI_count_', '', i) for i in umi_count.columns]

        # NOTE ASF USAGE DEPENDING ON MODEL
        # Extract w_sf factor (spot specific abundances) rename columns via regex (edited to new data)
        spot_factors = adata_vis_res.obsm['means_gnn_cell_abundance_a_sf']
        spot_factors.columns =  [sub('meansgnn_cell_abundance_a_sf_', '', i) for i in spot_factors.columns] # remove the meanscell_abundance bit in front of the name to get raw cell type names

        spot_factors_sd = adata_vis_res.obsm['stds_gnn_cell_abundance_a_sf']
        spot_factors_sd.columns =  [sub('stdsgnn_cell_abundance_a_sf_', '', i) for i in spot_factors_sd.columns]

    else:
        raise ValueError("input sitename not implemented")


    infer_cell_count = spot_factors[cell_count.columns]
    #not sure it could also by mean detection or some other awfully named thing
    # Current other guess is n_s_cells_per_location
    nUMI_factors = adata_vis_res.obs[['cell_count_' + i for i in cell_count_columns]]
    nUMI_factors.columns =  [sub('cell_count_', '', i) for i in nUMI_factors.columns]

    infer_cell_count = spot_factors[cell_count.columns]
    infer_nUMI_count = nUMI_factors[cell_count.columns]

    cell_proportions = (cell_count.T / cell_count.sum(1)).T
    cell_proportions.iloc[np.isnan(cell_proportions.values)] = 0
    infer_cell_proportions = (infer_cell_count.T / infer_cell_count.sum(1)).T
    umi_count_proportions = (umi_count.T / umi_count.sum(1)).T
    umi_count_proportions.iloc[np.isnan(umi_count_proportions.values)] = 0
    infer_nUMI_factors = (nUMI_factors.T / nUMI_factors.sum(1)).T

    # mean number of cell types per location
    # MATCHES THE PAPER NOTEBOOK!
    (cell_count.values > 1).sum(1).mean(), (cell_count.values).sum(1).mean(), (infer_cell_count.values > 1).sum(1).mean(), (infer_cell_count.values).sum(1).mean()

    # Metric 1: Np.corrcoeff between inferred cell density and cell count
    inf_cellDensity_cellCount_correlation = np.round(np.corrcoef(cell_count.values.flatten(), infer_cell_count.values.flatten()), 3)[0,1]

    # Metric 2: np.corrcoeff between inferred mRNA and simulated mRNA counts
    inf_mRNA_sim_mRNA_correlation = np.round(np.corrcoef(umi_count.values.flatten(), nUMI_factors.values.flatten()), 3)[0,1]

    # Metric 3: np.corrcoeff between inferred and simulated cell proportions 
    # (this seems to be the main result presented in the first big figure of C2L paper)
    inf_cellproportion_sim_cellproportion_correlation = np.round(np.corrcoef(cell_proportions.values.flatten(), infer_cell_proportions.values.flatten()), 3)[0,1]

    # Metric 4: np.corrcoeff between inferred and simulated mRNA proportions
    inf_mRNAProportion_sim_mRNAProportion_correlation = np.round(np.corrcoef(umi_count_proportions.values.flatten(), infer_nUMI_factors.values.flatten()), 3)[0,1]

    # Metric 5: np.corrcoeff between mRNA count and simulated cell count
    inf_mRNACount_sim_cellCount_correlation = np.round(np.corrcoef(cell_count.values.flatten(), infer_nUMI_count.values.flatten()), 3)[0,1]

    # Metric 6: np.corrcoeff between simulated mRNA count and simulated cell count
    sim_mRNACount_sim_cellCount_proportion = np.round(np.corrcoef(cell_count.values.flatten(), infer_nUMI_count.values.flatten()), 3)[0,1]

    ## Extracting
    ## Precision recall
    ## curve values for cell count
    # Get binary matrix of positive cell activity
    mode = "macro"
    pos_cell_count = cell_count.values > 0.1
    design = adata_vis_res.uns['design']['cell_types2zones']
    design_loc = design.loc[cell_count.columns,:]
    
    # Spatially uniform and high abundance cell types
    ind = (design_loc['is_uniform'] * design_loc['is_high_density']).values.astype(bool)
    cellcount_su_ha_mode_average_score, cellcount_su_ha_precision, cellcount_su_ha_recall, cellcount_su_ha_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_count.loc[:, ind], 
                                                                mode='macro', curve='PR')    
    # Spatially uniform and low abundance cell types
    ind = (design_loc['is_uniform'] * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellcount_su_la_mode_average_score, cellcount_su_la_precision, cellcount_su_la_recall, cellcount_su_la_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_count.loc[:, ind], 
                                                                mode='macro', curve='PR')
    # Sparse and high abundance cell types
    ind = ((1 - design_loc['is_uniform']) * design_loc['is_high_density']).values.astype(bool)
    cellcount_sparse_ha_mode_average_score, cellcount_sparse_ha_precision, cellcount_sparse_ha_recall, cellcount_sparse_ha_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_count.loc[:, ind], 
                                                                mode='macro', curve='PR')
    # Sparse and low abundance cell types
    ind = ((1 - design_loc['is_uniform']) * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellcount_sparse_la_mode_average_score, cellcount_sparse_la_precision, cellcount_sparse_la_recall, cellcount_sparse_la_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_count.loc[:, ind], 
                                                                mode='macro', curve='PR')
    # All cell types
    cellcount_all_mode_average_score, cellcount_all_precision, cellcount_all_recall, cellcount_all_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_count.loc[:, ind], 
                                                                mode='macro', curve='PR')

    ## PR values
    ## With cell proportions
    ##
    # Spatially uniform and high abundance cell types
    ind = (design_loc['is_uniform'] * design_loc['is_high_density']).values.astype(bool)
    cellproportion_su_ha_mode_average_score, cellproportion_su_ha_precision, cellproportion_su_ha_recall, cellproportion_su_ha_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_proportions.loc[:, ind], 
                                                                mode='macro', curve='PR')
    # Spatially uniform and low abundance cell types
    ind = (design_loc['is_uniform'] * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellproportion_su_la_mode_average_score, cellproportion_su_la_precision, cellproportion_su_la_recall, cellproportion_su_la_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_proportions.loc[:, ind], 
                                                                mode='macro', curve='PR')
    # Sparse and high abundance cell types
    ind = ((1 - design_loc['is_uniform']) * design_loc['is_high_density']).values.astype(bool)
    cellproportion_sparse_ha_mode_average_score, cellproportion_sparse_ha_precision, cellproportion_sparse_ha_recall, cellproportion_sparse_ha_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_proportions.loc[:, ind], 
                                                                mode='macro', curve='PR')
    # Sparse and low abundance cell types
    ind = ((1 - design_loc['is_uniform']) * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellproportion_sparse_la_mode_average_score, cellproportion_sparse_la_precision, cellproportion_sparse_la_recall, cellproportion_sparse_la_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_proportions.loc[:, ind], 
                                                                mode='macro', curve='PR')
    # All cell types
    cellproportion_all_mode_average_score, cellproportion_all_precision, cellproportion_all_recall, cellproportion_all_average_precision = pr_by_category_values(pos_cell_count[:, ind], 
                                                                infer_cell_proportions.loc[:, ind], 
                                                                mode='macro', curve='PR')

    #
    # Compute JSD for each of data modes (cell proportion only)
    #
    cellproportion_all_jsd = hist_obs_sim_jsd(cell_proportions, infer_cell_proportions)

    # Spatially uniform and high abundance cell types
    ind = (design_loc['is_uniform'] * design_loc['is_high_density']).values.astype(bool)
    cellproportion_su_ha_jsd = hist_obs_sim_jsd(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])
    
    # Spatially uniform and low abundance cell types
    ind = (design_loc['is_uniform'] * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellproportion_su_la_jsd = hist_obs_sim_jsd(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])

    # Sparse and high abundance cell types
    ind = ((1 - design_loc['is_uniform']) * design_loc['is_high_density']).values.astype(bool)
    cellproportion_sparse_ha_jsd = hist_obs_sim_jsd(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])

    # Sparse and low abundance cell types
    ind = ((1 - design_loc['is_uniform']) * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellproportion_sparse_la_jsd = hist_obs_sim_jsd(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])

    #
    # Compute Pearson for each of data modes (cell proportions)
    #
    # Spatially uniform and high abundance cell types
    ind = (design_loc['is_uniform'] * design_loc['is_high_density']).values.astype(bool)
    cellproportion_su_ha_correlation = hist_obs_sim_corr(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])
    
    # Spatially uniform and low abundance cell types
    ind = (design_loc['is_uniform'] * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellproportion_su_la_correlation = hist_obs_sim_corr(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])

    # Sparse and high abundance cell types
    ind = ((1 - design_loc['is_uniform']) * design_loc['is_high_density']).values.astype(bool)
    cellproportion_sparse_ha_correlation = hist_obs_sim_corr(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])

    # Sparse and low abundance cell types
    ind = ((1 - design_loc['is_uniform']) * (1 - design_loc['is_high_density'])).values.astype(bool)
    cellproportion_sparse_la_correlation = hist_obs_sim_corr(cell_proportions.loc[:, ind], infer_cell_proportions.loc[:, ind])


    metrics = {"inf_cellDensity_cellCount_correlation":inf_cellDensity_cellCount_correlation, 
        "inf_mRNA_sim_mRNA_correlation":inf_mRNA_sim_mRNA_correlation, 
        "inf_cellproportion_sim_cellproportion_correlation":inf_cellproportion_sim_cellproportion_correlation,
        "inf_mRNAProportion_sim_mRNAProportion_correlation":inf_mRNAProportion_sim_mRNAProportion_correlation,
        "inf_mRNACount_sim_cellCount_correlation":inf_mRNACount_sim_cellCount_correlation,
        "sim_mRNACount_sim_cellCount_proportion":sim_mRNACount_sim_cellCount_proportion,
        
        "cellproportion_all_jsd": cellproportion_all_jsd,
        "cellproportion_su_ha_jsd": cellproportion_su_ha_jsd,
        "cellproportion_su_la_jsd": cellproportion_su_la_jsd,
        "cellproportion_sparse_ha_jsd": cellproportion_sparse_ha_jsd,
        "cellproportion_sparse_la_jsd": cellproportion_sparse_la_jsd,

        "cellproportion_su_ha_correlation":cellproportion_su_ha_correlation,
        "cellproportion_su_la_correlation":cellproportion_su_la_correlation,
        "cellproportion_sparse_ha_correlation":cellproportion_sparse_ha_correlation,
        "cellproportion_sparse_la_correlation":cellproportion_sparse_la_correlation,

        "cellcount_su_ha_mode_average_score": cellcount_su_ha_mode_average_score,
        "cellcount_su_ha_precision": cellcount_su_ha_precision,
        "cellcount_su_ha_recall": cellcount_su_ha_recall,
        "cellcount_su_ha_average_precision": cellcount_su_ha_average_precision,
        "cellcount_su_la_mode_average_score": cellcount_su_la_mode_average_score,
        "cellcount_su_la_precision": cellcount_su_la_precision,
        "cellcount_su_la_recall": cellcount_su_la_recall,
        "cellcount_su_la_average_precision": cellcount_su_la_average_precision,
        "cellcount_sparse_ha_mode_average_score": cellcount_sparse_ha_mode_average_score,
        "cellcount_sparse_ha_precision": cellcount_sparse_ha_precision,
        "cellcount_sparse_ha_recall": cellcount_sparse_ha_recall,
        "cellcount_sparse_ha_average_precision": cellcount_sparse_ha_average_precision,
        "cellcount_sparse_la_mode_average_score": cellcount_sparse_la_mode_average_score,
        "cellcount_sparse_la_precision": cellcount_sparse_la_precision,
        "cellcount_sparse_la_recall": cellcount_sparse_la_recall,
        "cellcount_sparse_la_average_precision": cellcount_sparse_la_average_precision,
        "cellcount_all_mode_average_score":cellcount_all_mode_average_score,
        "cellcount_all_precision":cellcount_all_precision,
        "cellcount_all_recall":cellcount_all_recall,
        "cellcount_all_average_precision":cellcount_all_average_precision,

        "cellproportion_su_ha_mode_average_score": cellproportion_su_ha_mode_average_score,
        "cellproportion_su_ha_precision": cellproportion_su_ha_precision,
        "cellproportion_su_ha_recall": cellproportion_su_ha_recall,
        "cellproportion_su_ha_average_precision": cellproportion_su_ha_average_precision,
        "cellproportion_su_la_mode_average_score": cellproportion_su_la_mode_average_score,
        "cellproportion_su_la_precision": cellproportion_su_la_precision,
        "cellproportion_su_la_recall": cellproportion_su_la_recall,
        "cellproportion_su_la_average_precision": cellproportion_su_la_average_precision,
        "cellproportion_sparse_ha_mode_average_score": cellproportion_sparse_ha_mode_average_score,
        "cellproportion_sparse_ha_precision": cellproportion_sparse_ha_precision,
        "cellproportion_sparse_ha_recall": cellproportion_sparse_ha_recall,
        "cellproportion_sparse_ha_average_precision": cellproportion_sparse_ha_average_precision,
        "cellproportion_sparse_la_mode_average_score": cellproportion_sparse_la_mode_average_score,
        "cellproportion_sparse_la_precision": cellproportion_sparse_la_precision,
        "cellproportion_sparse_la_recall": cellproportion_sparse_la_recall,
        "cellproportion_sparse_la_average_precision": cellproportion_sparse_la_average_precision,
        "cellproportion_all_mode_average_score":cellproportion_all_mode_average_score,
        "cellproportion_all_precision":cellproportion_all_precision,
        "cellproportion_all_recall":cellproportion_all_recall,
        "cellproportion_all_average_precision":cellproportion_all_average_precision,

        }



    return metrics