import scanpy as sc
import scipy
import numpy as np

def preprocess_data(adata, adata_snrna_raw):
    """Preprocesses the synthetic data and the snrna data
    
    Args:
        adata (anndata): the synthetic dataset
        adata_snrna_raw (anndata): anndata of the snrna data 
    """
    sc.pp.calculate_qc_metrics(adata, inplace=True) # computes values for obs such as total_counts, pct_count_in_top_etc.

    # Sparsify data
    adata_snrna_raw.X = scipy.sparse.csr_matrix(adata_snrna_raw.X)
    adata.X = scipy.sparse.csr_matrix(adata.X)

    # remove cells and genes with 0 counts everywhere
    sc.pp.filter_cells(adata_snrna_raw, min_genes=1) 
    sc.pp.filter_genes(adata_snrna_raw, min_cells=1) # 8111 x 31053 to 8111 x 24594

    # calculate the mean of each gene across non-zero cells
    adata_snrna_raw.var['n_cells'] = (adata_snrna_raw.X.toarray() > 0).sum(0)
    adata_snrna_raw.var['nonz_mean'] = adata_snrna_raw.X.toarray().sum(0) / adata_snrna_raw.var['n_cells']
    
    nonz_mean_cutoff = np.log10(1.12) # cut off for expression in non-zero cells
    cell_count_cutoff = np.log10(adata_snrna_raw.shape[0] * 0.0005) # cut off percentage for cells with higher expression
    cell_count_cutoff2 = np.log10(adata_snrna_raw.shape[0] * 0.03)# cut off percentage for cells with small expression

    # select genes based on mean expression in non-zero cells
    adata_snrna_raw = adata_snrna_raw[:,(np.array(np.log10(adata_snrna_raw.var['nonz_mean']) > nonz_mean_cutoff)
             | np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff2))
          & np.array(np.log10(adata_snrna_raw.var['n_cells']) > cell_count_cutoff)]

    # Add counts matrix as adata.raw
    adata_snrna_raw.raw = adata_snrna_raw
    adata_vis = adata.copy()
    adata_vis.raw = adata_vis

    # Rename the sample names according to regex
    from re import sub
    adata_snrna_raw.obs['sample'] = [sub('_.+$','', i) for i in adata_snrna_raw.obs.index]

    return adata, adata_snrna_raw