# GNN-C2L

Code associated with "Relational inductive biases in spatial cell type deconvolution"

# Installation

Due to a combination of packages involving Torch, Pytorch Geometric, Pyro, alongside tools for single-cell data processing, we have to follow a specific installation procedure. These sets of instructions are for constructing the necessary `conda` environment, which we recommend as the package manager for development and evaluation.

1. Create a new conda environment
    ```bash
    conda create -n gnnc2l_env python=3.7
    conda activate gnnc2l_env
    ```
2. First please install PyTorch based on your hardware availability (https://pytorch.org/get-started/locally/) (below command applies if you have an CUDA-enabled card)
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
3. Install PyTorch Geometric (https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
    ```bash
    conda install pyg -c pyg
    conda install pytorch-scatter -c pyg
    ```
4. Install scanpy, plotnine, and squidpy
    ```bash
    pip install scanpy
    pip install squidpy
    pip install plotnine
    ```
5. Install `gnnc2l` at the root of the project folder
    ```bash
    pip install -e .
    ```

This will also allow you to extend and build upon this work.

# Running the experiments as in paper

Simply navigate to `experiments/comparative_eval/` and run all the models on all the datasets.

```bash
cd experiments/comparative_eval/
bash all_run.sh
```

# Citation

If any of the theory or code is useful to your research please consider citing our pre-print.

```
@article {GNNC2L,
	author = {Ramon Vinas and Paul Scherer and Nikola Simidjievski and Mateja Jamnik and Pietro Lio},
	title = {Spatio-relational inductive biases in spatial cell-type deconvolution},
	elocation-id = {2023.05.19.541474},
	year = {2023},
	doi = {10.1101/2023.05.19.541474},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Spatial transcriptomic technologies profile gene expression in-situ, facilitating the spatial characterisation of molecular phenomena within tissues, yet often at multi-cellular resolution. Computational approaches have been developed to infer fine-grained cell-type compositions across locations, but they frequently treat neighbouring spots independently of each other. Here we present GNN-C2L, a flexible deconvolution approach that leverages proximal inductive biases to propagate information along adjacent spots. In performance comparison on simulated and semi-simulated datasets, GNN-C2L achieves increased deconvolution performance over spatial-agnostic variants. We believe that accounting for spatial inductive biases can yield improved characterisation of cell-type heterogeneity in tissues.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/05/22/2023.05.19.541474},
	eprint = {https://www.biorxiv.org/content/early/2023/05/22/2023.05.19.541474.full.pdf},
	journal = {bioRxiv}
}
```

# Acknowledgements

We would like to acknowledge the authors of Cell2Location for the original cell2location package available at: https://github.com/BayraktarLab/cell2location 
