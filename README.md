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

TBA

# Acknowledgements

We would like to acknowledge the authors of Cell2Location for the original cell2location package available at: https://github.com/BayraktarLab/cell2location