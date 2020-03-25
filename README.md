
This is the implementation of paper:

> Ripple Walk Training: A Subgraph-based training framework for Large and Deep Graph Neural
> Network

### Requirements
The code is implemented in Python 3.7. Package used for development are just below.
```
networkx           
numpy              
scipy              
torch              
```

### Datasets

Pubmed


###Instructions for running the code

1, Run the subgraph sampling code
```
python3 subgraph_sample_pubmed.py
```
, the results will be stored in `./sampled_subgraph/`.

2, Run the GCN or GAT model training/testing code
```
python3 train_rw_gcn_pubmed.py
```
or
```
python3 train_rw_gat_pubmed.py
```
, the results will be shown on screen and stored in `./results/`.


###Note:

1, If no GPU is available, add config `--no-cuda True` when running the GCN/GAT models.
2, To change epoch numbers (default as 10) of training to NUMBER, add config `--epochs NUMBER`.
