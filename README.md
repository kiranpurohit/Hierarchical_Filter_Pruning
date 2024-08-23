## A Greedy Hierarchical Approach to Whole-Network Filter-Pruning in CNNs

This repository contains the implementation details of the paper "A Greedy Hierarchical Approach to Whole-Network Filter-Pruning in CNNs" accepted in TMLR 2024.
https://openreview.net/forum?id=WzHuebRSgQ


## Requirements
The code is written for python `3.6.3`, but should work for other version with some modifications.
Create a conda environment with python version `3.6.3`.  Install cudatoolkit according to gpu compatibility.
```
pip install -r requirements.txt
```

## Data Preparation

Not required for CIFAR10 and CIFAR100


## Python script overview

`omp.py` - It contains the code for pruning using FP-OMP approach.\
`omp_search.py` - It contains the code for pruning using HBGS approach.\
`omp_tree.py` - It contains the code for pruning using HBGTS approach.\
`backward.py` - It contains the code for pruning using FP-Backward approach.\
`backward_search.py` - It contains the code for pruning using HBGS-B approach.\
`backward_tree.py` - It contains the code for pruning using HBGTS-B approach.


### Key Parameters:

 `gpu`: cuda gpu device number \
 `batch_size`: training batch size \ 
 `step_ft`: number of fine tuning rounds \
 `ft_lr`: learning rate \ 
 `ratio`: whole network pruning ratio \
 `workers`: number of workers 
