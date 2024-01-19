## Hierarchical Backward-Greedy Algorithms for Filter Pruning

This repository contains the implementation details of our Hierarchical Backward-Greedy Algorithms for Scalable Whole-network
Filter-Pruning in CNNs approach.



## Requirements
The code is written for python `3.6.3`, but should work for other version with some modifications.
Create a conda environment with python version `3.6.3`.  Install cudatoolkit according to gpu compatibility.
```
pip install -r requirements.txt
```

## Data Preparation

Not required for CIFAR10 and CIFAR100


## Python script overview

`omp.py` - It contains the code for pruning using FP-OMP approach with the default args.
`omp_search.py` - It contains the code for pruning using HBGS approach with the default args.
`omp_tree.py` - It contains the code for pruning using HBGTS approach with the default args.
`backward.py` - It contains the code for pruning using FP-Backward approach with the default args.
`backward_search.py` - It contains the code for pruning using HBGS-B approach with the default args.
`backward_tree.py` - It contains the code for pruning using HBGTS-B approach with the default args.


### Key Parameters:

 `gpu`: cuda gpu device number \
 `batch_size`: training batch size \ 
 `step_ft`: number of fine tuning rounds \
 `ft_lr`: learning rate \ 
 `ratio`: whole network pruning ratio \
 `workers`: number of workers \
