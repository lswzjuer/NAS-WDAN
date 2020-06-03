# MixSearch: A Generalized Network Architecture Search for Medical Image Segmentation

In this work, we propose a lightweight weaved deep aggregation network named NAS-WDAN, which contains built-in learnable horizontal and vertical feature fusion. By combining differentiable cell-level and network-level search, NAS-WDAN can automatically determine the optimal architecture and get rid of the limitations of human design. Most importantly, in order to search a high-performance generalized network, we have produced a composite dataset containing multiple domain centers and smooth transitions between domains by directly mixing several different datasets and generating additional virtual examples. Differentiable architecture search performed on this composite dataset is called "MixSearch", which can solve the generalization problem of differentiable search.

## models

Implementation of the benchmark models, including unet,att-unet,multires-unet,r2t-unet,unet++.

## datasets

A folder containing the data read interface, in which each dataset corresponds to a script.

## img

Visualization of the baseline model

## network_pruning

This folder contains several kinds of pruning algorithm implementation, including the bnlim, L1pruning, softpruning,fpgmpruning...


## tools & utils

Helper functions and scripts.

* `train_baseline_xxx.py` &nbsp;
  Scripts for training the benchmark model on individual datasets

* `eval_baseline_xxx.py`  &nbsp;
  Scripts that validate the accuracy of the model on the validation set


e.g. `python eval_baseline_chaos.py --val_batch=8`
<<<<<<< HEAD
=======

>>>>>>> 67d7023a5ec875b65babad4a92dab826fc8539bf


## network search

The implementation of searchable weaved deep aggregation network

* cell_visualize &nbsp;
Searchable cells visualization

* nas_model &nbsp;
two searchable weaved network, one with a depth of 5 and one with a depth of 6.

**cell.py**  &nbsp;
Implementation of searchable cells.

**genotypes.py**  &nbsp;
Structure of searchable cells

**train_stage_search_xx.py**  &nbsp;
Perform architecture search on the three sub datasets and the composite dataset, respectively

e.g. Mixsearch on the composite dataset
`train_stage_search_mixup.py --train_batch=24 --val_batch=24 --epoch=80 --loss=bcelog --note=xxx`  

**retrain_xxx.py**  Retrain the models from different search configurations on each subdataset.

e.g. `retrain_cvc.py --train_batch=8 --val_batch=8 --loss=bcedice --epoch=1600 --lr=4e-3 --model=alpha0_5_stage1_double_deep_ep80 --note=xx `

`retrain_chao.py --train_batch=8 --val_batch=8 --loss=bcedice --epoch=1600 --lr=2e-3 --model=alpha0_5_stage1_double_deep_ep80 --note=xx `

**eval_prune_model_xx.py**  &nbsp;
Verify the trained search model on the corresponding dataset

e.g. Verify V<sub>isic</sub>'s performance on the CHAOS dataset `eval_prune_chaos.py --model=double_deep_isictrans/max_stage1_double_deep/alpha0_5_double_deep`

***note*** The search models for  V<sub>isic</sub>,V<sub>cvc</sub>,V<sub>chaos</sub>  in the genotypes.py file are *stage1_layer9_110epoch_double_deep_final*,*layer7_double_deep* and *stage0_double_deep_ep80_newim* respectively.


**nas_search_unet_prune.py** &nbsp; The implementation of the search model, during retraining/validation.

**operations.py**  Implementation of search space.

**model_time_test.py**  Measuring the inference time of single picture.

**get_arch_parameters.py** Network structure parameter analysis.
