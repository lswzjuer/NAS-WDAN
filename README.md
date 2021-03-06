# MixSearch: Searching for Domain Generalized Medical Image Segmentation Architectures 

Considering the scarcity of medical data, most datasets in medical image analysis are an order of magnitude smaller than those of natural images. However, most Network Architecture Search (NAS) approaches in medical images focused on specific datasets and did not take into account the generalization ability of the learned architectures on unseen datasets as well as different domains. In this paper, we address this point by proposing to search for generalizable U-shape architectures on a composited dataset that mixes medical images from multiple segmentation tasks and domains creatively, which is named MixSearch. Specifically, we propose a novel approach to mix multiple small-scale datasets from multiple domains and segmentation tasks to produce a large-scale dataset. Then, a novel weaved encoder-decoder structure is designed to search for a generalized segmentation network in both cell-level and network-level. The network produced by the proposed MixSearch framework achieves state-of-the-art results compared with advanced encoder-decoder networks across various datasets. Moreover, we also evaluate the learned network architectures on three additional datasets, which are unseen in the searching process. Extensive experiments show that the architectures automatically learned by our proposed MixSearch surpass U-Net and its variants by a significant margin, verifying the generalization ability and practicability of our proposed method.

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


e.g. `python eval_baseline_chaos.py --val_batch=1


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

## How do we perform MixSearch ?


`sudo CUDA_VISIBLE_DEVICES=0,1  python train_stage_search_mixup.py --epoch=80  --train_batch=12 --val_batch=12  --loss=bcelog --train_portio=0.5  --arch_lr=2e-4
--arch_weight_decay=1e-3 --lr=0.025 --weight_decay=3e-4 --init_channel=16 --arch_after=10 --gpus=2 --double_down_channel --deepsupervision --alpha=0.5 --note=ep80_double_deep_mixup`

**epoch：**  Number of training epochs per stage.

**xx_batch:** Train pr val batch.

**loss:** Loss function we choosed in search.

**train_portio:** The training set is divided equally to optimize the `w` and (alpha,beta).

**arch_lr:** The learning rate for (alpha,beta).

**arch_weight_decay:** L2 normal for (alpha,beta).

**lr&weight_decay:** learning rate and weight decay for `w`.

**arch_after:** In order to avoid falling into local optimization too early in the search process, the optimization of (alpha,beta) was carried out after training `w` a certain epochs.

**init_channel & double_down_channel:** Important parameters that determine network size and performance.

**deepsupervision:** Deep supervision training.

**alpha:** Control the sampling distribution when three dataset are mixup.

**dropout_prob:** In order to reduce the risk of overfitting, dropout can be added to the corresponding operation in mix-operation
