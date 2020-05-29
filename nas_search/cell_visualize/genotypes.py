from collections import namedtuple
import numpy as np
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal_down normal_down_concat normal_up normal_up_concat normal_normal normal_normal_concat')


layer7_deepvision_doublechannel_output2=Genotype(normal_down=[('down_cweight', 1), ('cweight', 0), ('down_cweight', 1), ('identity', 0), ('down_dep_conv', 1), ('cweight', 2), ('down_conv', 1), ('dep_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('cweight', 0), ('up_dep_conv', 1), ('up_cweight', 1), ('identity', 0), ('up_cweight', 1), ('cweight', 2), ('dep_conv', 4), ('shuffle_conv', 3)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('cweight', 0), ('identity', 0), ('identity', 1), ('cweight', 2), ('identity', 0), ('dep_conv', 4), ('shuffle_conv', 3)], normal_normal_concat=range(2, 6))
layer7_doublechannel_output2=Genotype(normal_down=[('down_cweight', 1), ('dep_conv', 0), ('shuffle_conv', 2), ('down_dep_conv', 1), ('dep_conv', 2), ('down_dil_conv', 1), ('down_cweight', 1), ('shuffle_conv', 3)], normal_down_concat=range(2, 6), normal_up=[('dep_conv', 0), ('up_cweight', 1), ('shuffle_conv', 2), ('up_cweight', 1), ('dep_conv', 2), ('shuffle_conv', 3), ('shuffle_conv', 3), ('conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('dep_conv', 0), ('shuffle_conv', 2), ('identity', 0), ('dep_conv', 2), ('shuffle_conv', 3), ('shuffle_conv', 3), ('conv', 4)], normal_normal_concat=range(2, 6))


layer7_deepvision_doublechannel=Genotype(normal_down=[('down_dep_conv', 1), ('cweight', 0), ('down_cweight', 1), ('conv', 2), ('identity', 0), ('cweight', 2), ('conv', 2), ('identity', 0)], normal_down_concat=range(2, 6), normal_up=[('up_cweight', 1), ('cweight', 0), ('conv', 2), ('cweight', 0), ('up_cweight', 1), ('identity', 0), ('conv', 2), ('identity', 0)], normal_up_concat=range(2, 6), normal_normal=[('cweight', 0), ('dil_conv', 1), ('conv', 2), ('dep_conv', 1), ('identity', 0), ('identity', 1), ('conv', 2), ('identity', 0)], normal_normal_concat=range(2, 6))
layer7_doublechannel=Genotype(normal_down=[('down_dil_conv', 1), ('identity', 0), ('conv', 0), ('down_dil_conv', 1), ('shuffle_conv', 0), ('down_conv', 1), ('shuffle_conv', 4), ('down_conv', 1)], normal_down_concat=range(2, 6), normal_up=[('up_dil_conv', 1), ('identity', 0), ('up_dep_conv', 1), ('conv', 0), ('shuffle_conv', 0), ('up_dil_conv', 1), ('shuffle_conv', 4), ('up_dep_conv', 1)], normal_up_concat=range(2, 6), normal_normal=[('identity', 1), ('identity', 0), ('conv', 0), ('identity', 1), ('shuffle_conv', 0), ('dep_conv', 2), ('shuffle_conv', 4), ('identity', 0)], normal_normal_concat=range(2, 6))


stage0_layer7_110epoch_double_final=Genotype(
    normal_down=[('down_dil_conv', 1), ('conv', 0), ('shuffle_conv', 0), ('down_cweight', 1), ('shuffle_conv', 0), ('dep_conv', 2), ('down_conv', 1), ('shuffle_conv', 4)],
    normal_down_concat=range(2, 6),
    normal_up=[('conv', 0), ('up_dil_conv', 1), ('shuffle_conv', 0), ('up_dep_conv', 1), ('shuffle_conv', 0), ('dep_conv', 2), ('up_dep_conv', 1), ('shuffle_conv', 4)],
    normal_up_concat=range(2, 6), normal_normal=[('identity', 1), ('conv', 0), ('shuffle_conv', 0), ('dep_conv', 1), ('shuffle_conv', 0), ('dep_conv', 2), ('shuffle_conv', 4), ('shuffle_conv', 0)],
    normal_normal_concat=range(2, 6))

stage0_layer7_110epoch_double_deep_final=Genotype(normal_down=[('down_dep_conv', 1), ('cweight', 0), ('down_cweight', 1), ('conv', 2), ('down_dep_conv', 1), ('dep_conv', 3), ('conv', 2), ('shuffle_conv', 3)],
         normal_down_concat=range(2, 6),
         normal_up=[('up_cweight', 1), ('cweight', 0), ('conv', 2), ('cweight', 0), ('up_cweight', 1), ('dep_conv', 3), ('conv', 2), ('shuffle_conv', 3)],
         normal_up_concat=range(2, 6),
         normal_normal=[('shuffle_conv', 1), ('cweight', 0), ('conv', 2), ('dep_conv', 1), ('dep_conv', 3), ('identity', 0), ('conv', 2), ('shuffle_conv', 3)],
         normal_normal_concat=range(2, 6))

stage0_layer7_110epoch_double_best=Genotype(normal_down=[('down_dil_conv', 1), ('conv', 0), ('shuffle_conv', 0), ('down_cweight', 1), ('shuffle_conv', 0), ('down_cweight', 1), ('down_conv', 1), ('shuffle_conv', 4)],
                    normal_down_concat=range(2, 6),
                    normal_up=[('conv', 0), ('up_dil_conv', 1), ('shuffle_conv', 0), ('up_dep_conv', 1), ('shuffle_conv', 0), ('dep_conv', 2), ('up_dep_conv', 1), ('shuffle_conv', 4)],
                    normal_up_concat=range(2, 6), normal_normal=[('conv', 0), ('identity', 1), ('shuffle_conv', 0), ('dep_conv', 1), ('shuffle_conv', 0), ('dep_conv', 2), ('shuffle_conv', 4), ('shuffle_conv', 0)],
                    normal_normal_concat=range(2, 6))

stage0_layer7_110epoch_double_deep_best=Genotype(
    normal_down=[('down_dep_conv', 1), ('cweight', 0), ('down_cweight', 1), ('conv', 2), ('down_dep_conv', 1), ('identity', 0), ('conv', 2), ('shuffle_conv', 3)],
    normal_down_concat=range(2, 6),
    normal_up=[('up_cweight', 1), ('cweight', 0), ('conv', 2), ('cweight', 0), ('up_cweight', 1), ('identity', 0), ('conv', 2), ('shuffle_conv', 3)],
    normal_up_concat=range(2, 6),
    normal_normal=[('shuffle_conv', 1), ('cweight', 0), ('conv', 2), ('dep_conv', 1), ('identity', 0), ('dep_conv', 3), ('conv', 2), ('shuffle_conv', 3)], normal_normal_concat=range(2, 6)
    )

# 0.934,0.888,0.808
stage1_layer9_110epoch_double_deep_final=Genotype(normal_down=[('down_dep_conv', 1), ('cweight', 0), ('conv', 2), ('down_dil_conv', 1), ('down_dep_conv', 1),('shuffle_conv', 3), ('conv', 3), ('down_dep_conv', 1)],
         normal_down_concat=range(2, 6),
         normal_up=[('up_cweight', 1), ('cweight', 0), ('conv', 2), ('up_conv', 1), ('up_cweight', 1), ('shuffle_conv', 3), ('conv', 3), ('dep_conv', 4)],
         normal_up_concat=range(2, 6),
        normal_normal=[('dep_conv', 1), ('cweight', 0), ('conv', 2), ('shuffle_conv', 0), ('dep_conv', 1), ('shuffle_conv', 3), ('conv', 3), ('dep_conv', 4)],
         normal_normal_concat=range(2, 6))


# 0.932,0.885,0.804  final output
stage1_layer9_110epoch_double_final=Genotype(normal_down=[('down_dil_conv', 1), ('shuffle_conv', 0), ('shuffle_conv', 0), ('down_cweight', 1), ('shuffle_conv', 3), ('down_conv', 1), ('down_conv', 1), ('conv', 3)],
         normal_down_concat=range(2, 6),
         normal_up=[('shuffle_conv', 0), ('up_dep_conv', 1), ('shuffle_conv', 0), ('shuffle_conv', 2), ('shuffle_conv', 3), ('cweight', 2), ('conv', 3), ('shuffle_conv', 2)],
         normal_up_concat=range(2, 6),
         normal_normal=[('shuffle_conv', 0), ('dil_conv', 1), ('shuffle_conv', 0), ('shuffle_conv', 2), ('shuffle_conv', 3), ('shuffle_conv', 1), ('conv', 3), ('shuffle_conv', 2)],
         normal_normal_concat=range(2, 6))

# 0.931 0.889 0.804
stage1_layer9_110epoch_deep_final=Genotype(normal_down=[('down_conv', 1), ('dep_conv', 0), ('shuffle_conv', 2), ('down_cweight', 1), ('down_cweight', 1), ('identity', 3), ('down_cweight', 1), ('dep_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('up_conv', 1), ('dep_conv', 0), ('shuffle_conv', 2), ('shuffle_conv', 0), ('identity', 3), ('up_dil_conv', 1), ('up_conv', 1), ('dep_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 0), ('shuffle_conv', 1), ('shuffle_conv', 2), ('shuffle_conv', 0), ('identity', 3), ('identity', 1), ('identity', 1), ('dep_conv', 4)], normal_normal_concat=range(2, 6))




stage1_layer9_110epoch_final=Genotype(
    normal_down=[('down_dep_conv', 1), ('conv', 0), ('shuffle_conv', 2), ('down_conv', 1), ('down_conv', 1), ('cweight', 3), ('down_dep_conv', 1), ('conv', 3)],
    normal_down_concat=range(2, 6),
    normal_up=[('conv', 0), ('up_dil_conv', 1), ('shuffle_conv', 2), ('shuffle_conv', 0), ('up_dep_conv', 1), ('cweight', 3), ('conv', 3), ('shuffle_conv', 2)],
    normal_up_concat=range(2, 6),
    normal_normal=[('conv', 1), ('conv', 0), ('shuffle_conv', 2), ('shuffle_conv', 0), ('shuffle_conv', 1), ('cweight', 3), ('conv', 3), ('shuffle_conv', 2)], normal_normal_concat=range(2, 6))








# CVC datasets
# ValLoss:0.892 ValAcc:0.985  ValDice:0.913 ValJc:0.844
layer7_double_deep = Genotype(normal_down=[('down_conv', 1), ('cweight', 0), ('dep_conv', 2), ('down_cweight', 1), ('dil_conv', 3), ('down_dil_conv', 1), ('dil_conv', 3), ('dil_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('up_cweight', 1), ('cweight', 0), ('dep_conv', 2), ('up_dil_conv', 1), ('dil_conv', 3), ('up_dil_conv', 1), ('dil_conv', 3), ('up_cweight', 1)], normal_up_concat=range(2, 6), normal_normal=[('dil_conv', 1), ('cweight', 0), ('dep_conv', 2), ('shuffle_conv', 1), ('dil_conv', 3), ('shuffle_conv', 2), ('dil_conv', 3), ('dil_conv', 4)], normal_normal_concat=range(2, 6))
#AM ValLoss:0.764 ValAcc:0.982  ValDice:0.894 ValJc:0.814
layer9_double_deep=Genotype(normal_down=[('avg_pool', 1), ('cweight', 0), ('down_cweight', 1), ('dil_conv', 2), ('down_dep_conv', 1), ('dep_conv', 3), ('dep_conv', 3),('down_conv', 1)], normal_down_concat=range(2, 6), normal_up=[('up_cweight', 1), ('cweight', 0), ('dil_conv', 2), ('dil_conv', 0), ('dep_conv', 3), ('up_cweight', 1), ('dep_conv', 3), ('up_cweight', 1)], normal_up_concat=range(2, 6), normal_normal=[('identity', 1), ('cweight', 0), ('dil_conv', 2), ('dil_conv', 0), ('dep_conv', 3), ('identity', 0), ('dep_conv', 3), ('identity', 1)], normal_normal_concat=range(2, 6))
#ValLoss:0.832 ValAcc:0.980  ValDice:0.882 ValJc:0.796
layer9_deep = Genotype(normal_down=[('down_conv', 1), ('dep_conv', 0), ('down_dep_conv', 1), ('dil_conv', 2), ('avg_pool', 1), ('shuffle_conv', 2), ('dep_conv', 4), ('dil_conv', 2)], normal_down_concat=range(2, 6), normal_up=[('up_dil_conv', 1), ('dep_conv', 0), ('up_dep_conv', 1), ('dil_conv', 2), ('shuffle_conv', 2), ('dep_conv', 0), ('dep_conv', 4), ('dil_conv', 2)], normal_up_concat=range(2, 6), normal_normal=[('conv', 1),('dep_conv', 0), ('dil_conv', 2), ('dep_conv', 1), ('shuffle_conv', 2), ('dep_conv', 0),('dep_conv', 4), ('dil_conv', 2)], normal_normal_concat=range(2, 6))
# ValLoss:0.152 ValAcc:0.983  ValDice:0.904 ValJc:0.828
layer9_double = Genotype(normal_down=[('avg_pool', 1), ('identity', 0), ('down_dep_conv', 1), ('conv', 2), ('down_dil_conv', 1), ('dil_conv', 3), ('dil_conv', 3), ('dep_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('identity', 0), ('up_dil_conv', 1), ('conv', 2), ('dil_conv', 0), ('dil_conv', 3), ('dep_conv', 2), ('dil_conv', 3), ('dep_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('identity', 0), ('cweight', 1), ('conv', 2), ('conv', 1), ('dil_conv', 3), ('identity', 1), ('dil_conv', 3), ('dep_conv', 4)], normal_normal_concat=range(2, 6))

# stage1 double deep, best
# 450 epoch stage1
#ValLoss:0.676 ValAcc:0.985  ValDice:0.912 ValJc:0.842
stage1_double_deep=Genotype(normal_down=[('down_dep_conv', 1), ('shuffle_conv',0), ('down_dep_conv', 1), ('dil_conv', 0), ('dil_conv', 3), ('shuffle_conv', 0), ('conv', 0), ('down_conv', 1)], normal_down_concat=range(2, 6), normal_up=[('up_dil_conv', 1), ('shuffle_conv', 0), ('dil_conv', 0), ('up_dil_conv', 1), ('dil_conv', 3), ('up_dil_conv',1), ('conv', 0), ('dil_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('shuffle_conv', 0), ('conv', 1), ('dil_conv', 0), ('conv', 2), ('dil_conv', 3), ('shuffle_conv', 0), ('conv', 0), ('dil_conv', 4)], normal_normal_concat=range(2, 6))
# AM ValLoss:0.761 ValAcc:0.982  ValDice:0.893 ValJc:0.814
stage1_deep = Genotype(normal_down=[('down_dil_conv', 1), ('dil_conv', 0), ('down_dep_conv', 1), ('shuffle_conv', 2), ('down_dil_conv', 1), ('dep_conv', 3), ('down_dep_conv', 1), ('identity', 0)], normal_down_concat=range(2, 6), normal_up=[('up_dil_conv', 1), ('dil_conv', 0), ('shuffle_conv', 2), ('up_cweight', 1), ('up_dil_conv', 1), ('dep_conv', 3), ('up_cweight', 1), ('identity', 0)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('dil_conv', 0), ('dep_conv', 1), ('shuffle_conv', 2), ('dep_conv', 3), ('dil_conv', 1), ('cweight', 1), ('identity', 0)], normal_normal_concat=range(2, 6))
# ValLoss:0.179 ValAcc:0.981  ValDice:0.887 ValJc:0.804
stage1_double = Genotype(normal_down=[('down_cweight', 1), ('cweight', 0), ('down_conv', 1), ('dep_conv', 2), ('down_dil_conv', 1), ('shuffle_conv', 0), ('down_cweight', 1), ('dil_conv', 2)], normal_down_concat=range(2, 6), normal_up=[('up_dil_conv', 1), ('cweight', 0), ('dep_conv', 2), ('up_dil_conv', 1), ('up_dil_conv', 1), ('shuffle_conv', 0), ('dil_conv', 2), ('dil_conv', 0)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('cweight', 0), ('identity', 1), ('dep_conv', 2), ('shuffle_conv', 0), ('conv', 3), ('dil_conv', 2), ('identity', 1)], normal_normal_concat=range(2, 6))
#









# refuge test
nodouble_deep_w10_layer7end=Genotype(normal_down=[('avg_pool', 1), ('identity', 0), ('conv', 0), ('identity', 2), ('dil_conv', 3), ('dep_conv', 2), ('dep_conv', 3), ('dep_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('up_dil_conv', 1), ('identity', 0), ('conv', 0), ('up_dep_conv', 1), ('dil_conv', 3), ('dep_conv', 2), ('dep_conv', 3), ('dep_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('identity', 0), ('dep_conv', 1), ('conv', 0), ('identity', 2), ('dil_conv', 3), ('dep_conv', 2), ('dep_conv', 3), ('dep_conv', 4)], normal_normal_concat=range(2, 6))
nodouble_deep_drop02_layer7end= Genotype(normal_down=[('cweight', 0), ('down_conv', 1), ('conv', 0), ('dil_conv', 2), ('shuffle_conv', 3), ('down_conv', 1), ('conv', 3), ('identity', 0)], normal_down_concat=range(2, 6), normal_up=[('cweight', 0), ('up_dil_conv', 1), ('conv', 0), ('dil_conv', 2), ('shuffle_conv', 3), ('identity', 0), ('conv', 3), ('identity', 0)], normal_up_concat=range(2, 6), normal_normal=[('cweight', 0), ('dil_conv', 1), ('conv', 0), ('dil_conv', 2), ('shuffle_conv', 3), ('identity', 0), ('conv', 3), ('identity', 0)], normal_normal_concat=range(2, 6))
nodouble_deep_w20_drop02_layer7end=Genotype(normal_down=[('down_conv', 1), ('cweight', 0), ('conv', 0), ('avg_pool', 1), ('dil_conv', 2), ('cweight', 0), ('conv', 3), ('down_dil_conv', 1)], normal_down_concat=range(2, 6), normal_up=[('up_dil_conv', 1), ('cweight', 0), ('conv', 0), ('up_dep_conv', 1), ('dil_conv', 2), ('cweight', 0), ('conv', 3), ('up_dil_conv', 1)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('cweight', 0), ('conv', 0), ('cweight', 1), ('dil_conv', 2), ('cweight', 0), ('conv', 3), ('identity', 1)], normal_normal_concat=range(2, 6))


stage1_nodouble_deep_a30_w10=Genotype(normal_down=[('identity', 0), ('down_cweight', 1), ('identity', 0), ('down_dep_conv', 1), ('dil_conv', 3), ('down_conv', 1), ('down_dep_conv', 1), ('dil_conv', 0)], normal_down_concat=range(2, 6), normal_up=[('identity', 0), ('up_dep_conv', 1), ('identity', 0), ('dil_conv', 2), ('dil_conv', 3), ('dil_conv', 2), ('dil_conv', 0), ('identity', 4)], normal_up_concat=range(2, 6), normal_normal=[('identity', 0), ('cweight', 1), ('identity', 0), ('dil_conv', 2), ('dil_conv', 3), ('dil_conv', 2), ('dil_conv', 0), ('identity', 4)], normal_normal_concat=range(2, 6))








# chaos datasets
nodouble_deep_init32_ep100=genotype = Genotype(normal_down=[('down_dil_conv', 1), ('shuffle_conv', 0), ('max_pool', 1), ('dil_conv', 2), ('max_pool', 1), ('identity', 3), ('max_pool', 1), ('conv', 3)], normal_down_concat=range(2, 6
), normal_up=[('up_dep_conv', 1), ('shuffle_conv', 0), ('dil_conv', 2), ('up_conv', 1), ('identity', 3), ('cweight', 2), ('up_dep_conv', 1), ('conv', 3)], normal_up_concat=range(2, 6), normal_normal=[('shuffle_conv', 0), (
'identity', 1), ('dil_conv', 2), ('dil_conv', 1), ('identity', 3), ('cweight', 2), ('conv', 3), ('dil_conv', 2)], normal_normal_concat=range(2, 6))

stage1_nodouble_deep_ep36=Genotype(normal_down=[('max_pool', 1), ('dil_conv', 0), ('dep_conv', 2),('down_dep_conv', 1), ('conv', 0), ('down_conv', 1), ('down_dil_conv', 1), ('dil_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('up_dep_conv', 1), ('dil_conv', 0), ('up_dep_conv', 1), ('dep_conv', 2), ('conv', 0), ('up_conv', 1), ('up_conv', 1), ('dil_conv', 4)], normal_up_concat=range(2, 6),normal_normal=[('dep_conv', 1), ('dil_conv', 0), ('cweight', 1), ('dep_conv', 2), ('conv', 0), ('shuffle_conv', 1), ('dil_conv', 1), ('dil_conv', 4)], normal_normal_concat=range(2, 6))

stage1_nodouble_deep_ep63=Genotype(normal_down=[('dil_conv', 0), ('max_pool', 1), ('down_dep_conv', 1), ('identity', 0), ('conv', 0), ('down_conv', 1), ('down_dil_conv', 1), ('identity', 0)], normal_down_concat=range(2,6), normal_up=[('dil_conv', 0), ('up_dep_conv', 1), ('up_dep_conv', 1), ('identity', 0), ('conv', 0), ('up_conv', 1), ('identity', 0), ('dil_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('dil_conv', 0), ('identity', 0), ('identity', 1), ('conv', 0), ('identity', 1), ('identity', 0), ('dil_conv', 1)], normal_normal_concat=range(2, 6))

stage1_nodouble_deep_ep83=Genotype(normal_down=[('dil_conv', 0), ('max_pool', 1), ('identity', 0), ('down_dep_conv', 1), ('down_cweight', 1), ('conv', 0), ('down_dil_conv', 1), ('identity', 0)], normal_down_concat=range(2, 6), normal_up=[('dil_conv', 0), ('up_dep_conv', 1), ('identity', 0), ('up_dep_conv', 1), ('up_conv', 1), ('conv', 0), ('identity', 0), ('dil_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('dil_conv', 0), ('identity', 1), ('identity', 0), ('identity', 1), ('conv', 0), ('identity', 0), ('dil_conv', 1)], normal_normal_concat=range(2, 6))

# 0.854
stage1_nodouble_deep_150awb= Genotype(normal_down=[('dil_conv', 0), ('max_pool', 1), ('identity', 0), ('down_dep_conv', 1), ('dil_conv', 0), ('down_cweight', 1), ('down_dil_conv', 1), ('identity', 0)], normal_down_concat=range(2, 6), normal_up=[('dil_conv', 0), ('up_dep_conv', 1), ('identity', 0), ('up_dep_conv', 1), ('dil_conv', 0), ('up_conv', 1), ('identity', 0), ('dil_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv',1), ('dil_conv', 0), ('identity', 1), ('identity', 0), ('dil_conv', 0), ('shuffle_conv', 1), ('identity', 0), ('dil_conv', 4)], normal_normal_concat=range(2, 6))

stage1_double_deep_ep80= Genotype(normal_down=[('dil_conv', 0), ('avg_pool', 1), ('identity', 0), ('avg_pool', 1), ('avg_pool', 1), ('cweight', 2), ('avg_pool', 1), ('identity', 4)], normal_down_concat=range(2, 6), normal_up=[('identity', 0), ('up_cweight', 1), ('identity', 0), ('up_cweight', 1), ('cweight', 2), ('cweight', 3), ('identity', 4), ('dil_conv', 0)], normal_up_concat=range(2, 6), normal_normal=[('identity', 1), ('identity', 0),('identity', 0), ('cweight', 2), ('cweight', 2), ('cweight', 3), ('cweight', 1), ('identity', 4)], normal_normal_concat=range(2, 6))
stage1_double_deep_ep80_ts=Genotype(normal_down=[('dil_conv', 0), ('max_pool', 1), ('down_dep_conv', 1), ('identity', 0), ('conv', 0), ('down_conv', 1), ('down_dil_conv', 1), ('identity', 0)], normal_down_concat=range(2,6), normal_up=[('dil_conv', 0), ('up_dep_conv', 1), ('up_dep_conv', 1), ('identity', 0), ('conv', 0), ('up_conv', 1), ('identity', 0), ('dil_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('dil_conv', 0), ('identity', 0), ('identity', 1), ('conv', 0), ('identity', 1), ('identity', 0), ('dil_conv', 1)], normal_normal_concat=range(2, 6))






# generalization


# isic mixup train

# alpha=1
alpha1_stage0_ep71=Genotype(normal_down=[('down_dil_conv', 1), ('dep_conv', 0), ('dil_conv', 2), ('down_conv', 1), ('dil_conv', 3), ('down_cweight', 1), ('dil_conv', 4), ('shuffle_conv', 2)], normal_down_concat=range(2, 6), normal_up=[('up_cweight', 1), ('dep_conv', 0), ('dil_conv', 2), ('up_cweight', 1), ('dil_conv', 3), ('shuffle_conv', 2), ('dil_conv', 4), ('shuffle_conv', 2)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 0), ('dil_conv', 1), ('dil_conv', 2), ('dil_conv', 1), ('dil_conv', 3), ('shuffle_conv', 2), ('dil_conv', 4), ('shuffle_conv', 2)], normal_normal_concat=range(2, 6))
alpha0_stage0_ep60= Genotype(normal_down=[('down_dil_conv', 1), ('dep_conv', 0), ('down_conv', 1), ('dil_conv', 2), ('down_cweight', 1), ('dil_conv', 3), ('conv', 2), ('dil_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('dep_conv', 0), ('up_cweight', 1), ('dil_conv', 2), ('up_cweight', 1), ('dil_conv', 3), ('shuffle_conv', 2), ('conv', 2), ('dil_conv', 4)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 0), ('conv', 1), ('dil_conv', 2), ('dep_conv', 1), ('dil_conv', 3), ('shuffle_conv', 2), ('conv', 2), ('dil_conv', 4)], normal_normal_concat=range(2, 6))

alpha1_stage1_double_deep_ep200 =Genotype(normal_down=[('down_cweight', 1), ('shuffle_conv', 0), ('down_conv', 1), ('dil_conv', 2), ('down_dep_conv', 1), ('shuffle_conv', 3), ('conv', 2), ('cweight', 4)], normal_down_concat=range(2, 6), normal_up=[('up_conv', 1), ('shuffle_conv', 0), ('dil_conv', 2), ('up_dil_conv', 1), ('shuffle_conv', 3), ('cweight', 2), ('conv', 2), ('cweight', 4)], normal_up_concat=range(2, 6), normal_normal=[('shuffle_conv', 1), ('shuffle_conv', 0), ('dil_conv', 2), ('shuffle_conv', 0), ('shuffle_conv', 3), ('conv', 1), ('conv', 2), ('cweight', 4)], normal_normal_concat=range(2, 6))
alpha0_stage1_double_deep_ep200= Genotype(normal_down=[('down_dil_conv', 1), ('dep_conv', 0), ('shuffle_conv', 2), ('down_conv', 1), ('dep_conv', 3), ('down_conv', 1), ('down_conv', 1), ('conv', 2)], normal_down_concat=range(2, 6),normal_up=[('up_conv', 1), ('dep_conv', 0), ('shuffle_conv', 2), ('up_cweight', 1), ('dep_conv', 3), ('up_dil_conv', 1), ('conv', 2), ('up_dep_conv', 1)], normal_up_concat=range(2, 6), normal_normal=[('dep_conv', 1), ('dep_conv', 0), ('shuffle_conv', 2), ('dep_conv', 0), ('dep_conv', 3), ('conv', 2), ('conv', 2), ('identity', 4)], normal_normal_concat=range(2, 6))

alpha0_5_stage1_double_deep_ep80=Genotype(normal_down=[('conv', 0), ('down_dil_conv', 1), ('down_dil_conv', 1), ('dil_conv', 2),('down_dil_conv', 1), ('shuffle_conv', 3), ('shuffle_conv', 2), ('shuffle_conv', 3)], normal_down_concat=range(2, 6), normal_up=[('conv', 0), ('up_dil_conv', 1), ('dil_conv', 2), ('shuffle_conv', 0), ('shuffle_conv', 3), ('cweight', 0), ('up_cweight', 1), ('shuffle_conv', 2)], normal_up_concat=range(2, 6), normal_normal=[('shuffle_conv', 1), ('conv', 0), ('dil_conv', 2), ('shuffle_conv', 0), ('shuffle_conv', 3), ('cweight',0), ('shuffle_conv', 1), ('shuffle_conv', 2)], normal_normal_concat=range(2, 6))

