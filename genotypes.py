from collections import namedtuple
import numpy as np

Genotype = namedtuple('Genotype', 'normal_down normal_down_concat normal_up normal_up_concat normal_normal normal_normal_concat')

CellLinkDownPos = [
    'avg_pool',
    'max_pool',
    'down_cweight',
    'down_dil_conv',
    'down_dep_conv',
    'down_conv'
]

CellLinkUpPos = [
    'up_cweight',
    'up_dep_conv',
    'up_conv',
    'up_dil_conv'
]

CellPos = [
    'none',
    'identity',
    'cweight',
    'dil_conv',
    'dep_conv',
    'shuffle_conv',
    'conv',
]

# class GenoParser:
#     def __init__(self, meta_node_num=4):
#         self._meta_node_num = meta_node_num
#
#     def parse(self, weights1, weights2, cell_type):
#         '''
#         :param weights1:  normal weights
#         :param weights2:  down or up weights
#         :param cell_type: normal_down, normal_up, normal_normal
#         :return:
#         '''
        # gene = []
        # n = 2  # indicate the all candidate index for current meta_node
        # start = 0
        # inp2changedim = 2 if cell_type == 'down' else 1
        # nc, no = weights1.shape
        # for i in range(self._meta_node_num):
        #     normal_op_end = start + n
        #     up_or_down_op_end = start + inp2changedim
        #
        #     mask1 = np.zeros(nc, dtype=bool)
        #     mask2 = np.zeros(nc, dtype=bool)
        #
        #     if cell_type == 'down':
        #         mask1[up_or_down_op_end:normal_op_end] = True
        #         mask2[start:up_or_down_op_end] = True
        #     else:
        #         # middle node connect is normal
        #         mask1[up_or_down_op_end + 1:normal_op_end] = True
        #         # first input  connect is normal
        #         mask1[start:up_or_down_op_end] = True
        #         # second input connet is up
        #         mask2[up_or_down_op_end] = True
        #     # normal weight : dowm is 6,up is 10     xxxx  1
        #     W1 = weights1[mask1].copy()
        #     # up or dwom weight: dowm is 8, up is 4  xxxx   2
        #     W2 = weights2[mask2].copy()
        #     gene_item1, gene_item2 = [], []
        #     # Get the k largest strength of mixed up or down edges, which k = 2
        #     # 找到这个中间节点对应的down/up连接里面概率最大的两条边
        #     if len(W2) >= 1:
        #         edges2 = sorted( range(inp2changedim),
        #                         key=lambda x: -max( W2[x][k] for k in range(len(W2[x])) ) )[:min(len(W2), 2)]
        #         # Get the best operation for up or down operation
        #         CellPrimitive = CellLinkUpPos if cell_type == 'up' else CellLinkDownPos
        #         for j in edges2:
        #             k_best = None
        #             for k in range(len(W2[j])):
        #                 if k_best is None or W2[j][k] > W2[j][k_best]:
        #                     k_best = k
        #             # Geno item: (weight_value, operation, node idx)
        #             gene_item2.append((W2[j][k_best], CellPrimitive[k_best],
        #                                j if cell_type == 'down' else j + 1))
        #
        #     # Get the k largest strength of mixed normal edges, which k = 2
        #     # 找到正常连接里面概率最大的两个连接
        #     if len(W1) > 0:
        #         edges1 = sorted(range(len(W1)), key=lambda x: -max(W1[x][k]
        #                                                            for k in range(len(W1[x])) if
        #                                                            k != CellPos.index('none')))[:min(len(W1), 2)]
        #         # Get the best operation for normal operation
        #         for j in edges1:
        #             k_best = None
        #             for k in range(len(W1[j])):
        #                 if k != CellPos.index('none'):
        #                     if k_best is None or W1[j][k] > W1[j][k_best]:
        #                         k_best = k
        #
        #             # Gene item: (weight_value, operation, node idx)
        #             gene_item1.append((W1[j][k_best], CellPos[k_best],
        #                                0 if j == 0 and cell_type == 'up' else j + inp2changedim))
        #     # normalize the weights value of gene_item1 and gene_item2
        #     if len(W1) > 0 and len(W2) > 0 and len(W1[0]) != len(W2[0]):
        #         normalize_scale = min(len(W1[0]), len(W2[0])) / max(len(W1[0]), len(W2[0]))
        #         if len(W1[0]) > len(W2[0]):
        #             gene_item2 = [(w * normalize_scale, po, fid) for (w, po, fid) in gene_item2]
        #         else:
        #             gene_item1 = [(w * normalize_scale, po, fid) for (w, po, fid) in gene_item1]
        #     # get the final k=2 best edges
        #     gene_item1 += gene_item2
        #     gene += [(po, fid) for (_, po, fid) in sorted(gene_item1)[-2:]]
        #     start = normal_op_end
        #     n += 1
        # return gene


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
# V_isic
stage1_layer9_110epoch_double_deep_final=Genotype(
    normal_down=[('down_dep_conv', 1), ('cweight', 0), ('conv', 2), ('down_dil_conv', 1), ('down_dep_conv', 1),('shuffle_conv', 3), ('conv', 3), ('down_dep_conv', 1)],
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


# 0.920 0.874
stage1_layer9_110epoch_final=Genotype(
    normal_down=[('down_dep_conv', 1), ('conv', 0), ('shuffle_conv', 2), ('down_conv', 1), ('down_conv', 1), ('cweight', 3), ('down_dep_conv', 1), ('conv', 3)],
    normal_down_concat=range(2, 6),
    normal_up=[('conv', 0), ('up_dil_conv', 1), ('shuffle_conv', 2), ('shuffle_conv', 0), ('up_dep_conv', 1), ('cweight', 3), ('conv', 3), ('shuffle_conv', 2)],
    normal_up_concat=range(2, 6),
    normal_normal=[('conv', 1), ('conv', 0), ('shuffle_conv', 2), ('shuffle_conv', 0), ('shuffle_conv', 1), ('cweight', 3), ('conv', 3), ('shuffle_conv', 2)], normal_normal_concat=range(2, 6))





#  CVC genotype
# ValLoss:0.892 ValAcc:0.985  ValDice:0.913 ValJc:0.844
# V_cvc
layer7_double_deep=Genotype(normal_down=[('down_conv', 1), ('cweight', 0), ('dep_conv', 2), ('down_cweight', 1), ('dil_conv', 3), ('down_dil_conv', 1), ('dil_conv', 3), ('dil_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('up_cweight', 1), ('cweight', 0), ('dep_conv', 2), ('up_dil_conv', 1), ('dil_conv', 3), ('up_dil_conv', 1), ('dil_conv', 3), ('up_cweight', 1)], normal_up_concat=range(2, 6), normal_normal=[('dil_conv', 1), ('cweight', 0), ('dep_conv', 2), ('shuffle_conv', 1), ('dil_conv', 3), ('shuffle_conv', 2), ('dil_conv', 3), ('dil_conv', 4)], normal_normal_concat=range(2, 6))
#layer7_double_deep = Genotype(normal_down=[('down_conv', 1), ('cweight', 0), ('dep_conv', 2), ('down_cweight', 1), ('dil_conv', 3), ('down_dil_conv', 1), ('dil_conv', 3), ('dil_conv', 4)], normal_down_concat=range(2, 6), normal_up=[('up_cweight', 1), ('cweight', 0), ('dep_conv', 2), ('up_dil_conv', 1), ('dil_conv', 3), ('up_dil_conv', 1), ('dil_conv', 3), ('up_cweight', 1)], normal_up_concat=range(2, 6), normal_normal=[('dil_conv', 1), ('cweight', 0), ('dep_conv', 2), ('shuffle_conv', 1), ('dil_conv', 3), ('shuffle_conv', 2), ('dil_conv', 3), ('dil_conv', 4)], normal_normal_concat=range(2, 6))
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
