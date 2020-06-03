
from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')




L7_stage1_v1 = Genotype(cell=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], cell_concat=range(2, 6))

L7_stage1_v2 = Genotype(cell=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], cell_concat=range(2, 6))


L9_epoch40_stage1 = Genotype(cell=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 4), ('dil_conv_3x3', 3)], cell_concat=range(2, 6))
