#coding:utf-8
import torch
import torch.nn as  nn
import torch.nn.functional as F
from genotypes import *
from operations import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(GlobalPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return F.interpolate(pool, (h,w), **self._up_kwargs)


class ConcurrentModule(nn.ModuleList):
    r"""Feed to a list of modules concurrently.
    The outputs of the layers are concatenated at channel dimension.

    Args:
        modules (iterable, optional): an iterable of modules to add
    """
    def __init__(self, modules=None):
        super(ConcurrentModule, self).__init__(modules)

    def forward(self, x):
        outputs = []
        for layer in self:
            outputs.append(layer(x))
        return torch.cat(outputs, 1)

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        if with_global:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       ConcurrentModule([
                                            Identity(),
                                            GlobalPooling(inter_channels, inter_channels,
                                                          norm_layer, self._up_kwargs),
                                       ]),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(2*inter_channels, out_channels, 1))
        else:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                       norm_layer(inter_channels),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class BuildCell(nn.Module):
    #c, stride, mixop_type, switch, dropout_prob
    def __init__(self, genotype, c_prev_prev, c_prev, c,cell_type,dp=0):
        super(BuildCell, self).__init__()
        self.c_prev_prev = c_prev_prev
        self.c_prev = c_prev
        self.c = c
        self.dropout_prob=dp
        self.genotype=genotype

        # the sanme feature map size (ck-2)
        if c_prev_prev !=-1:
            self.preprocess0=ConvOps(c_prev_prev,c,kernel_size=1, affine=True, ops_order='act_weight_norm')
        else:
            self.preprocess0=None
        # must be exits!
        self.preprocess1=ConvOps(c_prev,c,kernel_size=1, affine=True, ops_order='act_weight_norm')

        # cell_type=normal_down,normal_normal,normal_up ,three types
        # c, stride, mixop_type, switch, dropout_prob
        if cell_type=="normal_normal":
            self.op_names, self.idxs = zip(*genotype.normal_normal)
            self.concat = genotype.normal_normal_concat
        elif cell_type=="normal_down":
            self.op_names, self.idxs = zip(*genotype.normal_down)
            self.concat = genotype.normal_down_concat
        else:
            self.op_names, self.idxs = zip(*genotype.normal_up)
            self.concat = genotype.normal_up_concat
        self._compile()

    def _compile(self):
        assert len(self.op_names) == len(self.idxs)
        self.num_meta_node = len(self.op_names) // 2
        self.multiplier = len(self.concat)
        self.ops = nn.ModuleList()
        for name, index in zip(self.op_names, self.idxs):
            op = OPS[name](self.c, None, affine=True, dp=self.dropout_prob)
            self.ops += [op]


    def forward(self, s0, s1):
        '''
        :param s0:  may be is None
        :param s1:
        :return:
        '''
        if s0 is None:
            assert self.preprocess0 is None,"The s0 is None but the Preprocess0 is Not None!"
            s1=self.preprocess1(s1)
        else:
            s0 = self.preprocess0(s0)
            s1 = self.preprocess1(s1)
        # s0 may be is None
        states = [s0, s1]
        for i in range(self.num_meta_node):
            # h1 and h2 may be is none
            h1 = states[self.idxs[2*i]]
            h2 = states[self.idxs[2*i+1]]
            # All 8 op connections must exist
            op1 = self.ops[2*i]
            op2 = self.ops[2*i+1]
            #print(op1.__class__.__name__,op2.__class__.__name__)
            if h1 is None and h2 is None:
                #  h1,h2 cannot be none at the same time,
                #  because an intermediate node cannot have both connections pointing to the 0 node
                raise ValueError("h1 is none and h2 is none ")
            # the size of h1 and h2 may be different, so we need interpolate
            if h1 is not None and h2 is not None:
                #print("h1 is not none and h2 is not none ")
                h1 = op1(h1)
                h2 = op2(h2)
                if h1.size() != h2.size() :
                    #print('h1.size{} and h2.size{}'.format(h1.size(),h2.size()))
                    _, _, height1, width1 = h1.size()
                    _, _, height2, width2 = h2.size()
                    if height1 > height2 or width1 > width2:
                        h2 = F.interpolate(h2, (height1, width1))
                    else:
                        h1 = F.interpolate(h1, (height2, width2))
                s = h1+h2
            elif h1 is not None:
                # h2 is none
                #print("h2 is none ")
                h1=op1(h1)
                s=h1
            else:
                # h1 is none
                #print('h1 is none')
                h2=op2(h2)
                s=h2
            states += [s]
        # for s in states:
        #     if s is not None:
        #         print(s.size())
        return torch.cat([states[i] for i in self.concat], dim=1)



class BuildNasUnet(nn.Module):
    def __init__(self,genotype, input_c=3,c=16,num_classes=1, meta_node_num=4, layers=9,dp=0,
                 use_sharing=True,double_down_channel=True,aux=False):
        super(BuildNasUnet, self).__init__()
        self.dropout_prob=dp
        self.input_c=input_c
        self.num_class=num_classes
        self.meta_node_num=meta_node_num
        self.layers=layers
        self.use_sharing=use_sharing
        self.double_down_channel=double_down_channel
        self.depth=(self.layers+1)//2
        self.c_prev_prev=32
        self.c_prev=64

        # 3-->32
        self.stem0 = ConvOps(input_c,self.c_prev_prev , kernel_size=3,stride=1,ops_order='weight_norm_act')
        # 32-->64
        self.stem1 = ConvOps(self.c_prev_prev,self.c_prev , kernel_size=3,  stride=2, ops_order='weight_norm_act')

        init_channel = c
        if self.double_down_channel:
            self.layers_channel = [self.meta_node_num * init_channel * pow(2, i) for i in range(0, self.depth)]
            self.cell_channels = [init_channel * pow(2, i) for i in range(0, self.depth)]
        else:
            self.layers_channel = [self.meta_node_num * init_channel for i in range(0, self.depth)]
            self.cell_channels = [init_channel for i in range(0, self.depth)]

        for i in range(1, self.layers):
            if i == 1:
                self.cell_1_1 = BuildCell(genotype,
                                     -1, self.c_prev, self.cell_channels[1],
                                     cell_type="normal_down", dp=self.dropout_prob)

            elif i == 2:
                self.cell_2_0_0 = BuildCell(genotype,
                                       -1, self.c_prev, self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)
                self.cell_2_0_1 = BuildCell(genotype,
                                       self.c_prev, self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_2_2 = BuildCell(genotype,
                                     -1, self.layers_channel[1], self.cell_channels[2],
                                     cell_type="normal_down", dp=self.dropout_prob)

            elif i == 3:

                self.cell_3_1_0 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_3_1_1 = BuildCell(genotype,
                                       -1, self.layers_channel[1], self.cell_channels[1],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_3_1_2 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_3_3 = BuildCell(genotype,
                                     -1, self.layers_channel[2], self.cell_channels[3],
                                     cell_type="normal_down", dp=self.dropout_prob)



            elif i == 4:
                self.cell_4_0_0 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_4_0_1 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_4_2_0 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[1], self.cell_channels[2],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_4_2_1 = BuildCell(genotype,
                                       -1, self.layers_channel[2], self.cell_channels[2],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_4_2_2 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[3], self.cell_channels[2],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_4_4 = BuildCell(genotype,
                                     -1, self.layers_channel[3], self.cell_channels[4],
                                     cell_type="normal_down", dp=self.dropout_prob)



            elif i == 5:

                self.cell_5_1_0 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_5_1_1 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[1], self.cell_channels[1],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_5_1_2 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_5_3_0 = BuildCell(genotype,
                                       self.layers_channel[3], self.layers_channel[2], self.cell_channels[3],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_5_3_1 = BuildCell(genotype,
                                       -1, self.layers_channel[3], self.cell_channels[3],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_5_3_2 = BuildCell(genotype,
                                       self.layers_channel[3], self.layers_channel[4], self.cell_channels[3],
                                       cell_type="normal_up", dp=self.dropout_prob)

            elif i == 6:
                self.cell_6_0_0 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_6_0_1 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_6_2_0 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[1], self.cell_channels[2],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_6_2_1 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[2], self.cell_channels[2],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_6_2_2 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[3], self.cell_channels[2],
                                       cell_type="normal_up", dp=self.dropout_prob)

            elif i == 7:

                self.cell_7_1_0 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_7_1_1 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[1], self.cell_channels[1],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_7_1_2 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       cell_type="normal_up", dp=self.dropout_prob)

            elif i == 8:
                self.cell_8_0_0 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_8_0_1 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

        self.cell_2_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_4_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_6_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_8_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')

        self._init_weight_parameters()

    def _init_weight_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if isinstance(module,nn.BatchNorm2d) or isinstance(module,nn.GroupNorm):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)


    def forward(self, network_weight,input):
        '''
        :param input:
        :return:
        '''
        _, _, h, w = input.size()
        # layer 0
        self.stem0_f = self.stem0(input)
        self.stem1_f = self.stem1(self.stem0_f)

        # layer 1
        self.cell_1_1_f = self.cell_1_1(None, self.stem1_f)

        # layer 2
        self.cell_2_0_f = self.cell_2_0_0(None, self.stem1_f) * \
                          network_weight[2][0][1] / (network_weight[2][0][1] + network_weight[2][0][2]) + \
                          self.cell_2_0_1(self.stem1_f, self.cell_1_1_f) * \
                          network_weight[2][0][2] / (network_weight[2][0][1] + network_weight[2][0][2])

        self.cell_2_2_f = self.cell_2_2(None, self.cell_1_1_f)

        # layer 3

        self.cell_3_1_f = self.cell_3_1_0(self.cell_1_1_f, self.cell_2_0_f) * \
                          network_weight[3][1][0] + \
                          self.cell_3_1_1(None, self.cell_1_1_f) * \
                          network_weight[3][1][1] + \
                          self.cell_3_1_2(self.cell_1_1_f, self.cell_2_2_f) * \
                          network_weight[3][1][2]

        self.cell_3_3_f = self.cell_3_3(None, self.cell_2_2_f)

        # layer 4
        self.cell_4_0_f = self.cell_4_0_0(self.stem1_f, self.cell_2_0_f) * \
                          network_weight[4][0][1] / (network_weight[4][0][1] + network_weight[4][0][2]) + \
                          self.cell_4_0_1(self.cell_2_0_f, self.cell_3_1_f * \
                          network_weight[4][0][2] / (network_weight[4][0][1] + network_weight[4][0][2]))

        self.cell_4_2_f = self.cell_4_2_0(self.cell_2_2_f, self.cell_3_1_f) * \
                          network_weight[4][2][0] + \
                          self.cell_4_2_1(None, self.cell_2_2_f) * \
                          network_weight[4][2][1] + \
                          self.cell_4_2_2(self.cell_2_2_f, self.cell_3_3_f) * \
                          network_weight[4][2][2]

        self.cell_4_4_f = self.cell_4_4(None, self.cell_3_3_f)

        # layer 5

        self.cell_5_1_f = self.cell_5_1_0(self.cell_3_1_f, self.cell_4_0_f) * \
                          network_weight[5][1][0] + \
                          self.cell_5_1_1(self.cell_1_1_f, self.cell_3_1_f) * \
                          network_weight[5][1][1] + \
                          self.cell_5_1_2(self.cell_3_1_f, self.cell_4_2_f) * \
                          network_weight[5][1][2]

        self.cell_5_3_f = self.cell_5_3_0(self.cell_3_3_f, self.cell_4_2_f) * \
                          network_weight[5][3][0] + \
                          self.cell_5_3_1(None, self.cell_3_3_f) * \
                          network_weight[5][3][1] + \
                          self.cell_5_3_2(self.cell_3_3_f, self.cell_4_4_f) * \
                          network_weight[5][3][2]

        # layer 6
        self.cell_6_0_f = self.cell_6_0_0(self.cell_2_0_f, self.cell_4_0_f) * \
                          network_weight[6][0][1] / (network_weight[6][0][1] + network_weight[6][0][2]) + \
                          self.cell_6_0_1(self.cell_4_0_f, self.cell_5_1_f) * \
                          network_weight[6][0][2] / (network_weight[6][0][1] + network_weight[6][0][2])

        self.cell_6_2_f = self.cell_6_2_0(self.cell_4_2_f, self.cell_5_1_f) * \
                          network_weight[6][2][0] + \
                          self.cell_6_2_1(self.cell_2_2_f, self.cell_4_2_f) * \
                          network_weight[6][2][1] + \
                          self.cell_6_2_2(self.cell_4_2_f, self.cell_5_3_f) * \
                          network_weight[6][2][2]

        # layer 7

        self.cell_7_1_f = self.cell_7_1_0(self.cell_5_1_f, self.cell_6_0_f) * \
                          network_weight[7][1][0] + \
                          self.cell_7_1_1(self.cell_3_1_f, self.cell_5_1_f) * \
                          network_weight[7][1][1] + \
                          self.cell_7_1_2(self.cell_5_1_f, self.cell_6_2_f) * \
                          network_weight[7][1][2]
        # layer 8
        self.cell_8_0_f = self.cell_8_0_0(self.cell_4_0_f, self.cell_6_0_f) * \
                          network_weight[8][0][1] / (network_weight[8][0][1] + network_weight[8][0][2]) + \
                          self.cell_8_0_1(self.cell_6_0_f, self.cell_7_1_f) * \
                          network_weight[8][0][2] / (network_weight[8][0][1] + network_weight[8][0][2])


        self.ouput_4_0 = self.cell_4_0_output(self.cell_4_0_f)
        self.ouput_6_0 = self.cell_6_0_output(self.cell_6_0_f)
        self.ouput_8_0 = self.cell_8_0_output(self.cell_8_0_f)
        self.ouput_4_0 = F.interpolate(self.ouput_4_0, size=(h, w), mode='bilinear', align_corners=False)
        self.ouput_6_0 = F.interpolate(self.ouput_6_0, size=(h, w), mode='bilinear', align_corners=False)
        self.ouput_8_0 = F.interpolate(self.ouput_8_0, size=(h, w), mode='bilinear', align_corners=False)
        return [self.ouput_4_0, self.ouput_6_0, self.ouput_8_0]



class BuildNasUnetPrune(nn.Module):
    def __init__(self,genotype, input_c=3,c=16,num_classes=1, meta_node_num=4, layers=9,dp=0,
                 use_sharing=True,double_down_channel=True,aux=False):
        super(BuildNasUnetPrune, self).__init__()
        self.dropout_prob=dp
        self.input_c=input_c
        self.num_class=num_classes
        self.meta_node_num=meta_node_num
        self.layers=layers
        self.use_sharing=use_sharing
        self.double_down_channel=double_down_channel
        self.depth=(self.layers+1)//2
        self.c_prev_prev=32
        self.c_prev=64

        # 3-->32
        self.stem0 = ConvOps(input_c,self.c_prev_prev , kernel_size=3,stride=1,ops_order='weight_norm_act')
        # 32-->64
        self.stem1 = ConvOps(self.c_prev_prev,self.c_prev , kernel_size=3,  stride=2, ops_order='weight_norm_act')

        init_channel = c
        if self.double_down_channel:
            self.layers_channel = [self.meta_node_num * init_channel * pow(2, i) for i in range(0, self.depth)]
            self.cell_channels = [init_channel * pow(2, i) for i in range(0, self.depth)]
        else:
            self.layers_channel = [self.meta_node_num * init_channel for i in range(0, self.depth)]
            self.cell_channels = [init_channel for i in range(0, self.depth)]

        for i in range(1, self.layers):
            if i == 1:
                self.cell_1_1 = BuildCell(genotype,
                                     -1, self.c_prev, self.cell_channels[1],
                                     cell_type="normal_down", dp=self.dropout_prob)

            elif i == 2:
                self.cell_2_0_0 = BuildCell(genotype,
                                       -1, self.c_prev, self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)
                self.cell_2_0_1 = BuildCell(genotype,
                                       self.c_prev, self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_2_2 = BuildCell(genotype,
                                     -1, self.layers_channel[1], self.cell_channels[2],
                                     cell_type="normal_down", dp=self.dropout_prob)

            elif i == 3:

                self.cell_3_1_0 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_3_1_1 = BuildCell(genotype,
                                       -1, self.layers_channel[1], self.cell_channels[1],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_3_1_2 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_3_3 = BuildCell(genotype,
                                     -1, self.layers_channel[2], self.cell_channels[3],
                                     cell_type="normal_down", dp=self.dropout_prob)

            elif i == 4:
                self.cell_4_0_0 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_4_0_1 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_4_2_0 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[1], self.cell_channels[2],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_4_2_1 = BuildCell(genotype,
                                       -1, self.layers_channel[2], self.cell_channels[2],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_4_2_2 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[3], self.cell_channels[2],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_4_4 = BuildCell(genotype,
                                     -1, self.layers_channel[3], self.cell_channels[4],
                                     cell_type="normal_down", dp=self.dropout_prob)

            elif i == 5:

                self.cell_5_1_0 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_5_1_1 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[1], self.cell_channels[1],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_5_1_2 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_5_3_0 = BuildCell(genotype,
                                       self.layers_channel[3], self.layers_channel[2], self.cell_channels[3],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_5_3_1 = BuildCell(genotype,
                                       -1, self.layers_channel[3], self.cell_channels[3],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_5_3_2 = BuildCell(genotype,
                                       self.layers_channel[3], self.layers_channel[4], self.cell_channels[3],
                                       cell_type="normal_up", dp=self.dropout_prob)

            elif i == 6:
                self.cell_6_0_0 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_6_0_1 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_6_2_0 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[1], self.cell_channels[2],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_6_2_1 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[2], self.cell_channels[2],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_6_2_2 = BuildCell(genotype,
                                       self.layers_channel[2], self.layers_channel[3], self.cell_channels[2],
                                       cell_type="normal_up", dp=self.dropout_prob)

            elif i == 7:

                self.cell_7_1_0 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_7_1_1 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[1], self.cell_channels[1],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_7_1_2 = BuildCell(genotype,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       cell_type="normal_up", dp=self.dropout_prob)

            elif i == 8:
                self.cell_8_0_0 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_8_0_1 = BuildCell(genotype,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       cell_type="normal_up", dp=self.dropout_prob)

        self.cell_2_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_4_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_6_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_8_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')

        self._init_weight_parameters()

    def _init_weight_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if isinstance(module,nn.BatchNorm2d) or isinstance(module,nn.GroupNorm):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)


    def forward(self, input):
        '''
        :param input:
        :return:
        '''
        _, _, h, w = input.size()
        # layer 0
        self.stem0_f = self.stem0(input)
        self.stem1_f = self.stem1(self.stem0_f)

        # layer 1
        self.cell_1_1_f = self.cell_1_1(None, self.stem1_f)

        # layer 2
        self.cell_2_0_f = self.cell_2_0_0(None, self.stem1_f) + \
                          self.cell_2_0_1(self.stem1_f, self.cell_1_1_f)

        self.cell_2_2_f = self.cell_2_2(None, self.cell_1_1_f)

        # layer 3

        self.cell_3_1_f = self.cell_3_1_0(self.cell_1_1_f, self.cell_2_0_f)  + \
                          self.cell_3_1_1(None, self.cell_1_1_f)  + \
                          self.cell_3_1_2(self.cell_1_1_f, self.cell_2_2_f)

        self.cell_3_3_f = self.cell_3_3(None, self.cell_2_2_f)

        # layer 4
        self.cell_4_0_f = self.cell_4_0_0(self.stem1_f, self.cell_2_0_f)  + \
                          self.cell_4_0_1(self.cell_2_0_f, self.cell_3_1_f)

        self.cell_4_2_f = self.cell_4_2_0(self.cell_2_2_f, self.cell_3_1_f)  + \
                          self.cell_4_2_1(None, self.cell_2_2_f)  + \
                          self.cell_4_2_2(self.cell_2_2_f, self.cell_3_3_f)

        self.cell_4_4_f = self.cell_4_4(None, self.cell_3_3_f)

        # layer 5

        self.cell_5_1_f = self.cell_5_1_0(self.cell_3_1_f, self.cell_4_0_f) + \
                          self.cell_5_1_1(self.cell_1_1_f, self.cell_3_1_f) + \
                          self.cell_5_1_2(self.cell_3_1_f, self.cell_4_2_f)

        self.cell_5_3_f = self.cell_5_3_0(self.cell_3_3_f, self.cell_4_2_f) + \
                          self.cell_5_3_1(None, self.cell_3_3_f) + \
                          self.cell_5_3_2(self.cell_3_3_f, self.cell_4_4_f)

        # layer 6
        self.cell_6_0_f = self.cell_6_0_0(self.cell_2_0_f, self.cell_4_0_f)+ \
                          self.cell_6_0_1(self.cell_4_0_f, self.cell_5_1_f)

        self.cell_6_2_f = self.cell_6_2_0(self.cell_4_2_f, self.cell_5_1_f)+ \
                          self.cell_6_2_1(self.cell_2_2_f, self.cell_4_2_f) + \
                          self.cell_6_2_2(self.cell_4_2_f, self.cell_5_3_f)

        # layer 7

        self.cell_7_1_f = self.cell_7_1_0(self.cell_5_1_f, self.cell_6_0_f) + \
                          self.cell_7_1_1(self.cell_3_1_f, self.cell_5_1_f) + \
                          self.cell_7_1_2(self.cell_5_1_f, self.cell_6_2_f)
        # layer 8
        self.cell_8_0_f = self.cell_8_0_0(self.cell_4_0_f, self.cell_6_0_f)  + \
                          self.cell_8_0_1(self.cell_6_0_f, self.cell_7_1_f)

        self.ouput_4_0 = self.cell_4_0_output(self.cell_4_0_f)
        self.ouput_6_0 = self.cell_6_0_output(self.cell_6_0_f)
        self.ouput_8_0 = self.cell_8_0_output(self.cell_8_0_f)
        self.ouput_4_0 = F.interpolate(self.ouput_4_0, size=(h, w), mode='bilinear', align_corners=False)
        self.ouput_6_0 = F.interpolate(self.ouput_6_0, size=(h, w), mode='bilinear', align_corners=False)
        self.ouput_8_0 = F.interpolate(self.ouput_8_0, size=(h, w), mode='bilinear', align_corners=False)

        return [self.ouput_4_0, self.ouput_6_0, self.ouput_8_0]



if __name__=="__main__":
    from genotypes import *
    layer7_doublechannel = Genotype(
        normal_down=[('down_dil_conv', 1), ('identity', 0), ('conv', 0), ('down_dil_conv', 1), ('shuffle_conv', 0),
                     ('down_conv', 1), ('shuffle_conv', 4), ('down_conv', 1)], normal_down_concat=range(2, 6),
        normal_up=[('up_dil_conv', 1), ('identity', 0), ('up_dep_conv', 1), ('conv', 0), ('shuffle_conv', 0),
                   ('up_dil_conv', 1), ('shuffle_conv', 4), ('up_dep_conv', 1)], normal_up_concat=range(2, 6),
        normal_normal=[('identity', 1), ('identity', 0), ('conv', 0), ('identity', 1), ('shuffle_conv', 0),
                       ('dep_conv', 2), ('shuffle_conv', 4), ('identity', 0)], normal_normal_concat=range(2, 6))

    cell=BuildCell(layer7_doublechannel,c_prev_prev=-1, c_prev=256, c=32,cell_type='normal_up',dp=0)
    # The final output size for normal up may not be a multiple if normal does not exist
    s0=torch.FloatTensor(torch.ones(1, 128, 16, 16))
    s1=torch.FloatTensor(torch.ones(1, 256, 8, 8))
    output=cell(None,s1)
    print(output.size())
    x = torch.FloatTensor(torch.ones(1, 3, 128, 128))
    network=BuildNasUnet(layer7_doublechannel,input_c=3,c=16,num_classes=1, meta_node_num=4, layers=9,dp=0,
                 use_sharing=True,double_down_channel=True)
    print(network)
    output=network(x)
    for o in output:
        print(o.size())

    for name,parameter in network.named_parameters():
        print(name,parameter.size())
    #
    # x = torch.FloatTensor(torch.ones(1, 3, 128, 128))
    # network=BuildNasUnetPrune(layer7_doublechannel,input_c=3,c=16,num_classes=1, meta_node_num=4, layers=9,dp=0,
    #              use_sharing=True,double_down_channel=True)
    # print(network)
    # output=network(x)
    # for o in output:
    #     print(o.size())
    #
    # for name,parameter in network.named_parameters():
    #     print(name,parameter.size())