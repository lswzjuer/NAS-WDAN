import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys

sys.path.append("../")

from cell import Cell
from genotypes import *
from operations import *



class UnetLayer9_v2(nn.Module):
    def __init__(self,input_c=3,c=16,num_classes=1, meta_node_num=4, layers=7,dp=0,
                 use_sharing=True,double_down_channel=True,use_softmax_head=False,
                 switches_normal=[],switches_down=[],switches_up=[]):
        super(UnetLayer9_v2, self).__init__()
        self.CellLinkDownPos=CellLinkDownPos
        self.CellPos=CellPos
        self.CellLinkUpPos=CellLinkUpPos
        self.switches_normal=switches_normal
        self.switches_down=switches_down
        self.switches_up=switches_up
        self.dropout_prob=dp
        self.input_c=input_c
        self.num_class=num_classes
        self.meta_node_num=meta_node_num
        self.layers=layers
        self.use_sharing=use_sharing
        self.double_down_channel=double_down_channel
        self.use_softmax_head=use_softmax_head
        self.depth=(self.layers+1)//2
        self.c_prev_prev=32
        self.c_prev=64
        # 3-->32
        self.stem0 = ConvOps(self.input_c,self.c_prev_prev , kernel_size=3,stride=1,ops_order='weight_norm_act')
        # 32-->64
        self.stem1 = ConvOps(self.c_prev_prev,self.c_prev , kernel_size=3,  stride=2, ops_order='weight_norm_act')

        init_channel=c
        if self.double_down_channel:
            self.layers_channel = [self.meta_node_num * init_channel * pow(2, i) for i in range(0, self.depth)]
            self.cell_channels=[ init_channel*pow(2,i) for i in range(0, self.depth)]
        else:
            self.layers_channel = [self.meta_node_num * init_channel for i in range(0, self.depth)]
            self.cell_channels=[ init_channel for i in range(0, self.depth)]


        for i in range(1, self.layers):
            if i == 1:
                self.cell_1_1 = Cell(self.meta_node_num,
                                     -1, self.c_prev, self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

            elif i == 2:
                self.cell_2_0_0 =Cell(self.meta_node_num,
                                     -1, self.c_prev, self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)
                self.cell_2_0_1 = Cell(self.meta_node_num,
                                     self.c_prev, self.layers_channel[1], self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)

                self.cell_2_2 = Cell(self.meta_node_num,
                                     -1, self.layers_channel[1], self.cell_channels[2],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

            elif i == 3:

                self.cell_3_1_0 = Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

                self.cell_3_1_1 =Cell(self.meta_node_num,
                                     -1, self.layers_channel[1], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_3_1_2 = Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)


                self.cell_3_3 = Cell(self.meta_node_num,
                                     -1, self.layers_channel[2], self.cell_channels[3],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)



            elif i == 4:
                self.cell_4_0_0 =  Cell(self.meta_node_num,
                                     self.c_prev, self.layers_channel[0], self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_4_0_1 =Cell(self.meta_node_num,
                                     self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)

                self.cell_4_2_0 =Cell(self.meta_node_num,
                                     self.layers_channel[2], self.layers_channel[1], self.cell_channels[2],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

                self.cell_4_2_1 =Cell(self.meta_node_num,
                                     -1, self.layers_channel[2], self.cell_channels[2],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_4_2_2 = Cell(self.meta_node_num,
                                     self.layers_channel[2], self.layers_channel[3], self.cell_channels[2],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)


                self.cell_4_4 = Cell(self.meta_node_num,
                                     -1, self.layers_channel[3], self.cell_channels[4],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)



            elif i == 5:

                self.cell_5_1_0 = Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

                self.cell_5_1_1 = Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[1], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_5_1_2 =Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)

                self.cell_5_3_0 = Cell(self.meta_node_num,
                                     self.layers_channel[3], self.layers_channel[2], self.cell_channels[3],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

                self.cell_5_3_1 = Cell(self.meta_node_num,
                                     -1, self.layers_channel[3], self.cell_channels[3],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_5_3_2 = Cell(self.meta_node_num,
                                     self.layers_channel[3], self.layers_channel[4], self.cell_channels[3],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)

            elif i == 6:
                self.cell_6_0_0 =Cell(self.meta_node_num,
                                     self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_6_0_1 = Cell(self.meta_node_num,
                                     self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)



                self.cell_6_2_0 =Cell(self.meta_node_num,
                                     self.layers_channel[2], self.layers_channel[1], self.cell_channels[2],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

                self.cell_6_2_1 = Cell(self.meta_node_num,
                                     self.layers_channel[2], self.layers_channel[2], self.cell_channels[2],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

                self.cell_6_2_2 =Cell(self.meta_node_num,
                                     self.layers_channel[2], self.layers_channel[3], self.cell_channels[2],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)

            elif i == 7:

                self.cell_7_1_0 = Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_down",dp=self.dropout_prob)

                self.cell_7_1_1 = Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[1], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_7_1_2 = Cell(self.meta_node_num,
                                     self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)

            elif i == 8:
                self.cell_8_0_0 = Cell(self.meta_node_num,
                                     self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_normal",dp=self.dropout_prob)

                self.cell_8_0_1 = Cell(self.meta_node_num,
                                     self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                     switch_normal=self.switches_normal,switch_down=self.switches_down,switch_up=self.switches_up,
                                     cell_type="normal_up",dp=self.dropout_prob)


        self.cell_2_0_output =ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1, ops_order='weight')
        self.cell_4_0_output =ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1, ops_order='weight')
        self.cell_6_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1, ops_order='weight')
        self.cell_8_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1, ops_order='weight')


        if self.use_softmax_head:
            self.softmax = nn.LogSoftmax(dim=1)

        self._init_arch_parameters()
        #self._init_weight_parameters()

    def _init_arch_parameters(self):
        '''
        :return:
        '''
        normal_num_ops =np.count_nonzero(self.switches_normal[0])
        down_num_ops = np.count_nonzero(self.switches_down[0])
        up_num_ops = np.count_nonzero(self.switches_up[0])


        k = sum(1 for i in range(self.meta_node_num) for n in range(2 + i))  # total number of input node
        self.alphas_down = nn.Parameter(1e-3 * torch.randn(k, down_num_ops))
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, normal_num_ops))
        self.alphas_up = nn.Parameter(1e-3 * torch.randn(k, up_num_ops))

        self.alphas_network = nn.Parameter(1e-3 * torch.randn(self.layers, self.depth, 3))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alphas' in n:
                self._alphas.append((n, p))

        self._arch_parameters = [
            # cell
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal,
            # network
            self.alphas_network,
        ]

    def _init_weight_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')


    def forward(self, input):
        '''
        :param input:
        :return:
        '''
        _,_,h,w=input.size()
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_down = F.softmax(self.alphas_down, dim=-1)
        weights_up=F.softmax(self.alphas_up, dim=-1)
        network_weight=F.softmax(self.alphas_network, dim=-1)

        # layer 0
        self.stem0_f=self.stem0(input)
        self.stem1_f = self.stem1(self.stem0_f)

        # layer 1
        self.cell_1_1_f = self.cell_1_1(None, self.stem1_f, weights_normal,weights_down,weights_up)


        # layer 2
        self.cell_2_0_f = self.cell_2_0_0(None,self.stem1_f, weights_normal,weights_down,weights_up) * network_weight[2][0][1]/(network_weight[2][0][1]+network_weight[2][0][2]) + \
                          self.cell_2_0_1(self.stem1_f, self.cell_1_1_f, weights_normal,weights_down,weights_up) * network_weight[2][0][2]/(network_weight[2][0][1]+network_weight[2][0][2])


        self.cell_2_2_f = self.cell_2_2(None, self.cell_1_1_f, weights_normal,weights_down,weights_up)


        # layer 3

        self.cell_3_1_f = self.cell_3_1_0(self.cell_1_1_f, self.cell_2_0_f, weights_normal,weights_down,weights_up) * network_weight[3][1][0] + \
                          self.cell_3_1_1(None,self.cell_1_1_f, weights_normal,weights_down,weights_up) * network_weight[3][1][1] + \
                          self.cell_3_1_2(self.cell_1_1_f, self.cell_2_2_f, weights_normal,weights_down,weights_up) * network_weight[3][1][2]

        self.cell_3_3_f = self.cell_3_3(None, self.cell_2_2_f, weights_normal,weights_down,weights_up)



        # layer 4
        self.cell_4_0_f = self.cell_4_0_0(self.stem1_f,self.cell_2_0_f, weights_normal,weights_down,weights_up) * network_weight[4][0][1]/(network_weight[4][0][1]+network_weight[4][0][2]) + \
                          self.cell_4_0_1(self.cell_2_0_f, self.cell_3_1_f, weights_normal,weights_down,weights_up) * network_weight[4][0][2]/(network_weight[4][0][1]+network_weight[4][0][2])


        self.cell_4_2_f = self.cell_4_2_0(self.cell_2_2_f, self.cell_3_1_f, weights_normal,weights_down,weights_up) * network_weight[4][2][0] + \
                          self.cell_4_2_1(None,self.cell_2_2_f, weights_normal,weights_down,weights_up) * network_weight[4][2][1] + \
                          self.cell_4_2_2(self.cell_2_2_f, self.cell_3_3_f, weights_normal,weights_down,weights_up) * network_weight[4][2][2]

        self.cell_4_4_f = self.cell_4_4(None, self.cell_3_3_f, weights_normal,weights_down,weights_up)

        # layer 5

        self.cell_5_1_f = self.cell_5_1_0(self.cell_3_1_f, self.cell_4_0_f, weights_normal,weights_down,weights_up) * network_weight[5][1][0] + \
                          self.cell_5_1_1(self.cell_1_1_f,self.cell_3_1_f, weights_normal,weights_down,weights_up) * network_weight[5][1][1] + \
                          self.cell_5_1_2(self.cell_3_1_f, self.cell_4_2_f, weights_normal,weights_down,weights_up) * network_weight[5][1][2]


        self.cell_5_3_f = self.cell_5_3_0(self.cell_3_3_f, self.cell_4_2_f, weights_normal,weights_down,weights_up) * network_weight[5][3][0] + \
                          self.cell_5_3_1(None,self.cell_3_3_f, weights_normal,weights_down,weights_up) * network_weight[5][3][1] + \
                          self.cell_5_3_2(self.cell_3_3_f, self.cell_4_4_f, weights_normal,weights_down,weights_up) * network_weight[5][3][2]

        # layer 6
        self.cell_6_0_f = self.cell_6_0_0(self.cell_2_0_f,self.cell_4_0_f, weights_normal,weights_down,weights_up) * network_weight[6][0][1]/(network_weight[6][0][1]+network_weight[6][0][2]) + \
                          self.cell_6_0_1(self.cell_4_0_f, self.cell_5_1_f, weights_normal,weights_down,weights_up) * network_weight[6][0][2]/(network_weight[6][0][1]+network_weight[6][0][2])


        self.cell_6_2_f = self.cell_6_2_0(self.cell_4_2_f, self.cell_5_1_f, weights_normal,weights_down,weights_up) * network_weight[6][2][0] + \
                          self.cell_6_2_1(self.cell_2_2_f, self.cell_4_2_f, weights_normal,weights_down,weights_up) * network_weight[6][2][1] + \
                          self.cell_6_2_2(self.cell_4_2_f, self.cell_5_3_f, weights_normal,weights_down,weights_up) * network_weight[6][2][2]

        # layer 7

        self.cell_7_1_f = self.cell_7_1_0(self.cell_5_1_f, self.cell_6_0_f, weights_normal,weights_down,weights_up) * network_weight[7][1][0] + \
                          self.cell_7_1_1(self.cell_3_1_f, self.cell_5_1_f, weights_normal,weights_down,weights_up) * network_weight[7][1][1] + \
                          self.cell_7_1_2(self.cell_5_1_f, self.cell_6_2_f, weights_normal,weights_down,weights_up) * network_weight[7][1][2]
        # layer 8
        self.cell_8_0_f = self.cell_8_0_0(self.cell_4_0_f, self.cell_6_0_f, weights_normal,weights_down,weights_up) * network_weight[8][0][1]/(network_weight[8][0][1]+network_weight[8][0][2]) + \
                          self.cell_8_0_1(self.cell_6_0_f, self.cell_7_1_f, weights_normal,weights_down,weights_up) * network_weight[8][0][2]/(network_weight[8][0][1]+network_weight[8][0][2])

        self.ouput_4_0 = self.cell_4_0_output(self.cell_4_0_f)
        self.ouput_6_0 = self.cell_6_0_output(self.cell_6_0_f)
        self.ouput_8_0 = self.cell_8_0_output(self.cell_8_0_f)
        self.ouput_4_0= F.interpolate(self.ouput_4_0,size=(h,w),mode='bilinear',align_corners=False)
        self.ouput_6_0=F.interpolate(self.ouput_6_0,size=(h,w),mode='bilinear',align_corners=False)
        self.ouput_8_0=F.interpolate(self.ouput_8_0,size=(h,w),mode='bilinear',align_corners=False)

        return [self.ouput_4_0, self.ouput_6_0, self.ouput_8_0]


    def load_alphas(self, alphas_dict):
        self.alphas_down = alphas_dict['alphas_down']
        self.alphas_up = alphas_dict['alphas_up']
        self.alphas_normal = alphas_dict['alphas_normal']
        self.alphas_network=alphas_dict['alphas_network']
        self._arch_parameters = [
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal,
            self.alphas_network
        ]

    def alphas_dict(self):
        return {
            'alphas_down': self.alphas_down,
            'alphas_normal': self.alphas_normal,
            'alphas_up': self.alphas_up,
            'alphas_network': self.alphas_network
        }

    def arch_parameters(self):
        return self._arch_parameters

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'alphas' not in name]


    def decode_network(self):
        '''
        :return: Get the path probability and the largest set of paths
        '''
        # Get path weights
        network_parameters = F.softmax(self.arch_parameters()[1], dim=-1).data.cpu().numpy() * 10
        # Take only valid path branches




    def genotype(self):
        '''
        :return: Get the structure of the cell
        '''
        weight_normal=F.softmax(self.alphas_normal,dim=-1).data.cpu().numpy()
        weight_down=F.softmax(self.alphas_down,dim=-1).data.cpu().numpy()
        weight_up=F.softmax(self.alphas_up,dim=-1).data.cpu().numpy()
        num_mixops = len(weight_normal)
        assert len(self.switches_normal) == num_mixops and len(self.switches_down) == num_mixops and len(self.switches_up) == num_mixops
        # get the normal_down
        keep_num_normal=np.count_nonzero(self.switches_normal[0])
        keep_num_down=np.count_nonzero(self.switches_down[0])
        keep_num_up = np.count_nonzero(self.switches_up[0])
        assert keep_num_normal==len(weight_normal[0]) and keep_num_down==len(weight_down[0]) and keep_num_up==len(weight_up[0])

        normal_down_gen=self.normal_downup_parser(weight_normal.copy(),weight_down.copy(),self.CellLinkDownPos,self.CellPos,
                                                  self.switches_normal,self.switches_down,self.meta_node_num)
        normal_up_gen=self.normal_downup_parser(weight_normal.copy(),weight_up.copy(),self.CellLinkUpPos,self.CellPos,
                                                self.switches_normal,self.switches_up,self.meta_node_num)
        normal_normal_gen= self.parser_normal_old(weight_normal.copy(),self.switches_normal,self.CellPos,self.meta_node_num)

        concat = range(2, self.meta_node_num+2)
        geno_type = Genotype(
            normal_down=normal_down_gen, normal_down_concat = concat,
            normal_up=normal_up_gen, normal_up_concat=concat,
            normal_normal=normal_normal_gen,normal_normal_concat=concat,
        )
        return geno_type

    def normal_downup_parser(self,weight_normal,weight_down,CellLinkDownPos,CellPos,switches_normal,switches_down,meta_node_name):
        # get the normal_down
        normalize_sacle_nd=min(len(weight_normal[0]),len(weight_down[0]))/max(len(weight_normal[0]),len(weight_down[0]))
        down_normalize= True if len(weight_down[0])<len(weight_normal[0]) else False
        normal_down_res=[]
        for i in range(len(weight_normal)):
            if i in [1,3,6,10]:
                if down_normalize:
                    mixop_array=weight_down[i]*normalize_sacle_nd
                else:
                    mixop_array = weight_down[i]
                keep_ops_index=[]
                for j in range(len(CellLinkDownPos)):
                    if switches_down[i][j]:
                        keep_ops_index.append(j)
                max_value, max_index = float(np.max(mixop_array)), int(np.argmax(mixop_array))
                max_index_pri = keep_ops_index[max_index]
                max_op_name=CellLinkDownPos[max_index_pri]
                assert max_op_name!='none'
                normal_down_res.append((max_value,max_op_name))
            else:
                if down_normalize:
                    mixop_array=weight_normal[i]
                else:
                    mixop_array = weight_normal[i]*normalize_sacle_nd
                keep_ops_index=[]
                for j in range(len(CellPos)):
                    if switches_normal[i][j]:
                        keep_ops_index.append(j)
                assert CellPos.index('none')==0
                # Excluding none
                if switches_normal[i][0]:
                    mixop_array[0]=0
                max_value, max_index = float(np.max(mixop_array)), int(np.argmax(mixop_array))
                max_index_pri = keep_ops_index[max_index]
                max_op_name=CellPos[max_index_pri]
                assert max_op_name!='none'
                normal_down_res.append((max_value,max_op_name))
        # get the final cell genotype based in normal_down_res
        # print(normal_down_res)
        n = 2
        start = 0
        normal_down_gen=[]
        for i in range(meta_node_name):
            end=start+n
            node_egdes=normal_down_res[start:end].copy()
            keep_edges=sorted(range(2 + i), key=lambda x: -node_egdes[x][0])[:2]
            for j in keep_edges:
                op_name=node_egdes[j][1]
                normal_down_gen.append((op_name,j))
            start=end
            n+=1
        return normal_down_gen

    def parser_normal_old(self,weights_normal,siwtches_normal,PRIMITIVES,meta_node_num=4):
        num_mixops=len(weights_normal)
        assert len(siwtches_normal)==len(weights_normal),"The mixop num is not right !"
        num_operations =np.count_nonzero(siwtches_normal[0])
        for i in range(num_mixops):
            if siwtches_normal[i][0] == True:
                weights_normal[i][0] = 0
        edge_keep=[]
        for i in range(num_mixops):
            keep_obs=[]
            none_index=PRIMITIVES.index("none")
            for j in range(len(PRIMITIVES)):
                if siwtches_normal[i][j]:
                    keep_obs.append(j)
            # find max operation
            assert len(keep_obs)==num_operations,"The mixop {}`s keep ops is wrong !".format(i)
            # get the max op index and the max value apart from  zero
            max_value,max_index=float(np.max(weights_normal[i])),int(np.argmax(weights_normal[i]))
            max_index_pri=keep_obs[max_index]
            #print("i:{} cur:{} Pro:{} operation:{} max_value:{}".format(i,max_index,max_index_pri,PRIMITIVES[max_index],max_value))
            assert max_index_pri!=none_index,"The none be choose !"
            edge_keep.append((max_value,PRIMITIVES[max_index_pri]))
        # keep two edge for every node
        start = 0
        n = 2
        keep_operations = []
        # 2,3,4,5
        for i in range(meta_node_num):
            end = start + n  # 0~1 2~4 5~8 9~14
            node_values = edge_keep[start:end].copy()
            # The edge num of the ith point is 2+i
            keep_edges = sorted(range(2 + i), key=lambda x: -node_values[x][0])[:2]
            for j in keep_edges:
                keep_op=node_values[j][1]
                keep_operations.append((keep_op, j))
            start=end
            n+=1
        # return
        return keep_operations




if __name__=="__main__":
    normal_num_ops = len(CellPos)
    down_num_ops = len(CellLinkDownPos)
    up_num_ops = len(CellLinkUpPos)

    switches_normal = []
    switches_down = []
    switches_up = []
    for i in range(14):
        single_one=[True for j in range(normal_num_ops)]
        for it in range(normal_num_ops//2):
            single_one[it]=False

        switches_normal.append(single_one)

    for i in range(14):
        single_one=[True for j in range(down_num_ops)]
        for it in range(down_num_ops//2):
            single_one[it]=False

        switches_down.append(single_one)

    for i in range(14):
        single_one=[True for j in range(up_num_ops)]
        for it in range(up_num_ops//2):
            single_one[it]=False
        switches_up.append(single_one)

    model=UnetLayer9_v2(input_c = 3, c = 16, num_classes = 1, meta_node_num = 4, layers = 9, dp = 0,
                        use_sharing = True, double_down_channel = True, use_softmax_head = False,
                        switches_normal = switches_normal, switches_down = switches_down, switches_up = switches_up)
    x = torch.FloatTensor(torch.ones(1, 3, 128, 128))
    ress=model(x)
    # for res in ress:
    #     print(res.size())
    for name,module in model.named_modules():
        print(module.__class__)
    # arch_para=model.arch_parameters()
    # print(model.genotype())


