import torch
import torch.nn as  nn

from operations import *
from genotypes import CellLinkDownPos,CellLinkUpPos,CellPos


class MixedOp (nn.Module):
    def __init__(self, c, stride,mixop_type,switch,dp):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._op_type = mixop_type
        if mixop_type=="down":
            PRIMITIVES=CellLinkDownPos
            assert stride==2 and len(PRIMITIVES)==len(switch),"the mixop type or nums is wrong "
        elif mixop_type=='up':
            PRIMITIVES=CellLinkUpPos
            assert stride==2 and len(PRIMITIVES)==len(switch),"the mixop type or nums is wrong "
        else:
            PRIMITIVES=CellPos
            assert stride==1 and len(PRIMITIVES)==len(switch),"the mixop type or nums is wrong "

        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](c, stride, affine=False, dp=dp)
                self._ops.append(op)


    def forward(self, x, weight_normal, weight_down,weight_up):
        # weight_normal: M * 1 where M is the number of normal primitive operations
        # weight_up_or_down: K * 1 where K is the number of up or down primitive operations
        # Todo: we have three different weights

        if self._op_type == 'down':
            rst = sum(w * op(x) for w, op in zip(weight_down, self._ops))
        elif self._op_type=="up":
            rst = sum(w * op(x) for w, op in zip(weight_up, self._ops))
        else:
            rst = sum(w * op(x) for w, op in zip(weight_normal, self._ops))
        return rst


class Cell(nn.Module):
    #c, stride, mixop_type, switch, dropout_prob
    def __init__(self, meta_node_num, c_prev_prev, c_prev, c,
                 switch_normal,switch_down,switch_up,cell_type,dp=0):
        super(Cell, self).__init__()
        self.c_prev_prev = c_prev_prev
        self.c_prev = c_prev
        self.c = c
        self._meta_node_num = meta_node_num
        self._multiplier = meta_node_num
        self._input_node_num = 2
        self._steps=meta_node_num

        # the sanme feature map size (ck-2)
        if c_prev_prev !=-1:
            self.preprocess0=ConvOps(c_prev_prev,c,kernel_size=1, affine=False, ops_order='act_weight_norm')
        else:
            self.preprocess0=None
        # must be exits!
        self.preprocess1=ConvOps(c_prev,c,kernel_size=1, affine=False, ops_order='act_weight_norm')


        self._ops = nn.ModuleList()
        # cell_type=normal_down,normal_normal,normal_up ,three types
        # c, stride, mixop_type, switch, dropout_prob
        if cell_type=="normal_normal":
            switch_count=0
            for i in range(self._meta_node_num):
                for j in range(self._input_node_num+i):
                    # the first input node is not exists!
                    if c_prev_prev==-1 and j==0:
                        op=None
                    else:
                        op=MixedOp(c,stride=1,mixop_type='normal',switch=switch_normal[switch_count],dp=dp)
                    self._ops.append(op)
                    switch_count+=1
        # input node 0 is normal,1 is down op /up op
        elif cell_type=='normal_down':
            switch_count=0
            for i in range(self._meta_node_num):
                for j in range(self._input_node_num+i):
                    # the first input node is not exists!
                    if c_prev_prev==-1 and j==0:
                        op=None
                    else:
                        if j==1:
                            op = MixedOp(c, stride=2, mixop_type='down', switch=switch_down[switch_count],dp=dp)
                        else:
                            op = MixedOp(c, stride=1, mixop_type='normal', switch=switch_normal[switch_count], dp=dp)

                    self._ops.append(op)
                    switch_count+=1

        elif cell_type=='normal_up':
            switch_count=0
            for i in range(self._meta_node_num):
                for j in range(self._input_node_num+i):
                    # the first input node is not exists!
                    if c_prev_prev==-1 and j==0:
                        op=None
                    else:
                        if j == 1:
                            op = MixedOp(c, stride=2, mixop_type='up', switch=switch_up[switch_count], dp=dp)
                        else:
                            op = MixedOp(c, stride=1, mixop_type='normal', switch=switch_normal[switch_count], dp=dp)
                    self._ops.append(op)
                    switch_count+=1


    def forward(self, s0, s1, weights_normal,weight_down,weights_up):

        if s0 is not None :
            s0 = self.preprocess0 (s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            tmp_list=[]
            for j, h in enumerate(states):
                if h is not None:
                    tmp_list.append(self._ops[offset+j](h,weights_normal[offset+j],weight_down[offset+j],weights_up[offset+j]))
            s = sum(consistent_dim(tmp_list))
            offset += len(states)
            states.append(s)
        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        #return  self.ReLUConvBN (concat_feature)
        return concat_feature

