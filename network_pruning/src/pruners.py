# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from .base_pruner import Pruner



logger = logging.getLogger('torch pruner')

# Different pruning algorithms are constructed

class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__
        self._forward = None



# The following  pruning algorithms belong to fixed pruning. Based on the pre-training model, the mask will not change after one calculation
class L1FilterPruner(Pruner):
    '''
    A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.
    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710
    '''
    def __init__(self,model, config_list):
        '''
        Example:
        configure_list = [{
        'sparsity': 0.5,
        'op_types': ['default'],
        #'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
         }]

        '''
        super(L1FilterPruner, self).__init__(model,config_list)
        #  self.mask_dict = {}  key:value  layername:mask
        #  Control dynamic update the pruning rate
        self.mask_calculated_ops=set()



    # the import func to compute the layer`s mask based the layer(layerinfo) and config(dict)
    def calc_mask(self, layer, config):
        """
        Pruners should overload this method to provide mask for weight tensors.
        The mask must have the same shape and type comparing to the weight.
        It will be applied with `mul()` operation on the weight.
        This method is effectively hooked to `forward()` method of the model.

        Parameters
        ----------
        layer : LayerInfo
            calculate mask for `layer`'s weight
        config : dict
            the configuration for generating the mask
        """
        # compute layer`s pruning mask with different pruning methods and design in different pruning class
        # note: add some recoder to avoid repeat compute
        weight=layer.module.weight.data
        op_name=layer.name
        op_type=layer.type
        assert op_type=="Conv2d","L1FilterPruner only supports 2d convolution layer pruning"
        assert op_type in config["op_types"] and (config.get('op_names') is None or op_name in config["op_names"]),\
            "{} not in config op_types or op_names".format(op_name)
        #Mask calculated ops accelerates the retrieval process
        if op_name in self.mask_calculated_ops:
            assert op_name in self.mask_dict,"The op should be in mask_dict"
            return self.mask_dict.get(op_name)
        mask = torch.ones(weight.size()).type_as(weight)
        prune_indexs=[]
        try:
            filters = weight.size(0)
            w_abs = weight.abs()
            k = int(filters * config['sparsity'])
            if k == 0:
                return mask
            w_abs_structured = w_abs.view(filters, -1).sum(dim=1)
            topk_values,prune_indexs = torch.topk(w_abs_structured.view(-1), k, largest=False)
            #keep_indexs=torch.topk(w_abs_structured.view(-1), filters-k, largest=True)[1]
            threshold=topk_values.max()
            mask = torch.gt(w_abs_structured, threshold)[:, None, None, None].expand_as(weight).type_as(weight)
            # print("total_filters:{}  sparsity:{}  pruning_nums:{} threshold:{}".format(filters,config["sparsity"],k,threshold))
            # print("min:{} max:{}".format(w_abs_structured.min(),w_abs_structured.max()))
            # print(prune_indexs.numpy(),len(prune_indexs.numpy()))
            # print(keep_indexs.numpy(),len(keep_indexs.numpy()))
        finally:
            self.mask_dict.update({op_name: mask})
            self.pruning_indexs_dict.update({op_name: prune_indexs})
            self.mask_calculated_ops.add(op_name)
        return mask




# TODO:Before applying the algorithm, sparse training of l1 regularization bn layer is needed
class SlimPruner(Pruner):
    """
    A structured pruning algorithm that prunes channels by pruning the weights of BN layers.
    Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan and Changshui Zhang
    "Learning Efficient Convolutional Networks through Network Slimming", 2017 ICCV
    https://arxiv.org/pdf/1708.06519.pdf

    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        config_list : list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        Example:
        configure_list = [{
        'sparsity': 0.7,
        'op_types': ['BatchNorm2d'],
        }]

        """
        super(SlimPruner, self).__init__(model,config_list)
        # Control dynamic update the pruning rate
        self.mask_calculated_ops = set()

        weight_list = []
        if len(config_list) > 1:
            logger.warning('Slim pruner only supports 1 configuration')
        config = config_list[0]
        assert "BatchNorm2d" in config["op_types"],"Config is wrong!"
        for (layer, config) in self.detect_modules_to_compress():
            assert layer.type == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
            weight_list.append(layer.module.weight.data.abs().clone())
        #print(weight_list)
        all_bn_weights = torch.cat(weight_list)
        k = int(all_bn_weights.shape[0] * config['sparsity'])
        topk_values,prune_index= torch.topk(all_bn_weights.view(-1), k, largest=False)
        self.global_threshold=topk_values.max()
        self.count=0

    def calc_mask(self, layer, config):
        """
        Calculate the mask of given layer.
        Scale factors with the smallest absolute value in the BN layer are masked.
        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            layer's pruning config
        Returns
        -------
        torch.Tensor
            mask of the layer's weight
        """

        weight = layer.module.weight.data
        op_name = layer.name
        op_type = layer.type
        assert op_type == 'BatchNorm2d', 'SlimPruner only supports 2d batch normalization layer pruning'
        if op_name in self.mask_calculated_ops:
            assert op_name in self.mask_dict
            return self.mask_dict.get(op_name)
        mask = torch.ones(weight.size()).type_as(weight)
        prune_index=[]
        try:
            w_abs = weight.abs()
            mask = torch.gt(w_abs, self.global_threshold).type_as(weight)
            prune_index=[index for index in range(w_abs.size(0)) if mask[index]==0]
            #print("layer-->{} sparsity-->{}".format(op_name,len(prune_index)/w_abs.size(0)))
            # print("prune index:{}".format(prune_index))
            # print(prune_index)
            # self.count+=len(prune_index)
            # print(self.count)
        finally:
            self.mask_dict.update({layer.name: mask})
            self.pruning_indexs_dict.update({layer.name: prune_index})
            self.mask_calculated_ops.add(layer.name)
        return mask






# soft pruning with fixed pruning rate based l1/l2 norm.
class SoftPruner(Pruner):
    '''
    A filter pruner via l2/l1 norm.
    "Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks",
    https://arxiv.org/abs/1808.06866
    '''
    def __init__(self,model,config_list):
        '''
        configure_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d']
        'start_epoch':0
        'end_epoch':100
        'frequency':1
        }]
        '''
        super(SoftPruner, self).__init__(model,config_list)
        # Control dynamic update the pruning rate
        # if you want to update the pruning rate and mask, you should make the set() empty.
        self.now_epoch=0
        if len(config_list) > 1:
            logger.warning('soft pruner only supports 1 configuration')
        config = config_list[0]
        assert "start_epoch" in config and 'end_epoch' in config and 'frequency' in config
        self.start_epoch=config['start_epoch']
        self.end_epoch=config["end_epoch"]
        self.frequency=config['frequency']
        assert 0<=self.start_epoch<self.end_epoch
        assert self.frequency>=1


    def calc_mask(self, layer, config):
        """
        Calculate the mask of given layer.
        Scale factors with the smallest absolute value in the BN layer are masked.
        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            layer's pruning config
        Returns
        -------
        torch.Tensor
            mask of the layer's weight
        """
        weight = layer.module.weight.data
        op_name = layer.name
        op_type = layer.type
        assert 0 <= config.get('sparsity') < 1
        # just pruning the conv layer
        assert op_type in ['Conv1d', 'Conv2d']
        assert op_type in config['op_types']

        if (self.now_epoch>=self.start_epoch) and (self.now_epoch<=self.end_epoch) and (self.now_epoch-self.start_epoch)%self.frequency==0:
            # update the mask
            mask=torch.ones(weight.size()).type_as(weight)
            prune_index=[]
            try:
                # calculate the pruning mask based on current sparsity
                filters=weight.size(0)
                k=int(filters*config["sparsity"])
                if k<1 or filters<2:
                    return mask
                # calculate mask
                mask,prune_index=self._calculated_layer_mask(weight.clone(),k)
            finally:
                self.mask_dict.update({op_name: mask})
                self.pruning_indexs_dict.update({op_name: prune_index})
            return mask
        else:
            # get old mask or get init mask(all num is 1)
            if self.now_epoch<self.start_epoch:
                assert op_name not in self.model_dict
            elif self.now_epoch>self.end_epoch:
                assert op_name in self.model_dict
            mask=self.model_dict.get(op_name,torch.ones(weight.size()).type_as(weight))
            return mask




    def _calculated_layer_mask(self,weight,k):
        '''
        Calculate the mask based on weight and k .
        Criterion: l1,l2 norm
        '''
        w_abs=weight.abs().view(weight.size(0),-1)
        # l1 norm
        #w_l1normal=torch.norm(w_abs,p=1,dim=1)
        # l2 norm
        w_l2normal=torch.norm(w_abs,p=2,dim=1)
        topk_values,prune_index=torch.topk(w_l2normal,k,largest=False)
        threshold=topk_values.max()
        mask=torch.gt(w_l2normal,threshold)[:,None,None,None].expand_as(weight).type_as(weight)
        # check
        # print(w_l2normal.numpy())
        # print(topk_values.numpy())
        # print(prune_index)
        # print(w_l2normal.min(),w_l2normal.max(),threshold,k,len(prune_index),w_l2normal.size(0))
        return mask,prune_index


    def update_epoch(self, epoch):
        '''
        Update the mask for different pruning epoch.
        Soft  pruning means that the mask is updated with fixed steps.
        Update epoch is externally controlled
        '''
        #
        self.now_epoch=epoch


    def pruning_grad(self,model):
        '''
        pruning gradient for pruning weight based on mask_dict
        '''
        pass






class FPGMPruner(Pruner):
    """
    A filter pruner via geometric median.
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration",
    https://arxiv.org/pdf/1811.00250.pdf
    """

    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list: list
            support key for each list item:
                - sparsity: percentage of convolutional filters to be pruned.
        configure_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d']
        }]
        """
        super().__init__(model, config_list)
        self.now_epoch=0
        if len(config_list) > 1:
            logger.warning('fpgm pruner only supports 1 configuration')
        config = config_list[0]
        assert "start_epoch" in config and 'end_epoch' in config and 'frequency' in config
        self.start_epoch=config['start_epoch']
        self.end_epoch=config["end_epoch"]
        self.frequency=config['frequency']
        assert self.frequency>=1
        assert 0<=self.start_epoch<self.end_epoch


    def calc_mask(self, layer, config):
        """
        Supports Conv1d, Conv2d
        filter dimensions for Conv1d:
        OUT: number of output channel
        IN: number of input channel
        LEN: filter length
        filter dimensions for Conv2d:
        OUT: number of output channel
        IN: number of input channel
        H: filter height
        W: filter width
        Parameters
        ----------
        layer : LayerInfo
            calculate mask for `layer`'s weight
        config : dict
            the configuration for generating the mask
        """
        #print("use the cal_mask func of {}".format(layer.name))
        weight = layer.module.weight.data
        op_name=layer.name
        op_type=layer.type
        assert 0 <= config.get('sparsity') < 1
        assert op_type in ['Conv1d', 'Conv2d']
        assert op_type in config['op_types']

        if (self.now_epoch>=self.start_epoch) and (self.now_epoch<=self.end_epoch) and (self.now_epoch-self.start_epoch)%self.frequency==0:
            # update mask
            mask = torch.ones(weight.size()).type_as(weight)
            prune_index=[]
            try:
                num_filters = weight.size(0)
                k = int(num_filters * config.get('sparsity'))
                if num_filters < 2 or k < 1:
                    return mask
                prune_index = self._get_min_gm_kernel_idx(weight, k)
                for idx in prune_index:
                    mask[idx] = 0.
            finally:
                self.mask_dict.update({op_name: mask})
                self.pruning_indexs_dict.update({op_name: prune_index})
            return mask
        else:
            # get old mask or get init mask(all num is 1)
            if self.now_epoch<self.start_epoch:
                assert op_name not in self.model_dict
            elif self.now_epoch>self.end_epoch:
                assert op_name in self.model_dict
            mask=self.model_dict.get(op_name,torch.ones(weight.size()).type_as(weight))
            return mask



    def _get_min_gm_kernel_idx(self, weight, k):
        # input weight shape must be 4: O,I,H,W
        assert len(weight.size()) in [3, 4]
        dist_list = []
        for out_i in range(weight.size(0)):
            # The sum of the distance between the ith channel and the other channels
            dist_sum = self._get_distance_sum(weight, out_i)
            dist_list.append((dist_sum, out_i))
        # The smaller the distance, the closer it is to the geometric median, the easier it is to replace
        # topk small
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:k]
        # pruning channel index
        return [x[1] for x in min_gm_kernels]

    def _get_distance_sum(self, weight, out_idx):
        """
        Calculate the total distance between a specified filter (by out_idex and in_idx) and
        all other filters.
        Optimized verision of following naive implementation:
        def _get_distance_sum(self, weight, in_idx, out_idx):
            w = weight.view(-1, weight.size(-2), weight.size(-1))
            dist_sum = 0.
            for k in w:
                dist_sum += torch.dist(k, weight[in_idx, out_idx], p=2)
            return dist_sum
        Parameters
        ----------
        weight: Tensor
            convolutional filter weight
        out_idx: int
            output channel index of specified filter, this method calculates the total distance
            between this specified filter and all other filters.
        Returns
        -------
        float32
            The total distance
        """
        logger.debug('weight size: %s', weight.size())
        assert len(weight.size()) in [3, 4], 'unsupported weight shape'
        w = weight.view(weight.size(0), -1)
        anchor_w = w[out_idx].unsqueeze(0).expand(w.size(0), w.size(1))
        x = w - anchor_w
        x = (x * x).sum(-1)
        x = torch.sqrt(x)
        return x.sum()

    def update_epoch(self, epoch):
        '''
        Update the mask for different pruning epoch.
        Soft  pruning means that the mask is updated with fixed steps.
        Update epoch is externally controlled
        '''
        self.now_epoch=epoch



    def pruning_grad(self,model):
        '''
        pruning gradient for pruning weight based on mask_dict
        '''
        pass





# Soft Gradual pruning
class SoftGradualPruner(Pruner):
    '''
    The pruning strategy, which increased with the epoch, was implemented on the basis of SoftPruner and FPGMPruner.
    '''
    def __init__(self, model, config_list):
        """
        Parameters
        ----------
        model : torch.nn.module
            Model to be pruned
        config_list : list
            List on pruning configs
        Example:
        'op_types': ['Conv2d']
        #'op_names':[]
        'start_epoch':0
        'end_epoch':100
        'frequency':1
        'initial_sparsity'0
        'final_sparsity':0
        'power':2
        'criterion':l2

        """

        super().__init__(model, config_list)
        self.now_epoch = 0
        if len(config_list) > 1:
            logger.warning('fpgm pruner only supports 1 configuration')
        config = config_list[0]
        assert "start_epoch" in config and 'end_epoch' in config and 'frequency' in config
        self.start_epoch=config['start_epoch']
        self.end_epoch=config["end_epoch"]
        self.frequency=config['frequency']
        assert "initial_sparsity" in config and 'final_sparsity' in config and 'power' in config
        self.init_sp=config['initial_sparsity']
        self.end_sp=config['final_sparsity']
        self.power=config['power']
        self.criterion=config['criterion']
        assert 0 <= self.init_sp<=self.end_sp < 1
        assert self.start_epoch<self.end_epoch
        assert self.frequency>=1
        assert self.power in [1,2,3]
        assert self.criterion in ['l1','l2','fpgm']

    def calc_mask(self, layer, config):
        """
        Calculate the mask of given layer
        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            layer's pruning config
        Returns
        -------
        torch.Tensor
            mask of the layer's weight
        """

        weight = layer.module.weight.data
        op_name = layer.name
        op_type=layer.type
        assert op_type in ['Conv1d', 'Conv2d']
        assert op_type in config['op_types']
        assert 0 <= self.init_sp<=self.end_sp < 1
        assert self.start_epoch<self.end_epoch
        assert self.frequency>=1

        if self.now_epoch >= self.start_epoch and self.now_epoch<=self.end_epoch \
                and (self.now_epoch - self.start_epoch) % self.frequency == 0:
            # update mask
            mask=torch.ones(weight.size()).type_as(weight)
            prune_index=[]
            # current epoch`s sparsity, update in every epoch.
            try:
                target_sparsity = self.compute_target_sparsity()
                filters=weight.size(0)
                k=int(filters*target_sparsity)
                if k<1 or  filters<2:
                    return mask
                # criterion: l1,l2,fpgm
                mask,prune_index=self.__calculated_layer_mask(weight.clone(),k,self.criterion)
            finally:
                self.mask_dict.update({op_name: mask})
                self.pruning_indexs_dict.update({op_name: prune_index})
            return mask
        else:
            if self.now_epoch<self.start_epoch:
                assert op_name not in self.model_dict
            elif self.now_epoch>self.end_epoch:
                assert op_name in self.model_dict
            mask=self.model_dict.get(op_name,torch.ones(weight.size()).type_as(weight))
            return mask

    def __calculated_layer_mask(self,weight,pruning_num,criterion):
        '''
        The selection of three criteria
        '''
        assert criterion in ['l1','l2','fpgm']
        assert pruning_num<=weight.size(0)

        if self.criterion=='l1':
            w_abs = weight.abs().view(weight.size(0), -1)
            # l1 norm
            w_l1normal=torch.norm(w_abs,p=1,dim=1)
            topk_values, prune_index = torch.topk(w_l1normal, pruning_num, largest=False)
            threshold = topk_values.max()
            mask = torch.gt(w_l1normal, threshold)[:, None, None, None].expand_as(weight).type_as(weight)

        elif self.criterion=='l2':
            w_abs = weight.abs().view(weight.size(0), -1)
            # l2 norm
            w_l2normal = torch.norm(w_abs, p=2, dim=1)
            topk_values, prune_index = torch.topk(w_l2normal, pruning_num, largest=False)
            threshold = topk_values.max()
            mask = torch.gt(w_l2normal, threshold)[:, None, None, None].expand_as(weight).type_as(weight)

        else:
            prune_index=self._get_min_gm_kernel_idx(weight,pruning_num)
            mask=torch.ones(weight.size()).type_as(weight)
            for idx in prune_index:
                mask[idx]=0.
        return mask, prune_index


    def _get_min_gm_kernel_idx(self, weight, k):
        # input weight shape must be 4: O,I,H,W
        assert len(weight.size()) in [3, 4]
        dist_list = []
        for out_i in range(weight.size(0)):
            # The sum of the distance between the ith channel and the other channels
            dist_sum = self._get_distance_sum(weight, out_i)
            dist_list.append((dist_sum, out_i))
        # The smaller the distance, the closer it is to the geometric median, the easier it is to replace
        # topk small
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:k]
        # pruning channel index
        return [x[1] for x in min_gm_kernels]

    def _get_distance_sum(self, weight, out_idx):
        """
        Calculate the total distance between a specified filter (by out_idex and in_idx) and
        all other filters.
        Optimized verision of following naive implementation:
        def _get_distance_sum(self, weight, in_idx, out_idx):
            w = weight.view(-1, weight.size(-2), weight.size(-1))
            dist_sum = 0.
            for k in w:
                dist_sum += torch.dist(k, weight[in_idx, out_idx], p=2)
            return dist_sum
        Parameters
        ----------
        weight: Tensor
            convolutional filter weight
        out_idx: int
            output channel index of specified filter, this method calculates the total distance
            between this specified filter and all other filters.
        Returns
        -------
        float32
            The total distance
        """
        logger.debug('weight size: %s', weight.size())
        assert len(weight.size()) in [3, 4], 'unsupported weight shape'
        w = weight.view(weight.size(0), -1)
        anchor_w = w[out_idx].unsqueeze(0).expand(w.size(0), w.size(1))
        x = w - anchor_w
        x = (x * x).sum(-1)
        x = torch.sqrt(x)
        return x.sum()


    def compute_target_sparsity(self):
        """
        Calculate the sparsity for pruning
        Parameters
        ----------
        config : dict
            Layer's pruning config
        Returns
        -------
        float
            Target sparsity to be pruned
        """
        span = ((self.end_epoch - self.start_epoch - 1) // self.frequency) * self.frequency
        assert span > 0
        target_sparsity = (self.end_sp +
                           (self.init_sp - self.end_sp) *
                           (  1.0 - ((self.now_epoch - self.start_epoch) / span)) ** self.power )
        return target_sparsity


    def update_epoch(self, epoch):
        """
        Update epoch
        Parameters
        ----------
        epoch : int
            current training epoch
        """
        self.now_epoch = epoch




if __name__=="__main__":
    from unet import U_Net
    model=U_Net(3,1)
    configure_list = [{
        'sparsity': 0.6,
        # 整个模型所有的带参数的op都是待剪枝类型
        'op_types': ['Conv2d'],
        # 对符合类型的所有的层进行修剪
        # 可以在这里专门的指定待剪枝的层,不专门指定的话，默认所有的符合类型要求的层
        #"op_names":["Up_conv2.conv.0"]
        'start_epoch': 0,
        'end_epoch': 100,
        'frequency': 1,


        # 'initial_sparsity':0.,
        # 'final_sparsity': 0.5,
        # 'power': 2,
        # 'criterion': 'l2'
    }]

    pruner=FPGMPruner(model,configure_list)
    model=pruner.compress()
    test_tensor=torch.randn(size=(1,3,128,128))
    output=model(test_tensor)

    print(pruner.mask_dict)
    print(pruner.pruning_indexs_dict)
    #
    # for name,module in model.named_modules():
    #     print(name+"  "+type(module).__name__)
    #
