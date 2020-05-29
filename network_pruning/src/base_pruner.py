# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import torch

# default pruning layer type
weighted_modules = [
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'PReLU',
    'Embedding', 'EmbeddingBag',
]


_logger = logging.getLogger(__name__)


# custom layertype
class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__
        self._forward = None


# base compressor class
class Compressor:
    """
    Abstract base PyTorch compressor
    """

    def __init__(self, model, config_list):
        """
        Record necessary info in class members

        Parameters
        ----------
        model : pytorch model
            the model user wants to compress
        config_list : list
            the configurations that users specify for compression
        """
        self.bound_model = model
        self.config_list = config_list
        self.modules_to_compress = None

    def detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                layer = LayerInfo(name, module)
                # get this layer`s pruning config, if none, don`t pruning
                config = self.select_config(layer)
                # this layer add the modules_to_compress list
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress


    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.
        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers
        """
        modules_to_compress = self.detect_modules_to_compress()
        for layer, config in modules_to_compress:
            # do pruning, build new forward func to do mask pruning.
            self._instrument_layer(layer, config)
        # mask pruning model
        return self.bound_model

    def get_modules_to_compress(self):
        """
        To obtain all the to-be-compressed layers.

        Returns
        -------
        list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def select_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.config_list`

        Parameters
        ----------
        layer : LayerInfo
            one layer

        Returns
        -------
        config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        for config in self.config_list:
            config = config.copy()
            # true pruning layer`s op types
            config['op_types'] = self._expand_config_op_types(config)
            if layer.type not in config['op_types']:
                continue
            # op_names may be is none, it means all the op_types`s op will be pruning
            # if op_names is not none, we just poruning the op in op_name list
            if config.get('op_names') and layer.name not in config['op_names']:
                continue
            ret = config
        if ret is None or ret.get('exclude'):
            return None
        return ret

    def update_epoch(self, epoch):
        """
        If user want to update model every epoch, user can override this method.
        This method should be called at the beginning of each epoch

        Parameters
        ----------
        epoch : num
            the current epoch number
        """

    def step(self):
        """
        If user want to update model every step, user can override this method
        """

    def _instrument_layer(self, layer, config):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def _expand_config_op_types(self, config):
        '''

        :param config:    if config inlcude default, we expand it as default_layers.weighted_modules
        :return:
        '''
        if config is None:
            return []
        expanded_op_types = []
        for op_type in config.get('op_types', []):
            if op_type == 'default':
                expanded_op_types.extend(weighted_modules)
            else:
                expanded_op_types.append(op_type)
        return expanded_op_types




class Pruner(Compressor):
    """
    Prune to an exact pruning level specification
    Attributes
    ----------
    mask_dict : dict
        Dictionary for saving masks, `key` should be layer name and
        `value` should be a tensor which has the same shape with layer's weight

    """

    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.mask_dict = {}
        self.pruning_indexs_dict={}



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
        raise NotImplementedError("Pruners must overload calc_mask()")

    def _instrument_layer(self, layer, config):
        """
        Create a wrapper forward function to replace the original one.

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for generating the mask
        """
        assert layer._forward is None, 'Each model can only be compressed once'
        if not _check_weight(layer.module):
            _logger.warning('Module %s does not have parameter "weight"', layer.name)
            return
        layer._forward = layer.module.forward

        def new_forward(*inputs):
            # apply mask to weight
            old_weight = layer.module.weight.data
            mask = self.calc_mask(layer, config)
            layer.module.weight.data = old_weight.mul(mask)
            # calculate forward
            ret = layer._forward(*inputs)
            print("new forward of {}".format(layer.name))
            return ret

        # new forward func
        layer.module.forward = new_forward


    def export_model(self, model_path, mask_path=None, onnx_path=None, input_shape=None):
        """
        Export pruned model weights, masks and onnx model(optional)

        Parameters
        ----------
        model_path : str
            path to save pruned model state_dict
        mask_path : str
            (optional) path to save mask dict
        onnx_path : str
            (optional) path to save onnx model
        input_shape : list or tuple
            input shape to onnx model
        """
        # some layer need to be pruning but we don`t have the pruning info
        if self.detect_modules_to_compress() and not self.mask_dict:
            _logger.warning('You may not use self.mask_dict in base Pruner class to record masks')
        assert model_path is not None, 'model_path must be specified'
        for name, m in self.bound_model.named_modules():
            if name == "":
                continue
            mask = self.mask_dict.get(name)
            if mask is not None:
                mask_sum = mask.sum().item()
                mask_num = mask.numel()
                _logger.info('Layer: %s  Sparsity: %.2f', name, 1 - mask_sum / mask_num)
                m.weight.data = m.weight.data.mul(mask)
            else:
                _logger.info('Layer: %s  NOT compressed', name)
        torch.save(self.bound_model.state_dict(), model_path)
        _logger.info('Model state_dict saved to %s', model_path)
        if mask_path is not None:
            mask_state={
                "mask_dict":self.mask_dict,
                'prune_indexs':self.pruning_indexs_dict
            }
            torch.save(mask_state, mask_path)
            _logger.info('Mask dict saved to %s', mask_path)

        if onnx_path is not None:
            assert input_shape is not None, 'input_shape must be specified to export onnx model'
            # input info needed
            input_data = torch.Tensor(*input_shape)
            torch.onnx.export(self.bound_model, input_data, onnx_path)
            _logger.info('Model in onnx with input shape %s saved to %s', input_data.shape, onnx_path)


def _check_weight(module):
    try:
        return isinstance(module.weight, torch.nn.Parameter) and isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False










